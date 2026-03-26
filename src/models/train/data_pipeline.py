"""Training data pipeline for loading real data from database.

This module provides the TrainingDataPipeline class that orchestrates:
- Reading node panel and underlying bars from database
- Computing forward-looking returns for label construction
- Building train/val/test SurfaceDataset instances via DatasetBuilder
- Creating PyTorch DataLoaders with proper collation

Architecture:
    Follows the orchestrator pattern (like FeatureEngine, IngestionOrchestrator).
    Dependencies are injected via constructor for testability.

Example:
    pipeline = TrainingDataPipeline(
        raw_repo=raw_repo,
        derived_repo=derived_repo,
        config=TrainingDataConfig(splits=splits),
    )
    training_data = pipeline.load()
    result = trainer.train(training_data.train_loader, training_data.val_loader)
"""

import gc
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import pandas as pd
import torch
from loguru import logger
from torch.utils.data import DataLoader

from src.data.storage.repository import DerivedRepository, RawRepository
from src.models.dataset import (
    DatasetBuilder,
    DatasetConfig,
    LabelsConfig,
    SplitsConfig,
    SurfaceDataset,
)
from src.models.train.collate import surface_collate_fn


@dataclass
class TrainingDataConfig:
    """Configuration for the training data pipeline.

    Attributes:
        feature_version: Version string for node_panel features (e.g., "v1.0")
        underlying_symbol: Underlying symbol for returns (e.g., "SPY")
        splits: Train/val/test date split configuration
        labels: Label construction configuration
        batch_size: DataLoader batch size
        num_workers: DataLoader worker count (0 for main process)
        lookback_days: Number of historical periods (consecutive timestamps)
            in each sample window
        normalize_features: Whether to z-score normalize features
        max_label_horizon: Maximum forward horizon in trading days (for buffer)
        underlying_timeframe: Preferred underlying bar timeframe for label
            construction (falls back to "1d" if unavailable).
        feature_cols: Override feature columns (None = DEFAULT_FEATURE_COLS).
            Used by GNN ablation to restrict to node-specific features.
    """

    splits: SplitsConfig
    feature_version: str = "v1.0"
    underlying_symbol: str = "SPY"
    labels: LabelsConfig = field(default_factory=LabelsConfig)
    batch_size: int = 32
    num_workers: int = 4
    lookback_days: int = 21
    normalize_features: bool = True
    max_label_horizon: int = 21
    underlying_timeframe: str = "1m"
    feature_cols: list[str] | None = None


@dataclass
class TrainingData:
    """Result of the training data pipeline.

    Holds DataLoaders and raw datasets for train/val/test splits,
    plus convenience accessors for the shared graph and summary metadata.
    """

    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    train_dataset: SurfaceDataset
    val_dataset: SurfaceDataset
    test_dataset: SurfaceDataset

    @property
    def graph(self):
        """Shared graph topology (from training dataset)."""
        return self.train_dataset.graph

    @property
    def metadata(self) -> dict:
        """Summary metadata about loaded data."""
        return {
            "train_samples": len(self.train_dataset),
            "val_samples": len(self.val_dataset),
            "test_samples": len(self.test_dataset),
            "num_nodes": self.train_dataset.num_nodes,
            "num_features": self.train_dataset.num_features,
            "num_horizons": self.train_dataset.num_horizons,
        }


class TrainingDataPipeline:
    """Orchestrates loading of real data from DB into training-ready DataLoaders.

    Follows the same orchestrator pattern as FeatureEngine and IngestionOrchestrator:
    - Dependencies injected via constructor
    - Single public method (load) that returns a result dataclass
    - All business logic encapsulated in private methods
    """

    def __init__(
        self,
        raw_repo: RawRepository,
        derived_repo: DerivedRepository,
        config: TrainingDataConfig,
    ) -> None:
        self._raw_repo = raw_repo
        self._derived_repo = derived_repo
        self._config = config

    def load(self) -> TrainingData:
        """Load data from DB and build training-ready DataLoaders.

        Steps:
        1. Read node panel (with lookback buffer before train_start)
        2. Read underlying bars (with forward buffer past test_end for labels)
        3. Compute daily returns from close prices
        4. Build train/val/test SurfaceDatasets via DatasetBuilder
        5. Validate no split is empty
        6. Wrap in DataLoaders with surface_collate_fn

        Returns:
            TrainingData with loaders, datasets, graph, and metadata

        Raises:
            ValueError: If node panel or bars are empty, or any split has 0 samples
        """
        splits = self._config.splits

        # Step 1: Read node panel (include lookback buffer before train_start)
        logger.info("Reading node panel from database...")
        panel_start = _subtract_business_days(
            splits.train_start, self._config.lookback_days + 5
        )
        panel_end = splits.test_end
        panel_df = self._read_node_panel(panel_start, panel_end)
        self._log_lookback_context(panel_df)

        # Step 2: Read underlying bars (include forward buffer for labels)
        logger.info("Reading underlying bars from database...")
        buffer_days = self._config.max_label_horizon + 10  # calendar-day buffer
        bars_end = splits.test_end + timedelta(days=buffer_days)
        bars_df = self._read_underlying_bars(splits.train_start, bars_end)

        # Step 3: Compute returns (include close prices for DHR labels)
        logger.info("Computing underlying returns...")
        returns_df = self._compute_returns(bars_df, include_close=True)
        del bars_df  # no longer needed

        # Step 4: Build datasets via DatasetBuilder
        logger.info("Building train/val/test datasets...")
        dataset_kwargs: dict = dict(
            lookback_days=self._config.lookback_days,
            splits=splits,
            normalize_features=self._config.normalize_features,
        )
        if self._config.feature_cols is not None:
            dataset_kwargs["feature_cols"] = self._config.feature_cols
        dataset_config = DatasetConfig(**dataset_kwargs)
        builder = DatasetBuilder(dataset_config)

        train_ds, val_ds, test_ds = builder.build_datasets(
            panel_df=panel_df,
            labels_config=self._config.labels,
            underlying_returns=returns_df,
        )

        # Free source DataFrames — all data now lives in memmap-backed datasets
        del panel_df, returns_df
        gc.collect()

        # Step 5: Validate split sizes
        self._validate_splits(train_ds, val_ds, test_ds)

        # Step 6: Create DataLoaders
        train_loader = self._create_loader(train_ds, shuffle=True)
        val_loader = self._create_loader(val_ds, shuffle=False)
        test_loader = self._create_loader(test_ds, shuffle=False)

        result = TrainingData(
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            train_dataset=train_ds,
            val_dataset=val_ds,
            test_dataset=test_ds,
        )

        meta = result.metadata
        logger.info(
            f"Data loaded: {meta['train_samples']} train, "
            f"{meta['val_samples']} val, "
            f"{meta['test_samples']} test samples "
            f"({meta['num_nodes']} nodes, "
            f"{meta['num_features']} features, "
            f"{meta['num_horizons']} horizons)"
        )

        return result

    def _read_node_panel(self, start: datetime, end: datetime) -> pd.DataFrame:
        """Read node panel from derived repository."""
        panel_df = self._derived_repo.read_node_panel(
            feature_version=self._config.feature_version,
            start=start,
            end=end,
        )

        if panel_df.empty:
            raise ValueError(
                f"No node panel data found for feature_version='{self._config.feature_version}' "
                f"between {start.date()} and {end.date()}. "
                f"Run 'python scripts/build_features.py' first."
            )

        logger.info(
            f"Loaded {len(panel_df)} node panel rows "
            f"({panel_df['ts_utc'].nunique()} unique timestamps)"
        )
        return panel_df

    def _log_lookback_context(self, panel_df: pd.DataFrame) -> None:
        """Log effective lookback interpretation from observed data density."""
        if panel_df.empty:
            return

        ts = pd.to_datetime(panel_df["ts_utc"])
        unique_ts = ts.drop_duplicates()
        per_day = unique_ts.dt.normalize().value_counts()
        if per_day.empty:
            return

        snapshots_per_day = float(per_day.median())
        lookback_periods = self._config.lookback_days
        approx_days = lookback_periods / max(snapshots_per_day, 1.0)

        logger.info(
            f"lookback_days: {lookback_periods} unique timestamps "
            f"(~{approx_days:.1f} trading days at "
            f"median {snapshots_per_day:.0f} snapshots/day)"
        )

    def _read_underlying_bars(self, start: datetime, end: datetime) -> pd.DataFrame:
        """Read underlying bars from raw repository."""
        requested = self._config.underlying_timeframe
        fallback_timeframes = [requested]
        if requested != "1d":
            fallback_timeframes.append("1d")

        bars_df = pd.DataFrame()
        used_timeframe = requested
        for timeframe in fallback_timeframes:
            bars_df = self._raw_repo.read_underlying_bars(
                symbol=self._config.underlying_symbol,
                start=start,
                end=end,
                timeframe=timeframe,
            )
            if not bars_df.empty:
                used_timeframe = timeframe
                break

        if bars_df.empty:
            raise ValueError(
                f"No underlying bars found for {self._config.underlying_symbol} "
                f"between {start.date()} and {end.date()} "
                f"(attempted timeframes: {fallback_timeframes}). "
                f"Run 'python scripts/ingest_raw.py' first."
            )

        if used_timeframe != requested:
            logger.warning(
                f"No {requested} bars found for {self._config.underlying_symbol}; "
                f"falling back to {used_timeframe} bars for label construction."
            )

        logger.info(
            f"Loaded {len(bars_df)} underlying bars "
            f"for {self._config.underlying_symbol} "
            f"(timeframe={used_timeframe})"
        )
        return bars_df

    @staticmethod
    def _compute_returns(bars_df: pd.DataFrame, include_close: bool = False) -> pd.DataFrame:
        """Compute simple returns from underlying bars.

        DatasetBuilder._build_dhr_labels() expects a DataFrame with:
        - ts_utc: timestamp
        - return: simple return between consecutive bars
        - close: underlying close price (required for DHR labels)

        Args:
            bars_df: DataFrame with underlying OHLCV bars (must have ts_utc, close)
            include_close: If True, include close prices in output (for DHR labels)

        Returns:
            DataFrame with ts_utc, return, and optionally close columns
        """
        df = bars_df[["ts_utc", "close"]].copy()
        df["ts_utc"] = pd.to_datetime(df["ts_utc"])
        df = df.sort_values("ts_utc").drop_duplicates(subset=["ts_utc"], keep="last")
        df["return"] = df["close"].pct_change()
        if not include_close:
            df = df.dropna(subset=["return"])
        cols = ["ts_utc", "return"]
        if include_close:
            cols.append("close")
        return df[cols]

    def _create_loader(self, dataset: SurfaceDataset, shuffle: bool) -> DataLoader:
        """Create a DataLoader from a SurfaceDataset."""
        workers = self._config.num_workers
        return DataLoader(
            dataset,
            batch_size=self._config.batch_size,
            shuffle=shuffle,
            num_workers=workers,
            collate_fn=surface_collate_fn,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=workers > 0,
        )

    def _validate_splits(
        self,
        train_ds: SurfaceDataset,
        val_ds: SurfaceDataset,
        test_ds: SurfaceDataset,
    ) -> None:
        """Validate that all splits have samples and usable labels."""
        for name, ds in [("train", train_ds), ("val", val_ds), ("test", test_ds)]:
            if len(ds) == 0:
                raise ValueError(
                    f"The {name} split produced 0 samples. "
                    f"Check that your data covers the configured date range "
                    f"and that lookback_days ({self._config.lookback_days} periods) "
                    f"does not exceed the available timestamps in this split. "
                    f"Use --train-start/--val-start/--test-start/--test-end "
                    f"to override dates, or use --synthetic for testing."
                )

            if ds.num_horizons > 0:
                valid_labels = ds.count_valid_labels()
                if valid_labels == 0:
                    raise ValueError(
                        f"The {name} split has 0 valid labels after masking. "
                        "Check quality masks, split boundaries, and label horizons."
                    )

                total_labels = len(ds) * ds.num_nodes * ds.num_horizons
                coverage = valid_labels / max(total_labels, 1)
                logger.info(
                    f"  {name}: {len(ds)} samples, {valid_labels}/{total_labels} "
                    f"valid labels ({coverage:.2%})"
                )
            else:
                logger.info(f"  {name}: {len(ds)} samples")


def build_splits_from_yaml(yaml_splits: "DatasetSplitsConfig") -> SplitsConfig:
    """Convert YAML 6-date splits config to the 4-date SplitsConfig.

    The YAML config (config/config.yaml) uses 6 dates:
        train_start, train_end, val_start, val_end, test_start, test_end

    The dataset module (SplitsConfig) uses 4 boundary dates:
        train_start, val_start, test_start, test_end

    The splits are contiguous: train runs from train_start to val_start,
    val from val_start to test_start, test from test_start to test_end.

    Args:
        yaml_splits: DatasetSplitsConfig from config/schema.py (date fields)

    Returns:
        SplitsConfig for DatasetBuilder (datetime fields)
    """
    from src.config.schema import DatasetSplitsConfig  # noqa: F811

    return SplitsConfig(
        train_start=datetime.combine(yaml_splits.train_start, datetime.min.time()),
        val_start=datetime.combine(yaml_splits.val_start, datetime.min.time()),
        test_start=datetime.combine(yaml_splits.test_start, datetime.min.time()),
        test_end=datetime.combine(yaml_splits.test_end, datetime.min.time()),
    )


def _subtract_business_days(dt: datetime, n: int) -> datetime:
    """Subtract approximately n business days from a datetime.

    Conservative estimate: subtracts n * 1.5 calendar days to account
    for weekends and holidays.

    Args:
        dt: Starting datetime
        n: Number of business days to subtract

    Returns:
        Datetime approximately n business days before dt
    """
    calendar_days = int(n * 1.5) + 2
    return dt - timedelta(days=calendar_days)
