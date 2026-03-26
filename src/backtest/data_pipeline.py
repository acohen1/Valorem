"""Backtest data pipeline for loading real data and generating signals.

This module provides the BacktestDataPipeline class that orchestrates:
- Loading surface snapshots from database (for BacktestEngine execution)
- Loading node panel features from database (for model inference)
- Reconstructing the trained model from a checkpoint
- Running batch inference to generate trading signals
- Packaging everything into the format BacktestEngine.run_backtest() expects

Architecture:
    Follows the orchestrator pattern (like TrainingDataPipeline, FeatureEngine).
    Dependencies are injected via constructor for testability.

Example:
    pipeline = BacktestDataPipeline(
        raw_repo=raw_repo,
        derived_repo=derived_repo,
        config=BacktestDataConfig(checkpoint_path="artifacts/checkpoints/best_model.pt"),
    )
    data = pipeline.load(start_date=date(2023, 8, 1), end_date=date(2023, 8, 31))
    result = engine.run_backtest(data.surfaces, data.signals_by_date)
"""

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from loguru import logger
from torch.utils.data import DataLoader

from src.config.constants import SurfaceConstants
from src.config.schema import SignalThresholdConfig
from src.data.storage.repository import DerivedRepository, RawRepository
from src.pricing import HistoricalQuoteSource, PositionPricer
from src.models.dataset import (
    DEFAULT_FEATURE_COLS,
    DatasetBuilder,
    DatasetConfig,
    SurfaceDataset,
)
from src.models.ensemble import PatchTST_GNN_Ensemble
from src.models.gnn.model import GNNModelConfig
from src.models.graph import SurfaceGraphConfig, build_surface_graph
from src.models.patchtst.model import PatchTSTModelConfig
from src.models.train.collate import surface_collate_fn
from src.strategy.types import Signal, SignalType


# Wing buckets for signal classification.
# P40/C40 are excluded — they are near-the-money and should fall through
# to the ELEVATED_IV check (high IV + low edge → Iron Condor).
_WING_BUCKETS = frozenset({"P25", "P10", "C10", "C25"})


@dataclass
class BacktestDataConfig:
    """Configuration for the backtest data pipeline.

    Attributes:
        checkpoint_path: Path to trained model checkpoint.
        feature_version: Node panel feature version to load.
        surface_version: Surface snapshot version to load.
        underlying_symbol: Underlying symbol for price data.
        lookback_days: Historical lookback in periods (consecutive timestamps)
            for model inference (must match training).
        batch_size: Batch size for inference DataLoader.
        device: Device for model inference.
        signal_threshold: Thresholds for signal filtering.
    """

    checkpoint_path: str = "artifacts/checkpoints/best_model.pt"
    feature_version: str = "v1.0"
    surface_version: str = "v1.0"
    underlying_symbol: str = "SPY"
    lookback_days: int = 22  # Must match training dataset convention (lookback+1 timestamps)
    batch_size: int = 32
    device: str = "cpu"
    signal_threshold: SignalThresholdConfig = field(
        default_factory=SignalThresholdConfig
    )


@dataclass
class BacktestData:
    """Result of the backtest data pipeline.

    Contains everything needed to call BacktestEngine.run_backtest().
    """

    surfaces: dict[date, pd.DataFrame]
    signals_by_date: dict[date, list[Signal]]
    trading_dates: list[date]
    pricer: PositionPricer


class BacktestDataPipeline:
    """Orchestrates loading real data from DB for backtesting.

    Follows the same orchestrator pattern as TrainingDataPipeline:
    - Dependencies injected via constructor
    - Single public method (load) that returns a result dataclass
    - All business logic encapsulated in private methods
    """

    def __init__(
        self,
        raw_repo: RawRepository,
        derived_repo: DerivedRepository,
        config: BacktestDataConfig,
    ) -> None:
        self._raw_repo = raw_repo
        self._derived_repo = derived_repo
        self._config = config

    def load(self, start_date: date, end_date: date) -> BacktestData:
        """Load data from DB, run inference, and build backtest inputs.

        Steps:
        1. Load surface snapshots → build surfaces dict for the engine
        2. Load underlying bars → merge underlying_price into surfaces
        3. Load node panel features → build SurfaceDataset for inference
        4. Load model from checkpoint → run batch inference
        5. Convert predictions → signals per date
        6. Return BacktestData

        Args:
            start_date: Backtest start date (inclusive).
            end_date: Backtest end date (inclusive).

        Returns:
            BacktestData with surfaces, signals, and trading dates.

        Raises:
            ValueError: If no surface or node panel data is found.
            FileNotFoundError: If checkpoint file does not exist.
        """
        start_dt = datetime.combine(start_date, datetime.min.time())
        # Use end of day for end_date to make it inclusive
        end_dt = datetime.combine(end_date + timedelta(days=1), datetime.min.time())

        # Step 0: Create pricer with historical quote source
        quote_source = HistoricalQuoteSource(self._raw_repo)
        pricer = PositionPricer(quote_source=quote_source)

        # Step 1: Load surface snapshots
        logger.info("Loading surface snapshots from database...")
        surfaces = self._load_surfaces(start_dt, end_dt)

        # Step 2: Load underlying bars and merge price
        logger.info("Loading underlying bars...")
        self._merge_underlying_price(surfaces, start_dt, end_dt)

        # Step 3: Load node panel with lookback buffer for model inference
        logger.info("Loading node panel features...")
        lookback_buffer = timedelta(days=int(self._config.lookback_days * 1.5) + 5)
        panel_start = start_dt - lookback_buffer
        panel_df = self._load_node_panel(panel_start, end_dt)
        self._log_lookback_context(panel_df)

        # DEBUG: Log actual panel date range
        if not panel_df.empty:
            actual_start = pd.to_datetime(panel_df["ts_utc"].min())
            actual_end = pd.to_datetime(panel_df["ts_utc"].max())
            logger.info(
                f"Panel data actually spans {actual_start.date()} to {actual_end.date()} "
                f"(requested from {panel_start.date()})"
            )

        # Step 4: Load model and run inference
        logger.info("Loading model and running inference...")
        checkpoint_path = Path(self._config.checkpoint_path)
        model, feature_cols, feature_stats, label_stats = self._load_model(checkpoint_path)
        self._label_stats = label_stats

        predictions_by_date = self._run_inference(
            model, panel_df, feature_cols, feature_stats, start_date, end_date
        )

        # Step 5: Convert predictions to signals
        logger.info("Converting predictions to signals...")
        signals_by_date = self._predictions_to_signals(predictions_by_date, surfaces)

        # Step 6: Build result
        trading_dates = sorted(
            d for d in surfaces if start_date <= d <= end_date
        )

        logger.info(
            f"Backtest data loaded: {len(trading_dates)} trading days, "
            f"{sum(len(s) for s in signals_by_date.values())} total signals"
        )

        return BacktestData(
            surfaces=surfaces,
            signals_by_date=signals_by_date,
            trading_dates=trading_dates,
            pricer=pricer,
        )

    # ------------------------------------------------------------------
    # Surface loading
    # ------------------------------------------------------------------

    def _load_surfaces(
        self, start_dt: datetime, end_dt: datetime
    ) -> dict[date, pd.DataFrame]:
        """Load surface snapshots from DB and group by date.

        Args:
            start_dt: Start datetime.
            end_dt: End datetime.

        Returns:
            Dict mapping date to surface DataFrame.

        Raises:
            ValueError: If no surface data found.
        """
        surface_df = self._derived_repo.read_surface_snapshots(
            start=start_dt,
            end=end_dt,
            version=self._config.surface_version,
        )

        if surface_df.empty:
            raise ValueError(
                f"No surface snapshots found for version='{self._config.surface_version}' "
                f"between {start_dt.date()} and {end_dt.date()}. "
                f"Run 'python scripts/build_features.py' first."
            )

        # Rename columns to match BacktestEngine expectations
        if "exp_date" in surface_df.columns and "expiry" not in surface_df.columns:
            surface_df = surface_df.rename(columns={"exp_date": "expiry"})

        # Convert expiry from string (SQLite TEXT) to datetime.date
        if "expiry" in surface_df.columns:
            surface_df["expiry"] = pd.to_datetime(surface_df["expiry"]).dt.date

        # Add iv column alias if only iv_mid exists
        if "iv_mid" in surface_df.columns and "iv" not in surface_df.columns:
            surface_df["iv"] = surface_df["iv_mid"]

        # Convert ts_utc to date for grouping
        surface_df["ts_utc"] = pd.to_datetime(surface_df["ts_utc"])
        surface_df["_date"] = surface_df["ts_utc"].dt.date

        # Group by date. If multiple snapshots exist intraday, keep only the
        # latest timestamp for that date so pricing/inference align to one
        # coherent close-like surface.
        surfaces: dict[date, pd.DataFrame] = {}
        for surface_date, group in surface_df.groupby("_date"):
            latest_ts = group["ts_utc"].max()
            daily = group[group["ts_utc"] == latest_ts].copy()
            # Defensive dedupe: should already be unique by (ts_utc, option_symbol, version).
            daily = daily.drop_duplicates(subset=["option_symbol"], keep="last")
            surfaces[surface_date] = daily.drop(columns=["_date"]).reset_index(drop=True)

        logger.info(
            f"Loaded {len(surface_df)} surface rows across {len(surfaces)} trading days"
        )
        return surfaces

    def _merge_underlying_price(
        self,
        surfaces: dict[date, pd.DataFrame],
        start_dt: datetime,
        end_dt: datetime,
    ) -> None:
        """Merge underlying close price into each surface DataFrame.

        Args:
            surfaces: Dict of date → surface DataFrame (modified in place).
            start_dt: Start datetime for bar query.
            end_dt: End datetime for bar query.
        """
        bars_df = self._raw_repo.read_underlying_bars(
            symbol=self._config.underlying_symbol,
            start=start_dt,
            end=end_dt,
            timeframe="1m",
        )

        if bars_df.empty:
            logger.warning(
                f"No underlying bars found for {self._config.underlying_symbol}. "
                f"Surfaces will not have underlying_price column."
            )
            return

        # Build date → close price mapping
        bars_df["ts_utc"] = pd.to_datetime(bars_df["ts_utc"])
        bars_df["_date"] = bars_df["ts_utc"].dt.date
        price_by_date = bars_df.groupby("_date")["close"].last().to_dict()

        for surface_date, surface in surfaces.items():
            if surface_date in price_by_date:
                surface["underlying_price"] = price_by_date[surface_date]

    # ------------------------------------------------------------------
    # Node panel loading
    # ------------------------------------------------------------------

    def _load_node_panel(
        self, start_dt: datetime, end_dt: datetime
    ) -> pd.DataFrame:
        """Load node panel features from DB.

        Args:
            start_dt: Start datetime (includes lookback buffer).
            end_dt: End datetime.

        Returns:
            Node panel DataFrame.

        Raises:
            ValueError: If no data found.
        """
        panel_df = self._derived_repo.read_node_panel(
            feature_version=self._config.feature_version,
            start=start_dt,
            end=end_dt,
        )

        if panel_df.empty:
            raise ValueError(
                f"No node panel data found for feature_version='{self._config.feature_version}' "
                f"between {start_dt.date()} and {end_dt.date()}. "
                f"Run 'python scripts/build_features.py' first."
            )

        logger.info(
            f"Loaded {len(panel_df)} node panel rows "
            f"({panel_df['ts_utc'].nunique()} unique timestamps)"
        )
        return panel_df

    def _log_lookback_context(self, panel_df: pd.DataFrame) -> None:
        """Log effective lookback interpretation from observed panel density."""
        if panel_df.empty:
            return

        ts = pd.to_datetime(panel_df["ts_utc"])
        per_day = ts.dt.normalize().value_counts()
        if per_day.empty:
            return

        bars_per_day = float(per_day.median())
        lookback_periods = self._config.lookback_days
        approx_days = lookback_periods / max(bars_per_day, 1.0)

        if bars_per_day > 1.5:
            logger.warning(
                "lookback_days is interpreted as timestamp periods for inference: "
                f"{lookback_periods} periods (~{approx_days:.2f} trading days at "
                f"median {bars_per_day:.0f} bars/day)"
            )
        else:
            logger.info(
                "lookback_days interpretation for inference: "
                f"{lookback_periods} periods (~{approx_days:.2f} trading days)"
            )

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(
        self, checkpoint_path: Path
    ) -> tuple[PatchTST_GNN_Ensemble, list[str], Optional[dict], Optional[dict]]:
        """Load trained model from a self-describing checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file.

        Returns:
            Tuple of (model, feature_columns, feature_stats, label_stats).

        Raises:
            FileNotFoundError: If checkpoint does not exist.
        """
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        device = self._config.device
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )

        state_dict = checkpoint.get("model_state_dict")
        if state_dict is None:
            raise RuntimeError("Checkpoint does not contain model_state_dict")

        metadata = checkpoint.get("model_metadata")
        if metadata is None:
            raise RuntimeError(
                f"Checkpoint '{checkpoint_path}' lacks model_metadata. "
                f"Retrain with 'python scripts/train_model.py --env dev' "
                f"to produce a self-describing checkpoint."
            )

        logger.info("Loading model from checkpoint metadata")
        patchtst_config = PatchTSTModelConfig(**metadata["patchtst_config"])
        gnn_config = GNNModelConfig(**metadata["gnn_config"])
        input_dim = metadata.get("input_dim", 13)
        output_horizons = metadata.get("output_horizons", 3)
        feature_columns = metadata.get("feature_columns", list(DEFAULT_FEATURE_COLS))
        feature_stats = metadata.get("feature_stats")
        label_stats = metadata.get("label_stats")
        volume_feature_idx = None
        if gnn_config.use_dynamic_volume_edges:
            if "log_volume" in feature_columns:
                volume_feature_idx = feature_columns.index("log_volume")
            else:
                logger.warning(
                    "Checkpoint enables dynamic volume edges but feature_columns "
                    "does not include 'log_volume'; dynamic edge augmentation "
                    "will be disabled for inference."
                )

        has_internal_edges = "_edge_index" in state_dict or "_edge_attr" in state_dict
        model_graph = build_surface_graph() if has_internal_edges else None

        model = PatchTST_GNN_Ensemble(
            patchtst_config=patchtst_config,
            gnn_config=gnn_config,
            input_dim=input_dim,
            output_horizons=output_horizons,
            graph=model_graph,
            volume_feature_idx=volume_feature_idx,
        )

        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()

        logger.info(
            f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters"
        )

        return model, feature_columns, feature_stats, label_stats

    # ------------------------------------------------------------------
    # Batch inference
    # ------------------------------------------------------------------

    def _run_inference(
        self,
        model: PatchTST_GNN_Ensemble,
        panel_df: pd.DataFrame,
        feature_columns: list[str],
        feature_stats: Optional[dict],
        start_date: date,
        end_date: date,
    ) -> dict[date, dict[tuple[int, str], float]]:
        """Run batch inference over the backtest period.

        Builds a SurfaceDataset (same code path as training) and iterates
        through it with the model in eval mode.

        Args:
            model: Trained model.
            panel_df: Node panel DataFrame (with lookback buffer).
            feature_columns: Feature column names.
            feature_stats: Normalization stats from training (mean, std per feature).
            start_date: Backtest start date.
            end_date: Backtest end date.

        Returns:
            Dict mapping date to {(tenor, bucket): prediction} dicts.
        """
        device = self._config.device

        # Build inference dataset using the same code as training
        dataset_config = DatasetConfig(
            lookback_days=self._config.lookback_days,
            feature_cols=list(feature_columns),
            label_cols=[],  # No labels for inference
            normalize_features=feature_stats is not None,
        )
        builder = DatasetBuilder(dataset_config)
        dataset = builder.build_single_dataset(
            panel_df=panel_df,
            feature_stats=feature_stats,
        )

        if len(dataset) == 0:
            logger.warning("Inference dataset has 0 samples")
            return {}

        # Build graph metadata (index mapping always needed for node -> (tenor, bucket))
        graph = build_surface_graph()
        index_mapping = graph.index_mapping

        # If model checkpoint carries internal edges (including learnable edge
        # attributes), do not override them with a fresh static graph at inference.
        use_internal_edges = getattr(model, "_edge_index", None) is not None
        if use_internal_edges:
            edge_index = None
            edge_attr = None
        else:
            edge_index = graph.edge_index.to(device)
            edge_attr = graph.edge_attr.to(device) if graph.edge_attr is not None else None

        # Create DataLoader for batch inference
        loader = DataLoader(
            dataset,
            batch_size=self._config.batch_size,
            shuffle=False,
            collate_fn=surface_collate_fn,
        )

        # Map sample index -> end timestamp (and date) via dataset internals.
        sample_end_ts: list[pd.Timestamp] = []
        for idx in range(len(dataset)):
            _, end_idx = dataset._samples[idx]
            sample_end_ts.append(pd.Timestamp(dataset._unique_timestamps[end_idx]))

        # DEBUG: Log sample date range
        if sample_end_ts:
            sample_dates = [ts.date() for ts in sample_end_ts]
            logger.info(
                f"Sample dates range from {min(sample_dates)} to {max(sample_dates)} "
                f"({len(sample_dates)} total samples)"
            )
            in_range = sum(1 for d in sample_dates if start_date <= d <= end_date)
            logger.info(
                f"Samples within backtest period ({start_date} to {end_date}): {in_range}"
            )
        else:
            logger.warning("No sample dates generated!")

        # Run inference
        latest_by_date: dict[date, tuple[pd.Timestamp, dict[tuple[int, str], float]]] = {}
        sample_offset = 0

        with torch.no_grad():
            for batch in loader:
                X = batch["X"].to(device)
                mask = batch["mask"].to(device)
                batch_size = X.size(0)

                if use_internal_edges:
                    output = model(X, mask=mask)
                else:
                    output = model(X, edge_index, edge_attr, mask)

                # output: (batch, nodes, horizons) — take first horizon
                if output.dim() == 3:
                    output = output[:, :, 0]

                preds = output.cpu().numpy()  # (batch, nodes)
                masks = mask.cpu().numpy()  # (batch, nodes)

                for i in range(batch_size):
                    sample_idx = sample_offset + i
                    if sample_idx >= len(sample_end_ts):
                        break

                    sample_ts = sample_end_ts[sample_idx]
                    pred_date = sample_ts.date()

                    # Only keep predictions within the backtest period
                    if start_date <= pred_date <= end_date:
                        node_preds: dict[tuple[int, str], float] = {}
                        for node_idx in range(preds.shape[1]):
                            if not masks[i, node_idx]:
                                continue
                            if node_idx in index_mapping:
                                tenor, bucket = index_mapping[node_idx]
                                pred_val = float(preds[i, node_idx])
                                if not np.isfinite(pred_val):
                                    continue
                                node_preds[(tenor, bucket)] = pred_val
                        prev = latest_by_date.get(pred_date)
                        if prev is None or sample_ts >= prev[0]:
                            latest_by_date[pred_date] = (sample_ts, node_preds)

                sample_offset += batch_size

        predictions_by_date = {
            d: latest_by_date[d][1] for d in sorted(latest_by_date)
        }
        logger.info(
            f"Inference complete: {len(predictions_by_date)} days with predictions"
        )
        return predictions_by_date

    # ------------------------------------------------------------------
    # Signal conversion
    # ------------------------------------------------------------------

    def _predictions_to_signals(
        self,
        predictions_by_date: dict[date, dict[tuple[int, str], float]],
        surfaces: dict[date, pd.DataFrame],
    ) -> dict[date, list[Signal]]:
        """Convert model predictions to trading signals.

        Applies threshold filtering and signal type classification.

        Args:
            predictions_by_date: Per-date node predictions.
            surfaces: Per-date surface DataFrames (for IV context).

        Returns:
            Dict mapping date to list of Signal objects.
        """
        threshold = self._config.signal_threshold
        signals_by_date: dict[date, list[Signal]] = {}

        # Denormalize predictions if label stats are available.
        # Model outputs are in z-score space; convert to raw DHR scale.
        label_stats = getattr(self, "_label_stats", None)
        if label_stats:
            first_label = list(label_stats.keys())[0]
            mean, std = label_stats[first_label]
            predictions_by_date = {
                d: {k: v * std + mean for k, v in preds.items()}
                for d, preds in predictions_by_date.items()
            }

        for pred_date, node_preds in predictions_by_date.items():
            signals: list[Signal] = []
            surface = surfaces.get(pred_date)
            ts = datetime.combine(pred_date, datetime.min.time(), tzinfo=timezone.utc)

            # Build set of tradeable nodes for O(1) lookup
            if surface is not None and not surface.empty:
                available_nodes: set[tuple[int, str]] | None = set(
                    zip(surface["tenor_days"], surface["delta_bucket"])
                )
            else:
                available_nodes = None

            for (tenor, bucket), edge in node_preds.items():
                if not np.isfinite(edge):
                    continue

                # Apply edge threshold
                if abs(edge) < threshold.min_edge:
                    continue

                # Compute confidence from prediction magnitude
                confidence = min(1.0, abs(edge) / 0.1)

                if confidence < threshold.min_confidence:
                    continue

                # Skip signals for nodes absent from the day's surface
                if available_nodes is not None and (tenor, bucket) not in available_nodes:
                    logger.debug(
                        f"Skipping signal at {tenor}d {bucket} — no surface data"
                    )
                    continue

                # Classify signal type
                signal_type = self._classify_signal(edge, tenor, bucket, surface)

                signals.append(
                    Signal(
                        signal_type=signal_type,
                        edge=edge,
                        confidence=confidence,
                        tenor_days=tenor,
                        delta_bucket=bucket,
                        timestamp=ts,
                    )
                )

            if signals:
                signals_by_date[pred_date] = signals

        return signals_by_date

    @staticmethod
    def _classify_signal(
        edge: float,
        tenor: int,
        bucket: str,
        surface: Optional[pd.DataFrame],
    ) -> SignalType:
        """Classify signal type based on prediction and surface context.

        Args:
            edge: Predicted edge value.
            tenor: Tenor in days.
            bucket: Delta bucket name.
            surface: Surface DataFrame for IV context (may be None).

        Returns:
            Appropriate SignalType.
        """
        # Get IV context if available
        iv = 0.20  # default
        if surface is not None:
            node_data = surface[
                (surface["tenor_days"] == tenor)
                & (surface["delta_bucket"] == bucket)
            ]
            if not node_data.empty:
                for col in ("iv_mid", "iv"):
                    if col in node_data.columns:
                        iv = node_data[col].iloc[0]
                        break

        if bucket == "ATM":
            if tenor <= 14:
                return SignalType.TERM_ANOMALY
            else:
                return SignalType.DIRECTIONAL_VOL
        elif bucket in _WING_BUCKETS:
            return SignalType.SKEW_ANOMALY
        else:
            if iv > 0.30 and abs(edge) < 0.03:
                return SignalType.ELEVATED_IV
            else:
                return SignalType.DIRECTIONAL_VOL
