"""PyTorch datasets for volatility surface learning.

This module provides:
- SurfaceDataset: PyTorch Dataset for node panel time series
- DatasetBuilder: Builds train/val/test datasets with proper splits
"""

import gc
import shutil
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

from src.models.graph import (
    SurfaceGraphConfig,
    build_surface_graph,
    DEFAULT_DELTA_BUCKETS,
    DEFAULT_TENORS_DAYS,
)


# Default feature columns (matching node_panel schema)
# NOTE: iv_mid/iv_bid/iv_ask are deliberately excluded — they live in
# surface_snapshots only. iv_change_1d/5d (differenced) are safe: they
# capture IV dynamics without leaking the level.
DEFAULT_FEATURE_COLS = [
    # Cross-sectional (surface structure — shape, not level)
    "delta",
    "gamma",
    "vega",
    "theta",
    "spread_pct",
    "skew_slope",
    "term_slope",
    "curvature",
    # IV dynamics (differenced — no level leakage)
    "iv_change_1d",
    "iv_change_5d",
    # Vol-of-vol (rolling std of IV — predicts DHR dispersion)
    "iv_vol_5d",
    "iv_vol_10d",
    "iv_vol_21d",
    # IV richness (z-score vs rolling mean — mean-reversion signal)
    "iv_zscore_5d",
    "iv_zscore_10d",
    "iv_zscore_21d",
    # Global (realized vol)
    "underlying_rv_5d",
    "underlying_rv_10d",
    "underlying_rv_21d",
    # Macro (regime context)
    "VIXCLS_level",
    "VIXCLS_change_1w",
    "DGS10_level",
    "DGS10_change_1w",
    "DGS2_level",
    "DGS2_change_1w",
    # Volume / liquidity
    "log_volume",
    "volume_ratio_5d",
    "log_oi",
    "oi_change_5d",
]

# Node-specific features for GNN-only ablation.
#
# Strips 9 global features that are identical across all 42 surface nodes
# at each timestamp: realized vol (3) and macro (6). Through message passing,
# the GNN trivially denoises these by averaging 42 identical copies, giving
# it an unfair advantage over PatchTST which processes nodes independently.
# Removing them makes the ablation comparison valid.
# 20 features = 29 (DEFAULT_FEATURE_COLS) - 3 (underlying_rv_*) - 6 (macro)
GNN_ABLATION_FEATURE_COLS = [
    # Cross-sectional (surface structure)
    "delta",
    "gamma",
    "vega",
    "theta",
    "spread_pct",
    "skew_slope",
    "term_slope",
    "curvature",
    # IV dynamics (differenced)
    "iv_change_1d",
    "iv_change_5d",
    # Vol-of-vol
    "iv_vol_5d",
    "iv_vol_10d",
    "iv_vol_21d",
    # IV richness (z-score)
    "iv_zscore_5d",
    "iv_zscore_10d",
    "iv_zscore_21d",
    # Volume / liquidity
    "log_volume",
    "volume_ratio_5d",
    "log_oi",
    "oi_change_5d",
]
assert len(GNN_ABLATION_FEATURE_COLS) == 20  # noqa: S101

# Default label columns (delta-hedged returns)
DEFAULT_LABEL_COLS = ["y_dhr_5d", "y_dhr_10d", "y_dhr_21d"]


@dataclass
class SplitsConfig:
    """Configuration for train/val/test date splits.

    Attributes:
        train_start: Start date for training data
        val_start: Start date for validation (end of training)
        test_start: Start date for test (end of validation)
        test_end: End date for test data
    """

    train_start: datetime
    val_start: datetime
    test_start: datetime
    test_end: datetime

    def __post_init__(self):
        """Validate chronological ordering."""
        if not (self.train_start < self.val_start < self.test_start < self.test_end):
            raise ValueError("Dates must be chronologically ordered: train_start < val_start < test_start < test_end")


@dataclass
class LabelsConfig:
    """Configuration for label construction.

    Attributes:
        horizons_days: Prediction horizons in trading days

    Labels are delta-hedged returns (DHR) — per-node P&L of a delta-hedged
    option position. Has genuine cross-sectional variation because different
    nodes have different gamma exposure.
    Ref: Bakshi & Kapadia (RFS 2003), Bali et al. (RFS 2023).
    """

    horizons_days: list[int] = field(default_factory=lambda: [5, 10, 21])


@dataclass
class DatasetConfig:
    """Configuration for dataset construction.

    Attributes:
        lookback_days: Number of historical periods (consecutive timestamps)
            in each sample window
        feature_cols: List of feature column names
        label_cols: List of label column names
        splits: Train/val/test date boundaries
        graph: Graph configuration
        normalize_features: Whether to z-score normalize features
    """

    lookback_days: int = 21
    feature_cols: list[str] = field(default_factory=lambda: DEFAULT_FEATURE_COLS.copy())
    label_cols: list[str] = field(default_factory=lambda: DEFAULT_LABEL_COLS.copy())
    splits: Optional[SplitsConfig] = None
    graph: SurfaceGraphConfig = field(default_factory=SurfaceGraphConfig)
    normalize_features: bool = True


class SurfaceDataset(Dataset):
    """PyTorch dataset for volatility surface node panel.

    Creates sliding window samples from node panel data. Each sample contains:
    - X: Feature tensor of shape (lookback_days + 1, num_nodes, num_features)
    - y: Label tensor of shape (num_nodes, num_horizons)
    - mask: Boolean tensor of shape (num_nodes,) indicating valid nodes
    - graph: Static graph data (same for all samples)

    The dataset handles:
    - Temporal windowing with configurable lookback
    - Node ordering consistent with graph topology
    - Missing node handling via masking
    """

    def __init__(
        self,
        panel_df: pd.DataFrame,
        feature_cols: list[str],
        label_cols: list[str],
        lookback_days: int,
        graph: Optional[Data] = None,
        graph_config: Optional[SurfaceGraphConfig] = None,
        normalize_features: bool = False,
        feature_stats: Optional[dict[str, tuple]] = None,
        normalize_labels: bool = True,
        label_stats: Optional[dict[str, tuple[float, float]]] = None,
        min_sample_end_ts: Optional[datetime] = None,
        use_memmap: bool = True,
    ):
        """Initialize dataset from node panel DataFrame.

        Args:
            panel_df: DataFrame with node panel data. Must have columns:
                - ts_utc: Timestamp
                - tenor_days: Tenor bin
                - delta_bucket: Delta bucket name
                - feature_cols: All specified feature columns
                - label_cols: All specified label columns
            feature_cols: List of feature column names to extract
            label_cols: List of label column names to extract
            lookback_days: Number of timesteps in lookback window
            graph: Pre-built graph. If None, will be built from graph_config.
            graph_config: Configuration for graph construction (used if graph is None)
            normalize_features: Whether to z-score normalize features
            feature_stats: Pre-computed normalization stats for each feature column
                (used for validation/test sets to use training stats).
                Per-node stats are arrays of shape (num_nodes,); scalar stats
                are also accepted and broadcast correctly.
            normalize_labels: Whether to z-score normalize DHR labels. Stats are
                computed globally (not per-node) on training data and reused for
                val/test. Predictions must be denormalized at inference time.
            label_stats: Pre-computed label normalization stats from training set.
                Dict mapping label column name to (mean, std) tuple of floats.
                Required for val/test sets when normalize_labels=True.
            min_sample_end_ts: If provided, excludes samples whose target
                timestamp (window end) is earlier than this boundary. Useful for
                keeping pre-split lookback rows while ensuring targets stay in
                the intended split period.
            use_memmap: If True, back dense arrays with memory-mapped files
                instead of in-memory numpy arrays. Enables datasets larger
                than available RAM with zero impact on GPU pipeline.
        """
        self._feature_cols = feature_cols
        self._label_cols = label_cols
        self._lookback_days = lookback_days
        self._normalize = normalize_features
        self._normalize_labels = normalize_labels
        self._min_sample_end_ts = (
            pd.Timestamp(min_sample_end_ts) if min_sample_end_ts is not None else None
        )
        self._use_memmap = use_memmap
        self._memmap_dir: Optional[str] = (
            tempfile.mkdtemp(prefix="valorem_dataset_") if use_memmap else None
        )

        try:
            # Sort panel by timestamp, tenor, bucket
            self._panel = panel_df.sort_values(["ts_utc", "tenor_days", "delta_bucket"]).reset_index(drop=True)

            # Build or use provided graph
            if graph is not None:
                self._graph = graph
            else:
                config = graph_config or SurfaceGraphConfig()
                self._graph = build_surface_graph(config)

            # Get surface dimensions from graph
            self._delta_buckets = self._graph.delta_buckets
            self._tenors_days = self._graph.tenors_days
            self._num_nodes = self._graph.num_nodes

            # Build unique timestamp index
            self._unique_timestamps = sorted(self._panel["ts_utc"].unique())

            # Compute or use provided feature stats for normalization
            if normalize_features:
                if feature_stats is not None:
                    self._feature_stats = feature_stats
                else:
                    self._feature_stats = self._compute_feature_stats()
            else:
                self._feature_stats = None

            # Compute or use provided label stats for normalization
            if normalize_labels:
                if label_stats is not None:
                    self._label_stats = label_stats
                else:
                    self._label_stats = self._compute_label_stats()
            else:
                self._label_stats = None

            # Precompute dense arrays for O(1) __getitem__ access.
            # Must run after feature_stats and label_stats are available.
            self._features, self._labels, self._masks, self._label_masks = self._precompute_tensors()

            # Build sample index as integer pairs into precomputed arrays
            self._samples = self._build_sample_index()

            # Release DataFrame — all data now lives in precomputed arrays
            self._panel = None
        except Exception:
            self._cleanup_memmap()
            raise

    def _build_sample_index(self) -> list[tuple[int, int]]:
        """Build list of (start_idx, end_idx) integer pairs for each sample.

        Each sample spans lookback_days consecutive timestamps. Indices
        reference the precomputed dense arrays directly.

        Returns:
            List of (start_idx, end_idx) pairs into the timestamp dimension
        """
        samples: list[tuple[int, int]] = []

        num_ts = len(self._unique_timestamps)
        if num_ts <= self._lookback_days:
            return samples

        for i in range(self._lookback_days, num_ts):
            if (
                self._min_sample_end_ts is not None
                and pd.Timestamp(self._unique_timestamps[i]) < self._min_sample_end_ts
            ):
                continue
            samples.append((i - self._lookback_days, i))

        return samples

    def _precompute_tensors(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Convert panel DataFrame into dense arrays for O(1) __getitem__.

        Performs a single vectorized pass over the DataFrame to fill three
        pre-allocated arrays. Normalization is applied once here rather than
        per-sample in __getitem__.

        Returns:
            Tuple of:
                features: (num_timestamps, num_nodes, num_features) float32
                labels: (num_timestamps, num_nodes, num_horizons) float32
                masks: (num_timestamps, num_nodes) bool
                label_masks: (num_timestamps, num_nodes, num_horizons) bool
        """
        num_ts = len(self._unique_timestamps)
        num_features = len(self._feature_cols)
        num_horizons = len(self._label_cols)
        ts_to_idx = {ts: i for i, ts in enumerate(self._unique_timestamps)}

        # Allocate dense arrays (memmap-backed or in-memory)
        feat_shape = (num_ts, self._num_nodes, num_features)
        label_shape = (num_ts, self._num_nodes, num_horizons)
        mask_shape = (num_ts, self._num_nodes)

        if self._use_memmap:
            memmap_dir = Path(self._memmap_dir)
            features = np.memmap(
                memmap_dir / "features.dat", dtype=np.float32,
                mode="w+", shape=feat_shape,
            )
            features[:] = np.nan
            labels = np.memmap(
                memmap_dir / "labels.dat", dtype=np.float32,
                mode="w+", shape=label_shape,
            )
            labels[:] = np.nan
            label_masks = np.memmap(
                memmap_dir / "label_masks.dat", dtype=bool,
                mode="w+", shape=label_shape,
            )
            label_masks[:] = False
            masks = np.memmap(
                memmap_dir / "masks.dat", dtype=bool,
                mode="w+", shape=mask_shape,
            )
        else:
            features = np.full(feat_shape, np.nan, dtype=np.float32)
            labels = np.full(label_shape, np.nan, dtype=np.float32)
            label_masks = np.zeros(label_shape, dtype=bool)
            masks = np.zeros(mask_shape, dtype=bool)

        panel = self._panel

        # Map every row to (t_idx, node_idx) in one pass
        t_indices = panel["ts_utc"].map(ts_to_idx).values.astype(int)
        node_indices = np.array(
            [
                self._graph.node_mapping.get((int(tenor), str(bucket)), -1)
                for tenor, bucket in zip(
                    panel["tenor_days"].values, panel["delta_bucket"].values
                )
            ],
            dtype=int,
        )

        # Keep only rows that map to a valid graph node
        valid = node_indices >= 0
        t_idx = t_indices[valid]
        n_idx = node_indices[valid]
        valid_panel = panel.iloc[valid]

        # Fill features via column-wise fancy indexing
        for f_i, col in enumerate(self._feature_cols):
            if col in valid_panel.columns:
                vals = valid_panel[col].values.astype(np.float32)
                good = np.isfinite(vals)
                features[t_idx[good], n_idx[good], f_i] = vals[good]

        # Pre-compute quality mask once and apply consistently to both
        # node-level masks and horizon-level label masks.
        is_masked = (
            valid_panel["is_masked"].fillna(False).values.astype(bool)
            if "is_masked" in valid_panel.columns
            else np.zeros(len(valid_panel), dtype=bool)
        )
        not_masked = ~is_masked

        # Fill labels (exclude NaN/inf and quality-masked rows)
        for h_i, col in enumerate(self._label_cols):
            if col in valid_panel.columns:
                vals = valid_panel[col].values.astype(np.float32)
                good = np.isfinite(vals) & not_masked
                labels[t_idx[good], n_idx[good], h_i] = vals[good]
                label_masks[t_idx[good], n_idx[good], h_i] = True

        label_cols_in_df = [c for c in self._label_cols if c in valid_panel.columns]
        if self._label_cols:
            if label_cols_in_df:
                has_label = (
                    valid_panel[label_cols_in_df]
                    .apply(lambda s: np.isfinite(s.astype(float)))
                    .any(axis=1)
                    .values
                )
                node_valid = not_masked & has_label
            else:
                # Label columns don't exist yet — all non-masked nodes valid
                node_valid = not_masked
        else:
            node_valid = not_masked

        masks[t_idx[node_valid], n_idx[node_valid]] = True

        # Normalize features once (not per-sample)
        if self._normalize and self._feature_stats:
            for f_i, col in enumerate(self._feature_cols):
                if col in self._feature_stats:
                    mean, std = self._feature_stats[col]
                    features[:, :, f_i] = (features[:, :, f_i] - mean) / std

        # Normalize labels once (global z-score, not per-node)
        if self._normalize_labels and self._label_stats:
            for h_i, col in enumerate(self._label_cols):
                if col in self._label_stats:
                    mean, std = self._label_stats[col]
                    labels[:, :, h_i] = (labels[:, :, h_i] - mean) / std

        # Replace NaN → 0 (after normalization)
        np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
        np.nan_to_num(labels, nan=0.0, posinf=0.0, neginf=0.0, copy=False)

        # Flush writes and reopen as read-only for fork-safe DataLoader workers
        if self._use_memmap:
            features.flush()
            labels.flush()
            label_masks.flush()
            masks.flush()
            memmap_dir = Path(self._memmap_dir)
            features = np.memmap(
                memmap_dir / "features.dat", dtype=np.float32,
                mode="r", shape=feat_shape,
            )
            labels = np.memmap(
                memmap_dir / "labels.dat", dtype=np.float32,
                mode="r", shape=label_shape,
            )
            label_masks = np.memmap(
                memmap_dir / "label_masks.dat", dtype=bool,
                mode="r", shape=label_shape,
            )
            masks = np.memmap(
                memmap_dir / "masks.dat", dtype=bool,
                mode="r", shape=mask_shape,
            )

        return features, labels, masks, label_masks

    def _compute_feature_stats(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Compute per-node mean and std for each feature column.

        Per-node normalization prevents cross-sectional structure from leaking
        into normalized feature values. Without this, globally z-scored features
        encode node identity (e.g., ATM nodes always get positive gamma z-scores),
        giving per-node models like PatchTST implicit cross-sectional information.

        Returns:
            Dict mapping feature column name to (means, stds) tuple where
            each is a numpy array of shape (num_nodes,).
        """
        stats: dict[str, tuple[np.ndarray, np.ndarray]] = {}

        # Map panel rows to node indices (same logic as _precompute_tensors)
        node_indices = np.array(
            [
                self._graph.node_mapping.get((int(tenor), str(bucket)), -1)
                for tenor, bucket in zip(
                    self._panel["tenor_days"].values,
                    self._panel["delta_bucket"].values,
                )
            ],
            dtype=int,
        )
        is_masked = (
            self._panel["is_masked"].fillna(False).values.astype(bool)
            if "is_masked" in self._panel.columns
            else np.zeros(len(self._panel), dtype=bool)
        )

        for col in self._feature_cols:
            means = np.zeros(self._num_nodes)
            stds = np.ones(self._num_nodes)

            if col not in self._panel.columns:
                stats[col] = (means, stds)
                continue

            values = self._panel[col].values.astype(np.float64)
            valid = np.isfinite(values) & (node_indices >= 0) & (~is_masked)
            valid_values = values[valid]
            valid_nodes = node_indices[valid]

            for n in range(self._num_nodes):
                mask = valid_nodes == n
                count = mask.sum()
                if count > 1:
                    node_vals = valid_values[mask]
                    means[n] = np.mean(node_vals)
                    std = np.std(node_vals, ddof=1)
                    stds[n] = std if std >= 1e-10 else 1.0
                elif count == 1:
                    means[n] = valid_values[mask][0]

            stats[col] = (means, stds)

        return stats

    def get_feature_stats(self) -> dict[str, tuple]:
        """Get feature normalization statistics.

        Returns:
            Dict mapping feature column name to (means, stds) tuple.
            For per-node normalization, each element is a numpy array
            of shape (num_nodes,). For externally provided scalar stats,
            elements may be floats (broadcasting handles both).
        """
        return self._feature_stats.copy() if self._feature_stats else {}

    def _compute_label_stats(self) -> dict[str, tuple[float, float]]:
        """Compute global mean and std for each label column.

        Unlike features (per-node normalization), labels are normalized
        globally because DHR scale is driven by the annualization factor
        (252/H) and gamma magnitude, not by node identity. Global z-scoring
        brings all horizons to comparable O(1) scale for stable training.

        Returns:
            Dict mapping label column name to (mean, std) tuple of floats.
        """
        stats: dict[str, tuple[float, float]] = {}

        is_masked = (
            self._panel["is_masked"].fillna(False).values.astype(bool)
            if "is_masked" in self._panel.columns
            else np.zeros(len(self._panel), dtype=bool)
        )

        for col in self._label_cols:
            if col not in self._panel.columns:
                stats[col] = (0.0, 1.0)
                continue

            values = self._panel[col].values.astype(np.float64)
            valid = np.isfinite(values) & (~is_masked)
            valid_values = values[valid]

            if len(valid_values) > 1:
                mean = float(np.mean(valid_values))
                std = float(np.std(valid_values, ddof=1))
                stats[col] = (mean, std if std >= 1e-10 else 1.0)
            else:
                stats[col] = (0.0, 1.0)

        return stats

    def get_label_stats(self) -> dict[str, tuple[float, float]]:
        """Get label normalization statistics (for checkpoint storage).

        Returns:
            Dict mapping label column name to (mean, std) tuple of floats.
            Empty dict if label normalization is disabled.
        """
        return self._label_stats.copy() if self._label_stats else {}

    def _cleanup_memmap(self) -> None:
        """Remove temporary memmap directory and files."""
        memmap_dir = getattr(self, "_memmap_dir", None)
        if memmap_dir is not None:
            shutil.rmtree(memmap_dir, ignore_errors=True)
            self._memmap_dir = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._cleanup_memmap()
        return False

    def __del__(self):
        self._cleanup_memmap()

    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return len(self._samples)

    def count_valid_labels(self) -> int:
        """Count valid (unmasked, finite) labels across all samples.

        Returns:
            Total number of valid label elements referenced by dataset samples.
            Returns 0 for inference datasets (num_horizons == 0) or empty datasets.
        """
        if self.num_horizons == 0 or not self._samples:
            return 0

        sample_end_indices = np.fromiter(
            (end_idx for _, end_idx in self._samples),
            dtype=np.int64,
            count=len(self._samples),
        )
        if sample_end_indices.size == 0:
            return 0

        return int(np.count_nonzero(self._label_masks[sample_end_indices]))

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Get a single sample via O(1) indexing into precomputed arrays.

        Args:
            idx: Sample index

            Returns:
            Dict with keys:
                - X: Feature tensor (lookback_days + 1, num_nodes, num_features)
                - y: Label tensor (num_nodes, num_horizons)
                - mask: Valid node mask (num_nodes,)
                - label_mask: Valid label mask (num_nodes, num_horizons)
                - graph: Graph Data object
        """
        start_idx, end_idx = self._samples[idx]

        # .copy() ensures writable tensors (required for pin_memory) and
        # materialises memmap slices into contiguous RAM for the DataLoader.
        return {
            "X": torch.from_numpy(self._features[start_idx : end_idx + 1].copy()),
            "y": torch.from_numpy(self._labels[end_idx].copy()),
            "mask": torch.from_numpy(self._masks[end_idx].copy()),
            "label_mask": torch.from_numpy(self._label_masks[end_idx].copy()),
            "graph": self._graph,
        }

    @property
    def num_nodes(self) -> int:
        """Return number of nodes in graph."""
        return self._num_nodes

    @property
    def num_features(self) -> int:
        """Return number of features per node."""
        return len(self._feature_cols)

    @property
    def num_horizons(self) -> int:
        """Return number of prediction horizons."""
        return len(self._label_cols)

    @property
    def graph(self) -> Data:
        """Return the static graph structure."""
        return self._graph


class DatasetBuilder:
    """Build train/val/test datasets with proper date splits.

    Handles:
    - Loading node panel from repository
    - Computing forward-looking labels (optional)
    - Splitting by date boundaries
    - Building graph structure
    - Creating normalized datasets
    """

    def __init__(
        self,
        config: DatasetConfig,
    ):
        """Initialize dataset builder.

        Args:
            config: Dataset configuration
        """
        self._config = config
        self._graph = None

    def build_datasets(
        self,
        panel_df: pd.DataFrame,
        labels_config: Optional[LabelsConfig] = None,
        underlying_returns: Optional[pd.DataFrame] = None,
    ) -> tuple["SurfaceDataset", "SurfaceDataset", "SurfaceDataset"]:
        """Build train, val, test datasets from node panel.

        Args:
            panel_df: DataFrame with node panel data
            labels_config: Configuration for label construction (optional)
            underlying_returns: DataFrame with underlying returns for RV calculation
                (required if labels need to be built)

        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)

        Raises:
            ValueError: If splits config is not provided
        """
        if self._config.splits is None:
            raise ValueError("SplitsConfig must be provided in DatasetConfig")

        splits = self._config.splits

        # Build DHR labels if config provided and columns don't exist
        if labels_config is not None:
            panel_df = self._build_dhr_labels(panel_df, labels_config, underlying_returns)

        # Build graph (shared across all datasets)
        self._graph = build_surface_graph(self._config.graph)

        # Split by date — .copy() ensures splits own their memory so
        # panel_df can be freed immediately after.
        panel_df["ts_utc"] = pd.to_datetime(panel_df["ts_utc"])
        # Keep full pre-boundary history in each split DataFrame so validation
        # and test samples can use lookback context from prior periods.
        train_df = panel_df[panel_df["ts_utc"] < splits.val_start].copy()
        val_df = panel_df[panel_df["ts_utc"] < splits.test_start].copy()
        test_df = panel_df[panel_df["ts_utc"] < splits.test_end].copy()

        # Nullify labels whose forward price lookups cross split boundaries.
        # Without this, training samples near val_start have DHR labels
        # computed from validation-period prices (up to max_horizon days
        # of leakage).
        if labels_config is not None and underlying_returns is not None:
            trading_dates = self._extract_trading_dates(underlying_returns)
            if trading_dates is not None:
                self._nullify_boundary_labels(
                    train_df, splits.val_start,
                    labels_config.horizons_days, trading_dates,
                )
                self._nullify_boundary_labels(
                    val_df, splits.test_start,
                    labels_config.horizons_days, trading_dates,
                )

        # Free the full panel — all data now lives in the split DataFrames
        del panel_df
        gc.collect()

        # Build datasets sequentially, freeing each split df after consumption.
        # SurfaceDataset.__init__ copies into memmap-backed arrays then sets
        # self._panel = None, so the split df is the last reference.

        # Training dataset (computes feature and label stats)
        train_ds = SurfaceDataset(
            panel_df=train_df,
            feature_cols=self._config.feature_cols,
            label_cols=self._config.label_cols,
            lookback_days=self._config.lookback_days,
            graph=self._graph,
            normalize_features=self._config.normalize_features,
            normalize_labels=True,
            min_sample_end_ts=splits.train_start,
        )
        del train_df
        gc.collect()

        feature_stats = train_ds.get_feature_stats() if self._config.normalize_features else None
        label_stats = train_ds.get_label_stats()

        # Validation dataset (uses training stats)
        val_ds = SurfaceDataset(
            panel_df=val_df,
            feature_cols=self._config.feature_cols,
            label_cols=self._config.label_cols,
            lookback_days=self._config.lookback_days,
            graph=self._graph,
            normalize_features=self._config.normalize_features,
            feature_stats=feature_stats,
            normalize_labels=True,
            label_stats=label_stats,
            min_sample_end_ts=splits.val_start,
        )
        del val_df
        gc.collect()

        # Test dataset (uses training stats)
        test_ds = SurfaceDataset(
            panel_df=test_df,
            feature_cols=self._config.feature_cols,
            label_cols=self._config.label_cols,
            lookback_days=self._config.lookback_days,
            graph=self._graph,
            normalize_features=self._config.normalize_features,
            feature_stats=feature_stats,
            normalize_labels=True,
            label_stats=label_stats,
            min_sample_end_ts=splits.test_start,
        )
        del test_df
        gc.collect()

        return train_ds, val_ds, test_ds

    def _build_dhr_labels(
        self,
        panel_df: pd.DataFrame,
        labels_config: LabelsConfig,
        underlying_returns: Optional[pd.DataFrame],
    ) -> pd.DataFrame:
        """Compute delta-hedged return (DHR) labels per node.

        For each horizon H and each (timestamp, node) pair, computes:
            DHR = (mid_price[t+H] - mid_price[t]) - delta[t] * (S[t+H] - S[t])

        where mid_price is the option mid price and S is the underlying price.
        This captures the full P&L of a delta-hedged option position including
        theta decay, which creates genuine sign variation: positive when
        realized vol exceeds implied (option was cheap), negative when it
        doesn't (option was expensive). This is the variance risk premium.

        The previous Taylor approximation (0.5 * gamma * dS^2) dropped the
        theta term, making all labels non-negative and temporal variation
        purely a function of dS^2 (identical across nodes). The actual P&L
        formulation restores meaningful temporal signal.

        Requires 'mid_price' and 'delta' in panel_df and underlying close
        prices in underlying_returns. Nodes with NaN mid_price or delta
        get NaN DHR (masked out during training).

        Ref: Bakshi & Kapadia (RFS 2003), Bali et al. (RFS 2023).

        Args:
            panel_df: Node panel DataFrame. Must have mid_price and delta
                columns, and ts_utc for timestamp alignment.
            labels_config: Label configuration with horizons_days.
            underlying_returns: DataFrame with ts_utc and close columns.
                Used for underlying price at t and t+H.

        Returns:
            Panel DataFrame with y_dhr_{H}d label columns added.
        """
        if underlying_returns is None:
            return panel_df

        # Build underlying price series
        if "close" in underlying_returns.columns:
            if "ts_utc" in underlying_returns.columns:
                prices = underlying_returns.set_index("ts_utc")["close"].sort_index()
            elif "date" in underlying_returns.columns:
                prices = underlying_returns.set_index("date")["close"].sort_index()
            else:
                prices = underlying_returns["close"].sort_index()
        else:
            raise ValueError(
                "DHR labels require underlying close prices (a 'close' column in "
                "underlying_returns), not just returns. Ensure the data pipeline "
                "passes close prices via include_close=True."
            )

        if "mid_price" not in panel_df.columns or "delta" not in panel_df.columns:
            return panel_df
        if prices.empty:
            return panel_df

        panel_df["ts_utc"] = pd.to_datetime(panel_df["ts_utc"])

        # --- Underlying price lookup (vectorized) ---
        # For each panel timestamp, find S(t) and S(t+H) from the underlying
        # bar series, preserving intraday offset alignment.
        price_idx = prices.index  # sorted DatetimeIndex
        price_vals = prices.values
        price_dates = price_idx.normalize().values
        trading_dates, day_start = np.unique(price_dates, return_index=True)
        day_end = np.r_[day_start[1:] - 1, len(price_idx) - 1]

        bar_positions = np.arange(len(price_idx))
        bar_day_idx = np.searchsorted(day_start, bar_positions, side="right") - 1

        ts_arr = panel_df["ts_utc"].values
        pos = price_idx.searchsorted(ts_arr, side="left")
        pos_clip = np.minimum(pos, len(price_idx) - 1)
        exact = (pos < len(price_idx)) & (price_idx.values[pos_clip] == ts_arr)

        s_t = np.where(exact, price_vals[pos_clip], np.nan)
        day_idx = bar_day_idx[pos_clip]
        intra_day_offset = pos_clip - day_start[day_idx]

        # --- Option mid-price forward lookup (per-node) ---
        # Build a lookup structure keyed by (node, trading_day_index) so we
        # can find mid_price at t+H for each node independently.
        node_keys = (
            panel_df["tenor_days"].astype(str) + "_" + panel_df["delta_bucket"]
        ).values
        mid_vals = panel_df["mid_price"].values.astype(np.float64)

        # Map each panel row's ts_utc to its trading-day index
        panel_dates = pd.to_datetime(ts_arr).normalize().values
        panel_day_idx = np.searchsorted(trading_dates, panel_dates, side="left")

        # Build dict: (node_key, day_idx) -> mid_price for forward lookup
        mid_lookup: dict[tuple[str, int], float] = {}
        for i in range(len(panel_df)):
            if exact[i] and np.isfinite(mid_vals[i]):
                key = (node_keys[i], int(panel_day_idx[i]))
                mid_lookup[key] = mid_vals[i]

        delta_vals = panel_df["delta"].values.astype(np.float64)

        for H in labels_config.horizons_days:
            dhr_col = f"y_dhr_{H}d"

            # Underlying forward price
            future_day_idx = day_idx + H
            valid_future = exact & (future_day_idx < len(trading_dates))
            future_day_clip = np.minimum(future_day_idx, len(trading_dates) - 1)
            future_start = day_start[future_day_clip]
            future_end = day_end[future_day_clip]
            future_pos = np.minimum(future_start + intra_day_offset, future_end)
            s_t_h = np.where(valid_future, price_vals[future_pos], np.nan)

            # Option mid-price forward lookup per node
            mid_t = mid_vals.copy()
            mid_t_h = np.full(len(panel_df), np.nan)
            for i in range(len(panel_df)):
                if valid_future[i]:
                    key = (node_keys[i], int(panel_day_idx[i] + H))
                    val = mid_lookup.get(key)
                    if val is not None:
                        mid_t_h[i] = val

            # DHR = option P&L - delta hedge P&L
            # = (mid[t+H] - mid[t]) - delta[t] * (S[t+H] - S[t])
            ds = s_t_h - s_t
            option_pnl = mid_t_h - mid_t
            hedge_pnl = delta_vals * ds
            raw_dhr = option_pnl - hedge_pnl

            # Winsorize at 1st/99th percentiles to prevent outliers from
            # dominating loss gradients
            finite = raw_dhr[np.isfinite(raw_dhr)]
            if len(finite) > 0:
                p1, p99 = np.percentile(finite, [1, 99])
                raw_dhr = np.clip(raw_dhr, p1, p99)
            panel_df[dhr_col] = raw_dhr

        return panel_df

    @staticmethod
    def _extract_trading_dates(
        underlying_returns: pd.DataFrame,
    ) -> Optional[np.ndarray]:
        """Extract sorted trading date index from underlying returns.

        Uses the same price series logic as _build_dhr_labels to ensure
        position indices are consistent.

        Returns:
            Sorted numpy datetime64 array of trading dates, or None if
            the price series cannot be constructed.
        """
        if "close" not in underlying_returns.columns:
            return None
        if "ts_utc" in underlying_returns.columns:
            prices = underlying_returns.set_index("ts_utc")["close"].sort_index()
        elif "date" in underlying_returns.columns:
            prices = underlying_returns.set_index("date")["close"].sort_index()
        else:
            prices = underlying_returns["close"].sort_index()
        return prices.index.normalize().unique().values

    @staticmethod
    def _nullify_boundary_labels(
        split_df: pd.DataFrame,
        boundary_date: datetime,
        horizons: list[int],
        trading_dates: np.ndarray,
    ) -> None:
        """NaN-ify DHR labels whose forward price lookup crosses a split boundary.

        For a sample at date t with horizon H, if t + H trading days falls
        on or after boundary_date, the label uses prices from the next split.
        This sets those labels to NaN so they're masked out during training.

        Args:
            split_df: DataFrame for one split (modified in place).
            boundary_date: Start date of the next split.
            horizons: List of horizon lengths in trading days.
            trading_dates: Sorted array of all trading dates.
        """
        boundary_ts = np.datetime64(pd.Timestamp(boundary_date), "ns")
        boundary_pos = np.searchsorted(trading_dates, boundary_ts, side="left")
        ts_arr = pd.to_datetime(split_df["ts_utc"]).dt.normalize().values
        pos = np.searchsorted(trading_dates, ts_arr, side="left")

        for H in horizons:
            col = f"y_dhr_{H}d"
            if col not in split_df.columns:
                continue
            valid = pos < len(trading_dates)
            contaminated = valid & (pos + H >= boundary_pos)
            if contaminated.any():
                vals = split_df[col].values.copy()
                vals[contaminated] = np.nan
                split_df[col] = vals

    def build_single_dataset(
        self,
        panel_df: pd.DataFrame,
        feature_stats: Optional[dict[str, tuple]] = None,
    ) -> SurfaceDataset:
        """Build a single dataset without splits.

        Useful for inference on new data.

        Args:
            panel_df: DataFrame with node panel data
            feature_stats: Optional pre-computed feature statistics

        Returns:
            SurfaceDataset instance
        """
        if self._graph is None:
            self._graph = build_surface_graph(self._config.graph)

        return SurfaceDataset(
            panel_df=panel_df,
            feature_cols=self._config.feature_cols,
            label_cols=self._config.label_cols,
            lookback_days=self._config.lookback_days,
            graph=self._graph,
            normalize_features=self._config.normalize_features,
            feature_stats=feature_stats,
        )
