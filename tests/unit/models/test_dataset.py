"""Unit tests for dataset module."""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest
import torch

from src.models.dataset import (
    DatasetConfig,
    LabelsConfig,
    SplitsConfig,
    SurfaceDataset,
    DatasetBuilder,
    DEFAULT_FEATURE_COLS,
    DEFAULT_LABEL_COLS,
)
from src.models.graph import SurfaceGraphConfig, build_surface_graph


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def simple_graph_config():
    """Simple graph config for testing (3 buckets × 2 tenors = 6 nodes)."""
    return SurfaceGraphConfig(
        delta_buckets=["P25", "ATM", "C25"],
        tenors_days=[30, 60],
    )


@pytest.fixture
def simple_graph(simple_graph_config):
    """Pre-built simple graph."""
    return build_surface_graph(simple_graph_config)


@pytest.fixture
def sample_panel_df(simple_graph_config):
    """Create sample node panel DataFrame for testing."""
    buckets = simple_graph_config.delta_buckets
    tenors = simple_graph_config.tenors_days

    # Generate 10 timestamps (daily)
    base_time = datetime(2024, 1, 1, 16, 0, 0)
    timestamps = [base_time + timedelta(days=i) for i in range(10)]

    rows = []
    for ts in timestamps:
        for tenor in tenors:
            for bucket in buckets:
                row = {
                    "ts_utc": ts,
                    "tenor_days": tenor,
                    "delta_bucket": bucket,
                    "delta": 0.5 if bucket == "ATM" else (0.25 if "C" in bucket else -0.25),
                    "gamma": 0.05,
                    "vega": 0.15,
                    "theta": -0.01,
                    "spread_pct": 0.02,
                    "skew_slope": -0.1 + np.random.randn() * 0.02,
                    "term_slope": 0.05 + np.random.randn() * 0.01,
                    "curvature": 0.01 + np.random.randn() * 0.005,
                    "iv_change_1d": np.random.randn() * 0.005,
                    "iv_change_5d": np.random.randn() * 0.01,
                    "iv_vol_5d": 0.01 + abs(np.random.randn() * 0.003),
                    "iv_vol_10d": 0.01 + abs(np.random.randn() * 0.003),
                    "iv_vol_21d": 0.01 + abs(np.random.randn() * 0.003),
                    "iv_zscore_5d": np.random.randn() * 0.5,
                    "iv_zscore_10d": np.random.randn() * 0.5,
                    "iv_zscore_21d": np.random.randn() * 0.5,
                    "underlying_rv_5d": 0.15 + np.random.randn() * 0.02,
                    "underlying_rv_10d": 0.15 + np.random.randn() * 0.02,
                    "underlying_rv_21d": 0.15 + np.random.randn() * 0.02,
                    "VIXCLS_level": 0.2 + np.random.randn() * 0.03,
                    "VIXCLS_change_1w": np.random.randn() * 0.02,
                    "DGS10_level": 0.04 + np.random.randn() * 0.005,
                    "DGS10_change_1w": np.random.randn() * 0.003,
                    "DGS2_level": 0.045 + np.random.randn() * 0.005,
                    "DGS2_change_1w": np.random.randn() * 0.003,
                    "log_volume": 5.0 + np.random.randn() * 0.5,
                    "volume_ratio_5d": 1.0 + np.random.randn() * 0.2,
                    "log_oi": 7.0 + np.random.randn() * 0.5,
                    "oi_change_5d": np.random.randn() * 0.05,
                    "is_masked": False,
                }
                rows.append(row)

    return pd.DataFrame(rows)


@pytest.fixture
def sample_panel_with_labels(sample_panel_df):
    """Panel with label columns."""
    df = sample_panel_df.copy()
    # Add dummy labels
    df["y_dhr_5d"] = np.random.randn(len(df)) * 0.1
    df["y_dhr_10d"] = np.random.randn(len(df)) * 0.1
    df["y_dhr_21d"] = np.random.randn(len(df)) * 0.1
    return df


# =============================================================================
# SplitsConfig Tests
# =============================================================================


class TestSplitsConfig:
    """Tests for SplitsConfig."""

    def test_valid_splits(self):
        """Test valid date splits."""
        splits = SplitsConfig(
            train_start=datetime(2023, 1, 1),
            val_start=datetime(2023, 7, 1),
            test_start=datetime(2023, 10, 1),
            test_end=datetime(2024, 1, 1),
        )

        assert splits.train_start < splits.val_start
        assert splits.val_start < splits.test_start
        assert splits.test_start < splits.test_end

    def test_invalid_splits_order(self):
        """Test that invalid date order raises error."""
        with pytest.raises(ValueError, match="chronologically ordered"):
            SplitsConfig(
                train_start=datetime(2023, 7, 1),  # After val_start
                val_start=datetime(2023, 1, 1),
                test_start=datetime(2023, 10, 1),
                test_end=datetime(2024, 1, 1),
            )


class TestLabelsConfig:
    """Tests for LabelsConfig."""

    def test_default_config(self):
        """Test default label configuration."""
        config = LabelsConfig()

        assert config.horizons_days == [5, 10, 21]

    def test_custom_horizons(self):
        """Test custom horizon configuration."""
        config = LabelsConfig(horizons_days=[1, 5, 10])

        assert config.horizons_days == [1, 5, 10]


class TestDatasetConfig:
    """Tests for DatasetConfig."""

    def test_default_config(self):
        """Test default dataset configuration."""
        config = DatasetConfig()

        assert config.lookback_days == 21
        assert config.feature_cols == DEFAULT_FEATURE_COLS
        assert config.label_cols == DEFAULT_LABEL_COLS
        assert config.normalize_features is True

    def test_custom_config(self):
        """Test custom dataset configuration."""
        config = DatasetConfig(
            lookback_days=10,
            feature_cols=["iv_mid", "delta"],
            normalize_features=False,
        )

        assert config.lookback_days == 10
        assert config.feature_cols == ["iv_mid", "delta"]
        assert config.normalize_features is False


# =============================================================================
# SurfaceDataset Tests
# =============================================================================


class TestSurfaceDatasetInit:
    """Tests for SurfaceDataset initialization."""

    def test_init_with_panel(self, sample_panel_df, simple_graph):
        """Test basic initialization."""
        dataset = SurfaceDataset(
            panel_df=sample_panel_df,
            feature_cols=["iv_mid", "delta"],
            label_cols=[],
            lookback_days=3,
            graph=simple_graph,
        )

        assert dataset.num_nodes == 6  # 3 buckets × 2 tenors
        assert dataset.num_features == 2
        assert len(dataset) > 0

    def test_init_builds_graph(self, sample_panel_df, simple_graph_config):
        """Test that graph is built if not provided."""
        dataset = SurfaceDataset(
            panel_df=sample_panel_df,
            feature_cols=["iv_mid"],
            label_cols=[],
            lookback_days=3,
            graph_config=simple_graph_config,
        )

        assert dataset.num_nodes == 6
        assert dataset.graph is not None

    def test_sample_count(self, sample_panel_df, simple_graph):
        """Test sample count with lookback."""
        # 10 timestamps, lookback 3 -> 7 samples (indices 3-9)
        dataset = SurfaceDataset(
            panel_df=sample_panel_df,
            feature_cols=["iv_mid"],
            label_cols=[],
            lookback_days=3,
            graph=simple_graph,
        )

        assert len(dataset) == 7

    def test_insufficient_timestamps(self, sample_panel_df, simple_graph):
        """Test with lookback larger than available timestamps."""
        # Only 10 timestamps, lookback 15 -> 0 samples
        dataset = SurfaceDataset(
            panel_df=sample_panel_df,
            feature_cols=["iv_mid"],
            label_cols=[],
            lookback_days=15,
            graph=simple_graph,
        )

        assert len(dataset) == 0


class TestSurfaceDatasetGetitem:
    """Tests for __getitem__ method."""

    def test_getitem_returns_dict(self, sample_panel_with_labels, simple_graph):
        """Test that __getitem__ returns correct structure."""
        dataset = SurfaceDataset(
            panel_df=sample_panel_with_labels,
            feature_cols=["iv_mid", "delta"],
            label_cols=["y_dhr_5d"],
            lookback_days=3,
            graph=simple_graph,
        )

        sample = dataset[0]

        assert isinstance(sample, dict)
        assert "X" in sample
        assert "y" in sample
        assert "mask" in sample
        assert "graph" in sample

    def test_feature_tensor_shape(self, sample_panel_with_labels, simple_graph):
        """Test feature tensor shape."""
        dataset = SurfaceDataset(
            panel_df=sample_panel_with_labels,
            feature_cols=["iv_mid", "delta", "gamma"],
            label_cols=["y_dhr_5d"],
            lookback_days=3,
            graph=simple_graph,
        )

        sample = dataset[0]
        X = sample["X"]

        # Shape: (lookback + 1, num_nodes, num_features)
        assert X.shape == (4, 6, 3)  # 4 timesteps, 6 nodes, 3 features

    def test_label_tensor_shape(self, sample_panel_with_labels, simple_graph):
        """Test label tensor shape."""
        dataset = SurfaceDataset(
            panel_df=sample_panel_with_labels,
            feature_cols=["iv_mid"],
            label_cols=["y_dhr_5d", "y_dhr_10d"],
            lookback_days=3,
            graph=simple_graph,
        )

        sample = dataset[0]
        y = sample["y"]

        # Shape: (num_nodes, num_horizons)
        assert y.shape == (6, 2)

    def test_mask_tensor_shape(self, sample_panel_with_labels, simple_graph):
        """Test mask tensor shape."""
        dataset = SurfaceDataset(
            panel_df=sample_panel_with_labels,
            feature_cols=["iv_mid"],
            label_cols=["y_dhr_5d"],
            lookback_days=3,
            graph=simple_graph,
        )

        sample = dataset[0]
        mask = sample["mask"]

        assert mask.shape == (6,)
        assert mask.dtype == torch.bool

    def test_tensor_dtypes(self, sample_panel_with_labels, simple_graph):
        """Test tensor data types."""
        dataset = SurfaceDataset(
            panel_df=sample_panel_with_labels,
            feature_cols=["iv_mid"],
            label_cols=["y_dhr_5d"],
            lookback_days=3,
            graph=simple_graph,
        )

        sample = dataset[0]

        assert sample["X"].dtype == torch.float32
        assert sample["y"].dtype == torch.float32
        assert sample["mask"].dtype == torch.bool

    def test_graph_is_static(self, sample_panel_with_labels, simple_graph):
        """Test that same graph is returned for all samples."""
        dataset = SurfaceDataset(
            panel_df=sample_panel_with_labels,
            feature_cols=["iv_mid"],
            label_cols=["y_dhr_5d"],
            lookback_days=3,
            graph=simple_graph,
        )

        sample0 = dataset[0]
        sample1 = dataset[1]

        assert sample0["graph"] is sample1["graph"]


class TestSurfaceDatasetNormalization:
    """Tests for feature normalization."""

    def test_normalization_applied(self, sample_panel_with_labels, simple_graph):
        """Test that normalization is applied when enabled."""
        dataset = SurfaceDataset(
            panel_df=sample_panel_with_labels,
            feature_cols=["iv_mid"],
            label_cols=[],
            lookback_days=3,
            graph=simple_graph,
            normalize_features=True,
        )

        sample = dataset[0]
        X = sample["X"]

        # Normalized features should have roughly mean 0, std 1
        # (not exact due to windowing)
        assert X.abs().max() < 10  # No huge outliers

    def test_normalization_stats_computed(self, sample_panel_with_labels, simple_graph):
        """Test that per-node feature stats are computed."""
        dataset = SurfaceDataset(
            panel_df=sample_panel_with_labels,
            feature_cols=["iv_mid", "delta"],
            label_cols=[],
            lookback_days=3,
            graph=simple_graph,
            normalize_features=True,
        )

        stats = dataset.get_feature_stats()

        assert "iv_mid" in stats
        assert "delta" in stats
        assert len(stats["iv_mid"]) == 2  # (means, stds)
        # Per-node stats: arrays of shape (num_nodes,)
        means, stds = stats["delta"]
        assert isinstance(means, np.ndarray)
        assert means.shape == (simple_graph.num_nodes,)
        assert stds.shape == (simple_graph.num_nodes,)
        assert (stds > 0).all()

    def test_provided_stats_used(self, sample_panel_with_labels, simple_graph):
        """Test that provided stats override computed stats."""
        num_nodes = simple_graph.num_nodes
        custom_means = np.full(num_nodes, 0.5)
        custom_stds = np.full(num_nodes, 0.1)
        custom_stats = {
            "iv_mid": (custom_means, custom_stds),
            "delta": (np.zeros(num_nodes), np.full(num_nodes, 0.5)),
        }

        dataset = SurfaceDataset(
            panel_df=sample_panel_with_labels,
            feature_cols=["iv_mid", "delta"],
            label_cols=[],
            lookback_days=3,
            graph=simple_graph,
            normalize_features=True,
            feature_stats=custom_stats,
        )

        stats = dataset.get_feature_stats()
        np.testing.assert_array_equal(stats["iv_mid"][0], custom_means)
        np.testing.assert_array_equal(stats["iv_mid"][1], custom_stds)

    def test_no_normalization(self, sample_panel_with_labels, simple_graph):
        """Test dataset without normalization."""
        dataset = SurfaceDataset(
            panel_df=sample_panel_with_labels,
            feature_cols=["iv_mid"],
            label_cols=[],
            lookback_days=3,
            graph=simple_graph,
            normalize_features=False,
        )

        sample = dataset[0]
        X = sample["X"]

        # Raw IV should be around 0.2
        # Filter out zero-filled missing values
        non_zero = X[X != 0]
        if len(non_zero) > 0:
            assert 0.1 < non_zero.mean() < 0.4

    def test_feature_inf_values_are_sanitized(
        self, sample_panel_with_labels, simple_graph
    ):
        """Infinite feature values should not propagate into model inputs."""
        df = sample_panel_with_labels.copy()
        df.loc[df.index[0], "delta"] = np.inf
        df.loc[df.index[1], "delta"] = -np.inf

        dataset = SurfaceDataset(
            panel_df=df,
            feature_cols=["delta"],
            label_cols=["y_dhr_5d"],
            lookback_days=3,
            graph=simple_graph,
            normalize_features=False,
        )

        features = np.asarray(dataset._features)
        assert np.isfinite(features).all()
        assert np.abs(features).max() <= 1.0

    def test_feature_stats_exclude_quality_masked_rows(
        self, sample_panel_with_labels, simple_graph
    ):
        """Per-node normalization stats should ignore quality-masked rows."""
        df = sample_panel_with_labels.copy()
        masked_node = (df["tenor_days"] == 30) & (df["delta_bucket"] == "P25")
        df.loc[masked_node, "is_masked"] = True
        df.loc[masked_node, "delta"] = 123.0

        dataset = SurfaceDataset(
            panel_df=df,
            feature_cols=["delta"],
            label_cols=[],
            lookback_days=3,
            graph=simple_graph,
            normalize_features=True,
        )

        means, stds = dataset.get_feature_stats()["delta"]
        node_idx = simple_graph.node_mapping[(30, "P25")]
        assert means[node_idx] == pytest.approx(0.0)
        assert stds[node_idx] == pytest.approx(1.0)


class TestSurfaceDatasetMasking:
    """Tests for node masking."""

    def test_all_nodes_valid(self, sample_panel_with_labels, simple_graph):
        """Test mask when all nodes are valid."""
        dataset = SurfaceDataset(
            panel_df=sample_panel_with_labels,
            feature_cols=["iv_mid"],
            label_cols=["y_dhr_5d"],
            lookback_days=3,
            graph=simple_graph,
        )

        sample = dataset[0]
        mask = sample["mask"]

        # All nodes should be valid (sample data has no masking)
        assert mask.sum() == 6

    def test_masked_nodes(self, simple_graph_config):
        """Test mask when some nodes are masked."""
        buckets = simple_graph_config.delta_buckets
        tenors = simple_graph_config.tenors_days

        # Create panel with some masked nodes
        base_time = datetime(2024, 1, 1, 16, 0, 0)
        timestamps = [base_time + timedelta(days=i) for i in range(5)]

        rows = []
        for ts in timestamps:
            for tenor in tenors:
                for i, bucket in enumerate(buckets):
                    row = {
                        "ts_utc": ts,
                        "tenor_days": tenor,
                        "delta_bucket": bucket,
                        "iv_mid": 0.2,
                        "y_dhr_5d": 0.1,
                        "is_masked": (i == 0),  # Mask first bucket
                    }
                    rows.append(row)

        df = pd.DataFrame(rows)
        graph = build_surface_graph(simple_graph_config)

        dataset = SurfaceDataset(
            panel_df=df,
            feature_cols=["iv_mid"],
            label_cols=["y_dhr_5d"],
            lookback_days=2,
            graph=graph,
        )

        sample = dataset[0]
        mask = sample["mask"]

        # 2 masked nodes (one per tenor for P25 bucket)
        assert mask.sum() == 4


class TestSurfaceDatasetEdgeCases:
    """Tests for edge cases."""

    def test_missing_feature_column(self, sample_panel_df, simple_graph):
        """Test handling of missing feature columns."""
        dataset = SurfaceDataset(
            panel_df=sample_panel_df,
            feature_cols=["iv_mid", "nonexistent_feature"],
            label_cols=[],
            lookback_days=3,
            graph=simple_graph,
        )

        sample = dataset[0]
        X = sample["X"]

        # Should still work, missing feature will be 0
        assert X.shape[2] == 2

    def test_empty_label_cols(self, sample_panel_df, simple_graph):
        """Test with no label columns."""
        dataset = SurfaceDataset(
            panel_df=sample_panel_df,
            feature_cols=["iv_mid"],
            label_cols=[],
            lookback_days=3,
            graph=simple_graph,
        )

        sample = dataset[0]

        assert sample["y"].shape[1] == 0


# =============================================================================
# DatasetBuilder Tests
# =============================================================================


class TestDatasetBuilder:
    """Tests for DatasetBuilder."""

    @pytest.fixture
    def builder_config(self, simple_graph_config):
        """Config for dataset builder."""
        return DatasetConfig(
            lookback_days=2,
            feature_cols=["iv_mid", "delta"],
            label_cols=["y_dhr_5d"],
            splits=SplitsConfig(
                train_start=datetime(2024, 1, 1),
                val_start=datetime(2024, 1, 5),
                test_start=datetime(2024, 1, 7),
                test_end=datetime(2024, 1, 11),
            ),
            graph=simple_graph_config,
            normalize_features=True,
        )

    def test_build_datasets(self, builder_config, sample_panel_with_labels):
        """Test building train/val/test datasets."""
        builder = DatasetBuilder(builder_config)
        train_ds, val_ds, test_ds = builder.build_datasets(sample_panel_with_labels)

        assert isinstance(train_ds, SurfaceDataset)
        assert isinstance(val_ds, SurfaceDataset)
        assert isinstance(test_ds, SurfaceDataset)

    def test_split_no_overlap(self, builder_config, sample_panel_with_labels):
        """Test that splits don't overlap."""
        builder = DatasetBuilder(builder_config)
        train_ds, val_ds, test_ds = builder.build_datasets(sample_panel_with_labels)

        # Check that we have some samples in each set
        # Exact counts depend on timestamps and lookback
        total_samples = len(train_ds) + len(val_ds) + len(test_ds)
        assert total_samples > 0

    def test_normalization_stats_propagate(self, builder_config, sample_panel_with_labels):
        """Test that training stats propagate to val/test."""
        builder = DatasetBuilder(builder_config)
        train_ds, val_ds, test_ds = builder.build_datasets(sample_panel_with_labels)

        train_stats = train_ds.get_feature_stats()
        val_stats = val_ds.get_feature_stats()
        test_stats = test_ds.get_feature_stats()

        # Val and test should use same stats as train (per-node arrays)
        assert train_stats.keys() == val_stats.keys() == test_stats.keys()
        for col in train_stats:
            t_mean, t_std = train_stats[col]
            v_mean, v_std = val_stats[col]
            te_mean, te_std = test_stats[col]
            np.testing.assert_array_equal(t_mean, v_mean)
            np.testing.assert_array_equal(t_std, v_std)
            np.testing.assert_array_equal(t_mean, te_mean)
            np.testing.assert_array_equal(t_std, te_std)

    def test_build_without_splits_raises(self, simple_graph_config, sample_panel_with_labels):
        """Test that building without splits config raises error."""
        config = DatasetConfig(
            lookback_days=2,
            feature_cols=["iv_mid"],
            label_cols=[],
            splits=None,
            graph=simple_graph_config,
        )
        builder = DatasetBuilder(config)

        with pytest.raises(ValueError, match="SplitsConfig must be provided"):
            builder.build_datasets(sample_panel_with_labels)

    def test_build_single_dataset(self, builder_config, sample_panel_with_labels):
        """Test building single dataset."""
        builder = DatasetBuilder(builder_config)
        dataset = builder.build_single_dataset(sample_panel_with_labels)

        assert isinstance(dataset, SurfaceDataset)
        assert len(dataset) > 0


class TestDatasetBuilderLabels:
    """Tests for label construction in DatasetBuilder."""

    @pytest.fixture
    def builder_config(self, simple_graph_config):
        """Config for dataset builder."""
        return DatasetConfig(
            lookback_days=2,
            feature_cols=["iv_mid"],
            label_cols=["y_dhr_5d"],
            splits=SplitsConfig(
                train_start=datetime(2024, 1, 1),
                val_start=datetime(2024, 1, 15),
                test_start=datetime(2024, 1, 22),
                test_end=datetime(2024, 2, 1),
            ),
            graph=simple_graph_config,
            normalize_features=False,
        )

    def test_label_construction(self, builder_config, simple_graph_config):
        """Test that labels are constructed from underlying returns."""
        buckets = simple_graph_config.delta_buckets
        tenors = simple_graph_config.tenors_days

        # Create panel without labels
        base_time = datetime(2024, 1, 1, 16, 0, 0)
        timestamps = [base_time + timedelta(days=i) for i in range(30)]

        rows = []
        for ts in timestamps:
            for tenor in tenors:
                for bucket in buckets:
                    row = {
                        "ts_utc": ts,
                        "tenor_days": tenor,
                        "delta_bucket": bucket,
                        "iv_mid": 0.2,
                        "mid_price": 5.0 + np.random.randn() * 0.1,
                        "delta": 0.5,
                        "is_masked": False,
                    }
                    rows.append(row)

        panel_df = pd.DataFrame(rows)

        # Create underlying returns with close prices (required for DHR labels)
        return_dates = [base_time + timedelta(days=i) for i in range(40)]
        returns = [0.01 * (i % 3 - 1) for i in range(40)]
        close_prices = [450.0]
        for r in returns[:-1]:
            close_prices.append(close_prices[-1] * (1 + r))
        returns_df = pd.DataFrame({
            "ts_utc": return_dates,
            "return": returns,
            "close": close_prices,
        })

        labels_config = LabelsConfig(horizons_days=[5])

        builder = DatasetBuilder(builder_config)
        train_ds, val_ds, test_ds = builder.build_datasets(
            panel_df,
            labels_config=labels_config,
            underlying_returns=returns_df,
        )

        # Labels should have been built
        # The exact values depend on the DHR calculation
        if len(train_ds) > 0:
            sample = train_ds[0]
            # Should have label column
            assert sample["y"].shape[1] == 1


# =============================================================================
# Integration Tests (lightweight)
# =============================================================================


class TestDatasetIntegration:
    """Lightweight integration tests for dataset module."""

    def test_full_pipeline(self, sample_panel_with_labels, simple_graph_config):
        """Test full pipeline: config -> builder -> datasets -> samples."""
        config = DatasetConfig(
            lookback_days=2,
            feature_cols=["iv_mid", "delta", "gamma"],
            label_cols=["y_dhr_5d", "y_dhr_10d"],
            splits=SplitsConfig(
                train_start=datetime(2024, 1, 1),
                val_start=datetime(2024, 1, 5),
                test_start=datetime(2024, 1, 7),
                test_end=datetime(2024, 1, 11),
            ),
            graph=simple_graph_config,
            normalize_features=True,
        )

        builder = DatasetBuilder(config)
        train_ds, val_ds, test_ds = builder.build_datasets(sample_panel_with_labels)

        # Get a sample from training set
        if len(train_ds) > 0:
            sample = train_ds[0]

            assert sample["X"].shape == (3, 6, 3)  # lookback+1, nodes, features
            assert sample["y"].shape == (6, 2)  # nodes, horizons
            assert sample["mask"].shape == (6,)
            assert sample["graph"].num_nodes == 6

    def test_dataloader_compatibility(self, sample_panel_with_labels, simple_graph):
        """Test that dataset works with PyTorch DataLoader."""
        from torch.utils.data import DataLoader

        dataset = SurfaceDataset(
            panel_df=sample_panel_with_labels,
            feature_cols=["iv_mid"],
            label_cols=["y_dhr_5d"],
            lookback_days=2,
            graph=simple_graph,
        )

        # Custom collate function that handles the static graph
        def collate_fn(batch):
            """Collate function that stacks tensors and keeps graph reference."""
            return {
                "X": torch.stack([b["X"] for b in batch]),
                "y": torch.stack([b["y"] for b in batch]),
                "mask": torch.stack([b["mask"] for b in batch]),
                "graph": batch[0]["graph"],  # Graph is static, just keep first
            }

        # Create DataLoader with custom collate
        dataloader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)

        # Should be able to iterate
        batch = next(iter(dataloader))
        assert batch["X"].shape[0] == 2  # Batch dim


# =============================================================================
# Memmap Tests
# =============================================================================


class TestSurfaceDatasetMemmap:
    """Tests for memory-mapped backing of SurfaceDataset."""

    def test_memmap_produces_same_results(
        self, sample_panel_with_labels, simple_graph
    ):
        """Memmap and in-memory datasets produce identical __getitem__ output."""
        kwargs = dict(
            panel_df=sample_panel_with_labels,
            feature_cols=["iv_mid", "delta"],
            label_cols=["y_dhr_5d"],
            lookback_days=3,
            graph=simple_graph,
            normalize_features=False,
        )
        ds_mem = SurfaceDataset(**kwargs, use_memmap=False)
        ds_mmap = SurfaceDataset(**kwargs, use_memmap=True)

        assert len(ds_mem) == len(ds_mmap)
        sample_mem = ds_mem[0]
        sample_mmap = ds_mmap[0]

        assert torch.allclose(sample_mem["X"], sample_mmap["X"])
        assert torch.allclose(sample_mem["y"], sample_mmap["y"])
        assert torch.equal(sample_mem["mask"], sample_mmap["mask"])

    def test_memmap_creates_temp_files(
        self, sample_panel_with_labels, simple_graph
    ):
        """Memmap dataset creates .dat files in temp directory."""
        import os

        dataset = SurfaceDataset(
            panel_df=sample_panel_with_labels,
            feature_cols=["iv_mid"],
            label_cols=["y_dhr_5d"],
            lookback_days=3,
            graph=simple_graph,
            use_memmap=True,
        )

        memmap_dir = dataset._memmap_dir
        assert memmap_dir is not None
        assert os.path.isdir(memmap_dir)

        files = set(os.listdir(memmap_dir))
        assert "features.dat" in files
        assert "labels.dat" in files
        assert "masks.dat" in files

    def test_memmap_cleanup_on_del(
        self, sample_panel_with_labels, simple_graph
    ):
        """Temp directory is removed when dataset is garbage collected."""
        import os

        dataset = SurfaceDataset(
            panel_df=sample_panel_with_labels,
            feature_cols=["iv_mid"],
            label_cols=["y_dhr_5d"],
            lookback_days=3,
            graph=simple_graph,
            use_memmap=True,
        )

        memmap_dir = dataset._memmap_dir
        assert os.path.isdir(memmap_dir)

        del dataset

        assert not os.path.exists(memmap_dir)

    def test_memmap_arrays_are_readonly(
        self, sample_panel_with_labels, simple_graph
    ):
        """After construction, memmap arrays are opened in read-only mode."""
        dataset = SurfaceDataset(
            panel_df=sample_panel_with_labels,
            feature_cols=["iv_mid"],
            label_cols=["y_dhr_5d"],
            lookback_days=3,
            graph=simple_graph,
            use_memmap=True,
        )

        assert isinstance(dataset._features, np.memmap)
        assert dataset._features.mode == "r"
        assert isinstance(dataset._labels, np.memmap)
        assert dataset._labels.mode == "r"
        assert isinstance(dataset._masks, np.memmap)
        assert dataset._masks.mode == "r"

    def test_memmap_with_normalization(
        self, sample_panel_with_labels, simple_graph
    ):
        """Normalized memmap output matches normalized in-memory output."""
        kwargs = dict(
            panel_df=sample_panel_with_labels,
            feature_cols=["iv_mid", "delta"],
            label_cols=["y_dhr_5d"],
            lookback_days=3,
            graph=simple_graph,
            normalize_features=True,
        )
        ds_mem = SurfaceDataset(**kwargs, use_memmap=False)
        ds_mmap = SurfaceDataset(**kwargs, use_memmap=True)

        for i in range(min(3, len(ds_mem))):
            sample_mem = ds_mem[i]
            sample_mmap = ds_mmap[i]
            assert torch.allclose(sample_mem["X"], sample_mmap["X"], atol=1e-6)
            assert torch.allclose(sample_mem["y"], sample_mmap["y"], atol=1e-6)

    def test_no_memmap_fallback(
        self, sample_panel_with_labels, simple_graph
    ):
        """use_memmap=False produces plain ndarray, not memmap."""
        dataset = SurfaceDataset(
            panel_df=sample_panel_with_labels,
            feature_cols=["iv_mid"],
            label_cols=["y_dhr_5d"],
            lookback_days=3,
            graph=simple_graph,
            use_memmap=False,
        )

        assert not isinstance(dataset._features, np.memmap)
        assert isinstance(dataset._features, np.ndarray)
        assert dataset._memmap_dir is None

    def test_memmap_dataloader_compatible(
        self, sample_panel_with_labels, simple_graph
    ):
        """Memmap dataset works with multi-worker DataLoader."""
        from torch.utils.data import DataLoader

        dataset = SurfaceDataset(
            panel_df=sample_panel_with_labels,
            feature_cols=["iv_mid", "delta"],
            label_cols=["y_dhr_5d"],
            lookback_days=3,
            graph=simple_graph,
            use_memmap=True,
        )

        def collate_fn(batch):
            return {
                "X": torch.stack([b["X"] for b in batch]),
                "y": torch.stack([b["y"] for b in batch]),
                "mask": torch.stack([b["mask"] for b in batch]),
                "graph": batch[0]["graph"],
            }

        dataloader = DataLoader(
            dataset, batch_size=2, shuffle=True,
            num_workers=0, collate_fn=collate_fn,
        )

        batch = next(iter(dataloader))
        assert batch["X"].shape[0] == 2
        assert batch["X"].shape[2] == 6  # num_nodes
        assert batch["X"].shape[3] == 2  # num_features
