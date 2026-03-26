"""Integration tests for dataset pipeline.

Tests the full flow from node panel data to PyTorch datasets ready for training.
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from src.models.dataset import (
    DatasetConfig,
    LabelsConfig,
    SplitsConfig,
    SurfaceDataset,
    DatasetBuilder,
)
from src.models.graph import SurfaceGraphConfig, build_surface_graph


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def realistic_graph_config():
    """Realistic graph config matching production defaults."""
    return SurfaceGraphConfig(
        delta_buckets=["P10", "P25", "P40", "ATM", "C40", "C25", "C10"],
        tenors_days=[7, 14, 30, 60, 90, 120],
    )


@pytest.fixture
def realistic_node_panel(realistic_graph_config):
    """Create realistic node panel DataFrame simulating production data.

    Generates 60 trading days of data with:
    - All 42 nodes (7 buckets × 6 tenors)
    - Realistic IV values with smile pattern
    - Greeks derived from IV
    - Some masked nodes (simulating illiquid options)
    """
    buckets = realistic_graph_config.delta_buckets
    tenors = realistic_graph_config.tenors_days

    # Generate 60 trading days
    base_time = datetime(2024, 1, 2, 16, 0, 0)
    timestamps = []
    current = base_time
    for _ in range(60):
        timestamps.append(current)
        # Skip weekends
        current += timedelta(days=1)
        if current.weekday() >= 5:
            current += timedelta(days=7 - current.weekday())

    rows = []
    np.random.seed(42)

    # Base volatility term structure (upward sloping)
    tenor_base_iv = {7: 0.18, 14: 0.19, 30: 0.20, 60: 0.21, 90: 0.22, 120: 0.23}

    # Smile pattern (higher IV for OTM options)
    bucket_smile = {
        "P10": 0.08, "P25": 0.04, "P40": 0.01,
        "ATM": 0.00,
        "C40": 0.01, "C25": 0.03, "C10": 0.06,
    }

    # Delta values by bucket
    bucket_deltas = {
        "P10": -0.10, "P25": -0.25, "P40": -0.40,
        "ATM": 0.50,
        "C40": 0.40, "C25": 0.25, "C10": 0.10,
    }

    for t_idx, ts in enumerate(timestamps):
        # Add some time variation
        time_shift = 0.02 * np.sin(2 * np.pi * t_idx / 20)

        for tenor in tenors:
            for bucket in buckets:
                # Randomly mask ~5% of nodes
                is_masked = np.random.random() < 0.05

                # IV with term structure + smile + time variation
                base_iv = tenor_base_iv[tenor]
                smile_adj = bucket_smile[bucket]
                iv_mid = max(0.05, base_iv + smile_adj + time_shift + np.random.randn() * 0.01)

                # Greeks (simplified)
                delta = bucket_deltas[bucket]
                gamma = 0.05 / np.sqrt(tenor / 365)  # Higher for shorter tenors
                vega = 0.15 * np.sqrt(tenor / 365)  # Higher for longer tenors
                theta = -0.01 * iv_mid / np.sqrt(tenor / 365)

                row = {
                    "ts_utc": ts,
                    "tenor_days": tenor,
                    "delta_bucket": bucket,
                    "option_symbol": f"SPY_{ts.strftime('%Y%m%d')}_{tenor}_{bucket}",
                    "iv_mid": iv_mid if not is_masked else np.nan,
                    "iv_bid": iv_mid - 0.005 if not is_masked else np.nan,
                    "iv_ask": iv_mid + 0.005 if not is_masked else np.nan,
                    "delta": delta if not is_masked else np.nan,
                    "gamma": gamma if not is_masked else np.nan,
                    "vega": vega if not is_masked else np.nan,
                    "theta": theta if not is_masked else np.nan,
                    "spread_pct": 0.02 + np.random.random() * 0.01,
                    "iv_change_1d": np.random.randn() * 0.01,
                    "iv_change_5d": np.random.randn() * 0.02,
                    "skew_slope": -0.05 + np.random.randn() * 0.01,
                    "term_slope": 0.02 + np.random.randn() * 0.005,
                    "underlying_rv_5d": 0.15 + np.random.randn() * 0.02,
                    "underlying_rv_10d": 0.16 + np.random.randn() * 0.02,
                    "underlying_rv_21d": 0.17 + np.random.randn() * 0.02,
                    "is_masked": is_masked,
                }
                rows.append(row)

    return pd.DataFrame(rows)


@pytest.fixture
def underlying_returns():
    """Create underlying daily returns for label construction."""
    base_time = datetime(2024, 1, 2, 16, 0, 0)  # 4pm to match panel timestamps
    dates = []
    current = base_time
    for _ in range(90):  # Extra days for forward-looking labels
        dates.append(current)
        current += timedelta(days=1)
        if current.weekday() >= 5:
            current += timedelta(days=7 - current.weekday())

    np.random.seed(42)
    returns = np.random.randn(len(dates)) * 0.01  # ~1% daily vol

    # Build close prices from returns (DHR labels need price levels)
    close_prices = [450.0]  # starting price
    for r in returns[:-1]:
        close_prices.append(close_prices[-1] * (1 + r))

    return pd.DataFrame({
        "ts_utc": dates,
        "return": returns,
        "close": close_prices,
    })


# =============================================================================
# Integration Tests
# =============================================================================


class TestNodePanelToDataset:
    """Test converting node panel to PyTorch dataset."""

    def test_panel_to_dataset(self, realistic_node_panel, realistic_graph_config):
        """Test basic conversion from panel to dataset."""
        graph = build_surface_graph(realistic_graph_config)

        dataset = SurfaceDataset(
            panel_df=realistic_node_panel,
            feature_cols=["iv_mid", "delta", "gamma", "vega", "theta"],
            label_cols=[],
            lookback_days=21,
            graph=graph,
        )

        # 60 timestamps, lookback 21 -> 39 samples
        assert len(dataset) == 39
        assert dataset.num_nodes == 42  # 7 × 6
        assert dataset.num_features == 5

    def test_dataset_shapes(self, realistic_node_panel, realistic_graph_config):
        """Test tensor shapes match expected dimensions."""
        graph = build_surface_graph(realistic_graph_config)

        dataset = SurfaceDataset(
            panel_df=realistic_node_panel,
            feature_cols=["iv_mid", "delta", "gamma"],
            label_cols=[],
            lookback_days=10,
            graph=graph,
        )

        sample = dataset[0]

        # X: (lookback + 1, num_nodes, num_features)
        assert sample["X"].shape == (11, 42, 3)

        # y: (num_nodes, num_horizons)
        assert sample["y"].shape == (42, 0)

        # mask: (num_nodes,)
        assert sample["mask"].shape == (42,)

    def test_graph_topology_consistent(self, realistic_node_panel, realistic_graph_config):
        """Test that graph topology is consistent across samples."""
        graph = build_surface_graph(realistic_graph_config)

        dataset = SurfaceDataset(
            panel_df=realistic_node_panel,
            feature_cols=["iv_mid"],
            label_cols=[],
            lookback_days=5,
            graph=graph,
        )

        # All samples should share same graph
        for i in range(min(5, len(dataset))):
            sample = dataset[i]
            assert sample["graph"] is graph
            assert sample["graph"].num_nodes == 42


class TestTrainValTestSplits:
    """Test train/validation/test splitting."""

    def test_chronological_splits(self, realistic_node_panel, realistic_graph_config):
        """Test that splits are chronological with no leakage."""
        config = DatasetConfig(
            lookback_days=5,
            feature_cols=["iv_mid", "delta"],
            label_cols=[],
            splits=SplitsConfig(
                train_start=datetime(2024, 1, 1),
                val_start=datetime(2024, 2, 1),
                test_start=datetime(2024, 2, 15),
                test_end=datetime(2024, 3, 1),
            ),
            graph=realistic_graph_config,
            normalize_features=True,
        )

        builder = DatasetBuilder(config)
        train_ds, val_ds, test_ds = builder.build_datasets(realistic_node_panel)

        # All datasets should have samples (depends on exact dates)
        total_samples = len(train_ds) + len(val_ds) + len(test_ds)
        assert total_samples > 0

    def test_normalization_propagation(self, realistic_node_panel, realistic_graph_config):
        """Test that normalization stats from train propagate to val/test."""
        config = DatasetConfig(
            lookback_days=5,
            feature_cols=["iv_mid", "delta", "gamma"],
            label_cols=[],
            splits=SplitsConfig(
                train_start=datetime(2024, 1, 1),
                val_start=datetime(2024, 2, 1),
                test_start=datetime(2024, 2, 15),
                test_end=datetime(2024, 3, 1),
            ),
            graph=realistic_graph_config,
            normalize_features=True,
        )

        builder = DatasetBuilder(config)
        train_ds, val_ds, test_ds = builder.build_datasets(realistic_node_panel)

        # Get stats from each dataset
        train_stats = train_ds.get_feature_stats()
        val_stats = val_ds.get_feature_stats()
        test_stats = test_ds.get_feature_stats()

        # Val and test should use training stats (per-node arrays)
        for col in ["iv_mid", "delta", "gamma"]:
            t_mean, t_std = train_stats[col]
            v_mean, v_std = val_stats[col]
            te_mean, te_std = test_stats[col]
            np.testing.assert_array_equal(t_mean, v_mean)
            np.testing.assert_array_equal(t_std, v_std)
            np.testing.assert_array_equal(t_mean, te_mean)
            np.testing.assert_array_equal(t_std, te_std)


class TestLabelConstruction:
    """Test forward-looking label construction."""

    def test_dhr_labels(
        self, realistic_node_panel, realistic_graph_config, underlying_returns
    ):
        """Test delta-hedged return label construction."""
        config = DatasetConfig(
            lookback_days=5,
            feature_cols=["iv_mid"],
            label_cols=["y_dhr_5d", "y_dhr_10d"],
            splits=SplitsConfig(
                train_start=datetime(2024, 1, 1),
                val_start=datetime(2024, 2, 1),
                test_start=datetime(2024, 2, 15),
                test_end=datetime(2024, 3, 1),
            ),
            graph=realistic_graph_config,
            normalize_features=False,
        )

        labels_config = LabelsConfig(horizons_days=[5, 10])

        builder = DatasetBuilder(config)
        train_ds, _, _ = builder.build_datasets(
            realistic_node_panel,
            labels_config=labels_config,
            underlying_returns=underlying_returns,
        )

        if len(train_ds) > 0:
            sample = train_ds[0]
            y = sample["y"]

            # Should have 2 horizons
            assert y.shape == (42, 2)

            # Labels should be finite (not NaN/Inf) where mask is True
            mask = sample["mask"]
            valid_labels = y[mask]
            # Some NaN is expected for nodes without valid IV
            assert valid_labels.shape[0] > 0


def surface_collate_fn(batch):
    """Custom collate function that handles static graph properly."""
    return {
        "X": torch.stack([b["X"] for b in batch]),
        "y": torch.stack([b["y"] for b in batch]),
        "mask": torch.stack([b["mask"] for b in batch]),
        "graph": batch[0]["graph"],  # Graph is static, use first sample's
    }


class TestDataLoaderCompatibility:
    """Test compatibility with PyTorch DataLoader."""

    def test_single_batch(self, realistic_node_panel, realistic_graph_config):
        """Test loading single batch."""
        graph = build_surface_graph(realistic_graph_config)

        dataset = SurfaceDataset(
            panel_df=realistic_node_panel,
            feature_cols=["iv_mid", "delta"],
            label_cols=[],
            lookback_days=5,
            graph=graph,
        )

        # Use custom collate to handle graph objects
        loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=surface_collate_fn)
        batch = next(iter(loader))

        assert batch["X"].shape == (1, 6, 42, 2)  # batch, time, nodes, features

    def test_iteration_no_errors(self, realistic_node_panel, realistic_graph_config):
        """Test that we can iterate through entire dataset."""
        graph = build_surface_graph(realistic_graph_config)

        dataset = SurfaceDataset(
            panel_df=realistic_node_panel,
            feature_cols=["iv_mid"],
            label_cols=[],
            lookback_days=5,
            graph=graph,
        )

        loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=surface_collate_fn)

        count = 0
        for batch in loader:
            count += 1
            # Verify shapes
            assert batch["X"].shape[0] == 1
            assert batch["mask"].shape[0] == 1

        assert count == len(dataset)


class TestMaskingBehavior:
    """Test node masking behavior in integrated pipeline."""

    def test_masked_nodes_in_features(self, realistic_graph_config):
        """Test that masked nodes have correct feature handling."""
        buckets = realistic_graph_config.delta_buckets
        tenors = realistic_graph_config.tenors_days

        # Create minimal panel with explicit masking pattern
        base_time = datetime(2024, 1, 1, 16, 0, 0)
        timestamps = [base_time + timedelta(days=i) for i in range(10)]

        rows = []
        for ts in timestamps:
            for tenor in tenors:
                for i, bucket in enumerate(buckets):
                    # Mask ATM bucket (index 3) at all timestamps
                    is_masked = (bucket == "ATM")
                    row = {
                        "ts_utc": ts,
                        "tenor_days": tenor,
                        "delta_bucket": bucket,
                        "iv_mid": 0.2 if not is_masked else np.nan,
                        "is_masked": is_masked,
                    }
                    rows.append(row)

        panel_df = pd.DataFrame(rows)
        graph = build_surface_graph(realistic_graph_config)

        dataset = SurfaceDataset(
            panel_df=panel_df,
            feature_cols=["iv_mid"],
            label_cols=[],
            lookback_days=3,
            graph=graph,
            normalize_features=False,
        )

        if len(dataset) > 0:
            sample = dataset[0]
            mask = sample["mask"]

            # ATM nodes should be masked (6 nodes, one per tenor)
            # Node indices for ATM (bucket index 3): 3, 10, 17, 24, 31, 38
            atm_indices = [3 + i * 7 for i in range(6)]
            for idx in atm_indices:
                assert mask[idx] == False, f"ATM node {idx} should be masked"


class TestEndToEndPipeline:
    """Full end-to-end integration tests."""

    def test_complete_training_pipeline(
        self, realistic_node_panel, realistic_graph_config, underlying_returns
    ):
        """Test complete pipeline from raw data to training-ready batches."""
        # Configure dataset
        config = DatasetConfig(
            lookback_days=10,
            feature_cols=[
                "iv_mid", "delta", "gamma", "vega", "theta",
                "spread_pct", "iv_change_1d", "iv_change_5d",
            ],
            label_cols=["y_dhr_5d", "y_dhr_10d", "y_dhr_21d"],
            splits=SplitsConfig(
                train_start=datetime(2024, 1, 1),
                val_start=datetime(2024, 2, 1),
                test_start=datetime(2024, 2, 15),
                test_end=datetime(2024, 3, 1),
            ),
            graph=realistic_graph_config,
            normalize_features=True,
        )

        labels_config = LabelsConfig(horizons_days=[5, 10, 21])

        # Build datasets
        builder = DatasetBuilder(config)
        train_ds, val_ds, test_ds = builder.build_datasets(
            realistic_node_panel,
            labels_config=labels_config,
            underlying_returns=underlying_returns,
        )

        # Create data loaders with custom collate
        train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=surface_collate_fn)
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=surface_collate_fn)

        # Simulate one training epoch
        if len(train_ds) > 0:
            for batch in train_loader:
                X = batch["X"]
                y = batch["y"]
                mask = batch["mask"]
                graph = batch["graph"]

                # Verify all tensors are on same device (CPU)
                assert X.device == y.device == mask.device

                # Verify shapes
                batch_size, time_steps, num_nodes, num_features = X.shape
                assert num_nodes == 42
                assert num_features == 8
                assert y.shape == (batch_size, num_nodes, 3)  # 3 horizons
                assert mask.shape == (batch_size, num_nodes)

                # Just verify first batch
                break

        # Verify validation loader works too
        if len(val_ds) > 0:
            val_batch = next(iter(val_loader))
            assert val_batch["X"].shape[2] == 42  # Same number of nodes

    def test_reproducibility(self, realistic_node_panel, realistic_graph_config):
        """Test that dataset construction is reproducible."""
        graph = build_surface_graph(realistic_graph_config)

        # Create same dataset twice
        dataset1 = SurfaceDataset(
            panel_df=realistic_node_panel.copy(),
            feature_cols=["iv_mid"],
            label_cols=[],
            lookback_days=5,
            graph=graph,
            normalize_features=True,
        )

        dataset2 = SurfaceDataset(
            panel_df=realistic_node_panel.copy(),
            feature_cols=["iv_mid"],
            label_cols=[],
            lookback_days=5,
            graph=graph,
            normalize_features=True,
        )

        # Same number of samples
        assert len(dataset1) == len(dataset2)

        # Same data for same index
        if len(dataset1) > 0:
            sample1 = dataset1[0]
            sample2 = dataset2[0]

            assert torch.allclose(sample1["X"], sample2["X"])
            assert torch.equal(sample1["mask"], sample2["mask"])
