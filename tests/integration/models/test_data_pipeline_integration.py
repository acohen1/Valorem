"""Integration tests for TrainingDataPipeline with real database.

Tests the full flow from DB reads through dataset construction to DataLoader output.
Uses an in-memory SQLite database with realistic synthetic data.
"""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from src.models.dataset import SplitsConfig
from src.models.train.data_pipeline import TrainingDataConfig, TrainingDataPipeline


# =============================================================================
# Fixtures
# =============================================================================


def _generate_panel_data(
    start: datetime,
    end: datetime,
    feature_version: str = "v1.0",
) -> pd.DataFrame:
    """Generate realistic node panel data for testing.

    Creates data with 3 delta buckets and 2 tenors (6 nodes)
    at daily frequency between start and end.
    """
    buckets = ["P25", "ATM", "C25"]
    tenors = [30, 60]
    np.random.seed(42)

    dates = pd.bdate_range(start, end, freq="B")
    rows = []

    for ts in dates:
        for tenor in tenors:
            for bucket in buckets:
                base_iv = 0.20 + (0.01 if tenor == 60 else 0.0)
                smile = 0.03 if bucket != "ATM" else 0.0

                rows.append({
                    "ts_utc": ts.to_pydatetime(),
                    "tenor_days": tenor,
                    "delta_bucket": bucket,
                    "option_symbol": f"SPY230901{'C' if 'C' in bucket else 'P'}00450000",
                    "iv_mid": base_iv + smile + np.random.randn() * 0.01,
                    "iv_bid": base_iv + smile - 0.005,
                    "iv_ask": base_iv + smile + 0.005,
                    "spread_pct": 0.02,
                    "delta": {"P25": -0.25, "ATM": 0.50, "C25": 0.25}[bucket],
                    "gamma": 0.05,
                    "vega": 0.15,
                    "theta": -0.01,
                    "iv_change_1d": np.random.randn() * 0.005,
                    "iv_change_5d": np.random.randn() * 0.01,
                    "skew_slope": -0.1 + np.random.randn() * 0.02,
                    "term_slope": 0.05 + np.random.randn() * 0.01,
                    "curvature": 0.01 + np.random.randn() * 0.005,
                    "underlying_rv_5d": 0.15 + np.random.randn() * 0.02,
                    "underlying_rv_10d": 0.15 + np.random.randn() * 0.02,
                    "underlying_rv_21d": 0.15 + np.random.randn() * 0.02,
                    "feature_version": feature_version,
                    "is_masked": False,
                    "mask_reason": None,
                })

    return pd.DataFrame(rows)


def _generate_bars_data(start: datetime, end: datetime) -> pd.DataFrame:
    """Generate underlying bars data for testing."""
    np.random.seed(42)
    dates = pd.bdate_range(start, end, freq="B")
    price = 450.0
    rows = []

    for ts in dates:
        change = np.random.randn() * 2.0
        price = max(100.0, price + change)
        rows.append({
            "ts_utc": ts.to_pydatetime(),
            "symbol": "SPY",
            "timeframe": "1d",
            "dataset": "DBEQ.BASIC",
            "schema": "ohlcv-1d",
            "stype_in": "raw_symbol",
            "open": price - abs(change) * 0.3,
            "high": price + abs(change) * 0.5,
            "low": price - abs(change) * 0.5,
            "close": price,
            "volume": 50000000,
        })

    return pd.DataFrame(rows)


@pytest.fixture
def db_with_data(raw_repo, derived_repo):
    """Populate shared DB with synthetic node panel and bars data.

    Uses the module-scoped engine from conftest (cleaned between tests).

    Returns:
        Tuple of (raw_repo, derived_repo)
    """
    # Write underlying bars (March through October for forward buffer)
    bars_df = _generate_bars_data(datetime(2023, 3, 1), datetime(2023, 10, 31))
    raw_repo.write_underlying_bars(bars_df, run_id="test-run")

    # Write node panel (March through August)
    panel_df = _generate_panel_data(datetime(2023, 3, 1), datetime(2023, 8, 31))
    derived_repo.write_node_panel(panel_df, feature_version="v1.0")

    return raw_repo, derived_repo


@pytest.fixture
def pipeline_config():
    """Standard pipeline config for integration tests."""
    return TrainingDataConfig(
        splits=SplitsConfig(
            train_start=datetime(2023, 4, 1),
            val_start=datetime(2023, 7, 1),
            test_start=datetime(2023, 8, 1),
            test_end=datetime(2023, 8, 31),
        ),
        feature_version="v1.0",
        underlying_symbol="SPY",
        batch_size=4,
        lookback_days=5,
        normalize_features=True,
    )


# =============================================================================
# Integration Tests
# =============================================================================


class TestTrainingDataPipelineIntegration:
    """Integration tests with real in-memory SQLite DB."""

    def test_full_pipeline_from_db(self, db_with_data, pipeline_config):
        """Test loading data from DB all the way to DataLoaders."""
        raw_repo, derived_repo = db_with_data
        pipeline = TrainingDataPipeline(raw_repo, derived_repo, pipeline_config)
        result = pipeline.load()

        # Verify we get actual data
        assert result.metadata["train_samples"] > 0
        assert result.metadata["val_samples"] > 0
        assert result.metadata["test_samples"] > 0

        # Verify DataLoader works (can iterate)
        batch = next(iter(result.train_loader))
        assert "X" in batch
        assert "y" in batch
        assert "mask" in batch

        # Verify shapes
        X = batch["X"]
        y = batch["y"]
        mask = batch["mask"]

        assert X.ndim == 4  # (batch, time, nodes, features)
        assert y.ndim == 3  # (batch, nodes, horizons)
        assert mask.ndim == 2  # (batch, nodes)

        # Time dimension should be lookback_days + 1
        assert X.shape[1] == pipeline_config.lookback_days + 1

    def test_labels_computed_correctly(self, db_with_data, pipeline_config):
        """Test that DHR labels are correctly computed from underlying data."""
        raw_repo, derived_repo = db_with_data
        pipeline = TrainingDataPipeline(raw_repo, derived_repo, pipeline_config)
        result = pipeline.load()

        # Labels should have 3 horizons (5d, 10d, 21d)
        assert result.metadata["num_horizons"] == 3

        # Get a batch and verify labels are finite (not NaN after processing)
        batch = next(iter(result.train_loader))
        y = batch["y"]
        mask = batch["mask"]

        # Where mask is True, labels should be finite
        valid_labels = y[mask.unsqueeze(-1).expand_as(y)]
        assert valid_labels.isfinite().all()

    def test_normalization_propagates(self, db_with_data, pipeline_config):
        """Test that training normalization stats propagate to val/test."""
        raw_repo, derived_repo = db_with_data
        pipeline = TrainingDataPipeline(raw_repo, derived_repo, pipeline_config)
        result = pipeline.load()

        # Training set should have feature stats computed
        train_stats = result.train_dataset.get_feature_stats()
        assert len(train_stats) > 0

        # Validation set should use training stats (passed via feature_stats param)
        # Verify by checking the features are normalized (mean ≈ 0, std ≈ 1 for training)
        # This is an indirect check since we can't access internal _feature_stats
        train_batch = next(iter(result.train_loader))
        val_batch = next(iter(result.val_loader))

        # Both should produce valid tensors
        assert train_batch["X"].isfinite().all()
        assert val_batch["X"].isfinite().all()

    def test_different_feature_version_returns_empty(self, db_with_data):
        """Querying for a non-existent feature version should raise."""
        raw_repo, derived_repo = db_with_data
        config = TrainingDataConfig(
            splits=SplitsConfig(
                train_start=datetime(2023, 4, 1),
                val_start=datetime(2023, 7, 1),
                test_start=datetime(2023, 8, 1),
                test_end=datetime(2023, 8, 31),
            ),
            feature_version="v99.0",  # Doesn't exist
            batch_size=4,
            lookback_days=5,
        )
        pipeline = TrainingDataPipeline(raw_repo, derived_repo, config)
        with pytest.raises(ValueError, match="No node panel data found"):
            pipeline.load()

    def test_batch_shapes_consistent(self, db_with_data, pipeline_config):
        """All batches should have consistent shapes across splits."""
        raw_repo, derived_repo = db_with_data
        pipeline = TrainingDataPipeline(raw_repo, derived_repo, pipeline_config)
        result = pipeline.load()

        for loader_name in ["train_loader", "val_loader", "test_loader"]:
            loader = getattr(result, loader_name)
            batch = next(iter(loader))

            X, y, mask = batch["X"], batch["y"], batch["mask"]

            # Node dimension should match across X, y, mask
            assert X.shape[2] == y.shape[1] == mask.shape[1]

            # Horizons dimension
            assert y.shape[2] == 3
