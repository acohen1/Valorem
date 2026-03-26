"""Unit tests for TrainingDataPipeline."""

from datetime import date, datetime, timedelta
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from src.config.schema import DatasetSplitsConfig
from src.models.dataset import LabelsConfig, SplitsConfig
from src.models.train.data_pipeline import (
    TrainingData,
    TrainingDataConfig,
    TrainingDataPipeline,
    _subtract_business_days,
    build_splits_from_yaml,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def splits():
    """Standard test splits covering a small date range."""
    return SplitsConfig(
        train_start=datetime(2023, 4, 1),
        val_start=datetime(2023, 7, 1),
        test_start=datetime(2023, 8, 1),
        test_end=datetime(2023, 8, 31),
    )


@pytest.fixture
def config(splits):
    """Standard TrainingDataConfig for tests."""
    return TrainingDataConfig(
        splits=splits,
        feature_version="v1.0",
        underlying_symbol="SPY",
        batch_size=4,
        lookback_days=5,
        normalize_features=True,
    )


def _make_panel_df(
    start: datetime,
    end: datetime,
    buckets: list[str] | None = None,
    tenors: list[int] | None = None,
) -> pd.DataFrame:
    """Create a realistic node panel DataFrame for testing.

    Args:
        start: Start date (inclusive)
        end: End date (exclusive)
        buckets: Delta bucket names (default: 3 buckets)
        tenors: Tenor days (default: 2 tenors)

    Returns:
        DataFrame matching node_panel schema
    """
    if buckets is None:
        buckets = ["P25", "ATM", "C25"]
    if tenors is None:
        tenors = [30, 60]

    # Generate daily timestamps (skip weekends)
    dates = pd.bdate_range(start, end, freq="B")

    rows = []
    for ts in dates:
        for tenor in tenors:
            for bucket in buckets:
                rows.append({
                    "ts_utc": ts.to_pydatetime(),
                    "tenor_days": tenor,
                    "delta_bucket": bucket,
                    "mid_price": 5.0 + np.random.randn() * 0.3,
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
                    "feature_version": "v1.0",
                })

    return pd.DataFrame(rows)


def _make_bars_df(start: datetime, end: datetime) -> pd.DataFrame:
    """Create underlying bars DataFrame for testing.

    Args:
        start: Start date (inclusive)
        end: End date (exclusive)

    Returns:
        DataFrame with ts_utc, close, and other OHLCV columns
    """
    dates = pd.bdate_range(start, end, freq="B")
    price = 450.0
    rows = []
    for ts in dates:
        change = np.random.randn() * 2.0
        price += change
        rows.append({
            "ts_utc": ts.to_pydatetime(),
            "symbol": "SPY",
            "open": price - abs(change) * 0.3,
            "high": price + abs(change) * 0.5,
            "low": price - abs(change) * 0.5,
            "close": price,
            "volume": 50000000,
        })

    return pd.DataFrame(rows)


@pytest.fixture
def mock_repos(config):
    """Create mock repositories that return synthetic DataFrames."""
    raw_repo = MagicMock()
    derived_repo = MagicMock()

    # Generate panel data covering the full range (with lookback buffer)
    panel_df = _make_panel_df(
        start=datetime(2023, 3, 1),
        end=datetime(2023, 8, 31),
        buckets=["P25", "ATM", "C25"],
        tenors=[30, 60],
    )
    derived_repo.read_node_panel.return_value = panel_df

    # Generate bars data covering range + forward buffer
    bars_df = _make_bars_df(
        start=datetime(2023, 4, 1),
        end=datetime(2023, 9, 30),
    )
    raw_repo.read_underlying_bars.return_value = bars_df

    return raw_repo, derived_repo


# =============================================================================
# TestTrainingDataConfig
# =============================================================================


class TestTrainingDataConfig:
    """Tests for TrainingDataConfig dataclass."""

    def test_defaults(self, splits):
        """Test default values."""
        config = TrainingDataConfig(splits=splits)
        assert config.feature_version == "v1.0"
        assert config.underlying_symbol == "SPY"
        assert config.batch_size == 32
        assert config.lookback_days == 21
        assert config.normalize_features is True
        assert config.max_label_horizon == 21
        assert config.num_workers == 4

    def test_custom_values(self, splits):
        """Test custom values are accepted."""
        config = TrainingDataConfig(
            splits=splits,
            feature_version="v2.0",
            underlying_symbol="QQQ",
            batch_size=16,
            lookback_days=10,
            normalize_features=False,
        )
        assert config.feature_version == "v2.0"
        assert config.underlying_symbol == "QQQ"
        assert config.batch_size == 16
        assert config.lookback_days == 10
        assert config.normalize_features is False

    def test_labels_default(self, splits):
        """Test default labels config has expected horizons."""
        config = TrainingDataConfig(splits=splits)
        assert config.labels.horizons_days == [5, 10, 21]


# =============================================================================
# TestBuildSplitsFromYaml
# =============================================================================


class TestBuildSplitsFromYaml:
    """Tests for build_splits_from_yaml function."""

    def test_basic_conversion(self):
        """Test 6-date to 4-date conversion."""
        yaml_splits = DatasetSplitsConfig(
            train_start=date(2023, 1, 1),
            train_end=date(2023, 6, 30),
            val_start=date(2023, 7, 1),
            val_end=date(2023, 9, 30),
            test_start=date(2023, 10, 1),
            test_end=date(2023, 12, 31),
        )
        splits = build_splits_from_yaml(yaml_splits)

        assert splits.train_start == datetime(2023, 1, 1)
        assert splits.val_start == datetime(2023, 7, 1)
        assert splits.test_start == datetime(2023, 10, 1)
        assert splits.test_end == datetime(2023, 12, 31)

    def test_dev_config_dates(self):
        """Test with actual dev.yaml date ranges."""
        yaml_splits = DatasetSplitsConfig(
            train_start=date(2023, 4, 1),
            train_end=date(2023, 6, 30),
            val_start=date(2023, 7, 1),
            val_end=date(2023, 7, 31),
            test_start=date(2023, 8, 1),
            test_end=date(2023, 8, 31),
        )
        splits = build_splits_from_yaml(yaml_splits)

        assert splits.train_start < splits.val_start
        assert splits.val_start < splits.test_start
        assert splits.test_start < splits.test_end

    def test_returns_datetime_not_date(self):
        """Splits should use datetime objects, not date."""
        yaml_splits = DatasetSplitsConfig(
            train_start=date(2023, 4, 1),
            train_end=date(2023, 6, 30),
            val_start=date(2023, 7, 1),
            val_end=date(2023, 7, 31),
            test_start=date(2023, 8, 1),
            test_end=date(2023, 8, 31),
        )
        splits = build_splits_from_yaml(yaml_splits)

        assert isinstance(splits.train_start, datetime)
        assert isinstance(splits.val_start, datetime)
        assert isinstance(splits.test_start, datetime)
        assert isinstance(splits.test_end, datetime)


# =============================================================================
# TestSubtractBusinessDays
# =============================================================================


class TestSubtractBusinessDays:
    """Tests for _subtract_business_days helper."""

    def test_subtracts_conservatively(self):
        """Result should be at least n calendar days before input."""
        dt = datetime(2023, 7, 1)
        result = _subtract_business_days(dt, 21)

        # Should be at least 21 calendar days before (conservative estimate)
        assert result < dt - timedelta(days=21)

    def test_zero_days(self):
        """Subtracting 0 business days still subtracts a small buffer."""
        dt = datetime(2023, 7, 1)
        result = _subtract_business_days(dt, 0)

        # Buffer of 2 calendar days
        assert result == dt - timedelta(days=2)


# =============================================================================
# TestTrainingDataPipeline
# =============================================================================


class TestTrainingDataPipeline:
    """Tests for TrainingDataPipeline with mocked repositories."""

    def test_load_returns_training_data(self, mock_repos, config):
        """load() should return a TrainingData instance."""
        raw_repo, derived_repo = mock_repos
        pipeline = TrainingDataPipeline(raw_repo, derived_repo, config)
        result = pipeline.load()

        assert isinstance(result, TrainingData)
        assert result.train_loader is not None
        assert result.val_loader is not None
        assert result.test_loader is not None

    def test_repos_called_with_correct_args(self, mock_repos, config):
        """Repos should be called with feature_version and date ranges."""
        raw_repo, derived_repo = mock_repos
        pipeline = TrainingDataPipeline(raw_repo, derived_repo, config)
        pipeline.load()

        # Node panel should be read with the configured feature_version
        derived_repo.read_node_panel.assert_called_once()
        call_kwargs = derived_repo.read_node_panel.call_args
        assert call_kwargs.kwargs["feature_version"] == "v1.0"

        # Underlying bars should be read for the configured symbol
        raw_repo.read_underlying_bars.assert_called_once()
        call_kwargs = raw_repo.read_underlying_bars.call_args
        assert call_kwargs.kwargs["symbol"] == "SPY"

    def test_empty_panel_raises(self, config):
        """Empty node panel should raise ValueError with helpful message."""
        raw_repo = MagicMock()
        derived_repo = MagicMock()
        derived_repo.read_node_panel.return_value = pd.DataFrame()

        pipeline = TrainingDataPipeline(raw_repo, derived_repo, config)
        with pytest.raises(ValueError, match="No node panel data found"):
            pipeline.load()

    def test_empty_bars_raises(self, config):
        """Empty underlying bars should raise ValueError with helpful message."""
        raw_repo = MagicMock()
        derived_repo = MagicMock()

        # Panel is non-empty
        panel_df = _make_panel_df(datetime(2023, 3, 1), datetime(2023, 8, 31))
        derived_repo.read_node_panel.return_value = panel_df

        # Bars are empty
        raw_repo.read_underlying_bars.return_value = pd.DataFrame()

        pipeline = TrainingDataPipeline(raw_repo, derived_repo, config)
        with pytest.raises(ValueError, match="No underlying bars found"):
            pipeline.load()

    def test_compute_returns(self):
        """_compute_returns should produce correct simple returns."""
        bars_df = pd.DataFrame({
            "ts_utc": [
                datetime(2023, 1, 2),
                datetime(2023, 1, 3),
                datetime(2023, 1, 4),
            ],
            "close": [100.0, 102.0, 101.0],
            "volume": [1000, 2000, 3000],
        })

        returns_df = TrainingDataPipeline._compute_returns(bars_df)

        assert "ts_utc" in returns_df.columns
        assert "return" in returns_df.columns
        # First row is dropped (NaN from pct_change)
        assert len(returns_df) == 2
        # Check values: 102/100 - 1 = 0.02, 101/102 - 1 ≈ -0.0098
        np.testing.assert_almost_equal(returns_df.iloc[0]["return"], 0.02, decimal=4)
        np.testing.assert_almost_equal(
            returns_df.iloc[1]["return"], -0.009804, decimal=4
        )

    def test_compute_returns_deduplicates(self):
        """_compute_returns should handle duplicate timestamps."""
        bars_df = pd.DataFrame({
            "ts_utc": [
                datetime(2023, 1, 2),
                datetime(2023, 1, 2),  # Duplicate
                datetime(2023, 1, 3),
            ],
            "close": [100.0, 100.5, 102.0],
            "volume": [1000, 500, 2000],
        })

        returns_df = TrainingDataPipeline._compute_returns(bars_df)

        # Should keep last duplicate, so 2 unique timestamps minus 1 for pct_change
        assert len(returns_df) == 1

    def test_loader_batch_size(self, mock_repos, config):
        """DataLoaders should use configured batch size."""
        raw_repo, derived_repo = mock_repos
        pipeline = TrainingDataPipeline(raw_repo, derived_repo, config)
        result = pipeline.load()

        assert result.train_loader.batch_size == config.batch_size
        assert result.val_loader.batch_size == config.batch_size
        assert result.test_loader.batch_size == config.batch_size

    def test_metadata_populated(self, mock_repos, config):
        """TrainingData.metadata should have expected keys."""
        raw_repo, derived_repo = mock_repos
        pipeline = TrainingDataPipeline(raw_repo, derived_repo, config)
        result = pipeline.load()

        meta = result.metadata
        assert "train_samples" in meta
        assert "val_samples" in meta
        assert "test_samples" in meta
        assert "num_nodes" in meta
        assert "num_features" in meta
        assert "num_horizons" in meta
        assert meta["train_samples"] > 0
        assert meta["num_nodes"] > 0

    def test_graph_accessible(self, mock_repos, config):
        """TrainingData.graph should return the shared graph."""
        raw_repo, derived_repo = mock_repos
        pipeline = TrainingDataPipeline(raw_repo, derived_repo, config)
        result = pipeline.load()

        graph = result.graph
        assert graph is not None
        assert graph.num_nodes > 0
        assert graph.edge_index.shape[0] == 2

    def test_train_loader_shuffles(self, mock_repos, config):
        """Train loader should shuffle, val/test should not."""
        raw_repo, derived_repo = mock_repos
        pipeline = TrainingDataPipeline(raw_repo, derived_repo, config)
        result = pipeline.load()

        # DataLoader stores shuffle state in the sampler
        assert isinstance(
            result.train_loader.sampler,
            torch.utils.data.sampler.RandomSampler
        )
        assert isinstance(
            result.val_loader.sampler,
            torch.utils.data.sampler.SequentialSampler
        )


# Need torch for sampler check
import torch  # noqa: E402
