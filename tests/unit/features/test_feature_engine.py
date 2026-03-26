"""Unit tests for the FeatureEngine orchestrator."""

from datetime import datetime, timedelta
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from src.features.engine import (
    FeatureEngine,
    FeatureEngineConfig,
    FeatureEngineResult,
    GlobalFeatureConfig,
    NodeFeatureConfig,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_surface_df():
    """Create sample surface snapshot data."""
    np.random.seed(42)
    n_days = 30
    n_tenors = 3
    n_buckets = 5

    rows = []
    for day in range(n_days):
        ts = datetime(2024, 1, 1) + timedelta(days=day)
        for tenor in [7, 30, 60]:
            for bucket in ["P10", "P25", "ATM", "C25", "C10"]:
                rows.append({
                    "ts_utc": ts,
                    "tenor_days": tenor,
                    "delta_bucket": bucket,
                    "iv_mid": 0.20 + np.random.randn() * 0.02,
                    "spread_pct": 0.01 + np.random.rand() * 0.02,
                    "delta": -0.1 + np.random.rand() * 0.8,
                    "volume": np.random.randint(100, 1000),
                    "open_interest": np.random.randint(1000, 10000),
                })

    return pd.DataFrame(rows)


@pytest.fixture
def sample_underlying_df():
    """Create sample underlying bar data."""
    np.random.seed(42)
    n = 60
    returns = np.random.randn(n) * 0.01
    prices = 100 * np.exp(np.cumsum(returns))

    return pd.DataFrame({
        "ts_utc": pd.date_range("2024-01-01", periods=n, freq="D"),
        "open": prices * (1 + np.random.randn(n) * 0.001),
        "high": prices * (1 + abs(np.random.randn(n) * 0.005)),
        "low": prices * (1 - abs(np.random.randn(n) * 0.005)),
        "close": prices,
        "volume": np.random.randint(50000000, 100000000, n),
    })


@pytest.fixture
def sample_fred_df():
    """Create sample FRED series data."""
    n = 100
    return pd.DataFrame({
        "obs_date": pd.date_range("2023-10-01", periods=n, freq="D"),
        "value": [4.5 + i * 0.01 + np.random.randn() * 0.02 for i in range(n)],
        "release_datetime_utc": pd.date_range("2023-10-01", periods=n, freq="D") + pd.Timedelta(days=1),
    })


@pytest.fixture
def mock_repos(sample_surface_df, sample_underlying_df, sample_fred_df):
    """Create mock repositories."""
    raw_repo = MagicMock()
    derived_repo = MagicMock()

    # Configure mock returns
    derived_repo.read_surface_snapshots.return_value = sample_surface_df
    raw_repo.read_underlying_bars.return_value = sample_underlying_df
    raw_repo.read_fred_series.return_value = sample_fred_df

    return raw_repo, derived_repo


# ============================================================================
# Config Tests
# ============================================================================


class TestFeatureEngineConfig:
    """Test FeatureEngineConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = FeatureEngineConfig()
        assert config.underlying_symbol == "SPY"
        assert set(config.fred_series) == {"DGS10", "DGS2", "VIXCLS"}
        assert config.lookback_buffer_days == 60
        assert config.include_node_features is True
        assert config.include_global_features is True
        assert config.include_macro_features is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = FeatureEngineConfig(
            underlying_symbol="QQQ",
            fred_series=["DGS10"],
            lookback_buffer_days=30,
            include_node_features=True,
            include_global_features=False,
            include_macro_features=False,
        )
        assert config.underlying_symbol == "QQQ"
        assert config.fred_series == ["DGS10"]
        assert config.lookback_buffer_days == 30
        assert config.include_global_features is False


class TestGlobalFeatureConfig:
    """Test GlobalFeatureConfig."""

    def test_default_config(self):
        """Test default global feature config."""
        config = GlobalFeatureConfig()
        assert config.include_returns is True
        assert config.include_realized_vol is True


# ============================================================================
# Initialization Tests
# ============================================================================


class TestFeatureEngineInit:
    """Test FeatureEngine initialization."""

    def test_init_with_defaults(self, mock_repos):
        """Test initialization with default config."""
        raw_repo, derived_repo = mock_repos

        engine = FeatureEngine(
            engine=MagicMock(),
            raw_repo=raw_repo,
            derived_repo=derived_repo,
        )

        assert engine._config is not None
        assert engine._node_generator is not None
        assert engine._returns_generator is not None
        assert engine._realized_vol_generator is not None
        assert engine._macro_generator is not None

    def test_init_requires_engine(self, mock_repos):
        """Test that engine parameter is required."""
        raw_repo, derived_repo = mock_repos

        with pytest.raises(ValueError, match="engine is required"):
            FeatureEngine(
                raw_repo=raw_repo,
                derived_repo=derived_repo,
            )

    def test_init_with_custom_config(self, mock_repos):
        """Test initialization with custom config."""
        raw_repo, derived_repo = mock_repos

        config = FeatureEngineConfig(
            underlying_symbol="QQQ",
            lookback_buffer_days=30,
        )

        engine = FeatureEngine(
            config=config,
            engine=MagicMock(),
            raw_repo=raw_repo,
            derived_repo=derived_repo,
        )

        assert engine._config.underlying_symbol == "QQQ"
        assert engine._config.lookback_buffer_days == 30


# ============================================================================
# Global Feature Generation Tests
# ============================================================================


class TestGlobalFeatureGeneration:
    """Test global feature generation."""

    def test_generate_global_features(self, mock_repos, sample_underlying_df):
        """Test global feature generation."""
        raw_repo, derived_repo = mock_repos

        engine = FeatureEngine(
            engine=MagicMock(),
            raw_repo=raw_repo,
            derived_repo=derived_repo,
        )

        result = engine._generate_global_features(sample_underlying_df)

        # Check expected columns
        assert "ts_utc" in result.columns
        assert "returns_1d" in result.columns
        assert "underlying_rv_5d" in result.columns
        assert "drawdown" in result.columns

    def test_generate_global_features_empty(self, mock_repos):
        """Test global feature generation with empty DataFrame."""
        raw_repo, derived_repo = mock_repos

        engine = FeatureEngine(
            engine=MagicMock(),
            raw_repo=raw_repo,
            derived_repo=derived_repo,
        )

        empty_df = pd.DataFrame(columns=["ts_utc", "close"])
        result = engine._generate_global_features(empty_df)

        assert "ts_utc" in result.columns
        assert len(result) == 0


# ============================================================================
# Merge Feature Tests
# ============================================================================


class TestMergeFeatures:
    """Test feature merging logic."""

    def test_merge_with_global_features(self, mock_repos):
        """Test merging node and global features."""
        raw_repo, derived_repo = mock_repos

        engine = FeatureEngine(
            engine=MagicMock(),
            raw_repo=raw_repo,
            derived_repo=derived_repo,
        )

        # Create test DataFrames
        node_df = pd.DataFrame({
            "ts_utc": pd.date_range("2024-01-01", periods=10, freq="D"),
            "tenor_days": [7] * 10,
            "delta_bucket": ["ATM"] * 10,
            "iv_mid": [0.2] * 10,
        })

        global_df = pd.DataFrame({
            "ts_utc": pd.date_range("2024-01-01", periods=10, freq="D"),
            "returns_1d": [0.01] * 10,
            "rv_5d": [0.02] * 10,
        })

        result = engine._merge_features(node_df, global_df, None)

        assert "iv_mid" in result.columns
        assert "returns_1d" in result.columns
        assert "rv_5d" in result.columns
        assert len(result) == 10

    def test_merge_with_macro_features(self, mock_repos):
        """Test merging with macro features."""
        raw_repo, derived_repo = mock_repos

        engine = FeatureEngine(
            engine=MagicMock(),
            raw_repo=raw_repo,
            derived_repo=derived_repo,
        )

        node_df = pd.DataFrame({
            "ts_utc": pd.date_range("2024-01-01", periods=10, freq="D"),
            "tenor_days": [7] * 10,
            "delta_bucket": ["ATM"] * 10,
            "iv_mid": [0.2] * 10,
        })

        macro_df = pd.DataFrame({
            "ts_utc": pd.date_range("2024-01-01", periods=10, freq="D"),
            "DGS10_level": [0.045] * 10,
        })

        result = engine._merge_features(node_df, None, macro_df)

        assert "iv_mid" in result.columns
        assert "DGS10_level" in result.columns

    def test_merge_uses_backward_direction(self, mock_repos):
        """Test that merge uses backward direction (no future leakage)."""
        raw_repo, derived_repo = mock_repos

        engine = FeatureEngine(
            engine=MagicMock(),
            raw_repo=raw_repo,
            derived_repo=derived_repo,
        )

        # Node data at 9am and 10am
        node_df = pd.DataFrame({
            "ts_utc": [
                datetime(2024, 1, 1, 9, 0),
                datetime(2024, 1, 1, 10, 0),
            ],
            "value": [1, 2],
        })

        # Global data only at 9am
        global_df = pd.DataFrame({
            "ts_utc": [datetime(2024, 1, 1, 9, 0)],
            "global_val": [100],
        })

        result = engine._merge_features(node_df, global_df, None)

        # Both rows should have global_val = 100 (forward-filled from 9am)
        assert result["global_val"].iloc[0] == 100
        assert result["global_val"].iloc[1] == 100  # Forward-filled

    def test_merge_all_none(self, mock_repos):
        """Test merging with all None optional dataframes."""
        raw_repo, derived_repo = mock_repos

        engine = FeatureEngine(
            engine=MagicMock(),
            raw_repo=raw_repo,
            derived_repo=derived_repo,
        )

        node_df = pd.DataFrame({
            "ts_utc": pd.date_range("2024-01-01", periods=5, freq="D"),
            "value": [1, 2, 3, 4, 5],
        })

        result = engine._merge_features(node_df, None, None)

        assert len(result) == 5
        assert "value" in result.columns


# ============================================================================
# Utility Method Tests
# ============================================================================


class TestUtilityMethods:
    """Test utility methods."""

    def test_generate_node_features_only(self, mock_repos, sample_surface_df):
        """Test node features only generation."""
        raw_repo, derived_repo = mock_repos

        engine = FeatureEngine(
            engine=MagicMock(),
            raw_repo=raw_repo,
            derived_repo=derived_repo,
        )

        result_df, result = engine.generate_node_features_only(sample_surface_df)

        assert result.feature_count > 0
        assert "iv_change_1d" in result_df.columns

    def test_generate_global_features_only(self, mock_repos, sample_underlying_df):
        """Test global features only generation."""
        raw_repo, derived_repo = mock_repos

        engine = FeatureEngine(
            engine=MagicMock(),
            raw_repo=raw_repo,
            derived_repo=derived_repo,
        )

        result = engine.generate_global_features_only(sample_underlying_df)

        assert "ts_utc" in result.columns
        assert "returns_1d" in result.columns

    def test_generate_macro_features_only(self, mock_repos, sample_fred_df):
        """Test macro features only generation."""
        raw_repo, derived_repo = mock_repos

        engine = FeatureEngine(
            engine=MagicMock(),
            raw_repo=raw_repo,
            derived_repo=derived_repo,
        )

        fred_data = {"DGS10": sample_fred_df}
        result = engine.generate_macro_features_only(fred_data)

        assert "ts_utc" in result.columns
        assert "DGS10_level" in result.columns


# ============================================================================
# Build Feature Panel Tests
# ============================================================================


class TestBuildFeaturePanel:
    """Test build_feature_panel method."""

    def test_build_empty_surface(self, mock_repos):
        """Test build with empty surface data."""
        raw_repo, derived_repo = mock_repos
        derived_repo.read_surface_snapshots.return_value = pd.DataFrame()

        engine = FeatureEngine(
            engine=MagicMock(),
            raw_repo=raw_repo,
            derived_repo=derived_repo,
        )

        panel_df, result = engine.build_feature_panel(
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 31),
            surface_version="v1.0",
            feature_version="v1.0",
            write_to_db=False,
        )

        assert len(panel_df) == 0
        assert result.row_count == 0

    def test_build_skips_writing_when_dry_run(self, mock_repos, sample_surface_df, sample_underlying_df, sample_fred_df):
        """Test that build doesn't write when write_to_db=False."""
        raw_repo, derived_repo = mock_repos
        derived_repo.read_surface_snapshots.return_value = sample_surface_df
        raw_repo.read_underlying_bars.return_value = sample_underlying_df
        raw_repo.read_fred_series.return_value = sample_fred_df

        engine = FeatureEngine(
            engine=MagicMock(),
            raw_repo=raw_repo,
            derived_repo=derived_repo,
        )

        panel_df, result = engine.build_feature_panel(
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 31),
            surface_version="v1.0",
            feature_version="v1.0",
            write_to_db=False,
        )

        # write_node_panel should not be called
        derived_repo.write_node_panel.assert_not_called()

    def test_build_includes_all_feature_types(self, mock_repos, sample_surface_df, sample_underlying_df, sample_fred_df):
        """Test that build includes node, global, and macro features."""
        raw_repo, derived_repo = mock_repos
        derived_repo.read_surface_snapshots.return_value = sample_surface_df
        raw_repo.read_underlying_bars.return_value = sample_underlying_df
        raw_repo.read_fred_series.return_value = sample_fred_df

        engine = FeatureEngine(
            engine=MagicMock(),
            raw_repo=raw_repo,
            derived_repo=derived_repo,
        )

        panel_df, result = engine.build_feature_panel(
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 31),
            surface_version="v1.0",
            feature_version="v1.0",
            write_to_db=False,
        )

        assert result.node_features_count > 0
        assert result.global_features_count > 0
        assert result.macro_features_count > 0

    def test_build_respects_feature_flags(self, mock_repos, sample_surface_df, sample_underlying_df, sample_fred_df):
        """Test that build respects include_* flags."""
        raw_repo, derived_repo = mock_repos
        derived_repo.read_surface_snapshots.return_value = sample_surface_df
        raw_repo.read_underlying_bars.return_value = sample_underlying_df
        raw_repo.read_fred_series.return_value = sample_fred_df

        config = FeatureEngineConfig(
            include_node_features=True,
            include_global_features=False,
            include_macro_features=False,
        )

        engine = FeatureEngine(
            config=config,
            engine=MagicMock(),
            raw_repo=raw_repo,
            derived_repo=derived_repo,
        )

        panel_df, result = engine.build_feature_panel(
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 31),
            surface_version="v1.0",
            feature_version="v1.0",
            write_to_db=False,
        )

        assert result.node_features_count > 0
        assert result.global_features_count == 0
        assert result.macro_features_count == 0


# ============================================================================
# Result Tests
# ============================================================================


class TestFeatureEngineResult:
    """Test FeatureEngineResult dataclass."""

    def test_result_attributes(self):
        """Test result attributes."""
        result = FeatureEngineResult(
            feature_version="v1.0",
            surface_version="v1.0",
            row_count=1000,
            feature_count=50,
            start_ts=datetime(2024, 1, 1),
            end_ts=datetime(2024, 1, 31),
            nodes_processed=15,
            node_features_count=30,
            global_features_count=10,
            macro_features_count=10,
            validation_passed=True,
        )

        assert result.feature_version == "v1.0"
        assert result.row_count == 1000
        assert result.feature_count == 50
        assert result.node_features_count == 30
        assert result.validation_passed is True
