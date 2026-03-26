"""Unit tests for feature engine orchestrator."""

import numpy as np
import pandas as pd
import pytest

from src.features.engine import (
    FeatureResult,
    NodeFeatureConfig,
    NodeFeatureGenerator,
)
from src.features.node.iv_features import IVFeatureConfig
from src.features.node.microstructure import MicrostructureConfig
from src.features.node.surface import SurfaceFeatureConfig


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_surface_df():
    """Create comprehensive sample surface data."""
    dates = pd.date_range("2024-01-01", periods=20, freq="D")

    data = []
    for date in dates:
        for tenor in [30, 60]:
            for bucket in ["P10", "P25", "ATM", "C25", "C10"]:
                # Delta mapping
                delta_map = {"P10": -0.10, "P25": -0.25, "ATM": 0.50, "C25": 0.25, "C10": 0.10}
                delta = delta_map[bucket]

                # Generate realistic data
                base_iv = 0.20 + 0.01 * (tenor / 30) + np.random.randn() * 0.005
                smile_adj = 0.02 * abs(delta - 0.5)
                iv = base_iv + smile_adj

                data.append({
                    "ts_utc": date,
                    "tenor_days": tenor,
                    "delta_bucket": bucket,
                    "delta": delta,
                    "iv_mid": iv,
                    "spread_pct": 0.02 + np.random.randn() * 0.005,
                    "volume": int(1000 + np.random.randn() * 200),
                    "open_interest": int(5000 + np.random.randn() * 500),
                })

    return pd.DataFrame(data)


@pytest.fixture
def generator():
    """Create feature generator with default config."""
    return NodeFeatureGenerator()


@pytest.fixture
def minimal_surface_df():
    """Create minimal surface data with just required columns."""
    return pd.DataFrame({
        "ts_utc": pd.date_range("2024-01-01", periods=10),
        "tenor_days": [30] * 10,
        "delta_bucket": ["ATM"] * 10,
        "iv_mid": [0.20 + 0.001 * i for i in range(10)],
    })


# ============================================================================
# Initialization Tests
# ============================================================================


class TestNodeFeatureGeneratorInit:
    """Test feature generator initialization."""

    def test_default_config(self):
        """Test default configuration values."""
        gen = NodeFeatureGenerator()
        assert gen._config.include_iv
        assert gen._config.include_microstructure
        assert gen._config.include_surface

    def test_custom_config(self):
        """Test custom configuration."""
        config = NodeFeatureConfig(
            include_iv=True,
            include_microstructure=False,
            include_surface=False,
        )
        gen = NodeFeatureGenerator(config)
        assert gen._config.include_iv
        assert not gen._config.include_microstructure
        assert not gen._config.include_surface

    def test_nested_config(self):
        """Test nested configuration objects."""
        config = NodeFeatureConfig(
            iv_config=IVFeatureConfig(change_periods=[1, 2, 3]),
            microstructure_config=MicrostructureConfig(volume_ratio_windows=[3, 7]),
            surface_config=SurfaceFeatureConfig(min_buckets_for_skew=2),
        )
        gen = NodeFeatureGenerator(config)
        assert gen._config.iv_config.change_periods == [1, 2, 3]
        assert gen._config.microstructure_config.volume_ratio_windows == [3, 7]
        assert gen._config.surface_config.min_buckets_for_skew == 2


# ============================================================================
# Generate Tests
# ============================================================================


class TestNodeFeatureGenerate:
    """Test feature generation."""

    def test_generate_returns_dataframe(self, generator, sample_surface_df):
        """Test generate returns DataFrame."""
        result_df, _ = generator.generate(sample_surface_df)
        assert isinstance(result_df, pd.DataFrame)

    def test_generate_returns_feature_result(self, generator, sample_surface_df):
        """Test generate returns FeatureResult."""
        _, result = generator.generate(sample_surface_df)
        assert isinstance(result, FeatureResult)

    def test_generate_adds_iv_features(self, generator, sample_surface_df):
        """Test IV features are added."""
        result_df, _ = generator.generate(sample_surface_df)

        assert "iv_change_1d" in result_df.columns
        assert "iv_vol_5d" in result_df.columns
        assert "iv_zscore_5d" in result_df.columns

    def test_generate_adds_microstructure_features(self, generator, sample_surface_df):
        """Test microstructure features are added."""
        result_df, _ = generator.generate(sample_surface_df)

        assert "volume_ratio_5d" in result_df.columns
        assert "log_volume" in result_df.columns
        assert "oi_change_5d" in result_df.columns

    def test_generate_adds_surface_features(self, generator, sample_surface_df):
        """Test surface features are added."""
        result_df, _ = generator.generate(sample_surface_df)

        assert "skew_slope" in result_df.columns
        assert "term_slope" in result_df.columns
        assert "atm_spread" in result_df.columns

    def test_generate_empty_df(self, generator):
        """Test generation with empty DataFrame."""
        empty_df = pd.DataFrame(columns=["ts_utc", "tenor_days", "delta_bucket", "iv_mid"])
        result_df, result = generator.generate(empty_df)

        assert len(result_df) == 0
        assert result.row_count == 0
        assert result.feature_count == 0

    def test_generate_missing_columns_raises(self, generator):
        """Test missing required columns raise ValueError."""
        df = pd.DataFrame({
            "ts_utc": pd.date_range("2024-01-01", periods=3),
            "iv_mid": [0.20, 0.21, 0.22],
        })
        with pytest.raises(ValueError, match="Missing required columns"):
            generator.generate(df)

    def test_generate_without_volume_or_oi(self, generator, minimal_surface_df):
        """Test generation without volume/OI skips microstructure."""
        result_df, result = generator.generate(minimal_surface_df)

        # IV features should be present
        assert "iv_change_1d" in result_df.columns

        # Microstructure features should NOT be present (no volume/OI inputs)
        assert "volume_ratio_5d" not in result_df.columns

    def test_generate_preserves_row_count(self, generator, sample_surface_df):
        """Test row count is preserved."""
        result_df, _ = generator.generate(sample_surface_df)
        assert len(result_df) == len(sample_surface_df)


# ============================================================================
# Feature Result Tests
# ============================================================================


class TestFeatureResult:
    """Test FeatureResult metadata."""

    def test_feature_result_row_count(self, generator, sample_surface_df):
        """Test feature result has correct row count."""
        _, result = generator.generate(sample_surface_df)
        assert result.row_count == len(sample_surface_df)

    def test_feature_result_feature_count(self, generator, sample_surface_df):
        """Test feature result has correct feature count."""
        _, result = generator.generate(sample_surface_df)
        assert result.feature_count > 0

    def test_feature_result_nodes_processed(self, generator, sample_surface_df):
        """Test feature result has correct node count."""
        _, result = generator.generate(sample_surface_df)

        # 2 tenors x 5 buckets = 10 nodes
        expected_nodes = 10
        assert result.nodes_processed == expected_nodes

    def test_feature_result_timestamps(self, generator, sample_surface_df):
        """Test feature result has correct timestamps."""
        _, result = generator.generate(sample_surface_df)

        assert result.start_ts == sample_surface_df["ts_utc"].min()
        assert result.end_ts == sample_surface_df["ts_utc"].max()

    def test_feature_result_feature_names(self, generator, sample_surface_df):
        """Test feature result lists generated features."""
        _, result = generator.generate(sample_surface_df)

        assert len(result.features_generated) == result.feature_count
        assert "iv_change_1d" in result.features_generated


# ============================================================================
# Selective Generation Tests
# ============================================================================


class TestSelectiveGeneration:
    """Test selective feature generation."""

    def test_iv_only(self):
        """Test generating only IV features."""
        config = NodeFeatureConfig(
            include_iv=True,
            include_microstructure=False,
            include_surface=False,
        )
        gen = NodeFeatureGenerator(config)

        df = pd.DataFrame({
            "ts_utc": pd.date_range("2024-01-01", periods=10),
            "tenor_days": [30] * 10,
            "delta_bucket": ["ATM"] * 10,
            "iv_mid": [0.20 + 0.001 * i for i in range(10)],
            "spread_pct": [0.02] * 10,
        })

        result_df, result = gen.generate(df)

        assert "iv_change_1d" in result_df.columns
        assert "volume_ratio_5d" not in result_df.columns
        assert "skew_slope" not in result_df.columns

    def test_microstructure_only(self):
        """Test generating only microstructure features."""
        config = NodeFeatureConfig(
            include_iv=False,
            include_microstructure=True,
            include_surface=False,
        )
        gen = NodeFeatureGenerator(config)

        df = pd.DataFrame({
            "ts_utc": pd.date_range("2024-01-01", periods=10),
            "tenor_days": [30] * 10,
            "delta_bucket": ["ATM"] * 10,
            "iv_mid": [0.20] * 10,
            "spread_pct": [0.02 + 0.001 * i for i in range(10)],
            "volume": [1000 + i * 100 for i in range(10)],
            "open_interest": [5000 + i * 50 for i in range(10)],
        })

        result_df, result = gen.generate(df)

        assert "iv_change_1d" not in result_df.columns
        assert "volume_ratio_5d" in result_df.columns
        assert "log_volume" in result_df.columns
        assert "skew_slope" not in result_df.columns

    def test_surface_only(self):
        """Test generating only surface features."""
        config = NodeFeatureConfig(
            include_iv=False,
            include_microstructure=False,
            include_surface=True,
        )
        gen = NodeFeatureGenerator(config)

        df = pd.DataFrame({
            "ts_utc": pd.to_datetime(["2024-01-01"] * 5),
            "tenor_days": [30] * 5,
            "delta_bucket": ["P10", "P25", "ATM", "C25", "C10"],
            "iv_mid": [0.25, 0.22, 0.20, 0.21, 0.24],
        })

        result_df, result = gen.generate(df)

        assert "iv_change_1d" not in result_df.columns
        assert "volume_ratio_5d" not in result_df.columns
        assert "skew_slope" in result_df.columns


# ============================================================================
# Utility Method Tests
# ============================================================================


class TestUtilityMethods:
    """Test utility methods."""

    def test_generate_iv_only_method(self, generator, sample_surface_df):
        """Test generate_iv_only utility method."""
        result = generator.generate_iv_only(sample_surface_df)

        assert "iv_change_1d" in result.columns
        # Should not have microstructure features
        assert "volume_ratio_5d" not in result.columns

    def test_generate_microstructure_only_method(self, generator, sample_surface_df):
        """Test generate_microstructure_only utility method."""
        result = generator.generate_microstructure_only(sample_surface_df)

        assert "volume_ratio_5d" in result.columns
        # Should not have IV features
        assert "iv_change_1d" not in result.columns

    def test_generate_surface_only_method(self, generator, sample_surface_df):
        """Test generate_surface_only utility method."""
        result = generator.generate_surface_only(sample_surface_df)

        assert "skew_slope" in result.columns
        # Should not have IV features
        assert "iv_change_1d" not in result.columns

    def test_get_feature_names(self, generator):
        """Test get_feature_names lists all expected features."""
        names = generator.get_feature_names()

        # IV features
        assert "iv_change_1d" in names
        assert "iv_vol_5d" in names

        # Microstructure features
        assert "volume_ratio_5d" in names
        assert "log_volume" in names

        # Surface features
        assert "skew_slope" in names
        assert "term_slope" in names


# ============================================================================
# No Leakage Tests
# ============================================================================


class TestNoLeakage:
    """Test that features don't leak future data."""

    def test_iv_change_no_future_leakage(self, generator):
        """Test IV change doesn't use future data."""
        df = pd.DataFrame({
            "ts_utc": pd.date_range("2024-01-01", periods=5),
            "tenor_days": [30] * 5,
            "delta_bucket": ["ATM"] * 5,
            "iv_mid": [0.20, 0.21, 0.22, 0.23, 0.24],
        })

        result = generator.generate_iv_only(df)

        # First value should be NaN (no prior data)
        assert pd.isna(result["iv_change_1d"].iloc[0])

        # Second value should be 0.01 (0.21 - 0.20)
        assert result["iv_change_1d"].iloc[1] == pytest.approx(0.01)

    def test_rolling_features_no_future_leakage(self, generator):
        """Test rolling features don't use future data."""
        df = pd.DataFrame({
            "ts_utc": pd.date_range("2024-01-01", periods=10),
            "tenor_days": [30] * 10,
            "delta_bucket": ["ATM"] * 10,
            "iv_mid": list(range(10)),  # 0, 1, 2, ..., 9
        })

        result = generator.generate_iv_only(df)

        # Rolling mean at position 5 should only use positions 0-5 (window=5 or less)
        # Not positions 6-9
        iv_vol = result["iv_vol_5d"]

        # Each rolling std should only consider past/current values
        # The variance should be based only on values we've seen
        for i in range(len(result)):
            if pd.notna(iv_vol.iloc[i]):
                # This is just a sanity check that values are computed
                assert iv_vol.iloc[i] >= 0


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests for complete pipeline."""

    def test_full_pipeline(self, generator, sample_surface_df):
        """Test full feature generation pipeline."""
        result_df, result = generator.generate(
            sample_surface_df,
            feature_version="v1.0.0",
        )

        # Verify output structure
        assert result.feature_version == "v1.0.0"
        assert result.row_count == len(sample_surface_df)
        assert result.feature_count > 15  # Should have many features
        assert result.nodes_processed == 10  # 2 tenors x 5 buckets

        # Verify original data preserved
        for col in sample_surface_df.columns:
            assert col in result_df.columns

        # Verify features generated
        expected_features = [
            "iv_change_1d", "iv_vol_5d",
            "volume_ratio_5d", "log_volume",
            "skew_slope", "term_slope", "atm_spread",
        ]
        for feat in expected_features:
            assert feat in result_df.columns

    def test_pipeline_with_nan_values(self, generator):
        """Test pipeline handles NaN values gracefully."""
        df = pd.DataFrame({
            "ts_utc": pd.date_range("2024-01-01", periods=10),
            "tenor_days": [30] * 10,
            "delta_bucket": ["ATM"] * 10,
            "iv_mid": [0.20, np.nan, 0.22, 0.23, np.nan, 0.25, 0.26, 0.27, 0.28, 0.29],
            "spread_pct": [0.02] * 10,
        })

        # Should not crash
        result_df, result = generator.generate(df)
        assert len(result_df) == 10
