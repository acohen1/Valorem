"""Unit tests for microstructure feature generator."""

import numpy as np
import pandas as pd
import pytest

from src.features.node.microstructure import (
    MicrostructureConfig,
    MicrostructureFeatureGenerator,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_surface_df():
    """Create sample surface data with microstructure fields."""
    dates = pd.date_range("2024-01-01", periods=20, freq="D")

    data = []
    for tenor in [30, 60]:
        for bucket in ["ATM", "P25"]:
            for i, date in enumerate(dates):
                data.append({
                    "ts_utc": date,
                    "tenor_days": tenor,
                    "delta_bucket": bucket,
                    "spread_pct": 0.02 + np.random.randn() * 0.005,
                    "volume": int(1000 + i * 50 + np.random.randn() * 100),
                    "open_interest": int(5000 + i * 100 + np.random.randn() * 200),
                })

    return pd.DataFrame(data)


@pytest.fixture
def generator():
    """Create microstructure feature generator with default config."""
    return MicrostructureFeatureGenerator()


@pytest.fixture
def custom_generator():
    """Create generator with custom config."""
    config = MicrostructureConfig(
        volume_ratio_windows=[3, 7],
        oi_change_periods=[1, 3],
        min_periods=2,
    )
    return MicrostructureFeatureGenerator(config)


# ============================================================================
# Initialization Tests
# ============================================================================


class TestMicrostructureInit:
    """Test microstructure generator initialization."""

    def test_default_config(self):
        """Test default configuration values."""
        gen = MicrostructureFeatureGenerator()
        assert gen._config.volume_ratio_windows == [5]
        assert gen._config.oi_change_periods == [5]
        assert gen._config.min_periods is None

    def test_custom_config(self):
        """Test custom configuration."""
        config = MicrostructureConfig(
            volume_ratio_windows=[7],
            oi_change_periods=[1, 5],
            min_periods=3,
        )
        gen = MicrostructureFeatureGenerator(config)
        assert gen._config.volume_ratio_windows == [7]


# ============================================================================
# Generate Tests
# ============================================================================


class TestMicrostructureGenerate:
    """Test microstructure feature generation."""

    def test_generate_adds_volume_columns(self, generator, sample_surface_df):
        """Test that generate adds volume-related columns."""
        result = generator.generate(sample_surface_df)

        assert "volume_ratio_5d" in result.columns
        assert "log_volume" in result.columns

    def test_generate_adds_oi_columns(self, generator, sample_surface_df):
        """Test that generate adds OI-related columns."""
        result = generator.generate(sample_surface_df)

        assert "oi_change_5d" in result.columns
        assert "log_oi" in result.columns

    def test_generate_preserves_original_columns(self, generator, sample_surface_df):
        """Test that original columns are preserved."""
        original_cols = sample_surface_df.columns.tolist()
        result = generator.generate(sample_surface_df)

        for col in original_cols:
            assert col in result.columns

    def test_generate_empty_df(self, generator):
        """Test generation with empty DataFrame."""
        empty_df = pd.DataFrame(columns=["ts_utc", "tenor_days", "delta_bucket"])
        result = generator.generate(empty_df)
        assert len(result) == 0

    def test_generate_missing_columns_raises(self, generator):
        """Test that missing columns raise ValueError."""
        df = pd.DataFrame({
            "ts_utc": pd.date_range("2024-01-01", periods=3),
        })
        with pytest.raises(ValueError, match="Missing required columns"):
            generator.generate(df)

    def test_generate_without_volume(self, generator):
        """Test generation when volume column is missing."""
        df = pd.DataFrame({
            "ts_utc": pd.date_range("2024-01-01", periods=10),
            "tenor_days": [30] * 10,
            "delta_bucket": ["ATM"] * 10,
        })

        result = generator.generate(df)

        # Volume features should NOT be present
        assert "volume_ratio_5d" not in result.columns
        assert "log_volume" not in result.columns

    def test_generate_without_open_interest(self, generator):
        """Test generation when OI column is missing."""
        df = pd.DataFrame({
            "ts_utc": pd.date_range("2024-01-01", periods=10),
            "tenor_days": [30] * 10,
            "delta_bucket": ["ATM"] * 10,
            "volume": [1000] * 10,
        })

        result = generator.generate(df)

        # Volume features should be present
        assert "volume_ratio_5d" in result.columns
        # OI features should NOT be present
        assert "oi_change_5d" not in result.columns

    def test_no_spread_features_computed(self, generator, sample_surface_df):
        """Test that dropped spread features are no longer computed."""
        result = generator.generate(sample_surface_df)

        assert "spread_pct_ma_5d" not in result.columns
        assert "spread_pct_ma_10d" not in result.columns
        assert "spread_pct_std_5d" not in result.columns
        assert "spread_pct_std_10d" not in result.columns
        assert "spread_pct_change_1d" not in result.columns

    def test_no_volume_ma_computed(self, generator, sample_surface_df):
        """Test that dropped volume_ma features are no longer computed."""
        result = generator.generate(sample_surface_df)

        assert "volume_ma_5d" not in result.columns
        assert "volume_ma_10d" not in result.columns

    def test_no_oi_ma_computed(self, generator, sample_surface_df):
        """Test that dropped oi_ma features are no longer computed."""
        result = generator.generate(sample_surface_df)

        assert "oi_ma_5d" not in result.columns
        assert "oi_ma_10d" not in result.columns


# ============================================================================
# Volume Feature Tests
# ============================================================================


class TestVolumeFeatures:
    """Test volume-related feature calculations."""

    def test_volume_ratio_correct(self, generator):
        """Test volume ratio is computed correctly."""
        df = pd.DataFrame({
            "ts_utc": pd.date_range("2024-01-01", periods=10),
            "tenor_days": [30] * 10,
            "delta_bucket": ["ATM"] * 10,
            "volume": [1000, 1200, 800, 1500, 1000, 1100, 900, 1400, 1000, 2000],
        })

        result = generator.generate(df)

        # Volume ratio = current / rolling mean (shift(1) excludes current obs)
        rolling_mean = pd.Series(df["volume"]).shift(1).rolling(5).mean().iloc[-1]
        expected_ratio = 2000 / rolling_mean
        assert result["volume_ratio_5d"].iloc[-1] == pytest.approx(expected_ratio)

    def test_log_volume_correct(self, generator):
        """Test log volume is computed correctly."""
        df = pd.DataFrame({
            "ts_utc": pd.date_range("2024-01-01", periods=5),
            "tenor_days": [30] * 5,
            "delta_bucket": ["ATM"] * 5,
            "volume": [100, 1000, 10000, 100000, 0],
        })

        result = generator.generate(df)

        expected_log = np.log1p(df["volume"].astype(float))
        np.testing.assert_allclose(result["log_volume"].values, expected_log.values)

    def test_volume_handles_zero(self, generator):
        """Test volume features handle zero values."""
        df = pd.DataFrame({
            "ts_utc": pd.date_range("2024-01-01", periods=5),
            "tenor_days": [30] * 5,
            "delta_bucket": ["ATM"] * 5,
            "volume": [0, 0, 100, 200, 0],
        })

        result = generator.generate(df)

        # Should not crash
        assert "volume_ratio_5d" in result.columns
        assert "log_volume" in result.columns


# ============================================================================
# Open Interest Feature Tests
# ============================================================================


class TestOIFeatures:
    """Test open interest feature calculations."""

    def test_oi_change_correct(self, generator):
        """Test OI percent change is correct."""
        df = pd.DataFrame({
            "ts_utc": pd.date_range("2024-01-01", periods=10),
            "tenor_days": [30] * 10,
            "delta_bucket": ["ATM"] * 10,
            "open_interest": [1000, 1100, 1200, 1100, 1000, 1100, 1200, 1300, 1400, 1500],
        })

        result = generator.generate(df)

        # OI change at position 5: (1100 - 1000) / 1000 = 0.1
        expected_oi_change = (1100 - 1000) / 1000
        assert result["oi_change_5d"].iloc[5] == pytest.approx(expected_oi_change)

    def test_log_oi_correct(self, generator):
        """Test log OI is computed correctly."""
        df = pd.DataFrame({
            "ts_utc": pd.date_range("2024-01-01", periods=5),
            "tenor_days": [30] * 5,
            "delta_bucket": ["ATM"] * 5,
            "open_interest": [100, 1000, 10000, 100000, 0],
        })

        result = generator.generate(df)

        expected_log = np.log1p(df["open_interest"].astype(float))
        np.testing.assert_allclose(result["log_oi"].values, expected_log.values)


# ============================================================================
# Per-Node Computation Tests
# ============================================================================


class TestPerNodeComputation:
    """Test that features are computed per node."""

    def test_features_computed_per_node(self, generator):
        """Test features are independent per node."""
        df = pd.DataFrame({
            "ts_utc": pd.to_datetime([
                "2024-01-01", "2024-01-02", "2024-01-03",
                "2024-01-01", "2024-01-02", "2024-01-03",
            ]),
            "tenor_days": [30, 30, 30, 60, 60, 60],
            "delta_bucket": ["ATM", "ATM", "ATM", "ATM", "ATM", "ATM"],
            "volume": [100, 200, 300, 1000, 2000, 3000],
        })

        result = generator.generate(df)

        # Node 1 (30 ATM): log_volume should use node 1's volume
        node1 = result[(result["tenor_days"] == 30) & (result["delta_bucket"] == "ATM")]
        expected_log = np.log1p(300.0)
        assert node1["log_volume"].iloc[-1] == pytest.approx(expected_log)

        # Node 2 (60 ATM): log_volume should use node 2's volume
        node2 = result[(result["tenor_days"] == 60) & (result["delta_bucket"] == "ATM")]
        expected_log = np.log1p(3000.0)
        assert node2["log_volume"].iloc[-1] == pytest.approx(expected_log)


# ============================================================================
# Standalone Utility Tests
# ============================================================================


class TestStandaloneUtilities:
    """Test standalone utility methods."""

    def test_compute_volume_features(self, generator):
        """Test standalone volume feature computation."""
        df = pd.DataFrame({
            "volume": [1000, 1200, 800, 1500, 1000],
        })

        result = generator.compute_volume_features(df, windows=[3])

        assert "volume_ratio_3d" in result.columns
        assert "log_volume" in result.columns

    def test_compute_volume_features_requires_column(self, generator):
        """Test volume features require volume column."""
        df = pd.DataFrame({"other": [1, 2, 3]})
        with pytest.raises(ValueError, match="volume"):
            generator.compute_volume_features(df)
