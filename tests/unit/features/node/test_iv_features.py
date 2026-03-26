"""Unit tests for IV feature generator."""

import numpy as np
import pandas as pd
import pytest

from src.features.node.iv_features import IVFeatureConfig, IVFeatureGenerator


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_surface_df():
    """Create sample surface data for testing."""
    # Generate 20 observations for 2 nodes
    dates = pd.date_range("2024-01-01", periods=20, freq="D")

    data = []
    for tenor in [30, 60]:
        for bucket in ["ATM", "P25"]:
            for i, date in enumerate(dates):
                # IV with some trend and noise
                base_iv = 0.20 + 0.001 * i + np.random.randn() * 0.01
                data.append({
                    "ts_utc": date,
                    "tenor_days": tenor,
                    "delta_bucket": bucket,
                    "iv_mid": base_iv,
                })

    return pd.DataFrame(data)


@pytest.fixture
def generator():
    """Create IV feature generator with default config."""
    return IVFeatureGenerator()


@pytest.fixture
def custom_generator():
    """Create IV feature generator with custom config."""
    config = IVFeatureConfig(
        change_periods=[1, 3, 5],
        rolling_windows=[3, 7],
        min_periods=2,
    )
    return IVFeatureGenerator(config)


# ============================================================================
# Initialization Tests
# ============================================================================


class TestIVFeatureGeneratorInit:
    """Test IV feature generator initialization."""

    def test_default_config(self):
        """Test default configuration values."""
        gen = IVFeatureGenerator()
        assert gen._config.change_periods == [1, 5]
        assert gen._config.rolling_windows == [5, 10, 21]
        assert gen._config.min_periods is None

    def test_custom_config(self):
        """Test custom configuration."""
        config = IVFeatureConfig(
            change_periods=[1, 2, 3],
            rolling_windows=[5, 10],
            min_periods=3,
        )
        gen = IVFeatureGenerator(config)
        assert gen._config.change_periods == [1, 2, 3]
        assert gen._config.rolling_windows == [5, 10]
        assert gen._config.min_periods == 3


# ============================================================================
# Generate Tests
# ============================================================================


class TestIVFeatureGenerate:
    """Test IV feature generation."""

    def test_generate_adds_change_columns(self, generator, sample_surface_df):
        """Test that generate adds IV change columns."""
        result = generator.generate(sample_surface_df)

        assert "iv_change_1d" in result.columns
        assert "iv_change_5d" in result.columns

    def test_generate_adds_volatility_columns(self, generator, sample_surface_df):
        """Test that generate adds IV volatility columns."""
        result = generator.generate(sample_surface_df)

        assert "iv_vol_5d" in result.columns
        assert "iv_vol_10d" in result.columns
        assert "iv_vol_21d" in result.columns

    def test_generate_adds_zscore_columns(self, generator, sample_surface_df):
        """Test that generate adds IV z-score columns."""
        result = generator.generate(sample_surface_df)

        assert "iv_zscore_5d" in result.columns
        assert "iv_zscore_10d" in result.columns
        assert "iv_zscore_21d" in result.columns

    def test_generate_preserves_original_columns(self, generator, sample_surface_df):
        """Test that original columns are preserved."""
        original_cols = sample_surface_df.columns.tolist()
        result = generator.generate(sample_surface_df)

        for col in original_cols:
            assert col in result.columns

    def test_generate_preserves_row_count(self, generator, sample_surface_df):
        """Test that row count is preserved."""
        result = generator.generate(sample_surface_df)
        assert len(result) == len(sample_surface_df)

    def test_generate_empty_df(self, generator):
        """Test generation with empty DataFrame."""
        empty_df = pd.DataFrame(columns=["ts_utc", "tenor_days", "delta_bucket", "iv_mid"])
        result = generator.generate(empty_df)
        assert len(result) == 0

    def test_generate_missing_columns_raises(self, generator):
        """Test that missing columns raise ValueError."""
        df = pd.DataFrame({
            "ts_utc": pd.date_range("2024-01-01", periods=3),
            "iv_mid": [0.20, 0.21, 0.22],
        })
        with pytest.raises(ValueError, match="Missing required columns"):
            generator.generate(df)

    def test_custom_config_columns(self, custom_generator, sample_surface_df):
        """Test that custom config produces expected columns."""
        result = custom_generator.generate(sample_surface_df)

        # Custom change periods
        assert "iv_change_1d" in result.columns
        assert "iv_change_3d" in result.columns
        assert "iv_change_5d" in result.columns

        # Custom rolling windows
        assert "iv_vol_3d" in result.columns
        assert "iv_vol_7d" in result.columns
        assert "iv_zscore_3d" in result.columns
        assert "iv_zscore_7d" in result.columns


# ============================================================================
# IV Change Tests
# ============================================================================


class TestIVChanges:
    """Test IV change calculations."""

    def test_iv_change_1d_correct(self, generator):
        """Test 1-day IV change is computed correctly."""
        df = pd.DataFrame({
            "ts_utc": pd.date_range("2024-01-01", periods=5),
            "tenor_days": [30] * 5,
            "delta_bucket": ["ATM"] * 5,
            "iv_mid": [0.20, 0.21, 0.22, 0.21, 0.23],
        })

        result = generator.generate(df)

        expected_changes = [np.nan, 0.01, 0.01, -0.01, 0.02]
        np.testing.assert_allclose(
            result["iv_change_1d"].values,
            expected_changes,
            rtol=1e-10,
            equal_nan=True,
        )

    def test_iv_change_5d_correct(self, generator):
        """Test 5-day IV change is computed correctly."""
        df = pd.DataFrame({
            "ts_utc": pd.date_range("2024-01-01", periods=10),
            "tenor_days": [30] * 10,
            "delta_bucket": ["ATM"] * 10,
            "iv_mid": [0.20, 0.21, 0.22, 0.21, 0.23, 0.24, 0.25, 0.24, 0.26, 0.27],
        })

        result = generator.generate(df)

        # 5-day change: iv[i] - iv[i-5]
        expected = [np.nan] * 5 + [0.04, 0.04, 0.02, 0.05, 0.04]
        np.testing.assert_allclose(
            result["iv_change_5d"].values,
            expected,
            rtol=1e-10,
            equal_nan=True,
        )

    def test_iv_change_per_node(self, generator):
        """Test IV changes are computed per node, not globally."""
        df = pd.DataFrame({
            "ts_utc": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-01", "2024-01-02"]),
            "tenor_days": [30, 30, 60, 60],
            "delta_bucket": ["ATM", "ATM", "ATM", "ATM"],
            "iv_mid": [0.20, 0.25, 0.30, 0.32],
        })

        result = generator.generate(df)

        # Node 1 (30 ATM): change should be 0.05
        node1 = result[(result["tenor_days"] == 30) & (result["delta_bucket"] == "ATM")]
        assert node1["iv_change_1d"].iloc[1] == pytest.approx(0.05)

        # Node 2 (60 ATM): change should be 0.02
        node2 = result[(result["tenor_days"] == 60) & (result["delta_bucket"] == "ATM")]
        assert node2["iv_change_1d"].iloc[1] == pytest.approx(0.02)


# ============================================================================
# IV Volatility Tests
# ============================================================================


class TestIVVolatility:
    """Test IV volatility calculations."""

    def test_iv_vol_rolling_std(self, generator):
        """Test rolling std is computed correctly."""
        df = pd.DataFrame({
            "ts_utc": pd.date_range("2024-01-01", periods=10),
            "tenor_days": [30] * 10,
            "delta_bucket": ["ATM"] * 10,
            "iv_mid": [0.20, 0.22, 0.21, 0.23, 0.20, 0.22, 0.21, 0.23, 0.20, 0.22],
        })

        result = generator.generate(df)

        # Verify rolling std is computed
        assert result["iv_vol_5d"].notna().sum() > 0

        # Manual check for last value with window=5 (shift(1) excludes current obs)
        expected_std = pd.Series(df["iv_mid"]).shift(1).rolling(5).std().iloc[-1]
        assert result["iv_vol_5d"].iloc[-1] == pytest.approx(expected_std)

    def test_iv_vol_min_periods(self):
        """Test min_periods is respected."""
        config = IVFeatureConfig(rolling_windows=[5], min_periods=3)
        gen = IVFeatureGenerator(config)

        df = pd.DataFrame({
            "ts_utc": pd.date_range("2024-01-01", periods=5),
            "tenor_days": [30] * 5,
            "delta_bucket": ["ATM"] * 5,
            "iv_mid": [0.20, 0.21, 0.22, 0.23, 0.24],
        })

        result = gen.generate(df)

        # shift(1) loses first row, then min_periods=3 needs 3 more valid values.
        # So first 3 values should be NaN, 4th should have data.
        assert pd.isna(result["iv_vol_5d"].iloc[0])
        assert pd.isna(result["iv_vol_5d"].iloc[1])
        assert pd.isna(result["iv_vol_5d"].iloc[2])
        assert pd.notna(result["iv_vol_5d"].iloc[3])


# ============================================================================
# Z-Score Tests
# ============================================================================


class TestIVZScore:
    """Test IV z-score calculations."""

    def test_zscore_computation(self, generator):
        """Test z-score is computed as (value - mean) / std."""
        df = pd.DataFrame({
            "ts_utc": pd.date_range("2024-01-01", periods=10),
            "tenor_days": [30] * 10,
            "delta_bucket": ["ATM"] * 10,
            "iv_mid": [0.20, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29],
        })

        result = generator.generate(df)

        # Manual z-score for last value with window=5 (shift(1) excludes current obs)
        iv_series = pd.Series(df["iv_mid"])
        iv_shifted = iv_series.shift(1)
        rolling_mean = iv_shifted.rolling(5).mean().iloc[-1]
        rolling_std = iv_shifted.rolling(5).std().iloc[-1]
        expected_zscore = (df["iv_mid"].iloc[-1] - rolling_mean) / rolling_std

        assert result["iv_zscore_5d"].iloc[-1] == pytest.approx(expected_zscore)

    def test_zscore_handles_zero_std(self, generator):
        """Test z-score handles zero std without error."""
        df = pd.DataFrame({
            "ts_utc": pd.date_range("2024-01-01", periods=5),
            "tenor_days": [30] * 5,
            "delta_bucket": ["ATM"] * 5,
            "iv_mid": [0.20, 0.20, 0.20, 0.20, 0.20],  # Constant IV
        })

        result = generator.generate(df)

        # Should not crash, values should be NaN for constant series
        # (0/0 = NaN)
        assert "iv_zscore_5d" in result.columns


# ============================================================================
# Standalone Utility Tests
# ============================================================================


class TestStandaloneUtilities:
    """Test standalone utility methods."""

    def test_compute_iv_changes(self, generator):
        """Test standalone IV changes computation."""
        df = pd.DataFrame({
            "iv_mid": [0.20, 0.21, 0.22, 0.21, 0.23],
        })

        result = generator.compute_iv_changes(df, periods=[1, 2])

        assert "iv_change_1d" in result.columns
        assert "iv_change_2d" in result.columns
        assert result["iv_change_1d"].iloc[1] == pytest.approx(0.01)

    def test_compute_iv_volatility(self, generator):
        """Test standalone IV volatility computation."""
        df = pd.DataFrame({
            "iv_mid": [0.20, 0.22, 0.21, 0.23, 0.20],
        })

        result = generator.compute_iv_volatility(df, windows=[3])

        assert "iv_vol_3d" in result.columns
        assert result["iv_vol_3d"].notna().sum() > 0
