"""Unit tests for realized volatility feature generator."""

import numpy as np
import pandas as pd
import pytest

from src.features.global_.realized_vol import RealizedVolConfig, RealizedVolGenerator


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_underlying_df():
    """Create sample underlying bar data with realistic volatility."""
    np.random.seed(42)
    n = 60
    returns = np.random.randn(n) * 0.01  # ~1% daily vol
    prices = 100 * np.exp(np.cumsum(returns))

    return pd.DataFrame({
        "ts_utc": pd.date_range("2024-01-01", periods=n, freq="D"),
        "close": prices,
    })


@pytest.fixture
def generator():
    """Create realized vol generator with default config."""
    return RealizedVolGenerator()


@pytest.fixture
def simple_vol_df():
    """Create simple data for volatility testing."""
    # Create known volatility pattern
    returns = [0.01, -0.01, 0.02, -0.02, 0.01, -0.01, 0.02, -0.02, 0.01, -0.01]
    prices = [100]
    for r in returns:
        prices.append(prices[-1] * (1 + r))

    return pd.DataFrame({
        "ts_utc": pd.date_range("2024-01-01", periods=len(prices), freq="D"),
        "close": prices,
    })


# ============================================================================
# Initialization Tests
# ============================================================================


class TestRealizedVolGeneratorInit:
    """Test realized vol generator initialization."""

    def test_default_config(self):
        """Test default configuration values."""
        gen = RealizedVolGenerator()
        assert gen._config.variance_windows == [5, 10, 21]
        assert gen._config.vol_of_vol_window == 21
        assert gen._config.drawdown_window == 252
        assert gen._config.annualization_factor == 252
        assert gen._config.min_periods == 2

    def test_custom_config(self):
        """Test custom configuration."""
        config = RealizedVolConfig(
            variance_windows=[5, 10],
            vol_of_vol_window=10,
            drawdown_window=60,
        )
        gen = RealizedVolGenerator(config)
        assert gen._config.variance_windows == [5, 10]
        assert gen._config.vol_of_vol_window == 10
        assert gen._config.drawdown_window == 60


# ============================================================================
# Generate Tests
# ============================================================================


class TestRealizedVolGenerate:
    """Test realized vol feature generation."""

    def test_generate_adds_rv_columns(self, generator, sample_underlying_df):
        """Test that generate adds realized variance columns."""
        result = generator.generate(sample_underlying_df)

        assert "rv_5d" in result.columns
        assert "rv_10d" in result.columns
        assert "rv_21d" in result.columns

    def test_generate_adds_realized_vol_columns(self, generator, sample_underlying_df):
        """Test that generate adds realized volatility columns."""
        result = generator.generate(sample_underlying_df)

        assert "realized_vol_5d" in result.columns
        assert "realized_vol_10d" in result.columns
        assert "realized_vol_21d" in result.columns

    def test_generate_adds_vol_of_vol(self, generator, sample_underlying_df):
        """Test that generate adds vol-of-vol column."""
        result = generator.generate(sample_underlying_df)

        assert "vol_of_vol_21d" in result.columns

    def test_generate_adds_drawdown(self, generator, sample_underlying_df):
        """Test that generate adds drawdown columns."""
        result = generator.generate(sample_underlying_df)

        assert "drawdown" in result.columns
        assert "max_drawdown_252d" in result.columns

    def test_generate_preserves_original_columns(self, generator, sample_underlying_df):
        """Test that original columns are preserved."""
        original_cols = sample_underlying_df.columns.tolist()
        result = generator.generate(sample_underlying_df)

        for col in original_cols:
            assert col in result.columns

    def test_generate_removes_temp_columns(self, generator, sample_underlying_df):
        """Test that temporary columns are removed."""
        result = generator.generate(sample_underlying_df)

        assert "_returns_1d" not in result.columns

    def test_generate_empty_df(self, generator):
        """Test generation with empty DataFrame."""
        empty_df = pd.DataFrame(columns=["ts_utc", "close"])
        result = generator.generate(empty_df)
        assert len(result) == 0

    def test_generate_missing_columns_raises(self, generator):
        """Test that missing columns raise ValueError."""
        df = pd.DataFrame({
            "ts_utc": pd.date_range("2024-01-01", periods=3),
        })
        with pytest.raises(ValueError, match="Missing required columns"):
            generator.generate(df)


# ============================================================================
# Realized Variance Tests
# ============================================================================


class TestRealizedVariance:
    """Test realized variance calculations."""

    def test_rv_is_annualized(self, generator, sample_underlying_df):
        """Test that realized variance is annualized."""
        result = generator.generate(sample_underlying_df)

        # Annualized variance should be roughly in the range of daily vol^2 * 252
        # For ~1% daily vol, annualized variance should be ~0.025 (2.5%)
        rv_5d = result["rv_5d"].dropna()
        assert rv_5d.mean() > 0.01  # Should be positive
        assert rv_5d.mean() < 0.5  # Should not be too large

    def test_realized_vol_is_sqrt_of_variance(self, generator, sample_underlying_df):
        """Test that realized vol equals sqrt of variance."""
        result = generator.generate(sample_underlying_df)

        np.testing.assert_allclose(
            result["realized_vol_5d"].values,
            np.sqrt(result["rv_5d"].values),
            equal_nan=True,
        )

    def test_rv_first_values_are_nan(self, generator, simple_vol_df):
        """Test that first values are NaN (insufficient data)."""
        result = generator.generate(simple_vol_df)

        # First few values should be NaN
        assert pd.isna(result["rv_5d"].iloc[0])

    def test_rv_custom_windows(self):
        """Test realized variance with custom windows."""
        config = RealizedVolConfig(variance_windows=[3, 7])
        gen = RealizedVolGenerator(config)

        df = pd.DataFrame({
            "ts_utc": pd.date_range("2024-01-01", periods=20, freq="D"),
            "close": [100 + i * 0.5 for i in range(20)],
        })
        result = gen.generate(df)

        assert "rv_3d" in result.columns
        assert "rv_7d" in result.columns
        assert "rv_5d" not in result.columns


# ============================================================================
# Vol-of-Vol Tests
# ============================================================================


class TestVolOfVol:
    """Test volatility-of-volatility calculations."""

    def test_vol_of_vol_computed(self, generator, sample_underlying_df):
        """Test vol-of-vol is computed."""
        result = generator.generate(sample_underlying_df)

        # Should have some non-NaN values
        assert result["vol_of_vol_21d"].notna().any()

    def test_vol_of_vol_custom_window(self):
        """Test vol-of-vol with custom window."""
        config = RealizedVolConfig(vol_of_vol_window=10)
        gen = RealizedVolGenerator(config)

        np.random.seed(42)
        df = pd.DataFrame({
            "ts_utc": pd.date_range("2024-01-01", periods=30, freq="D"),
            "close": 100 * np.exp(np.cumsum(np.random.randn(30) * 0.01)),
        })
        result = gen.generate(df)

        assert "vol_of_vol_10d" in result.columns


# ============================================================================
# Drawdown Tests
# ============================================================================


class TestDrawdown:
    """Test drawdown calculations."""

    def test_drawdown_at_peak_is_zero(self, generator):
        """Test drawdown is zero at price peaks."""
        # Create data with clear peak at end
        df = pd.DataFrame({
            "ts_utc": pd.date_range("2024-01-01", periods=10, freq="D"),
            "close": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
        })
        result = generator.generate(df)

        # All drawdowns should be 0 (always at new high)
        assert (result["drawdown"] == 0).all()

    def test_drawdown_negative_after_peak(self, generator):
        """Test drawdown is negative after price peak."""
        # Create data with peak then decline
        df = pd.DataFrame({
            "ts_utc": pd.date_range("2024-01-01", periods=5, freq="D"),
            "close": [100, 110, 100, 90, 80],
        })
        result = generator.generate(df)

        # At index 2: 100/110 - 1 = -0.0909
        assert result["drawdown"].iloc[2] == pytest.approx(-10/110, rel=1e-4)

        # At index 4: 80/110 - 1 = -0.2727
        assert result["drawdown"].iloc[4] == pytest.approx(-30/110, rel=1e-4)

    def test_max_drawdown(self, generator):
        """Test max drawdown captures worst point."""
        df = pd.DataFrame({
            "ts_utc": pd.date_range("2024-01-01", periods=10, freq="D"),
            "close": [100, 110, 90, 95, 85, 90, 88, 92, 100, 105],
        })
        result = generator.generate(df)

        # Max drawdown should be at index 4: 85/110 - 1 = -0.227
        min_dd = result["max_drawdown_252d"].iloc[-1]
        assert min_dd == pytest.approx(-25/110, rel=1e-4)


# ============================================================================
# Standalone Utility Tests
# ============================================================================


class TestStandaloneUtilities:
    """Test standalone utility methods."""

    def test_compute_realized_variance(self, generator):
        """Test standalone compute_realized_variance method."""
        returns = pd.Series([0.01, -0.01, 0.02, -0.02, 0.01, -0.01, 0.02])
        variance = generator.compute_realized_variance(returns, window=5)

        assert len(variance) == len(returns)
        assert variance.notna().any()

    def test_compute_realized_variance_no_annualize(self, generator):
        """Test compute_realized_variance without annualization."""
        returns = pd.Series([0.01, -0.01, 0.02, -0.02, 0.01])
        var_annual = generator.compute_realized_variance(returns, window=3, annualize=True)
        var_raw = generator.compute_realized_variance(returns, window=3, annualize=False)

        # Annualized should be 252x raw
        ratio = var_annual.dropna() / var_raw.dropna()
        assert ratio.mean() == pytest.approx(252, rel=1e-6)

    def test_compute_drawdown(self, generator):
        """Test standalone compute_drawdown method."""
        prices = pd.Series([100, 110, 100, 90, 95])
        drawdown = generator.compute_drawdown(prices)

        assert drawdown.iloc[0] == 0  # First price is peak
        assert drawdown.iloc[1] == 0  # New peak
        assert drawdown.iloc[2] == pytest.approx(-10/110)


# ============================================================================
# Edge Cases
# ============================================================================


class TestRealizedVolEdgeCases:
    """Test edge cases for realized volatility."""

    def test_constant_prices(self, generator):
        """Test with constant prices (zero volatility)."""
        df = pd.DataFrame({
            "ts_utc": pd.date_range("2024-01-01", periods=30, freq="D"),
            "close": [100] * 30,
        })
        result = generator.generate(df)

        # Variance should be zero (except NaN for first few)
        rv_5d = result["rv_5d"].dropna()
        assert (rv_5d == 0).all()

    def test_single_large_move(self, generator):
        """Test volatility spike from single large move."""
        prices = [100] * 10 + [120] + [120] * 10  # Jump in middle
        df = pd.DataFrame({
            "ts_utc": pd.date_range("2024-01-01", periods=len(prices), freq="D"),
            "close": prices,
        })
        result = generator.generate(df)

        # Variance should spike around the jump
        rv_5d = result["rv_5d"]
        max_idx = rv_5d.idxmax()
        assert max_idx >= 10  # Peak should be at or after jump
