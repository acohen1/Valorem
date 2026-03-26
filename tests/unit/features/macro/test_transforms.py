"""Unit tests for macro transform feature generator."""

import numpy as np
import pandas as pd
import pytest

from src.features.macro.alignment import AlignmentConfig
from src.features.macro.transforms import MacroTransformConfig, MacroTransformGenerator


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_fred_df():
    """Create sample FRED series data (treasury rates)."""
    return pd.DataFrame({
        "obs_date": pd.date_range("2024-01-01", periods=60, freq="D"),
        "value": [4.5 + i * 0.01 + np.random.randn() * 0.02 for i in range(60)],
    })


@pytest.fixture
def sample_vix_df():
    """Create sample VIX data (not in percent)."""
    return pd.DataFrame({
        "obs_date": pd.date_range("2024-01-01", periods=60, freq="D"),
        "value": [15 + i * 0.1 + np.random.randn() * 0.5 for i in range(60)],
    })


@pytest.fixture
def generator():
    """Create macro transform generator with default config."""
    return MacroTransformGenerator()


@pytest.fixture
def simple_series_df():
    """Create simple series for testing."""
    return pd.DataFrame({
        "obs_date": pd.date_range("2024-01-01", periods=40, freq="D"),
        "value": list(range(40)),  # 0, 1, 2, ..., 39
    })


# ============================================================================
# Initialization Tests
# ============================================================================


class TestMacroTransformInit:
    """Test macro transform generator initialization."""

    def test_default_config(self):
        """Test default configuration values."""
        gen = MacroTransformGenerator()
        assert gen._config.include_level is True
        assert gen._config.include_change_1w is True
        assert gen._config.include_change_1m is True
        assert gen._config.include_zscore is True
        assert gen._config.zscore_window == 252
        assert gen._config.percent_to_decimal is False

    def test_custom_config(self):
        """Test custom configuration."""
        config = MacroTransformConfig(
            include_level=True,
            include_change_1w=False,
            include_change_1m=True,
            include_zscore=False,
            zscore_window=100,
        )
        gen = MacroTransformGenerator(config)
        assert gen._config.include_change_1w is False
        assert gen._config.include_zscore is False
        assert gen._config.zscore_window == 100


# ============================================================================
# Generate Tests
# ============================================================================


class TestMacroTransformGenerate:
    """Test macro transform feature generation."""

    def test_generate_adds_level(self, generator, sample_fred_df):
        """Test that generate adds level column."""
        result = generator.generate(sample_fred_df, "DGS10")
        assert "DGS10_level" in result.columns

    def test_generate_adds_change_1w(self, generator, sample_fred_df):
        """Test that generate adds 1-week change column."""
        result = generator.generate(sample_fred_df, "DGS10")
        assert "DGS10_change_1w" in result.columns

    def test_generate_adds_change_1m(self, generator, sample_fred_df):
        """Test that generate adds 1-month change column."""
        result = generator.generate(sample_fred_df, "DGS10")
        assert "DGS10_change_1m" in result.columns

    def test_generate_adds_zscore(self, generator, sample_fred_df):
        """Test that generate adds z-score column."""
        result = generator.generate(sample_fred_df, "DGS10")
        assert "DGS10_zscore" in result.columns

    def test_generate_adds_ts_utc(self, generator, sample_fred_df):
        """Test that generate adds timestamp column."""
        result = generator.generate(sample_fred_df, "DGS10")
        assert "ts_utc" in result.columns

    def test_generate_empty_df(self, generator):
        """Test generation with empty DataFrame."""
        empty_df = pd.DataFrame(columns=["obs_date", "value"])
        result = generator.generate(empty_df, "TEST")
        assert len(result) == 0
        assert "ts_utc" in result.columns

    def test_generate_missing_columns_raises(self, generator):
        """Test that missing columns raise ValueError."""
        df = pd.DataFrame({"obs_date": pd.date_range("2024-01-01", periods=3)})
        with pytest.raises(ValueError, match="Missing required columns"):
            generator.generate(df, "TEST")


# ============================================================================
# Percent Conversion Tests
# ============================================================================


class TestPercentConversion:
    """Test percent to decimal conversion."""

    def test_percent_series_not_converted_by_default(self, generator, sample_fred_df):
        """Test that percent series values are NOT converted by default.

        The FRED provider already converts percent to decimal, so
        percent_to_decimal defaults to False.
        """
        # DGS10 is in percent_series but percent_to_decimal=False by default
        result = generator.generate(sample_fred_df.copy(), "DGS10")

        # Original values are ~4.5, should remain unconverted
        assert result["DGS10_level"].mean() > 1

    def test_non_percent_series_not_converted(self, generator, sample_vix_df):
        """Test that non-percent series values are not converted."""
        # VIX is not in percent_series
        result = generator.generate(sample_vix_df.copy(), "VIXCLS")

        # Original values are ~15-20, should stay the same
        assert result["VIXCLS_level"].mean() > 10

    def test_custom_percent_series(self, sample_fred_df):
        """Test custom percent series list."""
        config = MacroTransformConfig(percent_series=["CUSTOM"])
        gen = MacroTransformGenerator(config)

        result = gen.generate(sample_fred_df.copy(), "DGS10")
        # DGS10 not in custom list, so not converted
        assert result["DGS10_level"].mean() > 1  # Should be ~4.5


# ============================================================================
# Level Tests
# ============================================================================


class TestLevel:
    """Test level feature generation."""

    def test_level_only(self, simple_series_df):
        """Test level-only generation."""
        config = MacroTransformConfig(
            include_level=True,
            include_change_1w=False,
            include_change_1m=False,
            include_zscore=False,
            percent_to_decimal=False,
        )
        gen = MacroTransformGenerator(config)
        result = gen.generate(simple_series_df.copy(), "TEST")

        assert "TEST_level" in result.columns
        assert "TEST_change_1w" not in result.columns


# ============================================================================
# Change Tests
# ============================================================================


class TestChange:
    """Test change feature generation."""

    def test_change_1w_correct(self, simple_series_df):
        """Test 1-week change calculation."""
        config = MacroTransformConfig(
            include_level=False,
            include_change_1w=True,
            include_change_1m=False,
            include_zscore=False,
            percent_to_decimal=False,
        )
        gen = MacroTransformGenerator(config)
        result = gen.generate(simple_series_df.copy(), "TEST")

        # Change at index 7: value[7] - value[0] = 7 - 0 = 7
        assert result["TEST_change_1w"].iloc[7] == pytest.approx(7)

    def test_change_1m_correct(self, simple_series_df):
        """Test 1-month change calculation."""
        config = MacroTransformConfig(
            include_level=False,
            include_change_1w=False,
            include_change_1m=True,
            include_zscore=False,
            percent_to_decimal=False,
        )
        gen = MacroTransformGenerator(config)
        result = gen.generate(simple_series_df.copy(), "TEST")

        # Change at index 30: value[30] - value[0] = 30 - 0 = 30
        assert result["TEST_change_1m"].iloc[30] == pytest.approx(30)

    def test_change_first_values_nan(self, simple_series_df):
        """Test that first values are NaN for changes."""
        config = MacroTransformConfig(
            include_level=False,
            include_change_1w=True,
            include_change_1m=False,
            include_zscore=False,
            percent_to_decimal=False,
        )
        gen = MacroTransformGenerator(config)
        result = gen.generate(simple_series_df.copy(), "TEST")

        # First 7 values should be NaN for 1w change
        assert result["TEST_change_1w"].iloc[:7].isna().all()


# ============================================================================
# Z-Score Tests
# ============================================================================


class TestZScore:
    """Test z-score feature generation."""

    def test_zscore_computation(self, generator, sample_fred_df):
        """Test z-score is computed."""
        result = generator.generate(sample_fred_df.copy(), "DGS10")

        # Z-score should have some non-NaN values
        assert result["DGS10_zscore"].notna().any()

    def test_zscore_centered_around_zero(self):
        """Test z-score is centered around zero for stationary data."""
        np.random.seed(42)
        df = pd.DataFrame({
            "obs_date": pd.date_range("2024-01-01", periods=300, freq="D"),
            "value": np.random.randn(300) * 0.5 + 5,  # Mean=5, std=0.5
        })
        config = MacroTransformConfig(
            include_level=False,
            include_change_1w=False,
            include_change_1m=False,
            include_zscore=True,
            zscore_window=100,
            percent_to_decimal=False,
        )
        gen = MacroTransformGenerator(config)
        result = gen.generate(df, "TEST")

        # Z-score mean should be close to 0
        zscore_mean = result["TEST_zscore"].dropna().mean()
        assert abs(zscore_mean) < 0.5

    def test_zscore_custom_window(self):
        """Test z-score with custom window."""
        df = pd.DataFrame({
            "obs_date": pd.date_range("2024-01-01", periods=50, freq="D"),
            "value": list(range(50)),
        })
        config = MacroTransformConfig(
            include_zscore=True,
            zscore_window=20,
            percent_to_decimal=False,
        )
        gen = MacroTransformGenerator(config)
        result = gen.generate(df, "TEST")

        # Should have values after window kicks in
        assert result["TEST_zscore"].notna().sum() > 0


# ============================================================================
# Multi-Series Tests
# ============================================================================


class TestMultiSeries:
    """Test multi-series generation."""

    def test_generate_multi_basic(self, generator, sample_fred_df, sample_vix_df):
        """Test generating features for multiple series."""
        series_data = {
            "DGS10": sample_fred_df,
            "VIXCLS": sample_vix_df,
        }
        result = generator.generate_multi(series_data)

        assert "DGS10_level" in result.columns
        assert "VIXCLS_level" in result.columns
        assert "ts_utc" in result.columns

    def test_generate_multi_empty(self, generator):
        """Test generate_multi with empty dict."""
        result = generator.generate_multi({})
        assert len(result) == 0
        assert "ts_utc" in result.columns

    def test_generate_multi_merges_correctly(self, generator):
        """Test that multi-series merge works correctly."""
        df1 = pd.DataFrame({
            "obs_date": pd.date_range("2024-01-01", periods=5, freq="D"),
            "value": [1, 2, 3, 4, 5],
        })
        df2 = pd.DataFrame({
            "obs_date": pd.date_range("2024-01-01", periods=5, freq="D"),
            "value": [10, 20, 30, 40, 50],
        })

        config = MacroTransformConfig(
            include_level=True,
            include_change_1w=False,
            include_change_1m=False,
            include_zscore=False,
            percent_to_decimal=False,
        )
        gen = MacroTransformGenerator(config)

        result = gen.generate_multi({"A": df1, "B": df2})

        assert len(result) == 5
        assert "A_level" in result.columns
        assert "B_level" in result.columns


# ============================================================================
# Standalone Utility Tests
# ============================================================================


class TestStandaloneUtilities:
    """Test standalone utility methods."""

    def test_compute_change(self, generator):
        """Test standalone compute_change method."""
        values = pd.Series([100, 101, 102, 103, 104, 105, 106, 107])
        change = generator.compute_change(values, "1w")

        # Change at index 7: 107 - 100 = 7
        assert change.iloc[7] == pytest.approx(7)

    def test_compute_zscore(self, generator):
        """Test standalone compute_zscore method."""
        values = pd.Series([10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
        zscore = generator.compute_zscore(values, window=5)

        # Should have non-NaN values
        assert zscore.notna().any()


# ============================================================================
# Alignment Integration Tests
# ============================================================================


class TestAlignmentIntegration:
    """Test integration with release-time alignment."""

    def test_conservative_alignment_applied(self, sample_fred_df):
        """Test that conservative alignment is applied."""
        config = MacroTransformConfig(
            alignment=AlignmentConfig(mode="conservative", conservative_delay_days=1),
            percent_to_decimal=False,
        )
        gen = MacroTransformGenerator(config)
        result = gen.generate(sample_fred_df, "TEST")

        # First ts_utc should be obs_date + 1 day
        expected = sample_fred_df["obs_date"].iloc[0] + pd.Timedelta(days=1)
        assert result["ts_utc"].iloc[0] == expected

    def test_strict_alignment_filters(self):
        """Test that strict alignment filters rows without release time."""
        df = pd.DataFrame({
            "obs_date": pd.date_range("2024-01-01", periods=5, freq="D"),
            "value": [1, 2, 3, 4, 5],
            "release_datetime_utc": [
                pd.Timestamp("2024-01-02"),
                None,
                pd.Timestamp("2024-01-04"),
                pd.NaT,
                pd.Timestamp("2024-01-06"),
            ],
        })
        config = MacroTransformConfig(
            alignment=AlignmentConfig(mode="strict"),
            percent_to_decimal=False,
        )
        gen = MacroTransformGenerator(config)
        result = gen.generate(df, "TEST")

        # Only 3 rows have valid release times
        assert len(result) == 3
