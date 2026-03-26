"""Unit tests for returns feature generator."""

import numpy as np
import pandas as pd
import pytest

from src.features.global_.returns import ReturnsConfig, ReturnsGenerator


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_underlying_df():
    """Create sample underlying bar data."""
    return pd.DataFrame({
        "ts_utc": pd.date_range("2024-01-01", periods=30, freq="D"),
        "close": [100 + i * 0.5 + np.random.randn() * 0.1 for i in range(30)],
        "open": [100 + i * 0.5 for i in range(30)],
        "high": [101 + i * 0.5 for i in range(30)],
        "low": [99 + i * 0.5 for i in range(30)],
        "volume": [1000000] * 30,
    })


@pytest.fixture
def generator():
    """Create returns generator with default config."""
    return ReturnsGenerator()


@pytest.fixture
def simple_price_df():
    """Create simple price data for testing."""
    return pd.DataFrame({
        "ts_utc": pd.date_range("2024-01-01", periods=10, freq="D"),
        "close": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
    })


# ============================================================================
# Initialization Tests
# ============================================================================


class TestReturnsGeneratorInit:
    """Test returns generator initialization."""

    def test_default_config(self):
        """Test default configuration values."""
        gen = ReturnsGenerator()
        assert gen._config.periods == [1, 5, 10, 21]
        assert gen._config.include_log_returns is True
        assert gen._config.include_simple_returns is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = ReturnsConfig(
            periods=[1, 3, 7],
            include_log_returns=False,
            include_simple_returns=True,
        )
        gen = ReturnsGenerator(config)
        assert gen._config.periods == [1, 3, 7]
        assert gen._config.include_log_returns is False


# ============================================================================
# Generate Tests
# ============================================================================


class TestReturnsGenerate:
    """Test returns feature generation."""

    def test_generate_adds_simple_returns(self, generator, sample_underlying_df):
        """Test that generate adds simple return columns."""
        result = generator.generate(sample_underlying_df)

        assert "returns_1d" in result.columns
        assert "returns_5d" in result.columns
        assert "returns_10d" in result.columns
        assert "returns_21d" in result.columns

    def test_generate_adds_log_returns(self, generator, sample_underlying_df):
        """Test that generate adds log return columns."""
        result = generator.generate(sample_underlying_df)

        assert "log_returns_1d" in result.columns
        assert "log_returns_5d" in result.columns
        assert "log_returns_10d" in result.columns
        assert "log_returns_21d" in result.columns

    def test_generate_preserves_original_columns(self, generator, sample_underlying_df):
        """Test that original columns are preserved."""
        original_cols = sample_underlying_df.columns.tolist()
        result = generator.generate(sample_underlying_df)

        for col in original_cols:
            assert col in result.columns

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

    def test_generate_only_simple_returns(self, sample_underlying_df):
        """Test generation with only simple returns."""
        config = ReturnsConfig(include_log_returns=False)
        gen = ReturnsGenerator(config)
        result = gen.generate(sample_underlying_df)

        assert "returns_1d" in result.columns
        assert "log_returns_1d" not in result.columns

    def test_generate_only_log_returns(self, sample_underlying_df):
        """Test generation with only log returns."""
        config = ReturnsConfig(include_simple_returns=False)
        gen = ReturnsGenerator(config)
        result = gen.generate(sample_underlying_df)

        assert "log_returns_1d" in result.columns
        assert "returns_1d" not in result.columns


# ============================================================================
# Return Calculation Tests
# ============================================================================


class TestReturnCalculations:
    """Test return calculations are correct."""

    def test_simple_return_1d_correct(self, generator, simple_price_df):
        """Test 1-day simple return calculation."""
        result = generator.generate(simple_price_df)

        # Second day: (101 - 100) / 100 = 0.01
        assert result["returns_1d"].iloc[1] == pytest.approx(0.01)
        # Third day: (102 - 101) / 101
        assert result["returns_1d"].iloc[2] == pytest.approx(1 / 101)

    def test_simple_return_5d_correct(self, generator, simple_price_df):
        """Test 5-day simple return calculation."""
        result = generator.generate(simple_price_df)

        # Day 6 (index 5): (105 - 100) / 100 = 0.05
        assert result["returns_5d"].iloc[5] == pytest.approx(0.05)

    def test_log_return_1d_correct(self, generator, simple_price_df):
        """Test 1-day log return calculation."""
        result = generator.generate(simple_price_df)

        # log(101/100)
        expected = np.log(101 / 100)
        assert result["log_returns_1d"].iloc[1] == pytest.approx(expected)

    def test_log_return_5d_correct(self, generator, simple_price_df):
        """Test 5-day log return calculation."""
        result = generator.generate(simple_price_df)

        # log(105/100)
        expected = np.log(105 / 100)
        assert result["log_returns_5d"].iloc[5] == pytest.approx(expected)

    def test_first_n_returns_are_nan(self, generator, simple_price_df):
        """Test that first N returns are NaN for N-day return."""
        result = generator.generate(simple_price_df)

        # First row should be NaN for 1d return
        assert pd.isna(result["returns_1d"].iloc[0])

        # First 5 rows should be NaN for 5d return
        assert result["returns_5d"].iloc[:5].isna().all()


# ============================================================================
# Standalone Utility Tests
# ============================================================================


class TestComputeReturns:
    """Test standalone compute_returns method."""

    def test_compute_returns_basic(self, generator):
        """Test basic compute_returns functionality."""
        prices = pd.Series([100, 101, 102, 103, 104, 105])
        result = generator.compute_returns(prices, periods=[1, 2])

        assert "returns_1d" in result.columns
        assert "returns_2d" in result.columns
        assert "log_returns_1d" in result.columns
        assert "log_returns_2d" in result.columns

    def test_compute_returns_custom_periods(self, generator):
        """Test compute_returns with custom periods."""
        prices = pd.Series([100, 105, 110, 115, 120])
        result = generator.compute_returns(prices, periods=[1, 3])

        # 1-day return at index 1: (105-100)/100 = 0.05
        assert result["returns_1d"].iloc[1] == pytest.approx(0.05)

        # 3-day return at index 3: (115-100)/100 = 0.15
        assert result["returns_3d"].iloc[3] == pytest.approx(0.15)


# ============================================================================
# Edge Cases
# ============================================================================


class TestReturnEdgeCases:
    """Test edge cases for returns calculation."""

    def test_constant_prices(self, generator):
        """Test returns with constant prices (zero returns)."""
        df = pd.DataFrame({
            "ts_utc": pd.date_range("2024-01-01", periods=10),
            "close": [100] * 10,
        })
        result = generator.generate(df)

        # All returns should be 0 (except first NaN)
        assert (result["returns_1d"].dropna() == 0).all()

    def test_negative_return(self, generator):
        """Test negative returns are computed correctly."""
        df = pd.DataFrame({
            "ts_utc": pd.date_range("2024-01-01", periods=5),
            "close": [100, 95, 90, 85, 80],  # Declining prices
        })
        result = generator.generate(df)

        # Returns should be negative
        assert result["returns_1d"].dropna().iloc[0] < 0

    def test_large_return(self, generator):
        """Test large returns are computed correctly."""
        df = pd.DataFrame({
            "ts_utc": pd.date_range("2024-01-01", periods=3),
            "close": [100, 200, 300],  # Doubling each day
        })
        result = generator.generate(df)

        # First return: (200-100)/100 = 1.0 (100%)
        assert result["returns_1d"].iloc[1] == pytest.approx(1.0)

    def test_unsorted_data_gets_sorted(self, generator):
        """Test that unsorted data is sorted by timestamp."""
        df = pd.DataFrame({
            "ts_utc": pd.to_datetime(["2024-01-03", "2024-01-01", "2024-01-02"]),
            "close": [103, 100, 101],
        })
        result = generator.generate(df)

        # Should be sorted by timestamp
        assert result["ts_utc"].iloc[0] == pd.Timestamp("2024-01-01")
        assert result["close"].iloc[0] == 100
