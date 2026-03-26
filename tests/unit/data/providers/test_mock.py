"""Unit tests for mock data providers."""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from src.data.providers.mock import MockMacroDataProvider, MockMarketDataProvider


class TestMockMarketDataProvider:
    """Test MockMarketDataProvider implementation."""

    @pytest.fixture
    def provider(self):
        """Create mock provider with fixed seed."""
        return MockMarketDataProvider(seed=42)

    def test_initialization(self, provider):
        """Test provider initialization."""
        assert provider.seed == 42
        assert provider._rng is not None

    def test_fetch_underlying_bars_returns_dataframe(self, provider):
        """Test that fetch_underlying_bars returns a DataFrame."""
        start = datetime(2023, 1, 1, 9, 30)
        end = datetime(2023, 1, 1, 10, 30)

        df = provider.fetch_underlying_bars("SPY", start, end, interval="1m")

        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_fetch_underlying_bars_has_correct_columns(self, provider):
        """Test that returned DataFrame has correct columns."""
        start = datetime(2023, 1, 1, 9, 30)
        end = datetime(2023, 1, 1, 10, 30)

        df = provider.fetch_underlying_bars("SPY", start, end, interval="1m")

        expected_columns = ["ts_utc", "open", "high", "low", "close", "volume"]
        assert list(df.columns) == expected_columns

    def test_fetch_underlying_bars_ohlc_consistency(self, provider):
        """Test that OHLC data is internally consistent."""
        start = datetime(2023, 1, 1, 9, 30)
        end = datetime(2023, 1, 1, 10, 30)

        df = provider.fetch_underlying_bars("SPY", start, end, interval="1m")

        # High should be >= open and close
        assert (df["high"] >= df["open"]).all()
        assert (df["high"] >= df["close"]).all()

        # Low should be <= open and close
        assert (df["low"] <= df["open"]).all()
        assert (df["low"] <= df["close"]).all()

        # High should be >= low
        assert (df["high"] >= df["low"]).all()

    def test_fetch_underlying_bars_different_intervals(self, provider):
        """Test fetching bars with different intervals."""
        start = datetime(2023, 1, 1, 9, 30)
        end = datetime(2023, 1, 1, 16, 0)

        intervals = ["1m", "5m", "1h"]
        for interval in intervals:
            df = provider.fetch_underlying_bars("SPY", start, end, interval=interval)
            assert len(df) > 0
            assert isinstance(df, pd.DataFrame)

    def test_fetch_underlying_bars_invalid_interval_raises_error(self, provider):
        """Test that invalid interval raises ValueError."""
        start = datetime(2023, 1, 1, 9, 30)
        end = datetime(2023, 1, 1, 10, 30)

        with pytest.raises(ValueError, match="Unsupported interval"):
            provider.fetch_underlying_bars("SPY", start, end, interval="15s")

    def test_fetch_underlying_bars_empty_range_returns_empty_df(self, provider):
        """Test that empty time range returns empty DataFrame."""
        start = datetime(2023, 1, 1, 10, 0)
        end = datetime(2023, 1, 1, 10, 0)  # Same as start

        df = provider.fetch_underlying_bars("SPY", start, end, interval="1m")

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
        assert list(df.columns) == ["ts_utc", "open", "high", "low", "close", "volume"]

    def test_fetch_underlying_bars_reproducible_with_seed(self):
        """Test that data is reproducible with same seed."""
        start = datetime(2023, 1, 1, 9, 30)
        end = datetime(2023, 1, 1, 10, 30)

        provider1 = MockMarketDataProvider(seed=42)
        provider2 = MockMarketDataProvider(seed=42)

        df1 = provider1.fetch_underlying_bars("SPY", start, end, interval="1m")
        df2 = provider2.fetch_underlying_bars("SPY", start, end, interval="1m")

        pd.testing.assert_frame_equal(df1, df2)

    def test_fetch_underlying_bars_different_symbols(self, provider):
        """Test fetching bars for different symbols."""
        start = datetime(2023, 1, 1, 9, 30)
        end = datetime(2023, 1, 1, 10, 30)

        df_spy = provider.fetch_underlying_bars("SPY", start, end, interval="1m")
        df_qqq = provider.fetch_underlying_bars("QQQ", start, end, interval="1m")

        # Both should return data
        assert len(df_spy) > 0
        assert len(df_qqq) > 0

        # Data should be different (different symbols)
        assert not df_spy["close"].equals(df_qqq["close"])

    def test_fetch_option_quotes_returns_dataframe(self, provider):
        """Test that fetch_option_quotes returns a DataFrame."""
        start = datetime(2023, 1, 1, 9, 30)
        end = datetime(2023, 1, 1, 10, 30)
        symbols = ["SPY230120C00400000", "SPY230120P00400000"]

        df = provider.fetch_option_quotes(symbols, start, end)

        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_fetch_option_quotes_has_correct_columns(self, provider):
        """Test that option quotes have correct columns."""
        start = datetime(2023, 1, 1, 9, 30)
        end = datetime(2023, 1, 1, 10, 30)
        symbols = ["SPY230120C00400000"]

        df = provider.fetch_option_quotes(symbols, start, end)

        expected_columns = ["ts_utc", "option_symbol", "bid", "ask", "bid_size", "ask_size"]
        assert list(df.columns) == expected_columns

    def test_fetch_option_quotes_bid_ask_consistency(self, provider):
        """Test that bid is always less than ask."""
        start = datetime(2023, 1, 1, 9, 30)
        end = datetime(2023, 1, 1, 10, 30)
        symbols = ["SPY230120C00400000"]

        df = provider.fetch_option_quotes(symbols, start, end)

        assert (df["bid"] < df["ask"]).all()

    def test_fetch_option_quotes_positive_prices(self, provider):
        """Test that option prices are positive."""
        start = datetime(2023, 1, 1, 9, 30)
        end = datetime(2023, 1, 1, 10, 30)
        symbols = ["SPY230120C00400000"]

        df = provider.fetch_option_quotes(symbols, start, end)

        assert (df["bid"] > 0).all()
        assert (df["ask"] > 0).all()

    def test_fetch_option_quotes_multiple_symbols(self, provider):
        """Test fetching quotes for multiple symbols."""
        start = datetime(2023, 1, 1, 9, 30)
        end = datetime(2023, 1, 1, 10, 30)
        symbols = ["SPY230120C00400000", "SPY230120P00400000", "SPY230127C00405000"]

        df = provider.fetch_option_quotes(symbols, start, end)

        # Should have data for all symbols
        assert set(df["option_symbol"].unique()) == set(symbols)

    def test_fetch_option_quotes_empty_symbols_returns_empty_df(self, provider):
        """Test that empty symbols list returns empty DataFrame."""
        start = datetime(2023, 1, 1, 9, 30)
        end = datetime(2023, 1, 1, 10, 30)

        df = provider.fetch_option_quotes([], start, end)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_fetch_option_quotes_invalid_schema_raises_error(self, provider):
        """Test that invalid schema raises ValueError."""
        start = datetime(2023, 1, 1, 9, 30)
        end = datetime(2023, 1, 1, 10, 30)
        symbols = ["SPY230120C00400000"]

        with pytest.raises(ValueError, match="Unsupported schema"):
            provider.fetch_option_quotes(symbols, start, end, schema="invalid")

    def test_estimate_cost_returns_zero(self, provider):
        """Test that estimate_cost returns 0.0 for mock."""
        start = datetime(2023, 1, 1, 9, 30)
        end = datetime(2023, 1, 1, 10, 30)

        cost = provider.estimate_cost("GLBX.MDP3", "ohlcv-1m", ["SPY"], start, end)

        assert cost == 0.0
        assert isinstance(cost, float)

    def test_resolve_option_symbols_returns_list(self, provider):
        """Test that resolve_option_symbols returns a list."""
        as_of = datetime(2023, 1, 1, 9, 30)

        symbols = provider.resolve_option_symbols(
            parent="SPY",
            as_of=as_of,
            dte_min=7,
            dte_max=30,
            moneyness_min=0.95,
            moneyness_max=1.05,
        )

        assert isinstance(symbols, list)
        assert len(symbols) > 0

    def test_resolve_option_symbols_format(self, provider):
        """Test that resolved symbols have correct format."""
        as_of = datetime(2023, 1, 1, 9, 30)

        symbols = provider.resolve_option_symbols(
            parent="SPY",
            as_of=as_of,
            dte_min=7,
            dte_max=30,
            moneyness_min=0.95,
            moneyness_max=1.05,
        )

        # Check format: SPY230120C00400000 or SPY230120P00400000
        for symbol in symbols:
            assert symbol.startswith("SPY")
            assert symbol[9] in ["C", "P"]  # Call or Put
            assert len(symbol) == 18

    def test_resolve_option_symbols_includes_calls_and_puts(self, provider):
        """Test that resolved symbols include both calls and puts."""
        as_of = datetime(2023, 1, 1, 9, 30)

        symbols = provider.resolve_option_symbols(
            parent="SPY",
            as_of=as_of,
            dte_min=7,
            dte_max=30,
            moneyness_min=0.95,
            moneyness_max=1.05,
        )

        calls = [s for s in symbols if "C" in s[9]]
        puts = [s for s in symbols if "P" in s[9]]

        assert len(calls) > 0
        assert len(puts) > 0

    def test_resolve_option_symbols_empty_parent_raises_error(self, provider):
        """Test that empty parent symbol raises ValueError."""
        as_of = datetime(2023, 1, 1, 9, 30)

        with pytest.raises(ValueError, match="Parent symbol cannot be empty"):
            provider.resolve_option_symbols(
                parent="",
                as_of=as_of,
                dte_min=7,
                dte_max=30,
                moneyness_min=0.95,
                moneyness_max=1.05,
            )


class TestMockMacroDataProvider:
    """Test MockMacroDataProvider implementation."""

    @pytest.fixture
    def provider(self):
        """Create mock provider with fixed seed."""
        return MockMacroDataProvider(seed=42)

    def test_initialization(self, provider):
        """Test provider initialization."""
        assert provider.seed == 42
        assert provider._rng is not None

    def test_fetch_series_returns_dataframe(self, provider):
        """Test that fetch_series returns a DataFrame."""
        start = datetime(2023, 1, 1)
        end = datetime(2023, 1, 31)

        df = provider.fetch_series("DGS10", start, end)

        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_fetch_series_has_correct_columns(self, provider):
        """Test that returned DataFrame has correct columns."""
        start = datetime(2023, 1, 1)
        end = datetime(2023, 1, 31)

        df = provider.fetch_series("DGS10", start, end)

        expected_columns = ["obs_date", "value", "release_datetime_utc"]
        assert list(df.columns) == expected_columns

    def test_fetch_series_positive_values(self, provider):
        """Test that series values are positive."""
        start = datetime(2023, 1, 1)
        end = datetime(2023, 1, 31)

        df = provider.fetch_series("DGS10", start, end)

        assert (df["value"] > 0).all()

    def test_fetch_series_different_series(self, provider):
        """Test fetching different series."""
        start = datetime(2023, 1, 1)
        end = datetime(2023, 1, 31)

        series_ids = ["DGS10", "VIXCLS", "CUSTOM"]
        for series_id in series_ids:
            df = provider.fetch_series(series_id, start, end)
            assert len(df) > 0
            assert isinstance(df, pd.DataFrame)

    def test_fetch_series_empty_series_id_raises_error(self, provider):
        """Test that empty series_id raises ValueError."""
        start = datetime(2023, 1, 1)
        end = datetime(2023, 1, 31)

        with pytest.raises(ValueError, match="Series ID cannot be empty"):
            provider.fetch_series("", start, end)

    def test_fetch_series_empty_range_returns_empty_df(self, provider):
        """Test that empty time range returns empty DataFrame."""
        start = datetime(2023, 1, 1)
        end = datetime(2023, 1, 1)  # Same as start

        df = provider.fetch_series("DGS10", start, end)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_fetch_series_reproducible_with_seed(self):
        """Test that data is reproducible with same seed."""
        start = datetime(2023, 1, 1)
        end = datetime(2023, 1, 31)

        provider1 = MockMacroDataProvider(seed=42)
        provider2 = MockMacroDataProvider(seed=42)

        df1 = provider1.fetch_series("DGS10", start, end)
        df2 = provider2.fetch_series("DGS10", start, end)

        pd.testing.assert_frame_equal(df1, df2)

    def test_get_latest_value_returns_tuple(self, provider):
        """Test that get_latest_value returns a tuple."""
        as_of = datetime(2023, 1, 15, 10, 0)

        result = provider.get_latest_value("DGS10", as_of)

        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_get_latest_value_correct_types(self, provider):
        """Test that get_latest_value returns correct types."""
        as_of = datetime(2023, 1, 15, 10, 0)

        release_dt, value = provider.get_latest_value("DGS10", as_of)

        assert isinstance(release_dt, datetime)
        assert isinstance(value, float)

    def test_get_latest_value_before_as_of(self, provider):
        """Test that returned value is released before as_of."""
        as_of = datetime(2023, 1, 15, 10, 0)

        release_dt, value = provider.get_latest_value("DGS10", as_of)

        assert release_dt <= as_of

    def test_get_latest_value_empty_series_id_raises_error(self, provider):
        """Test that empty series_id raises ValueError."""
        as_of = datetime(2023, 1, 15, 10, 0)

        with pytest.raises(ValueError, match="Series ID cannot be empty"):
            provider.get_latest_value("", as_of)

    def test_get_latest_value_no_data_raises_error(self, provider):
        """Test that no data before as_of raises ValueError."""
        # Use a very early date where no data exists
        as_of = datetime(1900, 1, 1, 10, 0)

        with pytest.raises(ValueError, match="No data"):
            provider.get_latest_value("DGS10", as_of)
