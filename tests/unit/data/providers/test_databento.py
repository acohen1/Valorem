"""Unit tests for Databento provider."""

from datetime import datetime
from unittest.mock import MagicMock, Mock, patch

import pandas as pd
import pytest

from src.data.providers.databento import DatabentoProvider
from src.data.providers.protocol import MarketDataProvider
from src.exceptions import ConfigError, ProviderError


class TestDatabentoProviderInitialization:
    """Test DatabentoProvider initialization."""

    @patch("src.data.providers.databento.db.Historical")
    def test_initialization_with_api_key(self, mock_historical):
        """Test provider initialization with explicit API key."""
        provider = DatabentoProvider(api_key="test_key")

        mock_historical.assert_called_once_with(key="test_key")
        assert provider._client is not None

    @patch("src.data.providers.databento.db.Historical")
    @patch.dict("os.environ", {"DATABENTO_API_KEY": "env_key"})
    def test_initialization_from_environment(self, mock_historical):
        """Test provider initialization from environment variable."""
        provider = DatabentoProvider()

        mock_historical.assert_called_once_with(key="env_key")
        assert provider._client is not None

    @patch.dict("os.environ", {}, clear=True)
    def test_initialization_without_key_raises_error(self):
        """Test that missing API key raises ConfigError."""
        with pytest.raises(ConfigError, match="Databento API key required"):
            DatabentoProvider()

    @patch("src.data.providers.databento.db.Historical")
    def test_provider_satisfies_protocol(self, mock_historical):
        """Test that DatabentoProvider satisfies MarketDataProvider protocol."""
        provider = DatabentoProvider(api_key="test_key")
        assert isinstance(provider, MarketDataProvider)


class TestFetchUnderlyingBars:
    """Test fetch_underlying_bars method."""

    @pytest.fixture
    def provider(self):
        """Create provider with mocked client."""
        with patch("src.data.providers.databento.db.Historical"):
            return DatabentoProvider(api_key="test_key")

    @pytest.fixture
    def mock_databento_response(self):
        """Create mock Databento response."""
        mock_data = Mock()
        mock_df = pd.DataFrame(
            {
                "ts_event": pd.date_range("2023-01-01 09:30", periods=5, freq="1min"),
                "open": [400.0, 400.5, 401.0, 400.8, 401.2],
                "high": [400.5, 401.0, 401.5, 401.0, 401.5],
                "low": [399.5, 400.0, 400.5, 400.5, 400.8],
                "close": [400.2, 400.8, 401.2, 400.9, 401.3],
                "volume": [1000, 1500, 1200, 1300, 1100],
            }
        )
        mock_data.to_df.return_value = mock_df
        return mock_data

    def test_fetch_underlying_bars_returns_dataframe(
        self, provider, mock_databento_response
    ):
        """Test that fetch_underlying_bars returns a DataFrame."""
        provider._client.timeseries.get_range = Mock(
            return_value=mock_databento_response
        )

        start = datetime(2023, 1, 1, 9, 30)
        end = datetime(2023, 1, 1, 10, 30)

        df = provider.fetch_underlying_bars("SPY", start, end, interval="1m")

        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_fetch_underlying_bars_normalizes_columns(
        self, provider, mock_databento_response
    ):
        """Test that column names are normalized."""
        provider._client.timeseries.get_range = Mock(
            return_value=mock_databento_response
        )

        start = datetime(2023, 1, 1, 9, 30)
        end = datetime(2023, 1, 1, 10, 30)

        df = provider.fetch_underlying_bars("SPY", start, end, interval="1m")

        expected_columns = ["ts_utc", "open", "high", "low", "close", "volume"]
        assert list(df.columns) == expected_columns

    def test_fetch_underlying_bars_calls_api_correctly(
        self, provider, mock_databento_response
    ):
        """Test that API is called with correct parameters."""
        mock_get_range = Mock(return_value=mock_databento_response)
        provider._client.timeseries.get_range = mock_get_range

        start = datetime(2023, 1, 1, 9, 30)
        end = datetime(2023, 1, 1, 10, 30)

        provider.fetch_underlying_bars("SPY", start, end, interval="1m")

        mock_get_range.assert_called_once_with(
            dataset="DBEQ.BASIC",
            schema="ohlcv-1m",
            symbols=["SPY"],
            stype_in="raw_symbol",
            start=start.isoformat(),
            end=end.isoformat(),
        )

    def test_fetch_underlying_bars_different_intervals(
        self, provider, mock_databento_response
    ):
        """Test fetching bars with different intervals."""
        provider._client.timeseries.get_range = Mock(
            return_value=mock_databento_response
        )

        start = datetime(2023, 1, 1, 9, 30)
        end = datetime(2023, 1, 1, 16, 0)

        intervals = {
            "1s": "ohlcv-1s",
            "1m": "ohlcv-1m",
            "1h": "ohlcv-1h",
            "1d": "ohlcv-1d",
        }

        for interval, schema in intervals.items():
            df = provider.fetch_underlying_bars("SPY", start, end, interval=interval)
            assert isinstance(df, pd.DataFrame)

            # Verify correct schema was used
            call_args = provider._client.timeseries.get_range.call_args
            assert call_args.kwargs["schema"] == schema

    def test_fetch_underlying_bars_invalid_interval_raises_error(self, provider):
        """Test that invalid interval raises ValueError."""
        start = datetime(2023, 1, 1, 9, 30)
        end = datetime(2023, 1, 1, 10, 30)

        with pytest.raises(ValueError, match="Unsupported interval"):
            provider.fetch_underlying_bars("SPY", start, end, interval="15s")

    def test_fetch_underlying_bars_empty_response(self, provider):
        """Test handling of empty API response."""
        mock_data = Mock()
        mock_data.to_df.return_value = pd.DataFrame()

        provider._client.timeseries.get_range = Mock(return_value=mock_data)

        start = datetime(2023, 1, 1, 9, 30)
        end = datetime(2023, 1, 1, 10, 30)

        df = provider.fetch_underlying_bars("SPY", start, end, interval="1m")

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
        assert list(df.columns) == ["ts_utc", "open", "high", "low", "close", "volume"]

    def test_fetch_underlying_bars_api_error_raises_runtime_error(self, provider):
        """Test that API errors are caught and re-raised as ProviderError."""
        provider._client.timeseries.get_range = Mock(
            side_effect=Exception("API connection failed")
        )

        start = datetime(2023, 1, 1, 9, 30)
        end = datetime(2023, 1, 1, 10, 30)

        with pytest.raises(ProviderError, match="Databento API error"):
            provider.fetch_underlying_bars("SPY", start, end, interval="1m")


class TestFetchOptionQuotes:
    """Test fetch_option_quotes method."""

    @pytest.fixture
    def provider(self):
        """Create provider with mocked client."""
        with patch("src.data.providers.databento.db.Historical"):
            return DatabentoProvider(api_key="test_key")

    @pytest.fixture
    def mock_databento_response(self):
        """Create mock Databento response for option quotes."""
        mock_data = Mock()
        mock_df = pd.DataFrame(
            {
                "ts_event": pd.date_range("2023-01-01 09:30", periods=5, freq="1min"),
                "symbol": ["SPY230120C00400000"] * 5,
                "bid_px": [5.0, 5.1, 5.05, 5.15, 5.2],
                "ask_px": [5.1, 5.2, 5.15, 5.25, 5.3],
                "bid_sz": [10, 15, 12, 13, 11],
                "ask_sz": [8, 12, 10, 11, 9],
            }
        )
        mock_data.to_df.return_value = mock_df
        return mock_data

    def test_fetch_option_quotes_returns_dataframe(
        self, provider, mock_databento_response
    ):
        """Test that fetch_option_quotes returns a DataFrame."""
        provider._client.timeseries.get_range = Mock(
            return_value=mock_databento_response
        )

        start = datetime(2023, 1, 1, 9, 30)
        end = datetime(2023, 1, 1, 10, 30)
        symbols = ["SPY230120C00400000"]

        df = provider.fetch_option_quotes(symbols, start, end)

        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_fetch_option_quotes_normalizes_columns(
        self, provider, mock_databento_response
    ):
        """Test that column names are normalized."""
        provider._client.timeseries.get_range = Mock(
            return_value=mock_databento_response
        )

        start = datetime(2023, 1, 1, 9, 30)
        end = datetime(2023, 1, 1, 10, 30)
        symbols = ["SPY230120C00400000"]

        df = provider.fetch_option_quotes(symbols, start, end)

        expected_columns = [
            "ts_utc",
            "option_symbol",
            "bid",
            "ask",
            "bid_size",
            "ask_size",
        ]
        assert list(df.columns) == expected_columns

    def test_fetch_option_quotes_calls_api_correctly(
        self, provider, mock_databento_response
    ):
        """Test that API is called with correct parameters."""
        mock_get_range = Mock(return_value=mock_databento_response)
        provider._client.timeseries.get_range = mock_get_range

        start = datetime(2023, 1, 1, 9, 30)
        end = datetime(2023, 1, 1, 10, 30)
        symbols = ["SPY230120C00400000", "SPY230120P00400000"]

        provider.fetch_option_quotes(symbols, start, end, schema="cbbo-1m")

        mock_get_range.assert_called_once_with(
            dataset="OPRA.PILLAR",
            schema="cbbo-1m",
            symbols=symbols,
            stype_in="raw_symbol",
            start=start.isoformat(),
            end=end.isoformat(),
        )

    def test_fetch_option_quotes_empty_symbols_returns_empty_df(self, provider):
        """Test that empty symbols list returns empty DataFrame."""
        start = datetime(2023, 1, 1, 9, 30)
        end = datetime(2023, 1, 1, 10, 30)

        df = provider.fetch_option_quotes([], start, end)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_fetch_option_quotes_api_error_raises_runtime_error(self, provider):
        """Test that API errors are caught and re-raised as ProviderError."""
        provider._client.timeseries.get_range = Mock(
            side_effect=Exception("API connection failed")
        )

        start = datetime(2023, 1, 1, 9, 30)
        end = datetime(2023, 1, 1, 10, 30)
        symbols = ["SPY230120C00400000"]

        with pytest.raises(ProviderError, match="Databento API error"):
            provider.fetch_option_quotes(symbols, start, end)


class TestEstimateCost:
    """Test estimate_cost method."""

    @pytest.fixture
    def provider(self):
        """Create provider with mocked client."""
        with patch("src.data.providers.databento.db.Historical"):
            return DatabentoProvider(api_key="test_key")

    def test_estimate_cost_returns_float(self, provider):
        """Test that estimate_cost returns a float."""
        # metadata.get_cost returns float directly
        provider._client.metadata.get_cost = Mock(return_value=123.45)

        start = datetime(2023, 1, 1, 9, 30)
        end = datetime(2023, 1, 1, 10, 30)

        cost = provider.estimate_cost(
            "GLBX.MDP3", "ohlcv-1m", ["SPY"], start, end
        )

        assert isinstance(cost, float)
        assert cost == 123.45

    def test_estimate_cost_calls_api_correctly(self, provider):
        """Test that cost API is called with correct parameters."""
        # metadata.get_cost returns float directly
        mock_get_cost = Mock(return_value=100.0)
        provider._client.metadata.get_cost = mock_get_cost

        start = datetime(2023, 1, 1, 9, 30)
        end = datetime(2023, 1, 1, 10, 30)
        symbols = ["SPY"]

        provider.estimate_cost("GLBX.MDP3", "ohlcv-1m", symbols, start, end)

        mock_get_cost.assert_called_once_with(
            dataset="GLBX.MDP3",
            schema="ohlcv-1m",
            symbols=symbols,
            stype_in="raw_symbol",
            start=start.isoformat(),
            end=end.isoformat(),
        )

    def test_estimate_cost_api_error_raises_runtime_error(self, provider):
        """Test that API errors are caught and re-raised as ProviderError."""
        provider._client.metadata.get_cost = Mock(
            side_effect=Exception("API connection failed")
        )

        start = datetime(2023, 1, 1, 9, 30)
        end = datetime(2023, 1, 1, 10, 30)

        with pytest.raises(ProviderError, match="Databento API error"):
            provider.estimate_cost("GLBX.MDP3", "ohlcv-1m", ["SPY"], start, end)


class TestResolveOptionSymbols:
    """Test resolve_option_symbols method."""

    @pytest.fixture
    def provider(self):
        """Create provider with mocked client."""
        with patch("src.data.providers.databento.db.Historical"):
            return DatabentoProvider(api_key="test_key")

    def test_resolve_option_symbols_returns_list(self, provider):
        """Test that resolve_option_symbols returns a list of symbols."""
        as_of = datetime(2023, 1, 1, 9, 30)

        # Mock the API response - method returns unfiltered symbols
        # Filtering is done by ManifestGenerator
        provider._client.timeseries.get_range.return_value.to_df.return_value = pd.DataFrame({
            "raw_symbol": ["SPY230120C00400000", "SPY230120P00390000"]
        })

        result = provider.resolve_option_symbols(parent="SPY", as_of=as_of)

        assert isinstance(result, list)
        assert len(result) == 2
        assert "SPY230120C00400000" in result


class TestNormalization:
    """Test data normalization methods."""

    @pytest.fixture
    def provider(self):
        """Create provider with mocked client."""
        with patch("src.data.providers.databento.db.Historical"):
            return DatabentoProvider(api_key="test_key")

    def test_normalize_bars(self, provider):
        """Test _normalize_bars method."""
        raw_df = pd.DataFrame(
            {
                "ts_event": pd.date_range("2023-01-01", periods=3, freq="1min"),
                "open": [400.0, 401.0, 402.0],
                "high": [401.0, 402.0, 403.0],
                "low": [399.0, 400.0, 401.0],
                "close": [400.5, 401.5, 402.5],
                "volume": [1000, 1500, 1200],
            }
        )

        normalized = provider._normalize_bars(raw_df)

        assert "ts_utc" in normalized.columns
        assert "ts_event" not in normalized.columns
        assert list(normalized.columns) == [
            "ts_utc",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]

    def test_normalize_quotes(self, provider):
        """Test _normalize_quotes method."""
        raw_df = pd.DataFrame(
            {
                "ts_event": pd.date_range("2023-01-01", periods=3, freq="1min"),
                "symbol": ["SPY230120C00400000"] * 3,
                "bid_px": [5.0, 5.1, 5.05],
                "ask_px": [5.1, 5.2, 5.15],
                "bid_sz": [10, 15, 12],
                "ask_sz": [8, 12, 10],
            }
        )

        normalized = provider._normalize_quotes(raw_df)

        assert "ts_utc" in normalized.columns
        assert "option_symbol" in normalized.columns
        assert "bid" in normalized.columns
        assert "ask" in normalized.columns
        assert "ts_event" not in normalized.columns
        assert "symbol" not in normalized.columns
        assert "bid_px" not in normalized.columns
