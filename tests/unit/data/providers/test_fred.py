"""Unit tests for FRED provider."""

from datetime import datetime
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from src.data.providers.fred import FREDProvider
from src.data.providers.protocol import MacroDataProvider
from src.exceptions import ConfigError, ProviderError


class TestFREDProviderInitialization:
    """Test FREDProvider initialization."""

    def test_initialization_with_api_key(self):
        """Test provider initialization with explicit API key."""
        provider = FREDProvider(api_key="test_key")

        assert provider._api_key == "test_key"
        assert provider._base_url == "https://api.stlouisfed.org/fred"
        assert provider._session is not None

    @patch.dict("os.environ", {"FRED_API_KEY": "env_key"})
    def test_initialization_from_environment(self):
        """Test provider initialization from environment variable."""
        provider = FREDProvider()

        assert provider._api_key == "env_key"

    @patch.dict("os.environ", {}, clear=True)
    def test_initialization_without_key_raises_error(self):
        """Test that missing API key raises ConfigError."""
        with pytest.raises(ConfigError, match="FRED API key required"):
            FREDProvider()

    def test_initialization_with_custom_base_url(self):
        """Test provider initialization with custom base URL."""
        provider = FREDProvider(api_key="test_key", base_url="https://custom.fred.api")

        assert provider._base_url == "https://custom.fred.api"

    def test_provider_satisfies_protocol(self):
        """Test that FREDProvider satisfies MacroDataProvider protocol."""
        provider = FREDProvider(api_key="test_key")
        assert isinstance(provider, MacroDataProvider)


class TestFetchSeries:
    """Test fetch_series method."""

    @pytest.fixture
    def provider(self):
        """Create provider for testing."""
        return FREDProvider(api_key="test_key")

    @pytest.fixture
    def mock_fred_response(self):
        """Create mock FRED API response."""
        return {
            "observations": [
                {
                    "date": "2023-01-01",
                    "value": "4.50",
                    "realtime_start": "2023-01-02",
                },
                {
                    "date": "2023-01-02",
                    "value": "4.55",
                    "realtime_start": "2023-01-03",
                },
                {
                    "date": "2023-01-03",
                    "value": "4.60",
                    "realtime_start": "2023-01-04",
                },
            ]
        }

    def test_fetch_series_returns_dataframe(self, provider, mock_fred_response):
        """Test that fetch_series returns a DataFrame."""
        with patch.object(provider._session, "get") as mock_get:
            mock_response = Mock()
            mock_response.json.return_value = mock_fred_response
            mock_response.raise_for_status = Mock()
            mock_get.return_value = mock_response

            start = datetime(2023, 1, 1)
            end = datetime(2023, 1, 10)

            df = provider.fetch_series("DGS10", start, end)

            assert isinstance(df, pd.DataFrame)
            assert len(df) > 0

    def test_fetch_series_has_correct_columns(self, provider, mock_fred_response):
        """Test that returned DataFrame has correct columns."""
        with patch.object(provider._session, "get") as mock_get:
            mock_response = Mock()
            mock_response.json.return_value = mock_fred_response
            mock_response.raise_for_status = Mock()
            mock_get.return_value = mock_response

            start = datetime(2023, 1, 1)
            end = datetime(2023, 1, 10)

            df = provider.fetch_series("DGS10", start, end)

            expected_columns = ["obs_date", "value", "release_datetime_utc"]
            assert list(df.columns) == expected_columns

    def test_fetch_series_converts_percent_to_decimal(self, provider):
        """Test that percent values are converted to decimal."""
        mock_response_data = {
            "observations": [
                {"date": "2023-01-01", "value": "4.50", "realtime_start": "2023-01-02"},
            ]
        }

        with patch.object(provider._session, "get") as mock_get:
            mock_response = Mock()
            mock_response.json.return_value = mock_response_data
            mock_response.raise_for_status = Mock()
            mock_get.return_value = mock_response

            start = datetime(2023, 1, 1)
            end = datetime(2023, 1, 10)

            df = provider.fetch_series("DGS10", start, end)

            # 4.50% should be converted to 0.045
            assert df["value"].iloc[0] == pytest.approx(0.045, rel=1e-5)

    def test_fetch_series_skips_missing_values(self, provider):
        """Test that missing values (.) are skipped."""
        mock_response_data = {
            "observations": [
                {"date": "2023-01-01", "value": "4.50", "realtime_start": "2023-01-02"},
                {"date": "2023-01-02", "value": ".", "realtime_start": "2023-01-03"},
                {"date": "2023-01-03", "value": "4.60", "realtime_start": "2023-01-04"},
            ]
        }

        with patch.object(provider._session, "get") as mock_get:
            mock_response = Mock()
            mock_response.json.return_value = mock_response_data
            mock_response.raise_for_status = Mock()
            mock_get.return_value = mock_response

            start = datetime(2023, 1, 1)
            end = datetime(2023, 1, 10)

            df = provider.fetch_series("DGS10", start, end)

            # Should have 2 rows, not 3 (missing value skipped)
            assert len(df) == 2

    def test_fetch_series_calls_api_correctly(self, provider, mock_fred_response):
        """Test that API is called with correct parameters."""
        with patch.object(provider._session, "get") as mock_get:
            mock_response = Mock()
            mock_response.json.return_value = mock_fred_response
            mock_response.raise_for_status = Mock()
            mock_get.return_value = mock_response

            start = datetime(2023, 1, 1, 9, 30)
            end = datetime(2023, 1, 10, 16, 0)

            provider.fetch_series("DGS10", start, end)

            mock_get.assert_called_once()
            call_args = mock_get.call_args

            assert call_args.args[0] == "https://api.stlouisfed.org/fred/series/observations"
            assert call_args.kwargs["params"]["series_id"] == "DGS10"
            assert call_args.kwargs["params"]["observation_start"] == "2023-01-01"
            # end is exclusive in our API; FRED's observation_end is inclusive,
            # so we subtract 1 day.
            assert call_args.kwargs["params"]["observation_end"] == "2023-01-09"
            assert call_args.kwargs["params"]["file_type"] == "json"

    def test_fetch_series_empty_series_id_raises_error(self, provider):
        """Test that empty series_id raises ValueError."""
        start = datetime(2023, 1, 1)
        end = datetime(2023, 1, 10)

        with pytest.raises(ValueError, match="Series ID cannot be empty"):
            provider.fetch_series("", start, end)

    def test_fetch_series_api_error_raises_provider_error(self, provider):
        """Test that API errors are caught and re-raised as ProviderError."""
        with patch.object(provider._session, "get") as mock_get:
            mock_get.side_effect = Exception("API connection failed")

            start = datetime(2023, 1, 1)
            end = datetime(2023, 1, 10)

            with pytest.raises(ProviderError, match="FRED API error"):
                provider.fetch_series("DGS10", start, end)

    def test_fetch_series_fred_error_message_raises_value_error(self, provider):
        """Test that FRED API error messages raise ValueError."""
        mock_response_data = {"error_message": "Series not found"}

        with patch.object(provider._session, "get") as mock_get:
            mock_response = Mock()
            mock_response.json.return_value = mock_response_data
            mock_response.raise_for_status = Mock()
            mock_get.return_value = mock_response

            start = datetime(2023, 1, 1)
            end = datetime(2023, 1, 10)

            with pytest.raises(ValueError, match="FRED API error: Series not found"):
                provider.fetch_series("INVALID", start, end)

    def test_fetch_series_empty_response_returns_empty_df(self, provider):
        """Test that empty API response returns empty DataFrame."""
        mock_response_data = {"observations": []}

        with patch.object(provider._session, "get") as mock_get:
            mock_response = Mock()
            mock_response.json.return_value = mock_response_data
            mock_response.raise_for_status = Mock()
            mock_get.return_value = mock_response

            start = datetime(2023, 1, 1)
            end = datetime(2023, 1, 10)

            df = provider.fetch_series("DGS10", start, end)

            assert isinstance(df, pd.DataFrame)
            assert len(df) == 0
            assert list(df.columns) == ["obs_date", "value", "release_datetime_utc"]

    def test_fetch_series_vix_not_converted(self, provider):
        """Test that VIX values are not converted (already in correct form)."""
        mock_response_data = {
            "observations": [
                {"date": "2023-01-01", "value": "20.50", "realtime_start": "2023-01-02"},
            ]
        }

        with patch.object(provider._session, "get") as mock_get:
            mock_response = Mock()
            mock_response.json.return_value = mock_response_data
            mock_response.raise_for_status = Mock()
            mock_get.return_value = mock_response

            start = datetime(2023, 1, 1)
            end = datetime(2023, 1, 10)

            df = provider.fetch_series("VIXCLS", start, end)

            # VIX value should NOT be converted (20.50 stays as 20.50, not 0.205)
            assert df["value"].iloc[0] == pytest.approx(20.50, rel=1e-5)


class TestGetLatestValue:
    """Test get_latest_value method."""

    @pytest.fixture
    def provider(self):
        """Create provider for testing."""
        return FREDProvider(api_key="test_key")

    def test_get_latest_value_returns_tuple(self, provider):
        """Test that get_latest_value returns a tuple."""
        mock_df = pd.DataFrame(
            {
                "obs_date": [pd.to_datetime("2023-01-01").date()],
                "value": [0.045],
                "release_datetime_utc": [pd.to_datetime("2023-01-02")],
            }
        )

        with patch.object(provider, "fetch_series", return_value=mock_df):
            as_of = datetime(2023, 1, 15, 10, 0)
            result = provider.get_latest_value("DGS10", as_of)

            assert isinstance(result, tuple)
            assert len(result) == 2

    def test_get_latest_value_correct_types(self, provider):
        """Test that get_latest_value returns correct types."""
        mock_df = pd.DataFrame(
            {
                "obs_date": [pd.to_datetime("2023-01-01").date()],
                "value": [0.045],
                "release_datetime_utc": [pd.to_datetime("2023-01-02")],
            }
        )

        with patch.object(provider, "fetch_series", return_value=mock_df):
            as_of = datetime(2023, 1, 15, 10, 0)
            release_dt, value = provider.get_latest_value("DGS10", as_of)

            assert isinstance(release_dt, pd.Timestamp)
            assert isinstance(value, float)

    def test_get_latest_value_before_as_of(self, provider):
        """Test that returned value is released before as_of."""
        mock_df = pd.DataFrame(
            {
                "obs_date": [
                    pd.to_datetime("2023-01-01").date(),
                    pd.to_datetime("2023-01-02").date(),
                ],
                "value": [0.045, 0.046],
                "release_datetime_utc": [
                    pd.to_datetime("2023-01-02"),
                    pd.to_datetime("2023-01-03"),
                ],
            }
        )

        with patch.object(provider, "fetch_series", return_value=mock_df):
            as_of = datetime(2023, 1, 2, 12, 0)
            release_dt, value = provider.get_latest_value("DGS10", as_of)

            assert release_dt <= as_of

    def test_get_latest_value_returns_most_recent(self, provider):
        """Test that get_latest_value returns the most recent value."""
        mock_df = pd.DataFrame(
            {
                "obs_date": [
                    pd.to_datetime("2023-01-01").date(),
                    pd.to_datetime("2023-01-02").date(),
                    pd.to_datetime("2023-01-03").date(),
                ],
                "value": [0.045, 0.046, 0.047],
                "release_datetime_utc": [
                    pd.to_datetime("2023-01-02"),
                    pd.to_datetime("2023-01-03"),
                    pd.to_datetime("2023-01-04"),
                ],
            }
        )

        with patch.object(provider, "fetch_series", return_value=mock_df):
            as_of = datetime(2023, 1, 5, 10, 0)
            release_dt, value = provider.get_latest_value("DGS10", as_of)

            # Should return the last value (0.047)
            assert value == pytest.approx(0.047, rel=1e-5)

    def test_get_latest_value_empty_series_id_raises_error(self, provider):
        """Test that empty series_id raises ValueError."""
        as_of = datetime(2023, 1, 15, 10, 0)

        with pytest.raises(ValueError, match="Series ID cannot be empty"):
            provider.get_latest_value("", as_of)

    def test_get_latest_value_no_data_raises_error(self, provider):
        """Test that no data before as_of raises ValueError."""
        # Return empty DataFrame
        mock_df = pd.DataFrame(columns=["obs_date", "value", "release_datetime_utc"])

        with patch.object(provider, "fetch_series", return_value=mock_df):
            as_of = datetime(2023, 1, 15, 10, 0)

            with pytest.raises(ValueError, match="No data available"):
                provider.get_latest_value("DGS10", as_of)

    def test_get_latest_value_filters_future_releases(self, provider):
        """Test that future releases are filtered out."""
        mock_df = pd.DataFrame(
            {
                "obs_date": [
                    pd.to_datetime("2023-01-01").date(),
                    pd.to_datetime("2023-01-02").date(),
                ],
                "value": [0.045, 0.046],
                "release_datetime_utc": [
                    pd.to_datetime("2023-01-02"),
                    pd.to_datetime("2023-01-10"),  # Future release
                ],
            }
        )

        with patch.object(provider, "fetch_series", return_value=mock_df):
            as_of = datetime(2023, 1, 5, 10, 0)
            release_dt, value = provider.get_latest_value("DGS10", as_of)

            # Should only return the first value (0.045), not the future one
            assert value == pytest.approx(0.045, rel=1e-5)


class TestIsPercentSeries:
    """Test _is_percent_series helper method."""

    @pytest.fixture
    def provider(self):
        """Create provider for testing."""
        return FREDProvider(api_key="test_key")

    def test_treasury_rates_are_percent(self, provider):
        """Test that treasury rate series are identified as percent."""
        assert provider._is_percent_series("DGS10") is True
        assert provider._is_percent_series("DGS2") is True
        assert provider._is_percent_series("DGS30") is True

    def test_fed_funds_rate_is_percent(self, provider):
        """Test that fed funds rate is identified as percent."""
        assert provider._is_percent_series("DFF") is True
        assert provider._is_percent_series("EFFR") is True

    def test_vix_is_not_percent(self, provider):
        """Test that VIX is not identified as percent."""
        assert provider._is_percent_series("VIXCLS") is False
        assert provider._is_percent_series("VIX") is False

    def test_unknown_series_is_not_percent(self, provider):
        """Test that unknown series default to not percent."""
        assert provider._is_percent_series("UNKNOWN123") is False
