"""Unit tests for live/symbols.py - Symbol discovery providers."""

import json
import tempfile
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.live.symbols import (
    DatabentoSymbolProvider,
    ManifestSymbolProvider,
    MockSymbolProvider,
    save_symbols_manifest,
)


class TestDatabentoSymbolProviderInit:
    """Tests for DatabentoSymbolProvider initialization."""

    def test_init_with_api_key(self):
        """Should initialize with provided API key."""
        with patch("src.live.symbols.db.Historical") as mock_client:
            provider = DatabentoSymbolProvider(api_key="test-key")
            mock_client.assert_called_once_with(key="test-key")

    def test_init_from_env(self):
        """Should load API key from environment variable."""
        with patch("src.live.symbols.db.Historical") as mock_client:
            with patch.dict("os.environ", {"DATABENTO_API_KEY": "env-key"}):
                provider = DatabentoSymbolProvider()
                mock_client.assert_called_once_with(key="env-key")

    def test_init_missing_api_key(self):
        """Should raise ValueError if no API key available."""
        with patch.dict("os.environ", {}, clear=True):
            # Remove DATABENTO_API_KEY if it exists
            with patch("os.getenv", return_value=None):
                with pytest.raises(ValueError, match="API key required"):
                    DatabentoSymbolProvider()

    def test_init_with_underlying_price(self):
        """Should accept optional underlying price."""
        with patch("src.live.symbols.db.Historical"):
            provider = DatabentoSymbolProvider(
                api_key="test-key", underlying_price=500.0
            )
            assert provider._underlying_price == 500.0


class TestDatabentoSymbolProviderOCCParser:
    """Tests for OCC symbol parsing."""

    def test_parse_valid_call_symbol(self):
        """Should parse valid call option symbol."""
        with patch("src.live.symbols.db.Historical"):
            provider = DatabentoSymbolProvider(api_key="test-key")
            result = provider._parse_occ_symbol("SPY240315C00450000")

            assert result is not None
            expiry, strike, right = result
            assert expiry == date(2024, 3, 15)
            assert strike == 450.0
            assert right == "C"

    def test_parse_valid_put_symbol(self):
        """Should parse valid put option symbol."""
        with patch("src.live.symbols.db.Historical"):
            provider = DatabentoSymbolProvider(api_key="test-key")
            result = provider._parse_occ_symbol("SPY240315P00440000")

            assert result is not None
            expiry, strike, right = result
            assert expiry == date(2024, 3, 15)
            assert strike == 440.0
            assert right == "P"

    def test_parse_decimal_strike(self):
        """Should handle decimal strikes correctly."""
        with patch("src.live.symbols.db.Historical"):
            provider = DatabentoSymbolProvider(api_key="test-key")
            # Strike of $450.50 = 450500
            result = provider._parse_occ_symbol("SPY240315C00450500")

            assert result is not None
            _, strike, _ = result
            assert strike == 450.5

    def test_parse_short_root(self):
        """Should parse symbols with short root symbols."""
        with patch("src.live.symbols.db.Historical"):
            provider = DatabentoSymbolProvider(api_key="test-key")
            result = provider._parse_occ_symbol("A240315C00100000")

            assert result is not None
            expiry, strike, _ = result
            assert expiry == date(2024, 3, 15)
            assert strike == 100.0

    def test_parse_long_root(self):
        """Should parse symbols with 6-character root."""
        with patch("src.live.symbols.db.Historical"):
            provider = DatabentoSymbolProvider(api_key="test-key")
            result = provider._parse_occ_symbol("GOOGL240315C00150000")

            assert result is not None
            expiry, strike, _ = result
            assert expiry == date(2024, 3, 15)
            assert strike == 150.0

    def test_parse_invalid_format(self):
        """Should return None for invalid format."""
        with patch("src.live.symbols.db.Historical"):
            provider = DatabentoSymbolProvider(api_key="test-key")
            assert provider._parse_occ_symbol("INVALID") is None
            assert provider._parse_occ_symbol("SPY") is None
            assert provider._parse_occ_symbol("") is None
            assert provider._parse_occ_symbol("SPY240315X00450000") is None  # Invalid right

    def test_parse_invalid_date(self):
        """Should return None for invalid date."""
        with patch("src.live.symbols.db.Historical"):
            provider = DatabentoSymbolProvider(api_key="test-key")
            # Invalid month 13
            assert provider._parse_occ_symbol("SPY241315C00450000") is None


class TestDatabentoSymbolProviderAPIFlow:
    """Tests for DatabentoSymbolProvider full API flow."""

    def _make_symbol(self, expiry: date, strike: float, right: str = "C") -> str:
        """Helper to create OCC symbol."""
        date_str = expiry.strftime("%y%m%d")
        strike_str = f"{int(strike * 1000):08d}"
        return f"SPY{date_str}{right}{strike_str}"

    def test_get_option_symbols_full_flow(self):
        """Should fetch and filter symbols through full API flow."""
        today = date.today()
        valid_expiry = today + timedelta(days=30)

        # Create mock definitions DataFrame
        mock_definitions = pd.DataFrame({
            "raw_symbol": [
                self._make_symbol(valid_expiry, 475),
                self._make_symbol(valid_expiry, 500),
                self._make_symbol(valid_expiry, 525),
            ]
        })

        # Mock the API client
        mock_result = MagicMock()
        mock_result.to_df.return_value = mock_definitions

        with patch("src.live.symbols.db.Historical") as mock_client_class:
            mock_client = MagicMock()
            mock_client.timeseries.get_range.return_value = mock_result
            mock_client_class.return_value = mock_client

            provider = DatabentoSymbolProvider(api_key="test-key", underlying_price=500.0)
            symbols = provider.get_option_symbols(
                underlying="SPY",
                min_dte=7,
                max_dte=90,
                moneyness_range=(0.9, 1.1),
            )

            assert len(symbols) == 3
            assert all("SPY" in s for s in symbols)

    def test_get_option_symbols_empty_definitions(self):
        """Should return empty list when no definitions found."""
        mock_result = MagicMock()
        mock_result.to_df.return_value = pd.DataFrame()

        with patch("src.live.symbols.db.Historical") as mock_client_class:
            mock_client = MagicMock()
            mock_client.timeseries.get_range.return_value = mock_result
            mock_client_class.return_value = mock_client

            provider = DatabentoSymbolProvider(api_key="test-key", underlying_price=500.0)
            symbols = provider.get_option_symbols("SPY")

            assert symbols == []

    def test_get_option_symbols_fetches_underlying_price(self):
        """Should fetch underlying price from API when not provided."""
        today = date.today()
        valid_expiry = today + timedelta(days=30)

        # Mock definitions
        mock_definitions = pd.DataFrame({
            "raw_symbol": [self._make_symbol(valid_expiry, 500)]
        })

        # Mock price data
        mock_price_df = pd.DataFrame({"close": [500.0]})

        mock_def_result = MagicMock()
        mock_def_result.to_df.return_value = mock_definitions

        mock_price_result = MagicMock()
        mock_price_result.to_df.return_value = mock_price_df

        with patch("src.live.symbols.db.Historical") as mock_client_class:
            mock_client = MagicMock()
            # First call is for definitions, second for price
            mock_client.timeseries.get_range.side_effect = [
                mock_def_result,
                mock_price_result,
            ]
            mock_client_class.return_value = mock_client

            provider = DatabentoSymbolProvider(api_key="test-key")
            symbols = provider.get_option_symbols("SPY", min_dte=7, max_dte=90)

            # Should have called API twice
            assert mock_client.timeseries.get_range.call_count == 2
            assert len(symbols) == 1

    def test_get_option_symbols_price_fetch_fails(self):
        """Should skip moneyness filter when price fetch fails."""
        today = date.today()
        valid_expiry = today + timedelta(days=30)

        # All strikes should pass without moneyness filter
        mock_definitions = pd.DataFrame({
            "raw_symbol": [
                self._make_symbol(valid_expiry, 300),  # Would fail 0.9-1.1 moneyness
                self._make_symbol(valid_expiry, 500),
                self._make_symbol(valid_expiry, 700),  # Would fail 0.9-1.1 moneyness
            ]
        })

        mock_def_result = MagicMock()
        mock_def_result.to_df.return_value = mock_definitions

        with patch("src.live.symbols.db.Historical") as mock_client_class:
            mock_client = MagicMock()
            # First call succeeds (definitions), second fails (price)
            mock_client.timeseries.get_range.side_effect = [
                mock_def_result,
                Exception("API error"),
            ]
            mock_client_class.return_value = mock_client

            provider = DatabentoSymbolProvider(api_key="test-key")
            symbols = provider.get_option_symbols(
                "SPY", min_dte=7, max_dte=90, moneyness_range=(0.9, 1.1)
            )

            # All symbols should pass since moneyness filter is skipped
            assert len(symbols) == 3

    def test_get_option_symbols_api_error_propagates(self):
        """Should propagate API errors from definitions fetch."""
        with patch("src.live.symbols.db.Historical") as mock_client_class:
            mock_client = MagicMock()
            mock_client.timeseries.get_range.side_effect = Exception("API connection failed")
            mock_client_class.return_value = mock_client

            provider = DatabentoSymbolProvider(api_key="test-key", underlying_price=500.0)

            with pytest.raises(Exception, match="API connection failed"):
                provider.get_option_symbols("SPY")

    def test_fetch_definitions_calls_api(self):
        """Should call Databento API with correct parameters."""
        mock_result = MagicMock()
        mock_result.to_df.return_value = pd.DataFrame()

        with patch("src.live.symbols.db.Historical") as mock_client_class:
            mock_client = MagicMock()
            mock_client.timeseries.get_range.return_value = mock_result
            mock_client_class.return_value = mock_client

            provider = DatabentoSymbolProvider(api_key="test-key")
            provider._fetch_definitions("SPY")

            # Verify API called with correct params
            call_kwargs = mock_client.timeseries.get_range.call_args.kwargs
            assert call_kwargs["dataset"] == "OPRA.PILLAR"
            assert call_kwargs["schema"] == "definition"
            assert call_kwargs["stype_in"] == "parent"
            assert "SPY.OPT" in call_kwargs["symbols"]

    def test_get_underlying_price_success(self):
        """Should extract price from API response."""
        mock_price_df = pd.DataFrame({"close": [505.50]})
        mock_result = MagicMock()
        mock_result.to_df.return_value = mock_price_df

        with patch("src.live.symbols.db.Historical") as mock_client_class:
            mock_client = MagicMock()
            mock_client.timeseries.get_range.return_value = mock_result
            mock_client_class.return_value = mock_client

            provider = DatabentoSymbolProvider(api_key="test-key")
            price = provider._get_underlying_price("SPY")

            assert price == 505.50

    def test_get_underlying_price_empty_response(self):
        """Should return None when no price data available."""
        mock_result = MagicMock()
        mock_result.to_df.return_value = pd.DataFrame()

        with patch("src.live.symbols.db.Historical") as mock_client_class:
            mock_client = MagicMock()
            mock_client.timeseries.get_range.return_value = mock_result
            mock_client_class.return_value = mock_client

            provider = DatabentoSymbolProvider(api_key="test-key")
            price = provider._get_underlying_price("SPY")

            assert price is None

    def test_get_underlying_price_api_error(self):
        """Should return None on API error."""
        with patch("src.live.symbols.db.Historical") as mock_client_class:
            mock_client = MagicMock()
            mock_client.timeseries.get_range.side_effect = Exception("API error")
            mock_client_class.return_value = mock_client

            provider = DatabentoSymbolProvider(api_key="test-key")
            price = provider._get_underlying_price("SPY")

            assert price is None


class TestDatabentoSymbolProviderFilteringEdgeCases:
    """Tests for edge cases in definition filtering."""

    def _make_symbol(self, expiry: date, strike: float, right: str = "C") -> str:
        """Helper to create OCC symbol."""
        date_str = expiry.strftime("%y%m%d")
        strike_str = f"{int(strike * 1000):08d}"
        return f"SPY{date_str}{right}{strike_str}"

    @pytest.fixture
    def provider(self):
        """Create provider with mocked client."""
        with patch("src.live.symbols.db.Historical"):
            return DatabentoSymbolProvider(api_key="test-key", underlying_price=500.0)

    def test_filter_with_symbol_column(self, provider):
        """Should handle 'symbol' column name."""
        today = date.today()
        expiry = today + timedelta(days=30)

        definitions = pd.DataFrame({
            "symbol": [self._make_symbol(expiry, 500)]
        })

        filtered = provider._filter_definitions(
            definitions, "SPY", 7, 90, (0.9, 1.1), 500.0
        )
        assert len(filtered) == 1

    def test_filter_with_index_symbol(self, provider):
        """Should handle symbol as index."""
        today = date.today()
        expiry = today + timedelta(days=30)

        definitions = pd.DataFrame({
            "other_col": [1]
        }, index=pd.Index([self._make_symbol(expiry, 500)], name="symbol"))

        filtered = provider._filter_definitions(
            definitions, "SPY", 7, 90, (0.9, 1.1), 500.0
        )
        assert len(filtered) == 1

    def test_filter_skips_na_symbols(self, provider):
        """Should skip NA symbol values."""
        today = date.today()
        expiry = today + timedelta(days=30)

        definitions = pd.DataFrame({
            "raw_symbol": [
                self._make_symbol(expiry, 500),
                None,  # NA value
                self._make_symbol(expiry, 510),
            ]
        })

        filtered = provider._filter_definitions(
            definitions, "SPY", 7, 90, (0.9, 1.1), 500.0
        )
        assert len(filtered) == 2

    def test_filter_skips_unparseable_symbols(self, provider):
        """Should skip symbols that can't be parsed."""
        today = date.today()
        expiry = today + timedelta(days=30)

        definitions = pd.DataFrame({
            "raw_symbol": [
                self._make_symbol(expiry, 500),
                "INVALID_SYMBOL",
                self._make_symbol(expiry, 510),
            ]
        })

        filtered = provider._filter_definitions(
            definitions, "SPY", 7, 90, (0.9, 1.1), 500.0
        )
        assert len(filtered) == 2


class TestDatabentoSymbolProviderFiltering:
    """Tests for definition filtering logic."""

    @pytest.fixture
    def provider(self):
        """Create provider with mocked client."""
        with patch("src.live.symbols.db.Historical"):
            return DatabentoSymbolProvider(api_key="test-key", underlying_price=500.0)

    def test_filter_by_dte(self, provider):
        """Should filter by DTE range."""
        today = date.today()

        # Create definitions with various DTEs
        definitions = pd.DataFrame(
            {
                "raw_symbol": [
                    self._make_symbol(today + timedelta(days=5), 500),  # DTE 5
                    self._make_symbol(today + timedelta(days=14), 500),  # DTE 14
                    self._make_symbol(today + timedelta(days=30), 500),  # DTE 30
                    self._make_symbol(today + timedelta(days=100), 500),  # DTE 100
                ]
            }
        )

        filtered = provider._filter_definitions(
            definitions,
            underlying="SPY",
            min_dte=7,
            max_dte=90,
            moneyness_range=(0.0, float("inf")),
            spot_price=500.0,
        )

        # Should include DTE 14 and 30, exclude 5 and 100
        assert len(filtered) == 2

    def test_filter_by_moneyness(self, provider):
        """Should filter by moneyness range."""
        today = date.today()
        expiry = today + timedelta(days=30)

        # Create definitions with various strikes
        definitions = pd.DataFrame(
            {
                "raw_symbol": [
                    self._make_symbol(expiry, 400),  # 0.8 moneyness
                    self._make_symbol(expiry, 475),  # 0.95 moneyness
                    self._make_symbol(expiry, 500),  # 1.0 moneyness
                    self._make_symbol(expiry, 525),  # 1.05 moneyness
                    self._make_symbol(expiry, 600),  # 1.2 moneyness
                ]
            }
        )

        filtered = provider._filter_definitions(
            definitions,
            underlying="SPY",
            min_dte=0,
            max_dte=90,
            moneyness_range=(0.9, 1.1),
            spot_price=500.0,
        )

        # Should include 475, 500, 525 (0.95, 1.0, 1.05)
        assert len(filtered) == 3

    def test_filter_combined(self, provider):
        """Should apply both DTE and moneyness filters."""
        today = date.today()

        definitions = pd.DataFrame(
            {
                "raw_symbol": [
                    # DTE 5 (too early)
                    self._make_symbol(today + timedelta(days=5), 500),
                    # DTE 14, moneyness 1.0 (valid)
                    self._make_symbol(today + timedelta(days=14), 500),
                    # DTE 30, moneyness 0.6 (too OTM)
                    self._make_symbol(today + timedelta(days=30), 300),
                    # DTE 100 (too late)
                    self._make_symbol(today + timedelta(days=100), 500),
                ]
            }
        )

        filtered = provider._filter_definitions(
            definitions,
            underlying="SPY",
            min_dte=7,
            max_dte=90,
            moneyness_range=(0.9, 1.1),
            spot_price=500.0,
        )

        # Only the DTE 14, moneyness 1.0 option should pass
        assert len(filtered) == 1

    def _make_symbol(self, expiry: date, strike: float, right: str = "C") -> str:
        """Helper to create OCC symbol."""
        date_str = expiry.strftime("%y%m%d")
        strike_str = f"{int(strike * 1000):08d}"
        return f"SPY{date_str}{right}{strike_str}"


class TestManifestSymbolProviderInit:
    """Tests for ManifestSymbolProvider initialization."""

    def test_init_with_list_format(self, tmp_path):
        """Should load manifest with list format."""
        manifest_path = tmp_path / "symbols.json"
        symbols = ["SPY240315C00450000", "SPY240315P00440000"]
        manifest_path.write_text(json.dumps(symbols))

        provider = ManifestSymbolProvider(manifest_path)
        assert provider._symbols == symbols
        assert provider._underlying is None

    def test_init_with_dict_format(self, tmp_path):
        """Should load manifest with dict format."""
        manifest_path = tmp_path / "symbols.json"
        manifest = {
            "symbols": ["SPY240315C00450000", "SPY240315P00440000"],
            "underlying": "SPY",
        }
        manifest_path.write_text(json.dumps(manifest))

        provider = ManifestSymbolProvider(manifest_path)
        assert provider._symbols == manifest["symbols"]
        assert provider._underlying == "SPY"

    def test_init_missing_file(self):
        """Should raise FileNotFoundError for missing manifest."""
        with pytest.raises(FileNotFoundError, match="Manifest file not found"):
            ManifestSymbolProvider("/nonexistent/path.json")

    def test_init_invalid_format(self, tmp_path):
        """Should raise ValueError for invalid format."""
        manifest_path = tmp_path / "symbols.json"
        manifest_path.write_text('"just a string"')

        with pytest.raises(ValueError, match="Invalid manifest format"):
            ManifestSymbolProvider(manifest_path)

    def test_init_empty_manifest(self, tmp_path):
        """Should warn but not fail for empty manifest."""
        manifest_path = tmp_path / "symbols.json"
        manifest_path.write_text("[]")

        # Should not raise, but will log warning
        provider = ManifestSymbolProvider(manifest_path)
        assert provider._symbols == []


class TestManifestSymbolProviderGetSymbols:
    """Tests for ManifestSymbolProvider.get_option_symbols."""

    @pytest.fixture
    def sample_manifest(self, tmp_path):
        """Create a sample manifest file."""
        today = date.today()

        def make_symbol(dte: int, strike: float, right: str = "C") -> str:
            expiry = today + timedelta(days=dte)
            date_str = expiry.strftime("%y%m%d")
            strike_str = f"{int(strike * 1000):08d}"
            return f"SPY{date_str}{right}{strike_str}"

        symbols = [
            make_symbol(5, 500),  # DTE 5
            make_symbol(14, 500),  # DTE 14
            make_symbol(30, 500),  # DTE 30
            make_symbol(100, 500),  # DTE 100
            "QQQ240315C00400000",  # Different underlying
        ]

        manifest_path = tmp_path / "symbols.json"
        manifest_path.write_text(json.dumps({"symbols": symbols, "underlying": "SPY"}))

        return manifest_path

    def test_filter_by_underlying(self, sample_manifest):
        """Should filter symbols by underlying prefix."""
        provider = ManifestSymbolProvider(sample_manifest)

        symbols = provider.get_option_symbols("SPY", min_dte=0, max_dte=365)

        # Should exclude QQQ symbol
        assert all(s.startswith("SPY") for s in symbols)

    def test_filter_by_dte(self, sample_manifest):
        """Should filter symbols by DTE range."""
        provider = ManifestSymbolProvider(sample_manifest)

        symbols = provider.get_option_symbols("SPY", min_dte=7, max_dte=90)

        # Should include DTE 14 and 30, exclude 5 and 100
        assert len(symbols) == 2

    def test_wrong_underlying_in_manifest(self, tmp_path):
        """Should return empty if manifest underlying doesn't match."""
        manifest_path = tmp_path / "symbols.json"
        manifest = {
            "symbols": ["SPY240315C00450000"],
            "underlying": "SPY",
        }
        manifest_path.write_text(json.dumps(manifest))

        provider = ManifestSymbolProvider(manifest_path)
        symbols = provider.get_option_symbols("QQQ")  # Request different underlying

        assert symbols == []

    def test_includes_unparseable_symbols(self, tmp_path):
        """Should include symbols that can't be parsed (lines 394-396)."""
        manifest_path = tmp_path / "symbols.json"
        # Include unparseable symbol with correct underlying prefix
        manifest = {
            "symbols": [
                "SPY_CUSTOM_FORMAT",  # Unparseable but starts with SPY
                "SPY-INVALID",  # Another unparseable format
            ],
        }
        manifest_path.write_text(json.dumps(manifest))

        provider = ManifestSymbolProvider(manifest_path)
        symbols = provider.get_option_symbols("SPY", min_dte=0, max_dte=365)

        # Both unparseable symbols should be included
        assert len(symbols) == 2
        assert "SPY_CUSTOM_FORMAT" in symbols
        assert "SPY-INVALID" in symbols


class TestManifestSymbolProviderOCCParser:
    """Tests for ManifestSymbolProvider._parse_occ_symbol edge cases."""

    def test_parse_invalid_format_returns_none(self, tmp_path):
        """Should return None for symbols that don't match pattern (line 424)."""
        manifest_path = tmp_path / "symbols.json"
        manifest_path.write_text(json.dumps([]))

        provider = ManifestSymbolProvider(manifest_path)

        # Various invalid formats
        assert provider._parse_occ_symbol("INVALID") is None
        assert provider._parse_occ_symbol("SPY") is None
        assert provider._parse_occ_symbol("") is None
        assert provider._parse_occ_symbol("SPY240315X00450000") is None  # Invalid right

    def test_parse_invalid_date_raises_exception(self, tmp_path):
        """Should return None when date parsing fails (lines 438-439)."""
        manifest_path = tmp_path / "symbols.json"
        manifest_path.write_text(json.dumps([]))

        provider = ManifestSymbolProvider(manifest_path)

        # Invalid month 13 - matches regex but ValueError on date()
        result = provider._parse_occ_symbol("SPY241315C00450000")
        assert result is None

        # Invalid day 32
        result = provider._parse_occ_symbol("SPY240132C00450000")
        assert result is None

    def test_parse_valid_symbol(self, tmp_path):
        """Should parse valid OCC symbol correctly."""
        manifest_path = tmp_path / "symbols.json"
        manifest_path.write_text(json.dumps([]))

        provider = ManifestSymbolProvider(manifest_path)

        result = provider._parse_occ_symbol("SPY240315C00450000")
        assert result is not None
        expiry, strike, right = result
        assert expiry == date(2024, 3, 15)
        assert strike == 450.0
        assert right == "C"


class TestMockSymbolProvider:
    """Tests for MockSymbolProvider."""

    def test_return_static_symbols(self):
        """Should return provided static symbols."""
        static_symbols = ["SPY240315C00450000", "SPY240315P00440000"]
        provider = MockSymbolProvider(symbols=static_symbols)

        symbols = provider.get_option_symbols("SPY")
        assert symbols == static_symbols

    def test_filter_static_by_underlying(self):
        """Should filter static symbols by underlying."""
        static_symbols = [
            "SPY240315C00450000",
            "QQQ240315C00400000",
        ]
        provider = MockSymbolProvider(symbols=static_symbols)

        symbols = provider.get_option_symbols("SPY")
        assert symbols == ["SPY240315C00450000"]

    def test_generate_symbols(self):
        """Should generate synthetic symbols when none provided."""
        provider = MockSymbolProvider(generate_count=50)

        symbols = provider.get_option_symbols("SPY", min_dte=7, max_dte=30)

        assert len(symbols) <= 50
        assert all(s.startswith("SPY") for s in symbols)

    def test_generated_symbols_are_valid_occ(self):
        """Generated symbols should be valid OCC format."""
        provider = MockSymbolProvider(generate_count=10)

        symbols = provider.get_option_symbols("SPY", min_dte=7, max_dte=30)

        # All should be parseable
        import re

        pattern = r"^([A-Z]{1,6})(\d{6})([CP])(\d{8})$"
        for symbol in symbols:
            assert re.match(pattern, symbol), f"Invalid OCC format: {symbol}"

    def test_generate_fewer_than_count(self):
        """Should return all generated symbols when fewer than generate_count (line 530).

        This tests the final return path when the loop completes without hitting
        the generate_count limit (narrow DTE range produces fewer symbols).
        """
        # Request a very large count with narrow filters
        # This should exhaust all combinations before hitting 10000
        provider = MockSymbolProvider(generate_count=10000)

        # Very narrow DTE range (7-14 = 1 expiration at 7-day interval)
        # Narrow moneyness (0.99-1.01 with $500 base = $495-$505, only ~2 strikes at $5 intervals)
        # 1 expiry * 2 strikes * 2 rights = 4 symbols max
        symbols = provider.get_option_symbols(
            "SPY",
            min_dte=7,
            max_dte=14,
            moneyness_range=(0.99, 1.01),
        )

        # Should return fewer than 10000 since we can't generate that many
        assert len(symbols) < 10000
        assert len(symbols) > 0
        # All should still be valid
        assert all(s.startswith("SPY") for s in symbols)

    def test_generate_exact_count(self):
        """Should stop generating at exactly generate_count."""
        provider = MockSymbolProvider(generate_count=5)

        # Wide range to ensure we can generate at least 5
        symbols = provider.get_option_symbols(
            "SPY",
            min_dte=7,
            max_dte=90,
            moneyness_range=(0.85, 1.15),
        )

        assert len(symbols) == 5


class TestSaveSymbolsManifest:
    """Tests for save_symbols_manifest utility."""

    def test_save_basic_manifest(self, tmp_path):
        """Should save symbols to JSON file."""
        output_path = tmp_path / "test_manifest.json"
        symbols = ["SPY240315C00450000", "SPY240315P00440000"]

        save_symbols_manifest(symbols, output_path)

        assert output_path.exists()
        data = json.loads(output_path.read_text())
        assert data["symbols"] == sorted(symbols)
        assert data["count"] == 2
        assert "created_at" in data

    def test_save_with_underlying(self, tmp_path):
        """Should include underlying in manifest."""
        output_path = tmp_path / "test_manifest.json"
        symbols = ["SPY240315C00450000"]

        save_symbols_manifest(symbols, output_path, underlying="SPY")

        data = json.loads(output_path.read_text())
        assert data["underlying"] == "SPY"

    def test_save_with_metadata(self, tmp_path):
        """Should include additional metadata."""
        output_path = tmp_path / "test_manifest.json"
        symbols = ["SPY240315C00450000"]
        metadata = {"source": "databento", "filters": {"min_dte": 7}}

        save_symbols_manifest(symbols, output_path, metadata=metadata)

        data = json.loads(output_path.read_text())
        assert data["source"] == "databento"
        assert data["filters"] == {"min_dte": 7}

    def test_create_parent_directories(self, tmp_path):
        """Should create parent directories if needed."""
        output_path = tmp_path / "nested" / "dir" / "manifest.json"
        symbols = ["SPY240315C00450000"]

        save_symbols_manifest(symbols, output_path)

        assert output_path.exists()


class TestSymbolProviderIntegration:
    """Integration tests for symbol providers."""

    def test_manifest_roundtrip(self, tmp_path):
        """Should be able to save and load symbols through manifest."""
        # Generate symbols with future dates
        today = date.today()

        def make_symbol(days_ahead: int, strike: float, right: str = "C") -> str:
            expiry = today + timedelta(days=days_ahead)
            date_str = expiry.strftime("%y%m%d")
            strike_str = f"{int(strike * 1000):08d}"
            return f"SPY{date_str}{right}{strike_str}"

        original_symbols = [
            make_symbol(30, 450),
            make_symbol(30, 440, "P"),
            make_symbol(60, 455),
        ]

        # Save
        manifest_path = tmp_path / "symbols.json"
        save_symbols_manifest(original_symbols, manifest_path, underlying="SPY")

        # Load
        provider = ManifestSymbolProvider(manifest_path)
        loaded = provider.get_option_symbols("SPY", min_dte=0, max_dte=365)

        assert sorted(loaded) == sorted(original_symbols)

    def test_mock_provider_for_testing(self):
        """MockSymbolProvider should work as drop-in replacement."""
        # This tests that MockSymbolProvider implements the SymbolProvider protocol

        def use_provider(provider):
            """Function that expects a SymbolProvider."""
            return provider.get_option_symbols("SPY", min_dte=7, max_dte=30)

        mock = MockSymbolProvider(generate_count=10)
        symbols = use_provider(mock)

        assert len(symbols) > 0
        assert all(isinstance(s, str) for s in symbols)
