"""Unit tests for data provider protocols."""

from datetime import datetime

import pandas as pd
import pytest

from src.data.providers.mock import MockMacroDataProvider, MockMarketDataProvider
from src.data.providers.protocol import MacroDataProvider, MarketDataProvider


class TestMarketDataProviderProtocol:
    """Test MarketDataProvider protocol compliance."""

    def test_mock_provider_satisfies_protocol(self):
        """Test that MockMarketDataProvider satisfies MarketDataProvider protocol."""
        provider = MockMarketDataProvider()
        assert isinstance(provider, MarketDataProvider)

    def test_protocol_has_required_methods(self):
        """Test that MarketDataProvider protocol defines required methods."""
        required_methods = [
            "fetch_underlying_bars",
            "fetch_option_quotes",
            "estimate_cost",
            "resolve_option_symbols",
        ]

        for method_name in required_methods:
            assert hasattr(MarketDataProvider, method_name)

    def test_protocol_is_runtime_checkable(self):
        """Test that MarketDataProvider is runtime checkable."""
        # Create a class that implements the protocol
        class CustomProvider:
            def fetch_underlying_bars(self, symbol, start, end, interval="1m"):
                return pd.DataFrame()

            def fetch_option_quotes(self, symbols, start, end, schema="cbbo-1m"):
                return pd.DataFrame()

            def fetch_option_bars(self, symbols, start, end, interval="1d"):
                return pd.DataFrame()

            def fetch_option_statistics(self, symbols, start, end, stat_types=None):
                return pd.DataFrame()

            def estimate_cost(self, dataset, schema, symbols, start, end):
                return 0.0

            def resolve_option_symbols(
                self, parent, as_of, dte_min, dte_max, moneyness_min, moneyness_max
            ):
                return []

        provider = CustomProvider()
        assert isinstance(provider, MarketDataProvider)

    def test_incomplete_implementation_fails_check(self):
        """Test that incomplete implementations fail protocol check."""

        class IncompleteProvider:
            def fetch_underlying_bars(self, symbol, start, end, interval="1m"):
                return pd.DataFrame()
            # Missing other required methods

        provider = IncompleteProvider()
        assert not isinstance(provider, MarketDataProvider)


class TestMacroDataProviderProtocol:
    """Test MacroDataProvider protocol compliance."""

    def test_mock_provider_satisfies_protocol(self):
        """Test that MockMacroDataProvider satisfies MacroDataProvider protocol."""
        provider = MockMacroDataProvider()
        assert isinstance(provider, MacroDataProvider)

    def test_protocol_has_required_methods(self):
        """Test that MacroDataProvider protocol defines required methods."""
        required_methods = [
            "fetch_series",
            "get_latest_value",
        ]

        for method_name in required_methods:
            assert hasattr(MacroDataProvider, method_name)

    def test_protocol_is_runtime_checkable(self):
        """Test that MacroDataProvider is runtime checkable."""

        class CustomProvider:
            def fetch_series(self, series_id, start, end):
                return pd.DataFrame()

            def get_latest_value(self, series_id, as_of):
                return (datetime.now(), 0.0)

        provider = CustomProvider()
        assert isinstance(provider, MacroDataProvider)

    def test_incomplete_implementation_fails_check(self):
        """Test that incomplete implementations fail protocol check."""

        class IncompleteProvider:
            def fetch_series(self, series_id, start, end):
                return pd.DataFrame()
            # Missing get_latest_value

        provider = IncompleteProvider()
        assert not isinstance(provider, MacroDataProvider)


class TestProtocolDocumentation:
    """Test that protocols have proper documentation."""

    def test_market_data_provider_has_docstrings(self):
        """Test that MarketDataProvider methods have docstrings."""
        methods_to_check = [
            "fetch_underlying_bars",
            "fetch_option_quotes",
            "estimate_cost",
            "resolve_option_symbols",
        ]

        for method_name in methods_to_check:
            method = getattr(MarketDataProvider, method_name)
            assert method.__doc__ is not None
            assert len(method.__doc__.strip()) > 0

    def test_macro_data_provider_has_docstrings(self):
        """Test that MacroDataProvider methods have docstrings."""
        methods_to_check = [
            "fetch_series",
            "get_latest_value",
        ]

        for method_name in methods_to_check:
            method = getattr(MacroDataProvider, method_name)
            assert method.__doc__ is not None
            assert len(method.__doc__.strip()) > 0

    def test_protocol_classes_have_docstrings(self):
        """Test that protocol classes have docstrings."""
        assert MarketDataProvider.__doc__ is not None
        assert len(MarketDataProvider.__doc__.strip()) > 0

        assert MacroDataProvider.__doc__ is not None
        assert len(MacroDataProvider.__doc__.strip()) > 0
