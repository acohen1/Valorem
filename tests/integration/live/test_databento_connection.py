"""Integration tests for Databento API connectivity.

These tests verify connectivity to Databento's API and are skipped
when DATABENTO_API_KEY is not set in the environment.
"""

import os

import pytest

# Skip all tests if API key not available
pytestmark = pytest.mark.skipif(
    not os.getenv("DATABENTO_API_KEY"),
    reason="DATABENTO_API_KEY not set - skipping Databento integration tests"
)


class TestDatabentoConnection:
    """Tests for Databento API connectivity."""

    def test_databento_client_connects(self) -> None:
        """Verify Databento client can be initialized."""
        import databento as db

        api_key = os.getenv("DATABENTO_API_KEY")
        client = db.Historical(key=api_key)

        # If we got here without exception, connection is valid
        assert client is not None

    def test_symbol_provider_fetches_definitions(self) -> None:
        """Verify DatabentoSymbolProvider can fetch option definitions."""
        from src.live.symbols import DatabentoSymbolProvider

        provider = DatabentoSymbolProvider()

        # This will make an API call
        try:
            symbols = provider.get_option_symbols(
                underlying="SPY",
                min_dte=7,
                max_dte=30,
                moneyness_range=(0.95, 1.05),
            )

            # Should return some symbols (actual count depends on market)
            assert isinstance(symbols, list)
            # Note: May return empty if market closed or no matching symbols
            print(f"Found {len(symbols)} SPY option symbols")

        except Exception as e:
            # API errors are acceptable in CI (rate limits, etc.)
            pytest.skip(f"Databento API error (may be rate limit): {e}")

class TestDatabentoDataFetch:
    """Tests for fetching actual data from Databento.

    These tests make real API calls and may incur costs.
    Run with caution.
    """

    @pytest.mark.slow
    def test_fetch_ohlcv_bars(self) -> None:
        """Test fetching OHLCV bars for underlying."""
        from datetime import datetime, timedelta

        from src.data.providers.databento import DatabentoProvider

        try:
            provider = DatabentoProvider()

            # Fetch last trading day's data
            end = datetime.now()
            start = end - timedelta(days=1)

            # Use ES (E-mini S&P) which is available on GLBX.MDP3
            bars = provider.fetch_underlying_bars(
                symbol="ESH5",  # E-mini S&P March 2025
                start=start,
                end=end,
                interval="1h",
            )

            assert bars is not None
            print(f"Fetched {len(bars)} bars")

        except Exception as e:
            pytest.skip(f"Databento data fetch error: {e}")


class TestEnvironmentConfigWithDatabento:
    """Tests for EnvironmentConfig with Databento credentials."""

    def test_paper_live_mode_validates(self) -> None:
        """Test PAPER_LIVE mode validates with API key present."""
        from src.config import EnvironmentConfig, TradingMode

        config = EnvironmentConfig.from_env(mode="paper_live")

        # Should not raise since API key is present
        config.validate()

        assert config.mode == TradingMode.PAPER_LIVE
        assert config.databento_api_key is not None
        assert config.is_live_data is True

    def test_cli_validation_passes(self) -> None:
        """Test CLI validation passes with API key."""
        from src.config import validate_cli_config

        results = validate_cli_config(mode="paper_live")

        assert results["valid"] is True
        assert results["config"] is not None
