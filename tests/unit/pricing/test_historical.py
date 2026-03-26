"""Unit tests for HistoricalQuoteSource."""

from datetime import date, datetime, timedelta, timezone
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.pricing.historical import HistoricalQuoteSource
from src.pricing.protocol import OptionQuote


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_raw_repo(df: pd.DataFrame | None = None) -> MagicMock:
    """Create a mock RawRepository that returns the given DataFrame."""
    repo = MagicMock()
    repo.read_option_quotes.return_value = (
        df if df is not None else pd.DataFrame()
    )
    return repo


def _make_quotes_df(rows: list[dict]) -> pd.DataFrame:
    """Build a raw_option_quotes-shaped DataFrame."""
    return pd.DataFrame(rows)


TRADE_DATE = date(2024, 3, 10)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestGetQuote:
    """Tests for HistoricalQuoteSource.get_quote."""

    def test_get_quote_returns_eod_bid_ask(self):
        """Latest timestamp per symbol is selected as the EOD quote."""
        df = _make_quotes_df(
            [
                {
                    "ts_utc": datetime(2024, 3, 10, 14, 0, tzinfo=timezone.utc),
                    "option_symbol": "SPY240315C00450000",
                    "bid": 5.00,
                    "ask": 5.30,
                },
                {
                    "ts_utc": datetime(2024, 3, 10, 16, 0, tzinfo=timezone.utc),
                    "option_symbol": "SPY240315C00450000",
                    "bid": 5.50,
                    "ask": 5.80,
                },
            ]
        )
        repo = _make_raw_repo(df)
        source = HistoricalQuoteSource(repo)

        quote = source.get_quote("SPY240315C00450000", TRADE_DATE)

        assert quote is not None
        assert quote.bid == pytest.approx(5.50)  # later timestamp
        assert quote.ask == pytest.approx(5.80)

    def test_get_quote_caches_per_date(self):
        """Second call for the same date+symbol doesn't hit DB."""
        df = _make_quotes_df(
            [
                {
                    "ts_utc": datetime(2024, 3, 10, 16, 0, tzinfo=timezone.utc),
                    "option_symbol": "SPY240315C00450000",
                    "bid": 5.50,
                    "ask": 5.80,
                },
            ]
        )
        repo = _make_raw_repo(df)
        source = HistoricalQuoteSource(repo)

        # First call loads from DB
        quote1 = source.get_quote("SPY240315C00450000", TRADE_DATE)
        # Second call should use cache
        quote2 = source.get_quote("SPY240315C00450000", TRADE_DATE)

        assert quote1 == quote2
        assert repo.read_option_quotes.call_count == 1

    def test_get_quote_returns_none_for_missing(self):
        """Unknown symbol returns None."""
        repo = _make_raw_repo(pd.DataFrame())
        source = HistoricalQuoteSource(repo)

        quote = source.get_quote("SPY240315C00999000", TRADE_DATE)

        assert quote is None


    def test_get_quote_loads_uncached_symbol_on_preloaded_date(self):
        """When date was preloaded for symbol A, get_quote(B) queries DB for B."""
        df_a = _make_quotes_df(
            [
                {
                    "ts_utc": datetime(2024, 3, 10, 16, 0, tzinfo=timezone.utc),
                    "option_symbol": "SPY240315C00450000",
                    "bid": 5.50,
                    "ask": 5.80,
                },
            ]
        )
        df_b = _make_quotes_df(
            [
                {
                    "ts_utc": datetime(2024, 3, 10, 16, 0, tzinfo=timezone.utc),
                    "option_symbol": "SPY240315P00440000",
                    "bid": 3.20,
                    "ask": 3.50,
                },
            ]
        )
        repo = MagicMock()
        repo.read_option_quotes.side_effect = [df_a, df_b]
        source = HistoricalQuoteSource(repo)

        # Preload only symbol A
        source.preload(TRADE_DATE, ["SPY240315C00450000"])
        assert repo.read_option_quotes.call_count == 1

        # get_quote for symbol B (not in original preload) should query DB
        quote_b = source.get_quote("SPY240315P00440000", TRADE_DATE)

        assert repo.read_option_quotes.call_count == 2
        assert quote_b is not None
        assert quote_b.bid == pytest.approx(3.20)
        assert quote_b.ask == pytest.approx(3.50)


class TestPreload:
    """Tests for HistoricalQuoteSource.preload."""

    def test_preload_batch_loads_symbols(self):
        """Single DB query for multiple symbols."""
        df = _make_quotes_df(
            [
                {
                    "ts_utc": datetime(2024, 3, 10, 16, 0, tzinfo=timezone.utc),
                    "option_symbol": "SPY240315C00450000",
                    "bid": 5.50,
                    "ask": 5.80,
                },
                {
                    "ts_utc": datetime(2024, 3, 10, 16, 0, tzinfo=timezone.utc),
                    "option_symbol": "SPY240315P00440000",
                    "bid": 3.20,
                    "ask": 3.50,
                },
            ]
        )
        repo = _make_raw_repo(df)
        source = HistoricalQuoteSource(repo)

        source.preload(
            TRADE_DATE, ["SPY240315C00450000", "SPY240315P00440000"]
        )

        # One DB call for both symbols
        assert repo.read_option_quotes.call_count == 1

        # Both symbols cached
        q1 = source.get_quote("SPY240315C00450000", TRADE_DATE)
        q2 = source.get_quote("SPY240315P00440000", TRADE_DATE)
        assert q1 is not None and q1.bid == pytest.approx(5.50)
        assert q2 is not None and q2.bid == pytest.approx(3.20)

        # No additional DB calls after cache hit
        assert repo.read_option_quotes.call_count == 1

    def test_preload_merges_with_cache(self):
        """Additional symbols extend the cache without re-fetching existing."""
        df1 = _make_quotes_df(
            [
                {
                    "ts_utc": datetime(2024, 3, 10, 16, 0, tzinfo=timezone.utc),
                    "option_symbol": "SPY240315C00450000",
                    "bid": 5.50,
                    "ask": 5.80,
                },
            ]
        )
        df2 = _make_quotes_df(
            [
                {
                    "ts_utc": datetime(2024, 3, 10, 16, 0, tzinfo=timezone.utc),
                    "option_symbol": "SPY240315P00440000",
                    "bid": 3.20,
                    "ask": 3.50,
                },
            ]
        )
        repo = MagicMock()
        repo.read_option_quotes.side_effect = [df1, df2]
        source = HistoricalQuoteSource(repo)

        # First preload: one symbol
        source.preload(TRADE_DATE, ["SPY240315C00450000"])
        assert repo.read_option_quotes.call_count == 1

        # Second preload: one new symbol + one already cached
        source.preload(
            TRADE_DATE, ["SPY240315C00450000", "SPY240315P00440000"]
        )
        # Should only query the new symbol
        assert repo.read_option_quotes.call_count == 2
        second_call = repo.read_option_quotes.call_args_list[1]
        assert second_call.kwargs["option_symbols"] == ["SPY240315P00440000"]

        # Both symbols accessible
        q1 = source.get_quote("SPY240315C00450000", TRADE_DATE)
        q2 = source.get_quote("SPY240315P00440000", TRADE_DATE)
        assert q1 is not None and q1.bid == pytest.approx(5.50)
        assert q2 is not None and q2.bid == pytest.approx(3.20)
