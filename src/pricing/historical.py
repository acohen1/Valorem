"""Historical quote source backed by raw_option_quotes.

Provides HistoricalQuoteSource, a QuoteSource implementation that reads
bid/ask from the raw_option_quotes table in the database. Used in
backtest mode to price positions whose contracts rotated out of the
analytical surface.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone

import pandas as pd

from ..data.storage.repository import RawRepository
from .protocol import OptionQuote

logger = logging.getLogger(__name__)


class HistoricalQuoteSource:
    """QuoteSource backed by raw_option_quotes in the database.

    Loads EOD (last timestamp per day) raw quotes on demand with
    symbol-filtered queries. Per-date caching avoids redundant
    DB round-trips.

    Usage:
        source = HistoricalQuoteSource(raw_repo)
        # Batch-load for efficiency
        source.preload(date(2023, 8, 15), ["SPY230915C00450000", ...])
        # Then lookup is O(1)
        quote = source.get_quote("SPY230915C00450000", date(2023, 8, 15))
    """

    def __init__(self, raw_repo: RawRepository) -> None:
        self._raw_repo = raw_repo
        self._cache: dict[date, dict[str, OptionQuote]] = {}

    def get_quote(self, symbol: str, trade_date: date) -> OptionQuote | None:
        """Get EOD quote for a symbol on a date.

        If the symbol has not been cached (even when other symbols for
        the same date have been), performs a single-symbol DB query.
        Callers should prefer ``preload`` for batch efficiency.

        Args:
            symbol: OCC option symbol
            trade_date: Trading date

        Returns:
            OptionQuote or None if no raw data exists.
        """
        cached_day = self._cache.get(trade_date)
        if cached_day is not None and symbol in cached_day:
            return cached_day[symbol]

        # Symbol not yet cached for this date — load it
        self.preload(trade_date, [symbol])
        return self._cache.get(trade_date, {}).get(symbol)

    def preload(self, trade_date: date, symbols: list[str]) -> None:
        """Batch-load quotes for multiple symbols on a date.

        Performs one symbol-filtered DB query. Results are cached.
        Subsequent calls for the same date with additional symbols
        will only query the new symbols.

        Args:
            trade_date: Trading date to load
            symbols: OCC option symbols to fetch
        """
        if not symbols:
            return

        # Filter to symbols not already cached
        cached_day = self._cache.get(trade_date, {})
        needed = [s for s in symbols if s not in cached_day]
        if not needed:
            return

        # Query DB: all quotes for these symbols on this date
        start_dt = datetime.combine(trade_date, datetime.min.time()).replace(
            tzinfo=timezone.utc
        )
        end_dt = datetime.combine(
            trade_date + timedelta(days=1), datetime.min.time()
        ).replace(tzinfo=timezone.utc)

        df = self._raw_repo.read_option_quotes(
            start=start_dt,
            end=end_dt,
            option_symbols=needed,
        )

        if trade_date not in self._cache:
            self._cache[trade_date] = {}

        if df.empty:
            return

        # Take last quote per symbol (EOD snapshot)
        df["ts_utc"] = pd.to_datetime(df["ts_utc"])
        eod = df.sort_values("ts_utc").groupby("option_symbol").last()

        for symbol, row in eod.iterrows():
            bid = row.get("bid")
            ask = row.get("ask")
            if bid is not None and ask is not None:
                self._cache[trade_date][symbol] = OptionQuote(
                    symbol=str(symbol),
                    bid=float(bid),
                    ask=float(ask),
                )

        logger.debug(
            f"Preloaded {len(eod)} quotes for {trade_date} "
            f"({len(needed)} requested)"
        )

    def clear_cache(self) -> None:
        """Clear the date cache."""
        self._cache.clear()
