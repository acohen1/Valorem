"""Abstract protocols for option quote providers.

This module defines the QuoteSource protocol and OptionQuote data class.
QuoteSource abstracts where option quotes come from — DB for backtest,
market feed for live — enabling dependency injection without tight coupling.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class OptionQuote:
    """Market quote for an option contract.

    Attributes:
        symbol: OCC option symbol
        bid: Best bid price
        ask: Best ask price
    """

    symbol: str
    bid: float
    ask: float


@runtime_checkable
class QuoteSource(Protocol):
    """Protocol for option quote providers.

    Implementations:
        HistoricalQuoteSource — reads raw_option_quotes from DB (backtest)
        LiveQuoteSource — reads from market data feed (future)
    """

    def get_quote(self, symbol: str, trade_date: date) -> OptionQuote | None:
        """Get latest bid/ask for a symbol on a given date.

        Args:
            symbol: OCC option symbol
            trade_date: Trading date

        Returns:
            OptionQuote with bid/ask, or None if not found.
        """
        ...
