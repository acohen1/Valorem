"""Centralised position pricing with cascading data sources.

This module provides PositionPricer, the single entry point for pricing
option legs. It implements a three-tier cascade:

1. Surface — exact option_symbol match (price + Greeks)
2. QuoteSource — exact symbol from raw quotes (price only, entry Greeks)
3. Entry fallback — stale entry price + entry Greeks (logged as warning)

All consumers (engine, manager, exit orders, execution) are injected with
a PositionPricer instance instead of performing inline surface lookups.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from enum import Enum

import pandas as pd

from ..config.constants import TradingConstants
from ..strategy.types import Greeks, OptionLeg
from ..utils.calculations import greeks_from_row
from .protocol import OptionQuote, QuoteSource

CONTRACT_MULTIPLIER = TradingConstants.CONTRACT_MULTIPLIER
logger = logging.getLogger(__name__)


class PriceSource(Enum):
    """Origin of a pricing result."""

    SURFACE = "surface"
    MARKET_DATA = "market_data"
    ENTRY_FALLBACK = "entry"


@dataclass(frozen=True, slots=True)
class PricedLeg:
    """Result of pricing an option leg.

    Attributes:
        symbol: OCC option symbol
        price: Market price (bid for longs, ask for shorts)
        greeks: Current Greeks (from surface or entry fallback)
        source: Which tier provided the price
    """

    symbol: str
    price: float
    greeks: Greeks
    source: PriceSource


class PositionPricer:
    """Centralizes option leg pricing with cascading data sources.

    Injected into any component that needs to price option legs.
    Surface is tried first (has price + Greeks), then QuoteSource
    (has price only), then entry price as final fallback.

    Args:
        quote_source: Optional secondary quote provider.
            None means surface-only mode (backward compatible).
    """

    def __init__(self, quote_source: QuoteSource | None = None) -> None:
        self._quote_source = quote_source

    def price_leg(
        self,
        leg: OptionLeg,
        surface: pd.DataFrame,
        as_of: date,
    ) -> PricedLeg:
        """Price a single option leg using cascading lookup.

        Lookup order:
        1. Surface (exact option_symbol match) — price + Greeks
        2. QuoteSource (exact symbol match) — price only, entry Greeks
        3. Entry fallback — stale price + entry Greeks

        Args:
            leg: Option leg to price
            surface: Current day's surface DataFrame
            as_of: Current trading date

        Returns:
            PricedLeg with price, Greeks, and source attribution
        """
        # Tier 1: Surface lookup
        row = self._find_in_surface(leg.symbol, surface)
        if row is not None:
            price = row["bid"] if leg.qty > 0 else row["ask"]
            greeks = greeks_from_row(row, fallback=leg.greeks)
            return PricedLeg(leg.symbol, price, greeks, PriceSource.SURFACE)

        # Tier 2: QuoteSource
        if self._quote_source is not None:
            quote = self._quote_source.get_quote(leg.symbol, as_of)
            if quote is not None:
                price = quote.bid if leg.qty > 0 else quote.ask
                return PricedLeg(
                    leg.symbol, price, leg.greeks, PriceSource.MARKET_DATA
                )

        # Tier 3: Entry fallback
        logger.warning(
            f"Option {leg.symbol} not found in surface or market data, "
            f"using entry price"
        )
        return PricedLeg(
            leg.symbol, leg.entry_price, leg.greeks, PriceSource.ENTRY_FALLBACK
        )

    def get_quote(
        self,
        symbol: str,
        surface: pd.DataFrame,
        as_of: date,
    ) -> OptionQuote | None:
        """Get bid/ask quote via cascading lookup (surface then QuoteSource).

        Used by ExecutionSimulator which needs raw bid/ask before
        applying slippage.

        Args:
            symbol: OCC option symbol
            surface: Current day's surface DataFrame
            as_of: Current trading date

        Returns:
            OptionQuote with bid/ask, or None if not found in any source.
        """
        # Tier 1: Surface
        row = self._find_in_surface(symbol, surface)
        if row is not None:
            return OptionQuote(
                symbol=symbol,
                bid=float(row["bid"]),
                ask=float(row["ask"]),
            )

        # Tier 2: QuoteSource
        if self._quote_source is not None:
            return self._quote_source.get_quote(symbol, as_of)

        return None

    def price_position(
        self,
        legs: list[OptionLeg],
        surface: pd.DataFrame,
        as_of: date,
    ) -> float:
        """Compute total MTM value for a set of legs.

        Args:
            legs: Option legs to price
            surface: Current day's surface DataFrame
            as_of: Current trading date

        Returns:
            Net position value (qty * price * multiplier, summed)
        """
        total = 0.0
        for leg in legs:
            priced = self.price_leg(leg, surface, as_of)
            total += leg.qty * priced.price * CONTRACT_MULTIPLIER
        return total

    def preload(self, trade_date: date, symbols: list[str]) -> None:
        """Hint to the quote source to batch-load symbols for a date.

        When the quote source supports ``preload`` (e.g.
        ``HistoricalQuoteSource``), this issues a single DB query for
        all requested symbols, avoiding per-symbol round-trips during
        pricing.  No-op when there is no QuoteSource or it does not
        support preloading.

        Args:
            trade_date: Trading date to preload quotes for
            symbols: OCC option symbols to preload
        """
        if (
            self._quote_source is not None
            and hasattr(self._quote_source, "preload")
        ):
            self._quote_source.preload(trade_date, symbols)

    @staticmethod
    def _find_in_surface(
        symbol: str, surface: pd.DataFrame
    ) -> pd.Series | None:
        """Exact option_symbol lookup in surface DataFrame."""
        if surface.empty:
            return None
        matches = surface[surface["option_symbol"] == symbol]
        if not matches.empty:
            return matches.iloc[0]
        return None
