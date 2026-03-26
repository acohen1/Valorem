"""Unit tests for PositionPricer."""

from datetime import date

import pandas as pd
import pytest

from src.config.constants import TradingConstants
from src.pricing.pricer import PositionPricer, PricedLeg, PriceSource
from src.pricing.protocol import OptionQuote, QuoteSource
from src.strategy.types import Greeks, OptionLeg, OptionRight

CONTRACT_MULTIPLIER = TradingConstants.CONTRACT_MULTIPLIER


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeQuoteSource:
    """QuoteSource implementation backed by a dict for testing."""

    def __init__(self, quotes: dict[str, OptionQuote] | None = None) -> None:
        self._quotes = quotes or {}

    def get_quote(self, symbol: str, trade_date: date) -> OptionQuote | None:
        return self._quotes.get(symbol)


def _make_surface(rows: list[dict]) -> pd.DataFrame:
    """Build a minimal surface DataFrame from a list of row dicts."""
    return pd.DataFrame(rows)


def _make_leg(
    symbol: str = "SPY240315C00450000",
    qty: int = 1,
    entry_price: float = 5.00,
    strike: float = 450.0,
    delta: float = 0.50,
    gamma: float = 0.02,
    vega: float = 0.30,
    theta: float = -0.05,
) -> OptionLeg:
    return OptionLeg(
        symbol=symbol,
        qty=qty,
        entry_price=entry_price,
        strike=strike,
        expiry=date(2024, 3, 15),
        right=OptionRight.CALL,
        greeks=Greeks(delta=delta, gamma=gamma, vega=vega, theta=theta),
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def surface() -> pd.DataFrame:
    """Surface with one representative option."""
    return _make_surface(
        [
            {
                "option_symbol": "SPY240315C00450000",
                "bid": 5.50,
                "ask": 5.80,
                "delta": 0.55,
                "gamma": 0.03,
                "vega": 0.35,
                "theta": -0.06,
            },
        ]
    )


@pytest.fixture
def empty_surface() -> pd.DataFrame:
    """Empty surface — correct schema, zero rows."""
    return pd.DataFrame(
        columns=["option_symbol", "bid", "ask", "delta", "gamma", "vega", "theta"]
    )


@pytest.fixture
def as_of() -> date:
    return date(2024, 3, 10)


# ---------------------------------------------------------------------------
# Tests: price_leg
# ---------------------------------------------------------------------------


class TestPriceLeg:
    """Tests for PositionPricer.price_leg."""

    def test_price_leg_surface_hit(self, surface, as_of):
        """Surface match returns PriceSource.SURFACE."""
        pricer = PositionPricer()
        leg = _make_leg()

        result = pricer.price_leg(leg, surface, as_of)

        assert result.source == PriceSource.SURFACE
        assert result.price == pytest.approx(5.50)  # bid for long

    def test_price_leg_quote_source_hit(self, empty_surface, as_of):
        """Surface miss + quote source hit returns PriceSource.MARKET_DATA."""
        quote = OptionQuote(symbol="SPY240315C00450000", bid=5.40, ask=5.70)
        source = FakeQuoteSource({"SPY240315C00450000": quote})
        pricer = PositionPricer(quote_source=source)
        leg = _make_leg()

        result = pricer.price_leg(leg, empty_surface, as_of)

        assert result.source == PriceSource.MARKET_DATA
        assert result.price == pytest.approx(5.40)  # bid for long

    def test_price_leg_entry_fallback(self, empty_surface, as_of):
        """Both miss returns PriceSource.ENTRY_FALLBACK."""
        pricer = PositionPricer(quote_source=FakeQuoteSource())
        leg = _make_leg(entry_price=5.00)

        result = pricer.price_leg(leg, empty_surface, as_of)

        assert result.source == PriceSource.ENTRY_FALLBACK
        assert result.price == pytest.approx(5.00)

    def test_price_leg_long_uses_bid_short_uses_ask(self, surface, as_of):
        """Direction-correct pricing: long → bid, short → ask."""
        pricer = PositionPricer()

        long_leg = _make_leg(qty=1)
        short_leg = _make_leg(qty=-1)

        long_result = pricer.price_leg(long_leg, surface, as_of)
        short_result = pricer.price_leg(short_leg, surface, as_of)

        assert long_result.price == pytest.approx(5.50)  # bid
        assert short_result.price == pytest.approx(5.80)  # ask

    def test_price_leg_greeks_from_surface(self, surface, as_of):
        """Surface hit uses surface Greeks."""
        pricer = PositionPricer()
        leg = _make_leg(delta=0.50)  # entry Greeks differ from surface

        result = pricer.price_leg(leg, surface, as_of)

        assert result.greeks.delta == pytest.approx(0.55)  # surface value
        assert result.greeks.gamma == pytest.approx(0.03)
        assert result.greeks.vega == pytest.approx(0.35)
        assert result.greeks.theta == pytest.approx(-0.06)

    def test_price_leg_greeks_fallback_from_quote_source(
        self, empty_surface, as_of
    ):
        """Quote source hit uses entry Greeks (raw quotes lack Greeks)."""
        quote = OptionQuote(symbol="SPY240315C00450000", bid=5.40, ask=5.70)
        source = FakeQuoteSource({"SPY240315C00450000": quote})
        pricer = PositionPricer(quote_source=source)
        leg = _make_leg(delta=0.50, gamma=0.02, vega=0.30, theta=-0.05)

        result = pricer.price_leg(leg, empty_surface, as_of)

        assert result.greeks.delta == pytest.approx(0.50)  # entry value
        assert result.greeks.gamma == pytest.approx(0.02)
        assert result.greeks.vega == pytest.approx(0.30)
        assert result.greeks.theta == pytest.approx(-0.05)

    def test_pricer_without_quote_source(self, empty_surface, as_of):
        """Surface-only mode (no quote source) falls back to entry."""
        pricer = PositionPricer()  # no quote_source
        leg = _make_leg(entry_price=5.00)

        result = pricer.price_leg(leg, empty_surface, as_of)

        assert result.source == PriceSource.ENTRY_FALLBACK
        assert result.price == pytest.approx(5.00)


# ---------------------------------------------------------------------------
# Tests: get_quote
# ---------------------------------------------------------------------------


class TestGetQuote:
    """Tests for PositionPricer.get_quote."""

    def test_get_quote_surface_then_quote_source(self, surface, as_of):
        """Surface hit returns surface bid/ask; miss falls to quote source."""
        quote = OptionQuote(symbol="SPY240315P00440000", bid=3.20, ask=3.50)
        source = FakeQuoteSource({"SPY240315P00440000": quote})
        pricer = PositionPricer(quote_source=source)

        # Surface hit
        result_surface = pricer.get_quote(
            "SPY240315C00450000", surface, as_of
        )
        assert result_surface is not None
        assert result_surface.bid == pytest.approx(5.50)
        assert result_surface.ask == pytest.approx(5.80)

        # Surface miss, quote source hit
        result_qs = pricer.get_quote("SPY240315P00440000", surface, as_of)
        assert result_qs is not None
        assert result_qs.bid == pytest.approx(3.20)
        assert result_qs.ask == pytest.approx(3.50)

    def test_get_quote_returns_none_when_both_miss(self, empty_surface, as_of):
        """Returns None when neither surface nor quote source has the symbol."""
        pricer = PositionPricer(quote_source=FakeQuoteSource())

        result = pricer.get_quote("SPY240315C00999000", empty_surface, as_of)

        assert result is None


# ---------------------------------------------------------------------------
# Tests: price_position
# ---------------------------------------------------------------------------


class TestPricePosition:
    """Tests for PositionPricer.price_position."""

    def test_price_position_aggregates_legs(self, as_of):
        """Total = sum(qty * price * multiplier) across legs."""
        surface = _make_surface(
            [
                {
                    "option_symbol": "SPY240315C00450000",
                    "bid": 5.50,
                    "ask": 5.80,
                    "delta": 0.55,
                    "gamma": 0.03,
                    "vega": 0.35,
                    "theta": -0.06,
                },
                {
                    "option_symbol": "SPY240315C00460000",
                    "bid": 2.10,
                    "ask": 2.40,
                    "delta": 0.30,
                    "gamma": 0.02,
                    "vega": 0.20,
                    "theta": -0.03,
                },
            ]
        )

        legs = [
            _make_leg(symbol="SPY240315C00450000", qty=1),   # long → bid 5.50
            _make_leg(symbol="SPY240315C00460000", qty=-1),   # short → ask 2.40
        ]

        pricer = PositionPricer()
        total = pricer.price_position(legs, surface, as_of)

        # (1 * 5.50 * 100) + (-1 * 2.40 * 100) = 550 - 240 = 310
        assert total == pytest.approx(310.0)
