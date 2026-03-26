"""Unit tests for CalendarSpread."""

from datetime import date

import pandas as pd
import pytest

from src.strategy.structures.base import CONTRACT_MULTIPLIER
from src.strategy.structures.calendar import CalendarSpread
from src.strategy.types import OptionRight, Signal, SignalType


@pytest.fixture
def sample_surface() -> pd.DataFrame:
    """Create a sample surface DataFrame for testing."""
    return pd.DataFrame(
        [
            # Near-term options (30 days)
            {
                "option_symbol": "SPY240215C00450000",
                "tenor_days": 30,
                "delta_bucket": "ATM",
                "strike": 450.0,
                "expiry": date(2024, 2, 15),
                "right": "C",
                "bid": 4.50,
                "ask": 4.70,
                "delta": 0.50,
                "gamma": 0.02,
                "vega": 0.15,
                "theta": -0.05,
            },
            {
                "option_symbol": "SPY240215P00440000",
                "tenor_days": 30,
                "delta_bucket": "P25",
                "strike": 440.0,
                "expiry": date(2024, 2, 15),
                "right": "P",
                "bid": 2.50,
                "ask": 2.70,
                "delta": -0.25,
                "gamma": 0.015,
                "vega": 0.12,
                "theta": -0.03,
            },
            # Far-term options (90 days)
            {
                "option_symbol": "SPY240515C00450000",
                "tenor_days": 90,
                "delta_bucket": "ATM",
                "strike": 450.0,
                "expiry": date(2024, 5, 15),
                "right": "C",
                "bid": 8.00,
                "ask": 8.30,
                "delta": 0.52,
                "gamma": 0.012,
                "vega": 0.25,
                "theta": -0.02,
            },
            {
                "option_symbol": "SPY240515P00440000",
                "tenor_days": 90,
                "delta_bucket": "P25",
                "strike": 440.0,
                "expiry": date(2024, 5, 15),
                "right": "P",
                "bid": 5.50,
                "ask": 5.80,
                "delta": -0.28,
                "gamma": 0.010,
                "vega": 0.22,
                "theta": -0.015,
            },
        ]
    )


class TestCalendarSpreadInit:
    """Tests for CalendarSpread initialization."""

    def test_default_tenor_gap(self) -> None:
        """Test default tenor gap is 30 days."""
        spread = CalendarSpread()
        assert spread._tenor_gap_days == 30

    def test_custom_tenor_gap(self) -> None:
        """Test custom tenor gap."""
        spread = CalendarSpread(tenor_gap_days=45)
        assert spread._tenor_gap_days == 45

    def test_default_strike_tolerance(self) -> None:
        """Test default strike tolerance is 1%."""
        spread = CalendarSpread()
        assert spread._strike_tolerance_pct == 0.01

    def test_custom_strike_tolerance(self) -> None:
        """Test custom strike tolerance."""
        spread = CalendarSpread(strike_tolerance_pct=0.02)
        assert spread._strike_tolerance_pct == 0.02


class TestCalendarSpreadCreateLegs:
    """Tests for create_legs method."""

    def test_positive_edge_sells_near_buys_far(
        self, sample_surface: pd.DataFrame
    ) -> None:
        """Test positive edge creates standard calendar (sell near, buy far)."""
        spread = CalendarSpread(tenor_gap_days=30)
        signal = Signal(
            signal_type=SignalType.TERM_ANOMALY,
            edge=0.05,  # Positive = sell near vol, buy far vol
            confidence=0.7,
            tenor_days=30,
            delta_bucket="ATM",
        )

        legs = spread.create_legs(signal, sample_surface)

        assert len(legs) == 2

        # Near leg should be short (sell)
        near_leg = min(legs, key=lambda l: l.expiry)
        assert near_leg.qty == -1
        assert near_leg.entry_price == 4.50  # Bid price for sell
        assert near_leg.expiry == date(2024, 2, 15)

        # Far leg should be long (buy)
        far_leg = max(legs, key=lambda l: l.expiry)
        assert far_leg.qty == 1
        assert far_leg.entry_price == 8.30  # Ask price for buy
        assert far_leg.expiry == date(2024, 5, 15)

    def test_negative_edge_buys_near_sells_far(
        self, sample_surface: pd.DataFrame
    ) -> None:
        """Test negative edge creates reverse calendar (buy near, sell far)."""
        spread = CalendarSpread(tenor_gap_days=30)
        signal = Signal(
            signal_type=SignalType.TERM_ANOMALY,
            edge=-0.05,  # Negative = buy near vol, sell far vol
            confidence=0.7,
            tenor_days=30,
            delta_bucket="ATM",
        )

        legs = spread.create_legs(signal, sample_surface)

        assert len(legs) == 2

        # Near leg should be long (buy)
        near_leg = min(legs, key=lambda l: l.expiry)
        assert near_leg.qty == 1
        assert near_leg.entry_price == 4.70  # Ask price for buy

        # Far leg should be short (sell)
        far_leg = max(legs, key=lambda l: l.expiry)
        assert far_leg.qty == -1
        assert far_leg.entry_price == 8.00  # Bid price for sell

    def test_exact_strike_match_when_available(
        self, sample_surface: pd.DataFrame
    ) -> None:
        """Test exact strike match is selected when far tenor has identical strike."""
        spread = CalendarSpread(tenor_gap_days=30)
        signal = Signal(
            signal_type=SignalType.TERM_ANOMALY,
            edge=0.05,
            confidence=0.7,
            tenor_days=30,
            delta_bucket="ATM",
        )

        legs = spread.create_legs(signal, sample_surface)

        # Fixture has near=450, far=450 — exact match
        assert legs[0].strike == legs[1].strike

    def test_same_option_type_for_both_legs(
        self, sample_surface: pd.DataFrame
    ) -> None:
        """Test both legs have the same option type."""
        spread = CalendarSpread(tenor_gap_days=30)
        signal = Signal(
            signal_type=SignalType.TERM_ANOMALY,
            edge=0.05,
            confidence=0.7,
            tenor_days=30,
            delta_bucket="ATM",
        )

        legs = spread.create_legs(signal, sample_surface)

        assert legs[0].right == legs[1].right

    def test_greeks_populated_correctly(self, sample_surface: pd.DataFrame) -> None:
        """Test Greeks are populated from surface data."""
        spread = CalendarSpread(tenor_gap_days=30)
        signal = Signal(
            signal_type=SignalType.TERM_ANOMALY,
            edge=0.05,
            confidence=0.7,
            tenor_days=30,
            delta_bucket="ATM",
        )

        legs = spread.create_legs(signal, sample_surface)

        for leg in legs:
            assert leg.greeks.delta != 0.0
            assert leg.greeks.gamma != 0.0
            assert leg.greeks.vega != 0.0
            assert leg.greeks.theta != 0.0

    def test_missing_near_option_raises(self, sample_surface: pd.DataFrame) -> None:
        """Test ValueError raised when near option not found."""
        spread = CalendarSpread()
        signal = Signal(
            signal_type=SignalType.TERM_ANOMALY,
            edge=0.05,
            confidence=0.7,
            tenor_days=45,  # No options at 45 days
            delta_bucket="ATM",
        )

        with pytest.raises(ValueError, match="No near option found"):
            spread.create_legs(signal, sample_surface)

    def test_missing_far_option_raises(self) -> None:
        """Test ValueError raised when far option not found."""
        # Create surface with only near-term options
        surface = pd.DataFrame(
            [
                {
                    "option_symbol": "SPY240215C00450000",
                    "tenor_days": 30,
                    "delta_bucket": "ATM",
                    "strike": 450.0,
                    "expiry": date(2024, 2, 15),
                    "right": "C",
                    "bid": 4.50,
                    "ask": 4.70,
                    "delta": 0.50,
                    "gamma": 0.02,
                    "vega": 0.15,
                    "theta": -0.05,
                },
            ]
        )

        spread = CalendarSpread(tenor_gap_days=30)
        signal = Signal(
            signal_type=SignalType.TERM_ANOMALY,
            edge=0.05,
            confidence=0.7,
            tenor_days=30,
            delta_bucket="ATM",
        )

        with pytest.raises(ValueError, match="No far option found"):
            spread.create_legs(signal, surface)

    def test_put_calendar_spread(self, sample_surface: pd.DataFrame) -> None:
        """Test calendar spread with put options."""
        spread = CalendarSpread(tenor_gap_days=30)
        signal = Signal(
            signal_type=SignalType.TERM_ANOMALY,
            edge=0.05,
            confidence=0.7,
            tenor_days=30,
            delta_bucket="P25",  # Put bucket
        )

        legs = spread.create_legs(signal, sample_surface)

        assert len(legs) == 2
        assert all(leg.right == OptionRight.PUT for leg in legs)

    def test_nearest_strike_within_tolerance(self) -> None:
        """Test nearest-strike matching when far tenor has a different strike."""
        # Near ATM at 450, far ATM at 452 (within 1% of 450 = 4.50 tolerance)
        surface = pd.DataFrame(
            [
                {
                    "option_symbol": "SPY240215C00450000",
                    "tenor_days": 30,
                    "delta_bucket": "ATM",
                    "strike": 450.0,
                    "expiry": date(2024, 2, 15),
                    "right": "C",
                    "bid": 4.50,
                    "ask": 4.70,
                    "delta": 0.50,
                    "gamma": 0.02,
                    "vega": 0.15,
                    "theta": -0.05,
                    "underlying_price": 450.0,
                },
                {
                    "option_symbol": "SPY240515C00452000",
                    "tenor_days": 90,
                    "delta_bucket": "ATM",
                    "strike": 452.0,
                    "expiry": date(2024, 5, 15),
                    "right": "C",
                    "bid": 8.00,
                    "ask": 8.30,
                    "delta": 0.48,
                    "gamma": 0.012,
                    "vega": 0.25,
                    "theta": -0.02,
                    "underlying_price": 452.0,
                },
            ]
        )

        spread = CalendarSpread(tenor_gap_days=30)
        signal = Signal(
            signal_type=SignalType.TERM_ANOMALY,
            edge=0.05,
            confidence=0.7,
            tenor_days=30,
            delta_bucket="ATM",
        )

        legs = spread.create_legs(signal, surface)

        assert len(legs) == 2
        near_leg = min(legs, key=lambda l: l.expiry)
        far_leg = max(legs, key=lambda l: l.expiry)
        assert near_leg.strike == 450.0
        assert far_leg.strike == 452.0

    def test_nearest_strike_exceeds_tolerance_raises(self) -> None:
        """Test rejection when nearest far strike exceeds tolerance."""
        # Near ATM at 450, far ATM at 460 (10 > 1% of 450 = 4.50)
        surface = pd.DataFrame(
            [
                {
                    "option_symbol": "SPY240215C00450000",
                    "tenor_days": 30,
                    "delta_bucket": "ATM",
                    "strike": 450.0,
                    "expiry": date(2024, 2, 15),
                    "right": "C",
                    "bid": 4.50,
                    "ask": 4.70,
                    "delta": 0.50,
                    "gamma": 0.02,
                    "vega": 0.15,
                    "theta": -0.05,
                    "underlying_price": 450.0,
                },
                {
                    "option_symbol": "SPY240515C00460000",
                    "tenor_days": 90,
                    "delta_bucket": "ATM",
                    "strike": 460.0,
                    "expiry": date(2024, 5, 15),
                    "right": "C",
                    "bid": 5.00,
                    "ask": 5.30,
                    "delta": 0.40,
                    "gamma": 0.010,
                    "vega": 0.20,
                    "theta": -0.015,
                    "underlying_price": 460.0,
                },
            ]
        )

        spread = CalendarSpread(tenor_gap_days=30)
        signal = Signal(
            signal_type=SignalType.TERM_ANOMALY,
            edge=0.05,
            confidence=0.7,
            tenor_days=30,
            delta_bucket="ATM",
        )

        with pytest.raises(ValueError, match="No far option within strike tolerance"):
            spread.create_legs(signal, surface)

    def test_wider_tolerance_accepts_larger_deviation(self) -> None:
        """Test that a wider tolerance permits larger strike deviations."""
        # Near ATM at 450, far ATM at 460 (10 within 3% of 450 = 13.50)
        surface = pd.DataFrame(
            [
                {
                    "option_symbol": "SPY240215C00450000",
                    "tenor_days": 30,
                    "delta_bucket": "ATM",
                    "strike": 450.0,
                    "expiry": date(2024, 2, 15),
                    "right": "C",
                    "bid": 4.50,
                    "ask": 4.70,
                    "delta": 0.50,
                    "gamma": 0.02,
                    "vega": 0.15,
                    "theta": -0.05,
                    "underlying_price": 450.0,
                },
                {
                    "option_symbol": "SPY240515C00460000",
                    "tenor_days": 90,
                    "delta_bucket": "ATM",
                    "strike": 460.0,
                    "expiry": date(2024, 5, 15),
                    "right": "C",
                    "bid": 5.00,
                    "ask": 5.30,
                    "delta": 0.40,
                    "gamma": 0.010,
                    "vega": 0.20,
                    "theta": -0.015,
                    "underlying_price": 460.0,
                },
            ]
        )

        # 3% tolerance: 450 * 0.03 = 13.50, deviation of 10 passes
        spread = CalendarSpread(tenor_gap_days=30, strike_tolerance_pct=0.03)
        signal = Signal(
            signal_type=SignalType.TERM_ANOMALY,
            edge=0.05,
            confidence=0.7,
            tenor_days=30,
            delta_bucket="ATM",
        )

        legs = spread.create_legs(signal, surface)

        near_leg = min(legs, key=lambda l: l.expiry)
        far_leg = max(legs, key=lambda l: l.expiry)
        assert near_leg.strike == 450.0
        assert far_leg.strike == 460.0

    def test_nearest_strike_prefers_closest(self) -> None:
        """Test that the closest strike is selected among multiple far candidates."""
        surface = pd.DataFrame(
            [
                # Near ATM at 450
                {
                    "option_symbol": "SPY240215C00450000",
                    "tenor_days": 30,
                    "delta_bucket": "ATM",
                    "strike": 450.0,
                    "expiry": date(2024, 2, 15),
                    "right": "C",
                    "bid": 4.50,
                    "ask": 4.70,
                    "delta": 0.50,
                    "gamma": 0.02,
                    "vega": 0.15,
                    "theta": -0.05,
                    "underlying_price": 450.0,
                },
                # Far ATM at 451 (closest)
                {
                    "option_symbol": "SPY240515C00451000",
                    "tenor_days": 90,
                    "delta_bucket": "ATM",
                    "strike": 451.0,
                    "expiry": date(2024, 5, 15),
                    "right": "C",
                    "bid": 8.00,
                    "ask": 8.30,
                    "delta": 0.49,
                    "gamma": 0.012,
                    "vega": 0.25,
                    "theta": -0.02,
                    "underlying_price": 451.0,
                },
                # Far ATM at 453 (further away)
                {
                    "option_symbol": "SPY240515C00453000",
                    "tenor_days": 90,
                    "delta_bucket": "ATM",
                    "strike": 453.0,
                    "expiry": date(2024, 5, 15),
                    "right": "C",
                    "bid": 7.80,
                    "ask": 8.10,
                    "delta": 0.47,
                    "gamma": 0.011,
                    "vega": 0.24,
                    "theta": -0.019,
                    "underlying_price": 453.0,
                },
            ]
        )

        spread = CalendarSpread(tenor_gap_days=30)
        signal = Signal(
            signal_type=SignalType.TERM_ANOMALY,
            edge=0.05,
            confidence=0.7,
            tenor_days=30,
            delta_bucket="ATM",
        )

        legs = spread.create_legs(signal, surface)

        far_leg = max(legs, key=lambda l: l.expiry)
        assert far_leg.strike == 451.0  # Closest to 450

    def test_tolerance_uses_underlying_price_when_available(self) -> None:
        """Test tolerance is computed from underlying_price, not strike."""
        # underlying_price=500, strike=480, so tolerance = 500 * 0.01 = 5.0
        # Far strike at 484 => deviation 4 < 5 => accepted
        surface = pd.DataFrame(
            [
                {
                    "option_symbol": "SPY240215P00480000",
                    "tenor_days": 30,
                    "delta_bucket": "P25",
                    "strike": 480.0,
                    "expiry": date(2024, 2, 15),
                    "right": "P",
                    "bid": 3.00,
                    "ask": 3.20,
                    "delta": -0.25,
                    "gamma": 0.015,
                    "vega": 0.12,
                    "theta": -0.03,
                    "underlying_price": 500.0,
                },
                {
                    "option_symbol": "SPY240515P00484000",
                    "tenor_days": 90,
                    "delta_bucket": "P25",
                    "strike": 484.0,
                    "expiry": date(2024, 5, 15),
                    "right": "P",
                    "bid": 5.50,
                    "ask": 5.80,
                    "delta": -0.26,
                    "gamma": 0.010,
                    "vega": 0.22,
                    "theta": -0.015,
                    "underlying_price": 500.0,
                },
            ]
        )

        spread = CalendarSpread(tenor_gap_days=30)
        signal = Signal(
            signal_type=SignalType.TERM_ANOMALY,
            edge=0.05,
            confidence=0.7,
            tenor_days=30,
            delta_bucket="P25",
        )

        legs = spread.create_legs(signal, surface)

        near_leg = min(legs, key=lambda l: l.expiry)
        far_leg = max(legs, key=lambda l: l.expiry)
        assert near_leg.strike == 480.0
        assert far_leg.strike == 484.0

    def test_tolerance_fallback_to_strike_without_underlying(self) -> None:
        """Test tolerance falls back to strike when underlying_price absent."""
        # No underlying_price column, so tolerance = strike * 0.01 = 450 * 0.01 = 4.50
        # Far strike at 454 => deviation 4 < 4.50 => accepted
        surface = pd.DataFrame(
            [
                {
                    "option_symbol": "SPY240215C00450000",
                    "tenor_days": 30,
                    "delta_bucket": "ATM",
                    "strike": 450.0,
                    "expiry": date(2024, 2, 15),
                    "right": "C",
                    "bid": 4.50,
                    "ask": 4.70,
                    "delta": 0.50,
                    "gamma": 0.02,
                    "vega": 0.15,
                    "theta": -0.05,
                },
                {
                    "option_symbol": "SPY240515C00454000",
                    "tenor_days": 90,
                    "delta_bucket": "ATM",
                    "strike": 454.0,
                    "expiry": date(2024, 5, 15),
                    "right": "C",
                    "bid": 7.80,
                    "ask": 8.10,
                    "delta": 0.48,
                    "gamma": 0.011,
                    "vega": 0.24,
                    "theta": -0.019,
                },
            ]
        )

        spread = CalendarSpread(tenor_gap_days=30)
        signal = Signal(
            signal_type=SignalType.TERM_ANOMALY,
            edge=0.05,
            confidence=0.7,
            tenor_days=30,
            delta_bucket="ATM",
        )

        legs = spread.create_legs(signal, surface)

        far_leg = max(legs, key=lambda l: l.expiry)
        assert far_leg.strike == 454.0


class TestCalendarSpreadComputeMaxLoss:
    """Tests for compute_max_loss method."""

    def test_debit_spread_max_loss(self, sample_surface: pd.DataFrame) -> None:
        """Test max loss for debit calendar spread."""
        spread = CalendarSpread(tenor_gap_days=30)
        signal = Signal(
            signal_type=SignalType.TERM_ANOMALY,
            edge=0.05,
            confidence=0.7,
            tenor_days=30,
            delta_bucket="ATM",
        )

        legs = spread.create_legs(signal, sample_surface)
        max_loss = spread.compute_max_loss(legs)

        # Sell near at 4.50, buy far at 8.30
        # Net premium = (-1 * 4.50 + 1 * 8.30) * 100 = -380 (debit)
        expected_debit = (8.30 - 4.50) * CONTRACT_MULTIPLIER
        assert max_loss == pytest.approx(expected_debit)

    def test_credit_spread_max_loss(self, sample_surface: pd.DataFrame) -> None:
        """Test max loss for credit calendar spread (reverse calendar)."""
        spread = CalendarSpread(tenor_gap_days=30)
        signal = Signal(
            signal_type=SignalType.TERM_ANOMALY,
            edge=-0.05,  # Negative = buy near, sell far
            confidence=0.7,
            tenor_days=30,
            delta_bucket="ATM",
        )

        legs = spread.create_legs(signal, sample_surface)
        max_loss = spread.compute_max_loss(legs)

        # Buy near at 4.70, sell far at 8.00
        # Net premium = (1 * 4.70 + -1 * 8.00) * 100 = 330 (credit)
        # Max loss = far leg premium = 8.00 * 100 = 800
        far_leg = max(legs, key=lambda l: l.expiry)
        expected_max_loss = abs(far_leg.entry_price * CONTRACT_MULTIPLIER)
        assert max_loss == pytest.approx(expected_max_loss)

    def test_max_loss_is_positive(self, sample_surface: pd.DataFrame) -> None:
        """Test max loss is always positive."""
        spread = CalendarSpread(tenor_gap_days=30)

        for edge in [0.05, -0.05]:
            signal = Signal(
                signal_type=SignalType.TERM_ANOMALY,
                edge=edge,
                confidence=0.7,
                tenor_days=30,
                delta_bucket="ATM",
            )

            legs = spread.create_legs(signal, sample_surface)
            max_loss = spread.compute_max_loss(legs)

            assert max_loss > 0


class TestCalendarSpreadComputeGreeks:
    """Tests for compute_greeks method."""

    def test_aggregate_greeks(self, sample_surface: pd.DataFrame) -> None:
        """Test Greeks are correctly aggregated across legs."""
        spread = CalendarSpread(tenor_gap_days=30)
        signal = Signal(
            signal_type=SignalType.TERM_ANOMALY,
            edge=0.05,
            confidence=0.7,
            tenor_days=30,
            delta_bucket="ATM",
        )

        legs = spread.create_legs(signal, sample_surface)
        greeks = spread.compute_greeks(legs)

        # Near leg: qty=-1, delta=0.50, gamma=0.02, vega=0.15, theta=-0.05
        # Far leg: qty=1, delta=0.52, gamma=0.012, vega=0.25, theta=-0.02
        # Aggregate (scaled by qty * 100):
        # delta = (-1 * 0.50 + 1 * 0.52) * 100 = 2
        # gamma = (-1 * 0.02 + 1 * 0.012) * 100 = -0.8
        # vega = (-1 * 0.15 + 1 * 0.25) * 100 = 10
        # theta = (-1 * -0.05 + 1 * -0.02) * 100 = 3

        assert greeks.delta == pytest.approx(2.0)
        assert greeks.gamma == pytest.approx(-0.8)
        assert greeks.vega == pytest.approx(10.0)
        assert greeks.theta == pytest.approx(3.0)

    def test_calendar_long_vega(self, sample_surface: pd.DataFrame) -> None:
        """Test standard calendar spread is long vega."""
        spread = CalendarSpread(tenor_gap_days=30)
        signal = Signal(
            signal_type=SignalType.TERM_ANOMALY,
            edge=0.05,  # Standard calendar
            confidence=0.7,
            tenor_days=30,
            delta_bucket="ATM",
        )

        legs = spread.create_legs(signal, sample_surface)
        greeks = spread.compute_greeks(legs)

        # Standard calendar (sell near, buy far) should be long vega
        assert greeks.vega > 0

    def test_calendar_short_gamma(self, sample_surface: pd.DataFrame) -> None:
        """Test standard calendar spread is short gamma."""
        spread = CalendarSpread(tenor_gap_days=30)
        signal = Signal(
            signal_type=SignalType.TERM_ANOMALY,
            edge=0.05,  # Standard calendar
            confidence=0.7,
            tenor_days=30,
            delta_bucket="ATM",
        )

        legs = spread.create_legs(signal, sample_surface)
        greeks = spread.compute_greeks(legs)

        # Standard calendar (sell near, buy far) should be short gamma
        # (near has higher gamma than far)
        assert greeks.gamma < 0
