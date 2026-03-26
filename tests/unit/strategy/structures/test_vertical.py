"""Unit tests for VerticalSpread."""

from datetime import date

import pandas as pd
import pytest

from src.strategy.structures.base import CONTRACT_MULTIPLIER
from src.strategy.structures.vertical import VerticalSpread
from src.strategy.types import OptionRight, Signal, SignalType


@pytest.fixture
def sample_surface() -> pd.DataFrame:
    """Create a sample surface DataFrame for testing."""
    return pd.DataFrame(
        [
            # ATM call options
            {
                "option_symbol": "SPY240315C00450000",
                "tenor_days": 30,
                "delta_bucket": "ATM",
                "strike": 450.0,
                "expiry": date(2024, 3, 15),
                "right": "C",
                "bid": 5.00,
                "ask": 5.20,
                "delta": 0.50,
                "gamma": 0.02,
                "vega": 0.18,
                "theta": -0.04,
                "underlying_price": 450.0,
            },
            # Higher strike call (for bull call spread)
            {
                "option_symbol": "SPY240315C00459000",
                "tenor_days": 30,
                "delta_bucket": "C25",
                "strike": 459.0,  # $9 above ATM (2% of 450)
                "expiry": date(2024, 3, 15),
                "right": "C",
                "bid": 2.50,
                "ask": 2.70,
                "delta": 0.35,
                "gamma": 0.018,
                "vega": 0.15,
                "theta": -0.03,
                "underlying_price": 450.0,
            },
            # ATM put options
            {
                "option_symbol": "SPY240315P00450000",
                "tenor_days": 30,
                "delta_bucket": "ATM",
                "strike": 450.0,
                "expiry": date(2024, 3, 15),
                "right": "P",
                "bid": 4.80,
                "ask": 5.00,
                "delta": -0.50,
                "gamma": 0.02,
                "vega": 0.18,
                "theta": -0.04,
                "underlying_price": 450.0,
            },
            # Lower strike put (for bear put spread)
            {
                "option_symbol": "SPY240315P00441000",
                "tenor_days": 30,
                "delta_bucket": "P25",
                "strike": 441.0,  # $9 below ATM (2% of 450)
                "expiry": date(2024, 3, 15),
                "right": "P",
                "bid": 2.20,
                "ask": 2.40,
                "delta": -0.30,
                "gamma": 0.015,
                "vega": 0.14,
                "theta": -0.025,
                "underlying_price": 450.0,
            },
        ]
    )


class TestVerticalSpreadInit:
    """Tests for VerticalSpread initialization."""

    def test_default_spread_width(self) -> None:
        """Test default spread width is 2%."""
        spread = VerticalSpread()
        assert spread._spread_width_pct == 0.02

    def test_custom_spread_width(self) -> None:
        """Test custom spread width."""
        spread = VerticalSpread(spread_width_pct=0.05)
        assert spread._spread_width_pct == 0.05


class TestVerticalSpreadCreateLegs:
    """Tests for create_legs method."""

    def test_positive_edge_creates_bull_call_spread(
        self, sample_surface: pd.DataFrame
    ) -> None:
        """Test positive edge creates bull call spread."""
        spread = VerticalSpread(spread_width_pct=0.02)
        signal = Signal(
            signal_type=SignalType.DIRECTIONAL_VOL,
            edge=0.05,  # Bullish
            confidence=0.7,
            tenor_days=30,
            delta_bucket="ATM",
        )

        legs = spread.create_legs(signal, sample_surface)

        assert len(legs) == 2

        # Should be all calls
        assert all(leg.right == OptionRight.CALL for leg in legs)

        # Buy lower strike, sell higher strike
        buy_leg = next(leg for leg in legs if leg.qty > 0)
        sell_leg = next(leg for leg in legs if leg.qty < 0)

        assert buy_leg.strike < sell_leg.strike
        assert buy_leg.entry_price == 5.20  # Ask for buy
        assert sell_leg.entry_price == 2.50  # Bid for sell

    def test_negative_edge_creates_bear_put_spread(
        self, sample_surface: pd.DataFrame
    ) -> None:
        """Test negative edge creates bear put spread."""
        spread = VerticalSpread(spread_width_pct=0.02)
        signal = Signal(
            signal_type=SignalType.DIRECTIONAL_VOL,
            edge=-0.05,  # Bearish
            confidence=0.7,
            tenor_days=30,
            delta_bucket="ATM",
        )

        legs = spread.create_legs(signal, sample_surface)

        assert len(legs) == 2

        # Should be all puts
        assert all(leg.right == OptionRight.PUT for leg in legs)

        # Buy higher strike put, sell lower strike put
        buy_leg = next(leg for leg in legs if leg.qty > 0)
        sell_leg = next(leg for leg in legs if leg.qty < 0)

        assert buy_leg.strike > sell_leg.strike
        assert buy_leg.entry_price == 5.00  # Ask for buy
        assert sell_leg.entry_price == 2.20  # Bid for sell

    def test_same_expiry_for_both_legs(self, sample_surface: pd.DataFrame) -> None:
        """Test both legs have the same expiry."""
        spread = VerticalSpread(spread_width_pct=0.02)
        signal = Signal(
            signal_type=SignalType.DIRECTIONAL_VOL,
            edge=0.05,
            confidence=0.7,
            tenor_days=30,
            delta_bucket="ATM",
        )

        legs = spread.create_legs(signal, sample_surface)

        assert legs[0].expiry == legs[1].expiry

    def test_greeks_populated(self, sample_surface: pd.DataFrame) -> None:
        """Test Greeks are populated from surface data."""
        spread = VerticalSpread(spread_width_pct=0.02)
        signal = Signal(
            signal_type=SignalType.DIRECTIONAL_VOL,
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

    def test_missing_anchor_option_raises(self, sample_surface: pd.DataFrame) -> None:
        """Test ValueError raised when anchor option not found."""
        spread = VerticalSpread()
        signal = Signal(
            signal_type=SignalType.DIRECTIONAL_VOL,
            edge=0.05,
            confidence=0.7,
            tenor_days=60,  # No options at 60 days
            delta_bucket="ATM",
        )

        with pytest.raises(ValueError, match="No option at"):
            spread.create_legs(signal, sample_surface)

    def test_missing_sell_strike_raises(self) -> None:
        """Test ValueError raised when sell strike not found."""
        # Surface with only one call strike
        surface = pd.DataFrame(
            [
                {
                    "option_symbol": "SPY240315C00450000",
                    "tenor_days": 30,
                    "delta_bucket": "ATM",
                    "strike": 450.0,
                    "expiry": date(2024, 3, 15),
                    "right": "C",
                    "bid": 5.00,
                    "ask": 5.20,
                    "delta": 0.50,
                    "gamma": 0.02,
                    "vega": 0.18,
                    "theta": -0.04,
                    "underlying_price": 450.0,
                },
            ]
        )

        spread = VerticalSpread(spread_width_pct=0.02)
        signal = Signal(
            signal_type=SignalType.DIRECTIONAL_VOL,
            edge=0.05,
            confidence=0.7,
            tenor_days=30,
            delta_bucket="ATM",
        )

        with pytest.raises(ValueError, match="No strikes above"):
            spread.create_legs(signal, surface)

    def test_uses_underlying_price_for_width(
        self, sample_surface: pd.DataFrame
    ) -> None:
        """Test spread width is calculated from underlying price."""
        spread = VerticalSpread(spread_width_pct=0.02)  # 2% = $9 for $450 underlying
        signal = Signal(
            signal_type=SignalType.DIRECTIONAL_VOL,
            edge=0.05,
            confidence=0.7,
            tenor_days=30,
            delta_bucket="ATM",
        )

        legs = spread.create_legs(signal, sample_surface)

        strikes = sorted([leg.strike for leg in legs])
        width = strikes[1] - strikes[0]

        # 2% of 450 = 9
        assert width == pytest.approx(9.0, rel=0.15)  # Allow 15% tolerance


class TestVerticalSpreadComputeMaxLoss:
    """Tests for compute_max_loss method."""

    def test_debit_spread_max_loss(self, sample_surface: pd.DataFrame) -> None:
        """Test max loss for debit vertical spread."""
        spread = VerticalSpread(spread_width_pct=0.02)
        signal = Signal(
            signal_type=SignalType.DIRECTIONAL_VOL,
            edge=0.05,  # Bull call spread (typically debit)
            confidence=0.7,
            tenor_days=30,
            delta_bucket="ATM",
        )

        legs = spread.create_legs(signal, sample_surface)
        max_loss = spread.compute_max_loss(legs)

        # Buy at 5.20, sell at 2.50
        # Net premium = (1 * 5.20 + -1 * 2.50) * 100 = -270 (debit)
        expected_debit = (5.20 - 2.50) * CONTRACT_MULTIPLIER
        assert max_loss == pytest.approx(expected_debit)

    def test_credit_spread_max_loss(self) -> None:
        """Test max loss for credit vertical spread."""
        # Create a surface where selling higher strike gives more premium
        surface = pd.DataFrame(
            [
                {
                    "option_symbol": "SPY240315C00450000",
                    "tenor_days": 30,
                    "delta_bucket": "ATM",
                    "strike": 450.0,
                    "expiry": date(2024, 3, 15),
                    "right": "C",
                    "bid": 2.00,
                    "ask": 2.20,
                    "delta": 0.50,
                    "gamma": 0.02,
                    "vega": 0.18,
                    "theta": -0.04,
                    "underlying_price": 450.0,
                },
                {
                    "option_symbol": "SPY240315C00460000",
                    "tenor_days": 30,
                    "delta_bucket": "C10",
                    "strike": 460.0,
                    "expiry": date(2024, 3, 15),
                    "right": "C",
                    "bid": 0.50,  # Much cheaper
                    "ask": 0.60,
                    "delta": 0.25,
                    "gamma": 0.015,
                    "vega": 0.12,
                    "theta": -0.02,
                    "underlying_price": 450.0,
                },
            ]
        )

        spread = VerticalSpread(spread_width_pct=0.02)
        signal = Signal(
            signal_type=SignalType.DIRECTIONAL_VOL,
            edge=0.05,
            confidence=0.7,
            tenor_days=30,
            delta_bucket="ATM",
        )

        # Force credit spread by manually creating legs
        # Buy lower strike at 2.20, sell higher strike at 0.50
        # This is actually still a debit spread, so let's test the logic differently

        # For a true credit spread, we need sell higher premium than buy
        # Let's just test the math directly
        from src.strategy.types import Greeks, OptionLeg, OptionRight

        legs = [
            OptionLeg(
                symbol="SPY240315P00450000",
                qty=-1,  # Sell
                entry_price=5.00,  # Higher premium
                strike=450.0,
                expiry=date(2024, 3, 15),
                right=OptionRight.PUT,
                greeks=Greeks(delta=-0.5, gamma=0.02, vega=0.18, theta=-0.04),
            ),
            OptionLeg(
                symbol="SPY240315P00440000",
                qty=1,  # Buy
                entry_price=2.00,  # Lower premium
                strike=440.0,
                expiry=date(2024, 3, 15),
                right=OptionRight.PUT,
                greeks=Greeks(delta=-0.25, gamma=0.015, vega=0.14, theta=-0.025),
            ),
        ]

        max_loss = spread.compute_max_loss(legs)

        # Net premium = (-1 * 5.00 + 1 * 2.00) * 100 = 300 (credit)
        # Spread width = (450 - 440) * 100 = 1000
        # Max loss = 1000 - 300 = 700
        assert max_loss == pytest.approx(700.0)

    def test_max_loss_is_positive(self, sample_surface: pd.DataFrame) -> None:
        """Test max loss is always positive."""
        spread = VerticalSpread(spread_width_pct=0.02)

        for edge in [0.05, -0.05]:
            signal = Signal(
                signal_type=SignalType.DIRECTIONAL_VOL,
                edge=edge,
                confidence=0.7,
                tenor_days=30,
                delta_bucket="ATM",
            )

            legs = spread.create_legs(signal, sample_surface)
            max_loss = spread.compute_max_loss(legs)

            assert max_loss > 0


class TestVerticalSpreadComputeGreeks:
    """Tests for compute_greeks method."""

    def test_aggregate_greeks_bull_call(self, sample_surface: pd.DataFrame) -> None:
        """Test Greeks aggregation for bull call spread."""
        spread = VerticalSpread(spread_width_pct=0.02)
        signal = Signal(
            signal_type=SignalType.DIRECTIONAL_VOL,
            edge=0.05,
            confidence=0.7,
            tenor_days=30,
            delta_bucket="ATM",
        )

        legs = spread.create_legs(signal, sample_surface)
        greeks = spread.compute_greeks(legs)

        # Buy ATM call: delta=0.50, gamma=0.02, vega=0.18, theta=-0.04
        # Sell 25C call: delta=0.35, gamma=0.018, vega=0.15, theta=-0.03
        # Aggregate (scaled by qty * 100):
        # delta = (1 * 0.50 + -1 * 0.35) * 100 = 15
        # gamma = (1 * 0.02 + -1 * 0.018) * 100 = 0.2
        # vega = (1 * 0.18 + -1 * 0.15) * 100 = 3
        # theta = (1 * -0.04 + -1 * -0.03) * 100 = -1

        assert greeks.delta == pytest.approx(15.0)
        assert greeks.gamma == pytest.approx(0.2)
        assert greeks.vega == pytest.approx(3.0)
        assert greeks.theta == pytest.approx(-1.0)

    def test_bull_call_spread_positive_delta(
        self, sample_surface: pd.DataFrame
    ) -> None:
        """Test bull call spread has positive delta."""
        spread = VerticalSpread(spread_width_pct=0.02)
        signal = Signal(
            signal_type=SignalType.DIRECTIONAL_VOL,
            edge=0.05,  # Bullish
            confidence=0.7,
            tenor_days=30,
            delta_bucket="ATM",
        )

        legs = spread.create_legs(signal, sample_surface)
        greeks = spread.compute_greeks(legs)

        assert greeks.delta > 0

    def test_bear_put_spread_negative_delta(
        self, sample_surface: pd.DataFrame
    ) -> None:
        """Test bear put spread has negative delta."""
        spread = VerticalSpread(spread_width_pct=0.02)
        signal = Signal(
            signal_type=SignalType.DIRECTIONAL_VOL,
            edge=-0.05,  # Bearish
            confidence=0.7,
            tenor_days=30,
            delta_bucket="ATM",
        )

        legs = spread.create_legs(signal, sample_surface)
        greeks = spread.compute_greeks(legs)

        # Bear put spread (buy higher put, sell lower put) should have negative delta
        assert greeks.delta < 0


class TestVerticalSpreadNetPremium:
    """Tests for net_premium and is_debit_spread methods."""

    def test_bull_call_is_debit(self, sample_surface: pd.DataFrame) -> None:
        """Test bull call spread is typically a debit spread."""
        spread = VerticalSpread(spread_width_pct=0.02)
        signal = Signal(
            signal_type=SignalType.DIRECTIONAL_VOL,
            edge=0.05,
            confidence=0.7,
            tenor_days=30,
            delta_bucket="ATM",
        )

        legs = spread.create_legs(signal, sample_surface)

        assert spread.is_debit_spread(legs)
        assert spread.net_premium(legs) < 0

    def test_bear_put_is_debit(self, sample_surface: pd.DataFrame) -> None:
        """Test bear put spread is typically a debit spread."""
        spread = VerticalSpread(spread_width_pct=0.02)
        signal = Signal(
            signal_type=SignalType.DIRECTIONAL_VOL,
            edge=-0.05,
            confidence=0.7,
            tenor_days=30,
            delta_bucket="ATM",
        )

        legs = spread.create_legs(signal, sample_surface)

        assert spread.is_debit_spread(legs)
        assert spread.net_premium(legs) < 0
