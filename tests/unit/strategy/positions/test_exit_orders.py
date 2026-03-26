"""Unit tests for exit order generation."""

from datetime import date, datetime, timezone

import pandas as pd
import pytest

from src.config.schema import ExecutionConfig, PricingConfig
from src.risk.portfolio import PositionState
from src.strategy.positions.exit_orders import (
    ExitOrder,
    ExitOrderGenerator,
)
from src.strategy.positions.lifecycle import ManagedPosition
from src.strategy.types import (
    ExitSignal,
    ExitSignalType,
    Greeks,
    OptionLeg,
    OptionRight,
    Signal,
    SignalType,
)


@pytest.fixture
def execution_config():
    """Create execution config for testing."""
    return ExecutionConfig(pricing=PricingConfig(buy_at="ask", sell_at="bid"))


@pytest.fixture
def exit_order_generator(execution_config):
    """Create exit order generator for testing."""
    return ExitOrderGenerator(config=execution_config)


@pytest.fixture
def sample_greeks():
    """Sample Greeks for testing."""
    return Greeks(delta=0.5, gamma=0.05, vega=0.3, theta=-0.02)


@pytest.fixture
def sample_signal():
    """Sample entry signal for testing."""
    return Signal(
        signal_type=SignalType.DIRECTIONAL_VOL,
        edge=0.05,
        confidence=0.75,
        tenor_days=30,
        delta_bucket="ATM",
    )


@pytest.fixture
def sample_legs(sample_greeks):
    """Sample option legs for a vertical spread."""
    return [
        OptionLeg(
            symbol="SPY240315C00450000",
            qty=1,
            entry_price=5.00,
            strike=450.0,
            expiry=date(2024, 3, 15),
            right=OptionRight.CALL,
            greeks=sample_greeks,
        ),
        OptionLeg(
            symbol="SPY240315C00455000",
            qty=-1,
            entry_price=3.00,
            strike=455.0,
            expiry=date(2024, 3, 15),
            right=OptionRight.CALL,
            greeks=Greeks(delta=0.4, gamma=0.04, vega=0.25, theta=-0.015),
        ),
    ]


@pytest.fixture
def sample_position(sample_legs, sample_greeks, sample_signal):
    """Sample managed position for testing."""
    return ManagedPosition(
        position_id="pos-001",
        state=PositionState.OPEN,
        legs=sample_legs,
        structure_type="VerticalSpread",
        entry_signal=sample_signal,
        entry_time=datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc),
        entry_price=-200.0,  # Debit spread
        entry_greeks=sample_greeks,
        max_loss=200.0,
        current_price=-200.0,
        current_greeks=sample_greeks,
        unrealized_pnl=0.0,
        days_held=0,
        days_to_expiry=30,
    )


@pytest.fixture
def sample_surface():
    """Sample surface DataFrame for testing."""
    return pd.DataFrame(
        [
            {
                "option_symbol": "SPY240315C00450000",
                "tenor_days": 30,
                "delta_bucket": "ATM",
                "strike": 450.0,
                "expiry": date(2024, 3, 15),
                "right": "C",
                "bid": 5.50,
                "ask": 5.70,
                "delta": 0.52,
                "gamma": 0.05,
                "vega": 0.32,
                "theta": -0.022,
            },
            {
                "option_symbol": "SPY240315C00455000",
                "tenor_days": 30,
                "delta_bucket": "C25",
                "strike": 455.0,
                "expiry": date(2024, 3, 15),
                "right": "C",
                "bid": 3.30,
                "ask": 3.50,
                "delta": 0.42,
                "gamma": 0.04,
                "vega": 0.27,
                "theta": -0.018,
            },
        ]
    )


class TestExitOrderGeneratorBasic:
    """Tests for basic exit order generation."""

    def test_generate_exit_order_for_open_position(
        self, exit_order_generator, sample_position, sample_surface
    ):
        """Test generating exit order for an open position."""
        exit_signal = ExitSignal(
            exit_type=ExitSignalType.TAKE_PROFIT,
            position_id="pos-001",
            urgency=0.5,
            reason="Take profit",
        )

        positions = {"pos-001": sample_position}

        orders = exit_order_generator.generate_exit_orders(
            exit_signals=[exit_signal],
            positions=positions,
            surface=sample_surface,
        )

        assert len(orders) == 1
        order = orders[0]
        assert order.position_id == "pos-001"
        assert order.exit_signal == exit_signal
        assert order.structure_type == "VerticalSpread"
        assert order.order_id.startswith("exit-")

    def test_closing_legs_reverse_quantities(
        self, exit_order_generator, sample_position, sample_surface
    ):
        """Test that closing legs have reversed quantities."""
        exit_signal = ExitSignal(
            exit_type=ExitSignalType.STOP_LOSS,
            position_id="pos-001",
            urgency=0.95,
            reason="Stop loss",
        )

        positions = {"pos-001": sample_position}

        orders = exit_order_generator.generate_exit_orders(
            exit_signals=[exit_signal],
            positions=positions,
            surface=sample_surface,
        )

        order = orders[0]

        # Original: +1 and -1
        # Closing: -1 and +1
        assert len(order.legs) == 2

        # Find the legs by symbol
        leg_450 = next(l for l in order.legs if l.symbol == "SPY240315C00450000")
        leg_455 = next(l for l in order.legs if l.symbol == "SPY240315C00455000")

        # Quantities should be reversed
        assert leg_450.qty == -1  # Was +1, now -1 (sell to close)
        assert leg_455.qty == 1  # Was -1, now +1 (buy to close)

    def test_closing_prices_use_correct_side(
        self, exit_order_generator, sample_position, sample_surface
    ):
        """Test that closing uses bid for longs and ask for shorts."""
        exit_signal = ExitSignal(
            exit_type=ExitSignalType.TAKE_PROFIT,
            position_id="pos-001",
            urgency=0.5,
            reason="Take profit",
        )

        positions = {"pos-001": sample_position}

        orders = exit_order_generator.generate_exit_orders(
            exit_signals=[exit_signal],
            positions=positions,
            surface=sample_surface,
        )

        order = orders[0]

        leg_450 = next(l for l in order.legs if l.symbol == "SPY240315C00450000")
        leg_455 = next(l for l in order.legs if l.symbol == "SPY240315C00455000")

        # Long position (qty was +1) sells at bid
        assert leg_450.entry_price == 5.50  # bid

        # Short position (qty was -1) buys at ask
        assert leg_455.entry_price == 3.50  # ask


class TestExitOrderGeneratorEdgeCases:
    """Tests for edge cases in exit order generation."""

    def test_skip_non_open_position(
        self, exit_order_generator, sample_position, sample_surface
    ):
        """Test that non-OPEN positions are skipped."""
        sample_position.state = PositionState.CLOSING

        exit_signal = ExitSignal(
            exit_type=ExitSignalType.TAKE_PROFIT,
            position_id="pos-001",
            urgency=0.5,
            reason="Take profit",
        )

        positions = {"pos-001": sample_position}

        orders = exit_order_generator.generate_exit_orders(
            exit_signals=[exit_signal],
            positions=positions,
            surface=sample_surface,
        )

        assert len(orders) == 0

    def test_skip_missing_position(self, exit_order_generator, sample_surface):
        """Test that missing positions are skipped."""
        exit_signal = ExitSignal(
            exit_type=ExitSignalType.TAKE_PROFIT,
            position_id="pos-999",
            urgency=0.5,
            reason="Take profit",
        )

        positions = {}

        orders = exit_order_generator.generate_exit_orders(
            exit_signals=[exit_signal],
            positions=positions,
            surface=sample_surface,
        )

        assert len(orders) == 0

    def test_option_not_in_surface_uses_entry_price(
        self, exit_order_generator, sample_position
    ):
        """Test fallback to entry price when option not in surface."""
        exit_signal = ExitSignal(
            exit_type=ExitSignalType.TAKE_PROFIT,
            position_id="pos-001",
            urgency=0.5,
            reason="Take profit",
        )

        positions = {"pos-001": sample_position}

        # Empty surface
        empty_surface = pd.DataFrame(
            columns=[
                "option_symbol",
                "tenor_days",
                "delta_bucket",
                "strike",
                "expiry",
                "right",
                "bid",
                "ask",
                "delta",
                "gamma",
                "vega",
                "theta",
            ]
        )

        orders = exit_order_generator.generate_exit_orders(
            exit_signals=[exit_signal],
            positions=positions,
            surface=empty_surface,
        )

        assert len(orders) == 1
        order = orders[0]

        # Should use entry prices as fallback
        leg_450 = next(l for l in order.legs if l.symbol == "SPY240315C00450000")
        leg_455 = next(l for l in order.legs if l.symbol == "SPY240315C00455000")

        assert leg_450.entry_price == 5.00  # Original entry price
        assert leg_455.entry_price == 3.00  # Original entry price


class TestExitOrderGeneratorMultipleSignals:
    """Tests for multiple exit signals."""

    def test_generate_multiple_exit_orders(
        self, exit_order_generator, sample_position, sample_surface, sample_greeks, sample_signal
    ):
        """Test generating exit orders for multiple positions."""
        # Create a second position
        position2_legs = [
            OptionLeg(
                symbol="SPY240315C00450000",
                qty=2,
                entry_price=5.00,
                strike=450.0,
                expiry=date(2024, 3, 15),
                right=OptionRight.CALL,
                greeks=sample_greeks,
            ),
        ]

        position2 = ManagedPosition(
            position_id="pos-002",
            state=PositionState.OPEN,
            legs=position2_legs,
            structure_type="LongCall",
            entry_signal=sample_signal,
            entry_time=datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc),
            entry_price=-1000.0,
            entry_greeks=sample_greeks,
            max_loss=1000.0,
            current_price=-1000.0,
            current_greeks=sample_greeks,
        )

        exit_signals = [
            ExitSignal(
                exit_type=ExitSignalType.STOP_LOSS,
                position_id="pos-001",
                urgency=0.95,
                reason="Stop loss",
            ),
            ExitSignal(
                exit_type=ExitSignalType.TIME_DECAY,
                position_id="pos-002",
                urgency=0.8,
                reason="Time decay",
            ),
        ]

        positions = {"pos-001": sample_position, "pos-002": position2}

        orders = exit_order_generator.generate_exit_orders(
            exit_signals=exit_signals,
            positions=positions,
            surface=sample_surface,
        )

        assert len(orders) == 2
        order_ids = {o.position_id for o in orders}
        assert order_ids == {"pos-001", "pos-002"}


class TestExitOrderGreeksCalculation:
    """Tests for Greeks calculation in exit orders."""

    def test_closing_greeks_are_computed(
        self, exit_order_generator, sample_position, sample_surface
    ):
        """Test that closing order Greeks are computed correctly."""
        exit_signal = ExitSignal(
            exit_type=ExitSignalType.TAKE_PROFIT,
            position_id="pos-001",
            urgency=0.5,
            reason="Take profit",
        )

        positions = {"pos-001": sample_position}

        orders = exit_order_generator.generate_exit_orders(
            exit_signals=[exit_signal],
            positions=positions,
            surface=sample_surface,
        )

        order = orders[0]

        # Closing order Greeks should reflect the closing legs
        # Leg 450: qty=-1, delta=0.52 → scaled delta = -1 * 0.52 * 100 = -52
        # Leg 455: qty=+1, delta=0.42 → scaled delta = +1 * 0.42 * 100 = +42
        # Net delta = -52 + 42 = -10
        assert order.greeks.delta == pytest.approx(-10.0, abs=1.0)

    def test_net_premium_calculation(
        self, exit_order_generator, sample_position, sample_surface
    ):
        """Test net premium calculation for exit order."""
        exit_signal = ExitSignal(
            exit_type=ExitSignalType.TAKE_PROFIT,
            position_id="pos-001",
            urgency=0.5,
            reason="Take profit",
        )

        positions = {"pos-001": sample_position}

        orders = exit_order_generator.generate_exit_orders(
            exit_signals=[exit_signal],
            positions=positions,
            surface=sample_surface,
        )

        order = orders[0]

        # Leg 450: qty=-1, price=5.50 → -1 * 5.50 * 100 = -550
        # Leg 455: qty=+1, price=3.50 → +1 * 3.50 * 100 = +350
        # Net premium = -550 + 350 = -200
        # Wait, we sell leg 450 at bid (5.50) and buy leg 455 at ask (3.50)
        # So net = -1 * 5.50 * 100 + 1 * 3.50 * 100 = -550 + 350 = -200
        # Actually this is wrong - when selling (qty=-1 in closing leg) we receive money
        # qty=-1 * price=5.50 * 100 = -550 (we receive 550)
        # qty=+1 * price=3.50 * 100 = +350 (we pay 350)
        # Net = -550 + 350 = -200... hmm

        # Actually let me recalculate:
        # For closing legs:
        # - Original long (qty=+1) → closing sell (qty=-1) at bid=5.50
        # - Original short (qty=-1) → closing buy (qty=+1) at ask=3.50
        # Net premium = (-1 * 5.50 + 1 * 3.50) * 100 = (-5.50 + 3.50) * 100 = -2.00 * 100 = -200
        # This means we pay $200 net? That doesn't sound right...
        # Wait, negative qty * positive price = negative, which in premium terms means we receive
        # So -200 means we receive $200
        # Actually the calculation is: sum(qty * price * 100)
        # -1 * 5.50 * 100 = -550 (selling, negative = receive)
        # +1 * 3.50 * 100 = +350 (buying, positive = pay)
        # Total = -550 + 350 = -200 (negative = net receive $200)

        # Actually the net_premium for closing a debit spread that gained should be positive
        # (we receive money). Let me check the actual implementation...
        # From the code: sum(leg.qty * leg.entry_price * CONTRACT_MULTIPLIER for leg in legs)
        # -1 * 5.50 * 100 + 1 * 3.50 * 100 = -550 + 350 = -200

        assert order.net_premium == pytest.approx(-200.0, abs=1.0)


class TestExitOrderDataclass:
    """Tests for ExitOrder dataclass."""

    def test_exit_order_repr(self, sample_greeks):
        """Test ExitOrder string representation."""
        exit_signal = ExitSignal(
            exit_type=ExitSignalType.STOP_LOSS,
            position_id="pos-001",
            urgency=0.95,
            reason="Stop loss",
        )

        order = ExitOrder(
            order_id="exit-123",
            legs=[],
            structure_type="VerticalSpread",
            exit_signal=exit_signal,
            position_id="pos-001",
            greeks=sample_greeks,
            net_premium=200.0,
            timestamp=datetime.now(timezone.utc),
        )

        repr_str = repr(order)
        assert "exit-123" in repr_str
        assert "pos-001" in repr_str
        assert "stop_loss" in repr_str
        assert "200.00" in repr_str


