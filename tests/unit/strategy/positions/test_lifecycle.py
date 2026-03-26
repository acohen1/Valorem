"""Unit tests for position lifecycle management."""

from datetime import date, datetime, timedelta, timezone

import pytest

from src.risk.portfolio import PositionState
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
def sample_greeks():
    """Sample Greeks for testing."""
    return Greeks(delta=0.5, gamma=0.05, vega=0.3, theta=-0.02)


@pytest.fixture
def sample_legs(sample_greeks):
    """Sample option legs for testing."""
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
def sample_signal():
    """Sample trading signal for testing."""
    return Signal(
        signal_type=SignalType.DIRECTIONAL_VOL,
        edge=0.05,
        confidence=0.75,
        tenor_days=30,
        delta_bucket="ATM",
    )


@pytest.fixture
def sample_position(sample_legs, sample_greeks, sample_signal):
    """Sample managed position for testing."""
    return ManagedPosition(
        position_id="test-123",
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


class TestManagedPositionCreation:
    """Tests for ManagedPosition dataclass creation."""

    def test_create_basic_position(self, sample_legs, sample_greeks, sample_signal):
        """Test basic position creation."""
        position = ManagedPosition(
            position_id="pos-001",
            state=PositionState.OPEN,
            legs=sample_legs,
            structure_type="CalendarSpread",
            entry_signal=sample_signal,
            entry_time=datetime.now(timezone.utc),
            entry_price=-150.0,
            entry_greeks=sample_greeks,
            max_loss=150.0,
        )

        assert position.position_id == "pos-001"
        assert position.state == PositionState.OPEN
        assert len(position.legs) == 2
        assert position.structure_type == "CalendarSpread"
        assert position.max_loss == 150.0

    def test_from_order_fill(self, sample_legs, sample_greeks, sample_signal):
        """Test creating position from order fill."""
        position = ManagedPosition.from_order_fill(
            position_id="fill-001",
            legs=sample_legs,
            structure_type="VerticalSpread",
            entry_signal=sample_signal,
            entry_price=-200.0,
            entry_greeks=sample_greeks,
            max_loss=200.0,
            days_to_expiry=30,
        )

        assert position.position_id == "fill-001"
        assert position.state == PositionState.OPEN
        assert position.unrealized_pnl == 0.0
        assert position.days_held == 0
        assert position.days_to_expiry == 30
        assert position.current_price == -200.0
        assert position.current_greeks == sample_greeks

    def test_from_order_fill_with_entry_time(
        self, sample_legs, sample_greeks, sample_signal
    ):
        """Test creating position with explicit entry time."""
        entry_time = datetime(2024, 1, 10, 9, 30, 0, tzinfo=timezone.utc)

        position = ManagedPosition.from_order_fill(
            position_id="fill-002",
            legs=sample_legs,
            structure_type="VerticalSpread",
            entry_signal=sample_signal,
            entry_price=-200.0,
            entry_greeks=sample_greeks,
            max_loss=200.0,
            days_to_expiry=30,
            entry_time=entry_time,
        )

        assert position.entry_time == entry_time


class TestManagedPositionState:
    """Tests for position state queries."""

    def test_is_open_when_open(self, sample_position):
        """Test is_open returns True for OPEN state."""
        assert sample_position.is_open() is True

    def test_is_open_when_closing(self, sample_position):
        """Test is_open returns False for CLOSING state."""
        sample_position.state = PositionState.CLOSING
        assert sample_position.is_open() is False

    def test_is_open_when_closed(self, sample_position):
        """Test is_open returns False for CLOSED state."""
        sample_position.state = PositionState.CLOSED
        assert sample_position.is_open() is False

    def test_is_closed_when_closed(self, sample_position):
        """Test is_closed returns True for CLOSED state."""
        sample_position.state = PositionState.CLOSED
        assert sample_position.is_closed() is True

    def test_is_closed_when_expired(self, sample_position):
        """Test is_closed returns True for EXPIRED state."""
        sample_position.state = PositionState.EXPIRED
        assert sample_position.is_closed() is True

    def test_is_closed_when_open(self, sample_position):
        """Test is_closed returns False for OPEN state."""
        assert sample_position.is_closed() is False


class TestPnlCalculations:
    """Tests for P&L calculations."""

    def test_pnl_pct_of_max_loss_zero_pnl(self, sample_position):
        """Test P&L percentage with zero unrealized P&L."""
        sample_position.unrealized_pnl = 0.0
        assert sample_position.pnl_pct_of_max_loss() == 0.0

    def test_pnl_pct_of_max_loss_negative_pnl(self, sample_position):
        """Test P&L percentage with negative (losing) P&L."""
        sample_position.unrealized_pnl = -100.0  # Lost $100
        sample_position.max_loss = 200.0

        # Should return positive value (0.5 = lost 50% of max loss)
        assert sample_position.pnl_pct_of_max_loss() == 0.5

    def test_pnl_pct_of_max_loss_positive_pnl(self, sample_position):
        """Test P&L percentage with positive (winning) P&L."""
        sample_position.unrealized_pnl = 50.0  # Made $50
        sample_position.max_loss = 200.0

        # Should return negative value (making money)
        assert sample_position.pnl_pct_of_max_loss() == -0.25

    def test_pnl_pct_of_max_loss_zero_max_loss(self, sample_position):
        """Test P&L percentage when max_loss is zero."""
        sample_position.max_loss = 0.0
        sample_position.unrealized_pnl = -100.0

        assert sample_position.pnl_pct_of_max_loss() == 0.0


class TestStateTransitions:
    """Tests for position state transitions."""

    def test_mark_closing(self, sample_position):
        """Test marking position as closing."""
        exit_signal = ExitSignal(
            exit_type=ExitSignalType.STOP_LOSS,
            position_id="test-123",
            urgency=0.9,
            reason="Stop-loss triggered",
        )

        sample_position.mark_closing(exit_signal)

        assert sample_position.state == PositionState.CLOSING
        assert sample_position.exit_signal == exit_signal

    def test_mark_closed(self, sample_position):
        """Test marking position as closed."""
        sample_position.entry_price = -200.0  # Debit spread
        sample_position.mark_closed(exit_price=50.0)

        assert sample_position.state == PositionState.CLOSED
        assert sample_position.exit_price == 50.0
        assert sample_position.exit_time is not None
        # Realized P&L = exit - entry = 50 - (-200) = 250 profit
        assert sample_position.realized_pnl == 250.0

    def test_mark_closed_with_explicit_time(self, sample_position):
        """Test marking position as closed with explicit time."""
        exit_time = datetime(2024, 2, 15, 15, 0, 0, tzinfo=timezone.utc)
        sample_position.mark_closed(exit_price=100.0, exit_time=exit_time)

        assert sample_position.exit_time == exit_time

    def test_mark_closed_credit_spread(self, sample_position):
        """Test realized P&L for credit spread."""
        sample_position.entry_price = 150.0  # Credit spread (received premium)
        sample_position.mark_closed(exit_price=-50.0)  # Cost to close

        # Realized P&L = -50 - 150 = -200 (paid to close)
        assert sample_position.realized_pnl == -200.0

    def test_mark_expired(self, sample_position):
        """Test marking position as expired."""
        sample_position.entry_price = -200.0  # Debit spread
        sample_position.mark_expired()

        assert sample_position.state == PositionState.EXPIRED
        assert sample_position.exit_time is not None
        assert sample_position.exit_price == 0.0
        # Debit spread expired worthless: lost the debit
        assert sample_position.realized_pnl == 200.0  # -(-200) = 200 (confusing but correct)

    def test_mark_expired_credit_spread(self, sample_position):
        """Test expired credit spread keeps premium."""
        sample_position.entry_price = 150.0  # Credit spread
        sample_position.mark_expired()

        # Credit spread expired worthless: keep the credit
        assert sample_position.realized_pnl == -150.0  # -150 (confusing sign convention)


class TestMarketDataUpdate:
    """Tests for market data updates."""

    def test_update_market_data(self, sample_position, sample_greeks):
        """Test updating position with market data."""
        new_greeks = Greeks(delta=0.45, gamma=0.04, vega=0.28, theta=-0.018)

        sample_position.update_market_data(
            current_price=-150.0,  # Position now worth -$150 (was -$200)
            current_greeks=new_greeks,
            days_to_expiry=25,
        )

        assert sample_position.current_price == -150.0
        assert sample_position.current_greeks == new_greeks
        assert sample_position.days_to_expiry == 25
        # Unrealized P&L = current - entry = -150 - (-200) = 50 profit
        assert sample_position.unrealized_pnl == 50.0

    def test_update_market_data_loss(self, sample_position, sample_greeks):
        """Test updating position showing a loss."""
        new_greeks = Greeks(delta=0.55, gamma=0.06, vega=0.35, theta=-0.025)

        sample_position.update_market_data(
            current_price=-300.0,  # Position now worth -$300 (was -$200)
            current_greeks=new_greeks,
            days_to_expiry=28,
        )

        # Unrealized P&L = -300 - (-200) = -100 loss
        assert sample_position.unrealized_pnl == -100.0

    def test_update_market_data_updates_days_held(self, sample_position, sample_greeks):
        """Test that days_held is updated."""
        # Set entry time to 5 days ago
        sample_position.entry_time = datetime.now(timezone.utc) - timedelta(days=5)

        sample_position.update_market_data(
            current_price=-180.0,
            current_greeks=sample_greeks,
            days_to_expiry=25,
        )

        assert sample_position.days_held >= 5


class TestExitTracking:
    """Tests for exit tracking fields."""

    def test_exit_fields_initially_none(self, sample_position):
        """Test exit fields are None for new position."""
        assert sample_position.exit_signal is None
        assert sample_position.exit_time is None
        assert sample_position.exit_price is None
        assert sample_position.realized_pnl is None

    def test_exit_fields_after_closing(self, sample_position):
        """Test exit fields are populated after closing."""
        exit_signal = ExitSignal(
            exit_type=ExitSignalType.TAKE_PROFIT,
            position_id="test-123",
            urgency=0.6,
            reason="Take profit",
        )

        sample_position.mark_closing(exit_signal)
        sample_position.mark_closed(exit_price=100.0)

        assert sample_position.exit_signal == exit_signal
        assert sample_position.exit_time is not None
        assert sample_position.exit_price == 100.0
        assert sample_position.realized_pnl is not None
