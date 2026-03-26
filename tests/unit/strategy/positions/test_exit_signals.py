"""Unit tests for exit signal generation."""

from datetime import date, datetime, timezone

import pandas as pd
import pytest

from src.config.schema import ExitSignalsConfig
from src.risk.portfolio import PositionState
from src.strategy.positions.exit_signals import ExitSignalGenerator
from src.strategy.positions.lifecycle import ManagedPosition
from src.strategy.types import (
    ExitSignalType,
    Greeks,
    OptionLeg,
    OptionRight,
    Signal,
    SignalType,
)


@pytest.fixture
def default_config():
    """Default exit signals configuration."""
    return ExitSignalsConfig(
        min_edge_retention=0.3,
        stop_loss_pct=0.8,
        take_profit_pct=0.5,
        min_dte_exit=3,
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
        entry_price=-200.0,
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
        {
            "option_symbol": ["SPY240315C00450000", "SPY240315C00455000"],
            "tenor_days": [30, 30],
            "delta_bucket": ["ATM", "ATM"],
            "bid": [4.80, 2.90],
            "ask": [5.00, 3.10],
        }
    )


@pytest.fixture
def sample_predictions():
    """Sample model predictions DataFrame."""
    return pd.DataFrame(
        {
            "tenor_days": [30, 30, 60],
            "delta_bucket": ["ATM", "C25", "ATM"],
            "edge": [0.04, 0.02, 0.03],
        }
    )


class TestExitSignalGeneratorInit:
    """Tests for ExitSignalGenerator initialization."""

    def test_init(self, default_config):
        """Test basic initialization."""
        generator = ExitSignalGenerator(default_config)
        assert generator._config == default_config


class TestStopLossCheck:
    """Tests for stop-loss exit trigger."""

    def test_stop_loss_triggered(self, default_config, sample_position, sample_surface):
        """Test stop-loss triggers when loss exceeds threshold."""
        generator = ExitSignalGenerator(default_config)

        # Position lost 80% of max loss
        sample_position.unrealized_pnl = -160.0  # -$160 of $200 max loss = 80%

        signals = generator.evaluate_positions(
            positions=[sample_position],
            surface=sample_surface,
        )

        assert len(signals) == 1
        assert signals[0].exit_type == ExitSignalType.STOP_LOSS
        assert signals[0].position_id == "test-123"
        assert signals[0].urgency == 0.95
        assert "80.0%" in signals[0].reason

    def test_stop_loss_not_triggered_below_threshold(
        self, default_config, sample_position, sample_surface
    ):
        """Test stop-loss does not trigger below threshold."""
        generator = ExitSignalGenerator(default_config)

        # Position lost 50% of max loss
        sample_position.unrealized_pnl = -100.0

        signals = generator.evaluate_positions(
            positions=[sample_position],
            surface=sample_surface,
        )

        assert len(signals) == 0

    def test_stop_loss_not_triggered_when_profitable(
        self, default_config, sample_position, sample_surface
    ):
        """Test stop-loss does not trigger when position is profitable."""
        generator = ExitSignalGenerator(default_config)

        sample_position.unrealized_pnl = 50.0  # Profitable

        signals = generator.evaluate_positions(
            positions=[sample_position],
            surface=sample_surface,
        )

        assert len(signals) == 0


class TestModelReversalCheck:
    """Tests for model reversal exit trigger."""

    def test_model_reversal_positive_to_negative(
        self, default_config, sample_position, sample_surface
    ):
        """Test reversal detection when edge flips from positive to negative."""
        generator = ExitSignalGenerator(default_config)

        # Entry edge was +0.05, now -0.04
        sample_position.entry_signal = Signal(
            signal_type=SignalType.DIRECTIONAL_VOL,
            edge=0.05,
            confidence=0.75,
            tenor_days=30,
            delta_bucket="ATM",
        )

        predictions = pd.DataFrame(
            {
                "tenor_days": [30],
                "delta_bucket": ["ATM"],
                "edge": [-0.04],
            }
        )

        signals = generator.evaluate_positions(
            positions=[sample_position],
            surface=sample_surface,
            model_predictions=predictions,
        )

        assert len(signals) == 1
        assert signals[0].exit_type == ExitSignalType.MODEL_REVERSAL
        assert "reversed" in signals[0].reason.lower()

    def test_model_reversal_negative_to_positive(
        self, default_config, sample_position, sample_surface
    ):
        """Test reversal detection when edge flips from negative to positive."""
        generator = ExitSignalGenerator(default_config)

        # Entry edge was -0.05, now +0.04
        sample_position.entry_signal = Signal(
            signal_type=SignalType.DIRECTIONAL_VOL,
            edge=-0.05,
            confidence=0.75,
            tenor_days=30,
            delta_bucket="ATM",
        )

        predictions = pd.DataFrame(
            {
                "tenor_days": [30],
                "delta_bucket": ["ATM"],
                "edge": [0.04],
            }
        )

        signals = generator.evaluate_positions(
            positions=[sample_position],
            surface=sample_surface,
            model_predictions=predictions,
        )

        assert len(signals) == 1
        assert signals[0].exit_type == ExitSignalType.MODEL_REVERSAL

    def test_model_reversal_not_significant(
        self, default_config, sample_position, sample_surface
    ):
        """Test no MODEL_REVERSAL when reversal magnitude is too small."""
        generator = ExitSignalGenerator(default_config)

        # Entry edge was +0.05, now -0.01 (not significant reversal)
        # Minimum reversal is 50% of entry = 0.025, so -0.01 is too small
        sample_position.entry_signal = Signal(
            signal_type=SignalType.DIRECTIONAL_VOL,
            edge=0.05,
            confidence=0.75,
            tenor_days=30,
            delta_bucket="ATM",
        )

        predictions = pd.DataFrame(
            {
                "tenor_days": [30],
                "delta_bucket": ["ATM"],
                "edge": [-0.01],  # Too small for reversal
            }
        )

        signals = generator.evaluate_positions(
            positions=[sample_position],
            surface=sample_surface,
            model_predictions=predictions,
        )

        # MODEL_REVERSAL should not trigger (magnitude too small)
        # But EDGE_DECAY will trigger (0.01 is 20% of 0.05, below 30% retention)
        assert len(signals) == 1
        assert signals[0].exit_type == ExitSignalType.EDGE_DECAY

    def test_no_reversal_same_sign(
        self, default_config, sample_position, sample_surface
    ):
        """Test no reversal when edges have same sign."""
        generator = ExitSignalGenerator(default_config)

        predictions = pd.DataFrame(
            {
                "tenor_days": [30],
                "delta_bucket": ["ATM"],
                "edge": [0.03],  # Same sign as entry
            }
        )

        signals = generator.evaluate_positions(
            positions=[sample_position],
            surface=sample_surface,
            model_predictions=predictions,
        )

        assert len(signals) == 0


class TestEdgeDecayCheck:
    """Tests for edge decay exit trigger."""

    def test_edge_decay_triggered(
        self, default_config, sample_position, sample_surface
    ):
        """Test edge decay triggers when edge falls below retention threshold."""
        generator = ExitSignalGenerator(default_config)

        # Entry edge was 0.05, retention threshold is 0.3 (30%)
        # So edge needs to fall below 0.05 * 0.3 = 0.015
        predictions = pd.DataFrame(
            {
                "tenor_days": [30],
                "delta_bucket": ["ATM"],
                "edge": [0.01],  # Only 20% retention
            }
        )

        signals = generator.evaluate_positions(
            positions=[sample_position],
            surface=sample_surface,
            model_predictions=predictions,
        )

        assert len(signals) == 1
        assert signals[0].exit_type == ExitSignalType.EDGE_DECAY
        assert "decayed" in signals[0].reason.lower()

    def test_edge_decay_not_triggered_above_threshold(
        self, default_config, sample_position, sample_surface
    ):
        """Test edge decay does not trigger above retention threshold."""
        generator = ExitSignalGenerator(default_config)

        # Entry edge was 0.05, 50% retention = 0.025
        predictions = pd.DataFrame(
            {
                "tenor_days": [30],
                "delta_bucket": ["ATM"],
                "edge": [0.03],  # 60% retention
            }
        )

        signals = generator.evaluate_positions(
            positions=[sample_position],
            surface=sample_surface,
            model_predictions=predictions,
        )

        assert len(signals) == 0


class TestTakeProfitCheck:
    """Tests for take-profit exit trigger."""

    def test_take_profit_triggered(
        self, default_config, sample_position, sample_surface
    ):
        """Test take-profit triggers when profit exceeds threshold."""
        generator = ExitSignalGenerator(default_config)

        # Position made 50% of max loss in profit
        sample_position.unrealized_pnl = 100.0  # $100 profit on $200 max loss

        signals = generator.evaluate_positions(
            positions=[sample_position],
            surface=sample_surface,
        )

        assert len(signals) == 1
        assert signals[0].exit_type == ExitSignalType.TAKE_PROFIT
        assert "50.0%" in signals[0].reason

    def test_take_profit_not_triggered_below_threshold(
        self, default_config, sample_position, sample_surface
    ):
        """Test take-profit does not trigger below threshold."""
        generator = ExitSignalGenerator(default_config)

        # Position made only 30% profit
        sample_position.unrealized_pnl = 60.0

        signals = generator.evaluate_positions(
            positions=[sample_position],
            surface=sample_surface,
        )

        assert len(signals) == 0


class TestTimeDecayCheck:
    """Tests for time decay exit trigger."""

    def test_time_decay_triggered(
        self, default_config, sample_position, sample_surface
    ):
        """Test time decay triggers when DTE below threshold."""
        generator = ExitSignalGenerator(default_config)

        sample_position.days_to_expiry = 2  # Below 3 day threshold

        signals = generator.evaluate_positions(
            positions=[sample_position],
            surface=sample_surface,
        )

        assert len(signals) == 1
        assert signals[0].exit_type == ExitSignalType.TIME_DECAY
        assert "2 DTE" in signals[0].reason

    def test_time_decay_at_threshold(
        self, default_config, sample_position, sample_surface
    ):
        """Test time decay triggers at threshold."""
        generator = ExitSignalGenerator(default_config)

        sample_position.days_to_expiry = 3  # Exactly at threshold

        signals = generator.evaluate_positions(
            positions=[sample_position],
            surface=sample_surface,
        )

        assert len(signals) == 1
        assert signals[0].exit_type == ExitSignalType.TIME_DECAY

    def test_time_decay_not_triggered_above_threshold(
        self, default_config, sample_position, sample_surface
    ):
        """Test time decay does not trigger above threshold."""
        generator = ExitSignalGenerator(default_config)

        sample_position.days_to_expiry = 10  # Well above threshold

        signals = generator.evaluate_positions(
            positions=[sample_position],
            surface=sample_surface,
        )

        assert len(signals) == 0


class TestExitPriority:
    """Tests for exit signal priority ordering."""

    def test_stop_loss_has_highest_priority(
        self, default_config, sample_position, sample_surface
    ):
        """Test stop-loss takes priority over other triggers."""
        generator = ExitSignalGenerator(default_config)

        # Multiple triggers active
        sample_position.unrealized_pnl = -160.0  # Stop-loss triggered
        sample_position.days_to_expiry = 2  # Time decay also triggered

        # Also add model reversal
        predictions = pd.DataFrame(
            {
                "tenor_days": [30],
                "delta_bucket": ["ATM"],
                "edge": [-0.04],  # Reversed
            }
        )

        signals = generator.evaluate_positions(
            positions=[sample_position],
            surface=sample_surface,
            model_predictions=predictions,
        )

        assert len(signals) == 1
        assert signals[0].exit_type == ExitSignalType.STOP_LOSS


class TestMultiplePositions:
    """Tests for evaluating multiple positions."""

    def test_evaluate_multiple_positions(
        self, default_config, sample_legs, sample_greeks, sample_signal, sample_surface
    ):
        """Test evaluating multiple positions."""
        generator = ExitSignalGenerator(default_config)

        pos1 = ManagedPosition(
            position_id="pos-1",
            state=PositionState.OPEN,
            legs=sample_legs,
            structure_type="VerticalSpread",
            entry_signal=sample_signal,
            entry_time=datetime.now(timezone.utc),
            entry_price=-200.0,
            entry_greeks=sample_greeks,
            max_loss=200.0,
            unrealized_pnl=-160.0,  # Stop-loss triggered
            days_to_expiry=30,
        )

        pos2 = ManagedPosition(
            position_id="pos-2",
            state=PositionState.OPEN,
            legs=sample_legs,
            structure_type="CalendarSpread",
            entry_signal=sample_signal,
            entry_time=datetime.now(timezone.utc),
            entry_price=-150.0,
            entry_greeks=sample_greeks,
            max_loss=150.0,
            unrealized_pnl=0.0,  # No trigger
            days_to_expiry=30,
        )

        pos3 = ManagedPosition(
            position_id="pos-3",
            state=PositionState.OPEN,
            legs=sample_legs,
            structure_type="VerticalSpread",
            entry_signal=sample_signal,
            entry_time=datetime.now(timezone.utc),
            entry_price=-100.0,
            entry_greeks=sample_greeks,
            max_loss=100.0,
            unrealized_pnl=0.0,
            days_to_expiry=2,  # Time decay triggered
        )

        signals = generator.evaluate_positions(
            positions=[pos1, pos2, pos3],
            surface=sample_surface,
        )

        assert len(signals) == 2
        signal_types = {s.exit_type for s in signals}
        assert ExitSignalType.STOP_LOSS in signal_types
        assert ExitSignalType.TIME_DECAY in signal_types

    def test_skip_non_open_positions(
        self, default_config, sample_legs, sample_greeks, sample_signal, sample_surface
    ):
        """Test that non-OPEN positions are skipped."""
        generator = ExitSignalGenerator(default_config)

        pos1 = ManagedPosition(
            position_id="pos-1",
            state=PositionState.CLOSING,  # Not OPEN
            legs=sample_legs,
            structure_type="VerticalSpread",
            entry_signal=sample_signal,
            entry_time=datetime.now(timezone.utc),
            entry_price=-200.0,
            entry_greeks=sample_greeks,
            max_loss=200.0,
            unrealized_pnl=-160.0,  # Would trigger stop-loss if OPEN
            days_to_expiry=30,
        )

        signals = generator.evaluate_positions(
            positions=[pos1],
            surface=sample_surface,
        )

        assert len(signals) == 0


class TestManualExit:
    """Tests for manual exit signal creation."""

    def test_create_manual_exit(self, default_config):
        """Test creating a manual exit signal."""
        generator = ExitSignalGenerator(default_config)

        signal = generator.create_manual_exit(
            position_id="test-123",
            reason="User requested exit",
        )

        assert signal.exit_type == ExitSignalType.MANUAL
        assert signal.position_id == "test-123"
        assert signal.urgency == 1.0
        assert "User requested exit" in signal.reason

    def test_create_manual_exit_default_reason(self, default_config):
        """Test creating a manual exit with default reason."""
        generator = ExitSignalGenerator(default_config)

        signal = generator.create_manual_exit(position_id="test-456")

        assert "Manual exit requested" in signal.reason


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_no_predictions_skips_model_checks(
        self, default_config, sample_position, sample_surface
    ):
        """Test that model checks are skipped when no predictions provided."""
        generator = ExitSignalGenerator(default_config)

        # Entry edge setup for reversal
        sample_position.entry_signal = Signal(
            signal_type=SignalType.DIRECTIONAL_VOL,
            edge=0.05,
            confidence=0.75,
            tenor_days=30,
            delta_bucket="ATM",
        )

        # No predictions provided - would normally trigger reversal
        signals = generator.evaluate_positions(
            positions=[sample_position],
            surface=sample_surface,
            model_predictions=None,  # No predictions
        )

        # No model-based signals should be generated
        assert len(signals) == 0

    def test_empty_predictions_dataframe(
        self, default_config, sample_position, sample_surface
    ):
        """Test handling of empty predictions DataFrame."""
        generator = ExitSignalGenerator(default_config)

        empty_predictions = pd.DataFrame(columns=["tenor_days", "delta_bucket", "edge"])

        signals = generator.evaluate_positions(
            positions=[sample_position],
            surface=sample_surface,
            model_predictions=empty_predictions,
        )

        assert len(signals) == 0

    def test_missing_node_in_predictions(
        self, default_config, sample_position, sample_surface
    ):
        """Test handling when position's node is not in predictions."""
        generator = ExitSignalGenerator(default_config)

        # Predictions for different node
        predictions = pd.DataFrame(
            {
                "tenor_days": [60],  # Different tenor
                "delta_bucket": ["C25"],  # Different bucket
                "edge": [-0.04],
            }
        )

        signals = generator.evaluate_positions(
            positions=[sample_position],
            surface=sample_surface,
            model_predictions=predictions,
        )

        # No model-based signals since node not found
        assert len(signals) == 0

    def test_zero_max_loss_position(
        self, default_config, sample_legs, sample_greeks, sample_signal, sample_surface
    ):
        """Test handling of position with zero max_loss."""
        generator = ExitSignalGenerator(default_config)

        position = ManagedPosition(
            position_id="test-zero",
            state=PositionState.OPEN,
            legs=sample_legs,
            structure_type="VerticalSpread",
            entry_signal=sample_signal,
            entry_time=datetime.now(timezone.utc),
            entry_price=-200.0,
            entry_greeks=sample_greeks,
            max_loss=0.0,  # Zero max loss
            unrealized_pnl=-100.0,
            days_to_expiry=30,
        )

        signals = generator.evaluate_positions(
            positions=[position],
            surface=sample_surface,
        )

        # No stop-loss or take-profit should trigger
        assert len(signals) == 0

    def test_zero_entry_edge(
        self, default_config, sample_legs, sample_greeks, sample_surface
    ):
        """Test handling of position with zero entry edge."""
        generator = ExitSignalGenerator(default_config)

        signal = Signal(
            signal_type=SignalType.DIRECTIONAL_VOL,
            edge=0.0,  # Zero edge
            confidence=0.75,
            tenor_days=30,
            delta_bucket="ATM",
        )

        position = ManagedPosition(
            position_id="test-zero-edge",
            state=PositionState.OPEN,
            legs=sample_legs,
            structure_type="VerticalSpread",
            entry_signal=signal,
            entry_time=datetime.now(timezone.utc),
            entry_price=-200.0,
            entry_greeks=sample_greeks,
            max_loss=200.0,
            unrealized_pnl=0.0,
            days_to_expiry=30,
        )

        predictions = pd.DataFrame(
            {
                "tenor_days": [30],
                "delta_bucket": ["ATM"],
                "edge": [0.01],
            }
        )

        # Should not crash on division by zero
        signals = generator.evaluate_positions(
            positions=[position],
            surface=sample_surface,
            model_predictions=predictions,
        )

        # Edge decay check should be skipped for zero entry edge
        assert len(signals) == 0
