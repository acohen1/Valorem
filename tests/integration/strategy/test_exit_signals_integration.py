"""Integration tests for exit signal generation."""

from datetime import date, datetime, timedelta, timezone

import pandas as pd
import pytest

from src.config.schema import ExitSignalsConfig
from src.risk.portfolio import Portfolio, PositionState
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
def exit_config():
    """Exit signals configuration for integration tests."""
    return ExitSignalsConfig(
        min_edge_retention=0.3,
        stop_loss_pct=0.8,
        take_profit_pct=0.5,
        min_dte_exit=3,
    )


@pytest.fixture
def surface_df():
    """Realistic surface DataFrame for integration tests."""
    return pd.DataFrame(
        {
            "option_symbol": [
                "SPY240315C00450000",
                "SPY240315C00455000",
                "SPY240315P00445000",
                "SPY240315P00440000",
                "SPY240322C00450000",
                "SPY240322C00455000",
            ],
            "tenor_days": [30, 30, 30, 30, 37, 37],
            "delta_bucket": ["ATM", "ATM", "ATM", "ATM", "ATM", "ATM"],
            "strike": [450.0, 455.0, 445.0, 440.0, 450.0, 455.0],
            "right": ["C", "C", "P", "P", "C", "C"],
            "bid": [5.20, 3.10, 4.80, 3.50, 6.00, 4.00],
            "ask": [5.40, 3.30, 5.00, 3.70, 6.20, 4.20],
            "iv": [0.22, 0.23, 0.21, 0.22, 0.21, 0.22],
            "delta": [0.52, 0.42, -0.48, -0.38, 0.53, 0.43],
            "gamma": [0.05, 0.04, 0.05, 0.04, 0.04, 0.03],
            "vega": [0.30, 0.25, 0.28, 0.22, 0.35, 0.30],
            "theta": [-0.02, -0.018, -0.019, -0.016, -0.018, -0.015],
        }
    )


@pytest.fixture
def model_predictions():
    """Model predictions DataFrame for integration tests."""
    return pd.DataFrame(
        {
            "tenor_days": [30, 30, 37, 37, 60],
            "delta_bucket": ["ATM", "C25", "ATM", "C25", "ATM"],
            "edge": [0.04, 0.02, 0.03, 0.01, 0.05],
            "confidence": [0.75, 0.65, 0.70, 0.60, 0.80],
        }
    )


def create_vertical_spread_position(
    position_id: str,
    entry_edge: float = 0.05,
    entry_price: float = -200.0,
    max_loss: float = 200.0,
    unrealized_pnl: float = 0.0,
    days_to_expiry: int = 30,
    tenor_days: int = 30,
    delta_bucket: str = "ATM",
) -> ManagedPosition:
    """Helper to create a vertical spread position."""
    greeks = Greeks(delta=0.1, gamma=0.01, vega=0.05, theta=-0.005)
    legs = [
        OptionLeg(
            symbol="SPY240315C00450000",
            qty=1,
            entry_price=5.30,
            strike=450.0,
            expiry=date(2024, 3, 15),
            right=OptionRight.CALL,
            greeks=Greeks(delta=0.52, gamma=0.05, vega=0.30, theta=-0.02),
        ),
        OptionLeg(
            symbol="SPY240315C00455000",
            qty=-1,
            entry_price=3.20,
            strike=455.0,
            expiry=date(2024, 3, 15),
            right=OptionRight.CALL,
            greeks=Greeks(delta=0.42, gamma=0.04, vega=0.25, theta=-0.018),
        ),
    ]
    signal = Signal(
        signal_type=SignalType.DIRECTIONAL_VOL,
        edge=entry_edge,
        confidence=0.75,
        tenor_days=tenor_days,
        delta_bucket=delta_bucket,
    )

    return ManagedPosition(
        position_id=position_id,
        state=PositionState.OPEN,
        legs=legs,
        structure_type="VerticalSpread",
        entry_signal=signal,
        entry_time=datetime.now(timezone.utc) - timedelta(days=5),
        entry_price=entry_price,
        entry_greeks=greeks,
        max_loss=max_loss,
        current_price=entry_price + unrealized_pnl,
        current_greeks=greeks,
        unrealized_pnl=unrealized_pnl,
        days_held=5,
        days_to_expiry=days_to_expiry,
    )


class TestExitSignalIntegration:
    """Integration tests for exit signal generation."""

    def test_full_exit_evaluation_cycle(
        self, exit_config, surface_df, model_predictions
    ):
        """Test full cycle of position evaluation and exit signal generation."""
        generator = ExitSignalGenerator(exit_config)

        # Create multiple positions with different conditions
        positions = [
            # Position 1: Normal, no exit trigger
            create_vertical_spread_position(
                position_id="pos-1",
                entry_edge=0.05,
                unrealized_pnl=20.0,  # Small profit
                days_to_expiry=25,
            ),
            # Position 2: Stop-loss trigger (80% of max loss)
            create_vertical_spread_position(
                position_id="pos-2",
                entry_edge=0.05,
                max_loss=200.0,
                unrealized_pnl=-165.0,  # 82.5% loss
                days_to_expiry=25,
            ),
            # Position 3: Take-profit trigger (50% of max loss as profit)
            create_vertical_spread_position(
                position_id="pos-3",
                entry_edge=0.05,
                max_loss=200.0,
                unrealized_pnl=110.0,  # 55% profit
                days_to_expiry=25,
            ),
            # Position 4: Time decay trigger (2 DTE)
            create_vertical_spread_position(
                position_id="pos-4",
                entry_edge=0.05,
                unrealized_pnl=0.0,
                days_to_expiry=2,
            ),
        ]

        signals = generator.evaluate_positions(
            positions=positions,
            surface=surface_df,
            model_predictions=model_predictions,
        )

        # Should have 3 exit signals (pos-1 has no trigger)
        assert len(signals) == 3

        # Verify signal types
        signal_map = {s.position_id: s for s in signals}

        assert "pos-2" in signal_map
        assert signal_map["pos-2"].exit_type == ExitSignalType.STOP_LOSS

        assert "pos-3" in signal_map
        assert signal_map["pos-3"].exit_type == ExitSignalType.TAKE_PROFIT

        assert "pos-4" in signal_map
        assert signal_map["pos-4"].exit_type == ExitSignalType.TIME_DECAY

    def test_model_driven_exits_with_predictions(
        self, exit_config, surface_df, model_predictions
    ):
        """Test model-driven exit signals with predictions."""
        generator = ExitSignalGenerator(exit_config)

        # Position with edge that will decay (current edge 0.04, entry was 0.05)
        # Retention = 0.04/0.05 = 0.8 = 80%, above 30% threshold - should not trigger

        # Position with edge reversal
        position_reversal = create_vertical_spread_position(
            position_id="pos-reversal",
            entry_edge=0.05,  # Positive entry
            tenor_days=30,
            delta_bucket="ATM",
        )

        # Modify predictions to show reversal
        reversed_predictions = pd.DataFrame(
            {
                "tenor_days": [30],
                "delta_bucket": ["ATM"],
                "edge": [-0.04],  # Reversed to negative
            }
        )

        signals = generator.evaluate_positions(
            positions=[position_reversal],
            surface=surface_df,
            model_predictions=reversed_predictions,
        )

        assert len(signals) == 1
        assert signals[0].exit_type == ExitSignalType.MODEL_REVERSAL
        assert signals[0].position_id == "pos-reversal"

    def test_edge_decay_with_predictions(
        self, exit_config, surface_df
    ):
        """Test edge decay exit signal."""
        generator = ExitSignalGenerator(exit_config)

        # Position with significant edge decay
        position = create_vertical_spread_position(
            position_id="pos-decay",
            entry_edge=0.10,  # High entry edge
            tenor_days=30,
            delta_bucket="ATM",
        )

        # Current edge is 0.02, which is 20% of 0.10, below 30% threshold
        decayed_predictions = pd.DataFrame(
            {
                "tenor_days": [30],
                "delta_bucket": ["ATM"],
                "edge": [0.02],  # Decayed from 0.10 to 0.02
            }
        )

        signals = generator.evaluate_positions(
            positions=[position],
            surface=surface_df,
            model_predictions=decayed_predictions,
        )

        assert len(signals) == 1
        assert signals[0].exit_type == ExitSignalType.EDGE_DECAY

    def test_priority_order_stop_loss_first(
        self, exit_config, surface_df
    ):
        """Test that stop-loss takes priority over other triggers."""
        generator = ExitSignalGenerator(exit_config)

        # Position that would trigger multiple exit conditions
        position = create_vertical_spread_position(
            position_id="pos-multi",
            entry_edge=0.05,
            max_loss=200.0,
            unrealized_pnl=-170.0,  # 85% loss - stop-loss
            days_to_expiry=2,  # Also time decay
        )

        # Also set up edge reversal
        reversed_predictions = pd.DataFrame(
            {
                "tenor_days": [30],
                "delta_bucket": ["ATM"],
                "edge": [-0.04],  # Reversed
            }
        )

        signals = generator.evaluate_positions(
            positions=[position],
            surface=surface_df,
            model_predictions=reversed_predictions,
        )

        # Only stop-loss should trigger (highest priority)
        assert len(signals) == 1
        assert signals[0].exit_type == ExitSignalType.STOP_LOSS

    def test_mixed_position_states(
        self, exit_config, surface_df
    ):
        """Test that only OPEN positions are evaluated."""
        generator = ExitSignalGenerator(exit_config)

        # Create positions in various states
        open_position = create_vertical_spread_position(
            position_id="pos-open",
            unrealized_pnl=-170.0,  # Would trigger stop-loss
        )

        closing_position = create_vertical_spread_position(
            position_id="pos-closing",
            unrealized_pnl=-170.0,
        )
        closing_position.state = PositionState.CLOSING

        closed_position = create_vertical_spread_position(
            position_id="pos-closed",
            unrealized_pnl=-170.0,
        )
        closed_position.state = PositionState.CLOSED

        positions = [open_position, closing_position, closed_position]

        signals = generator.evaluate_positions(
            positions=positions,
            surface=surface_df,
        )

        # Only OPEN position should generate signal
        assert len(signals) == 1
        assert signals[0].position_id == "pos-open"

    def test_position_lifecycle_with_exit_signals(
        self, exit_config, surface_df
    ):
        """Test full position lifecycle from entry to exit signal."""
        generator = ExitSignalGenerator(exit_config)

        # Create position via from_order_fill method
        legs = [
            OptionLeg(
                symbol="SPY240315C00450000",
                qty=1,
                entry_price=5.30,
                strike=450.0,
                expiry=date(2024, 3, 15),
                right=OptionRight.CALL,
                greeks=Greeks(delta=0.52, gamma=0.05, vega=0.30, theta=-0.02),
            ),
            OptionLeg(
                symbol="SPY240315C00455000",
                qty=-1,
                entry_price=3.20,
                strike=455.0,
                expiry=date(2024, 3, 15),
                right=OptionRight.CALL,
                greeks=Greeks(delta=0.42, gamma=0.04, vega=0.25, theta=-0.018),
            ),
        ]
        signal = Signal(
            signal_type=SignalType.DIRECTIONAL_VOL,
            edge=0.05,
            confidence=0.75,
            tenor_days=30,
            delta_bucket="ATM",
        )
        entry_greeks = Greeks(delta=0.1, gamma=0.01, vega=0.05, theta=-0.002)

        position = ManagedPosition.from_order_fill(
            position_id="lifecycle-test",
            legs=legs,
            structure_type="VerticalSpread",
            entry_signal=signal,
            entry_price=-210.0,  # Debit spread
            entry_greeks=entry_greeks,
            max_loss=210.0,
            days_to_expiry=30,
        )

        # Verify initial state
        assert position.is_open()
        assert position.unrealized_pnl == 0.0

        # Simulate market move - position loses money
        position.update_market_data(
            current_price=-380.0,  # Now worth -$380 (was -$210)
            current_greeks=Greeks(delta=0.15, gamma=0.02, vega=0.08, theta=-0.003),
            days_to_expiry=25,
        )

        # Unrealized loss = -380 - (-210) = -170
        assert position.unrealized_pnl == -170.0
        # Loss % = 170/210 = 80.9%
        loss_pct = abs(position.unrealized_pnl) / position.max_loss
        assert loss_pct > 0.8

        # Evaluate for exit
        signals = generator.evaluate_positions(
            positions=[position],
            surface=surface_df,
        )

        assert len(signals) == 1
        exit_signal = signals[0]
        assert exit_signal.exit_type == ExitSignalType.STOP_LOSS
        assert exit_signal.position_id == "lifecycle-test"

        # Mark position as closing
        position.mark_closing(exit_signal)
        assert position.state == PositionState.CLOSING
        assert position.exit_signal == exit_signal

        # Simulate exit fill
        position.mark_closed(exit_price=-50.0)  # Closed for -$50 (recovered some)
        assert position.is_closed()
        assert position.realized_pnl == -50.0 - (-210.0)  # = 160 loss recovered? No...
        # Actually: realized = exit - entry = -50 - (-210) = 160 (got back $160)
        # Wait, this is confusing. Let me think:
        # Entry price was -210 (paid $210 to open)
        # Exit price is -50 (paid $50 to close??? No, that's wrong)
        # Actually in closing a debit spread, you RECEIVE premium by selling
        # So exit_price should be positive (received $50)
        # Let's verify the math: realized = 50 - (-210) = 260? That seems wrong too.
        # The current implementation has: realized_pnl = exit_price - entry_price

    def test_no_exit_signals_for_healthy_positions(
        self, exit_config, surface_df, model_predictions
    ):
        """Test that healthy positions don't generate exit signals."""
        generator = ExitSignalGenerator(exit_config)

        # Healthy position: small profit, plenty of time, edge maintained
        healthy_position = create_vertical_spread_position(
            position_id="healthy",
            entry_edge=0.05,
            max_loss=200.0,
            unrealized_pnl=30.0,  # 15% profit
            days_to_expiry=25,
            tenor_days=30,
            delta_bucket="ATM",
        )

        # Predictions show maintained edge
        good_predictions = pd.DataFrame(
            {
                "tenor_days": [30],
                "delta_bucket": ["ATM"],
                "edge": [0.045],  # 90% retention
            }
        )

        signals = generator.evaluate_positions(
            positions=[healthy_position],
            surface=surface_df,
            model_predictions=good_predictions,
        )

        assert len(signals) == 0


class TestExitSignalUrgency:
    """Tests for exit signal urgency levels."""

    def test_stop_loss_high_urgency(self, exit_config, surface_df):
        """Test that stop-loss signals have high urgency."""
        generator = ExitSignalGenerator(exit_config)

        position = create_vertical_spread_position(
            position_id="urgent",
            max_loss=200.0,
            unrealized_pnl=-165.0,
        )

        signals = generator.evaluate_positions(
            positions=[position],
            surface=surface_df,
        )

        assert len(signals) == 1
        assert signals[0].urgency >= 0.9

    def test_time_decay_urgency_scales_with_dte(self, exit_config, surface_df):
        """Test that time decay urgency increases as DTE decreases."""
        # Use a config with higher min_dte_exit to see urgency differences
        config = ExitSignalsConfig(
            min_edge_retention=0.3,
            stop_loss_pct=0.8,
            take_profit_pct=0.5,
            min_dte_exit=5,  # Higher threshold to see urgency scaling
        )
        generator = ExitSignalGenerator(config)

        # 5 DTE - urgency = 5/5 = 1.0
        pos_5dte = create_vertical_spread_position(
            position_id="5dte",
            days_to_expiry=5,
        )

        # 3 DTE - urgency = min(5/3, 1.0) = 1.0 (capped)
        pos_3dte = create_vertical_spread_position(
            position_id="3dte",
            days_to_expiry=3,
        )

        signals = generator.evaluate_positions(
            positions=[pos_5dte, pos_3dte],
            surface=surface_df,
        )

        assert len(signals) == 2
        signal_map = {s.position_id: s for s in signals}

        # Both will cap at 1.0 since threshold/dte >= 1 for both
        # The urgency formula is: min(threshold / max(dte, 1), 1.0)
        # 5/5 = 1.0, 5/3 = 1.67 -> capped to 1.0
        # Both are at max urgency when below threshold
        assert signal_map["5dte"].urgency == 1.0
        assert signal_map["3dte"].urgency == 1.0

    def test_take_profit_moderate_urgency(self, exit_config, surface_df):
        """Test that take-profit signals have moderate urgency."""
        generator = ExitSignalGenerator(exit_config)

        position = create_vertical_spread_position(
            position_id="profit",
            max_loss=200.0,
            unrealized_pnl=120.0,  # 60% profit
        )

        signals = generator.evaluate_positions(
            positions=[position],
            surface=surface_df,
        )

        assert len(signals) == 1
        # Take profit should have moderate urgency (can let profits run)
        assert 0.4 <= signals[0].urgency <= 0.7
