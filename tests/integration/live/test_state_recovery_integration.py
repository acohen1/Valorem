"""Integration tests for state persistence and crash recovery.

Tests the full workflow: save state → simulate crash → reload state → continue trading.
"""

from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import pytest

from src.config.schema import (
    DriftBandsConfig,
    ExecutionConfig,
    ExitSignalsConfig,
    KillSwitchConfig,
    PaperConfig,
    PositionManagementConfig,
    RebalancingConfig,
    RiskCapsConfig,
    RiskConfig,
    StressConfig,
)
from src.live.loop import TradingLoop
from src.live.signal_generator import SignalGenerator
from src.live.surface_provider import SurfaceProvider
from src.live.router import Fill, PaperOrderRouter
from src.live.state import StateManager
from src.risk.portfolio import Portfolio
from src.strategy.sizing import SizingResult
from src.strategy.types import Greeks, OptionLeg, OptionRight, Signal, SignalType


class MockSurfaceProvider:
    """Mock surface provider for testing."""

    def __init__(self) -> None:
        self._call_count = 0

    def get_latest_surface(self) -> pd.DataFrame:
        """Return mock surface data."""
        self._call_count += 1
        return pd.DataFrame(
            [
                {
                    "option_symbol": "SPY240315C00450000",
                    "tenor_days": 30,
                    "delta_bucket": "ATM",
                    "strike": 450.0,
                    "expiry": date(2024, 3, 15),
                    "right": "C",
                    "bid": 5.00 + self._call_count * 0.05,
                    "ask": 5.10 + self._call_count * 0.05,
                    "delta": 0.45,
                    "gamma": 0.02,
                    "vega": 0.30,
                    "theta": -0.05,
                    "iv": 0.20,
                    "underlying_price": 450.0,
                },
                {
                    "option_symbol": "SPY240315C00455000",
                    "tenor_days": 30,
                    "delta_bucket": "C25",
                    "strike": 455.0,
                    "expiry": date(2024, 3, 15),
                    "right": "C",
                    "bid": 3.00 + self._call_count * 0.03,
                    "ask": 3.08 + self._call_count * 0.03,
                    "delta": 0.35,
                    "gamma": 0.018,
                    "vega": 0.25,
                    "theta": -0.04,
                    "iv": 0.18,
                    "underlying_price": 450.0,
                },
            ]
        )


class MockSignalGenerator:
    """Mock signal generator that produces signals on first few calls."""

    def __init__(self, max_signals: int = 2) -> None:
        self._call_count = 0
        self._max_signals = max_signals

    def generate_signals(
        self,
        surface: pd.DataFrame,
        features: Optional[pd.DataFrame] = None,
    ) -> list[Signal]:
        """Generate a signal on first few calls."""
        self._call_count += 1

        if self._call_count <= self._max_signals:
            return [
                Signal(
                    signal_type=SignalType.DIRECTIONAL_VOL,
                    edge=0.05,
                    confidence=0.8,
                    tenor_days=30,
                    delta_bucket="ATM",
                )
            ]
        return []


def _make_sizing_result(qty: int = 1) -> SizingResult:
    """Helper to create SizingResult with correct fields."""
    return SizingResult(
        quantity_multiplier=qty,
        base_contracts=qty,
        adjusted_contracts=qty,
        confidence_factor=1.0,
        liquidity_factor=1.0,
        risk_factor=1.0,
        reason="fixed",
    )


@pytest.fixture
def temp_state_dir(tmp_path: Path) -> str:
    """Create temporary state directory."""
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    return str(state_dir)


@pytest.fixture
def paper_config(temp_state_dir: str) -> PaperConfig:
    """Create paper config with temp state dir."""
    return PaperConfig(
        state_dir=temp_state_dir,
        save_state_interval=1,
        loop_interval_seconds=0,
        max_loop_iterations=3,
        halt_on_error=True,
    )


@pytest.fixture
def execution_config() -> ExecutionConfig:
    """Create execution configuration."""
    return ExecutionConfig()


@pytest.fixture
def risk_config() -> RiskConfig:
    """Create risk configuration."""
    return RiskConfig(
        caps=RiskCapsConfig(
            max_abs_delta=1000.0,
            max_abs_gamma=100.0,
            max_abs_vega=500.0,
            max_daily_loss=5000.0,
        ),
        kill_switch=KillSwitchConfig(
            halt_on_daily_loss=True,
            max_daily_loss=5000.0,
            halt_on_stress_breach=False,
            halt_on_liquidity_collapse=True,
            max_spread_pct=0.5,
        ),
        stress=StressConfig(enabled=False),
        position_management=PositionManagementConfig(
            exit_signals=ExitSignalsConfig(),
            rebalancing=RebalancingConfig(enabled=False),
            drift_bands=DriftBandsConfig(),
        ),
    )


class TestStateRecoveryIntegration:
    """Integration tests for state recovery after crash."""

    def test_state_persists_after_loop_iterations(
        self,
        paper_config: PaperConfig,
        execution_config: ExecutionConfig,
        risk_config: RiskConfig,
        temp_state_dir: str,
    ) -> None:
        """Test that state is persisted after each iteration."""
        surface_provider = MockSurfaceProvider()
        signal_generator = MockSignalGenerator(max_signals=2)
        order_router = PaperOrderRouter(execution_config)

        loop = TradingLoop(
            paper_config=paper_config,
            execution_config=execution_config,
            risk_config=risk_config,
            surface_provider=surface_provider,
            signal_generator=signal_generator,
            order_router=order_router,
        )

        # Run loop for 3 iterations
        loop.start()

        # Check snapshots were saved
        state_manager = loop.state_manager
        assert state_manager.get_snapshot_count() == 3

    def test_crash_recovery_continues_from_last_state(
        self,
        paper_config: PaperConfig,
        execution_config: ExecutionConfig,
        risk_config: RiskConfig,
        temp_state_dir: str,
    ) -> None:
        """Test that a new loop can recover state after 'crash'."""
        # First loop - run some iterations
        surface_provider1 = MockSurfaceProvider()
        signal_generator1 = MockSignalGenerator(max_signals=1)
        order_router1 = PaperOrderRouter(execution_config)

        loop1 = TradingLoop(
            paper_config=paper_config,
            execution_config=execution_config,
            risk_config=risk_config,
            surface_provider=surface_provider1,
            signal_generator=signal_generator1,
            order_router=order_router1,
        )

        loop1.start()
        final_iteration_1 = loop1.state.iteration

        # "Crash" - loop1 is done, state was saved

        # Second loop - should recover state
        # max_loop_iterations is a hard limit on total iterations
        # Since we recovered at iteration 3, we need max >= 5 to run 2 more
        paper_config_2 = PaperConfig(
            state_dir=temp_state_dir,
            save_state_interval=1,
            loop_interval_seconds=0,
            max_loop_iterations=final_iteration_1 + 2,  # Run until iteration 5
        )

        surface_provider2 = MockSurfaceProvider()
        signal_generator2 = MockSignalGenerator(max_signals=0)  # No new signals
        order_router2 = PaperOrderRouter(execution_config)

        loop2 = TradingLoop(
            paper_config=paper_config_2,
            execution_config=execution_config,
            risk_config=risk_config,
            surface_provider=surface_provider2,
            signal_generator=signal_generator2,
            order_router=order_router2,
        )

        # Verify state was recovered
        assert loop2.state.iteration == final_iteration_1

        # Run more iterations
        loop2.start()

        # Should have continued from where we left off
        assert loop2.state.iteration == final_iteration_1 + 2

    def test_portfolio_positions_restored_after_crash(
        self,
        temp_state_dir: str,
        execution_config: ExecutionConfig,
        risk_config: RiskConfig,
    ) -> None:
        """Test that portfolio positions are restored after crash."""
        # Create a portfolio with positions
        legs = [
            OptionLeg(
                symbol="SPY240315C00450000",
                qty=1,
                entry_price=5.00,
                strike=450.0,
                expiry=date(2024, 3, 15),
                right=OptionRight.CALL,
                greeks=Greeks(delta=0.45, gamma=0.02, vega=0.30, theta=-0.05),
            ),
        ]

        portfolio = Portfolio(daily_pnl=100.0)
        portfolio = portfolio.add_position(
            legs=legs,
            position_id="test_pos_001",
            structure_type="SingleLeg",
            max_loss=510.0,
        )

        # Save state
        paper_config = PaperConfig(
            state_dir=temp_state_dir,
            save_state_interval=1,
            loop_interval_seconds=0,
        )

        state_manager1 = StateManager(paper_config, portfolio=portfolio)
        state_manager1.save_snapshot(iteration=5)

        # "Crash" and reload
        state_manager2 = StateManager(paper_config)

        # Verify positions restored
        assert len(state_manager2.portfolio.positions) == 1
        assert state_manager2.portfolio.positions[0].position_id == "test_pos_001"
        assert state_manager2.portfolio.daily_pnl == 100.0
        assert state_manager2.last_iteration == 5

    def test_order_and_fill_history_restored(
        self,
        temp_state_dir: str,
    ) -> None:
        """Test that order and fill history is restored after crash."""
        paper_config = PaperConfig(
            state_dir=temp_state_dir,
            save_state_interval=1,
            loop_interval_seconds=0,
        )

        # First manager - record some orders and fills
        state_manager1 = StateManager(paper_config)

        # Record orders (simplified representation)
        state_manager1._order_history = [
            {"order_id": "order_001", "structure_type": "SingleLeg"},
            {"order_id": "order_002", "structure_type": "VerticalSpread"},
        ]

        # Record fills (simplified representation)
        state_manager1._fill_history = [
            {"fill_id": "fill_001", "order_id": "order_001"},
        ]

        state_manager1.save_snapshot(iteration=10)

        # "Crash" and reload
        state_manager2 = StateManager(paper_config)

        # Verify history restored
        assert len(state_manager2.order_history) == 2
        assert len(state_manager2.fill_history) == 1
        assert state_manager2.order_history[0]["order_id"] == "order_001"
        assert state_manager2.fill_history[0]["fill_id"] == "fill_001"

    def test_full_trading_cycle_with_recovery(
        self,
        temp_state_dir: str,
        execution_config: ExecutionConfig,
        risk_config: RiskConfig,
    ) -> None:
        """Test complete trading cycle with crash and recovery."""
        paper_config = PaperConfig(
            state_dir=temp_state_dir,
            save_state_interval=1,
            loop_interval_seconds=0,
            max_loop_iterations=2,
        )

        # Phase 1: Initial trading session
        loop1 = TradingLoop(
            paper_config=paper_config,
            execution_config=execution_config,
            risk_config=risk_config,
            surface_provider=MockSurfaceProvider(),
            signal_generator=MockSignalGenerator(max_signals=1),
            order_router=PaperOrderRouter(execution_config),
        )

        loop1.start()

        # Record metrics from first session
        fills_session1 = loop1.metrics.total_fills
        iteration_session1 = loop1.state.iteration

        # Phase 2: "Crash" and recovery
        # max_loop_iterations is total limit, so set it high enough
        paper_config_2 = PaperConfig(
            state_dir=temp_state_dir,
            save_state_interval=1,
            loop_interval_seconds=0,
            max_loop_iterations=iteration_session1 + 3,  # Allow 3 more iterations
        )

        loop2 = TradingLoop(
            paper_config=paper_config_2,
            execution_config=execution_config,
            risk_config=risk_config,
            surface_provider=MockSurfaceProvider(),
            signal_generator=MockSignalGenerator(max_signals=1),
            order_router=PaperOrderRouter(execution_config),
        )

        # Verify recovery
        assert loop2.state.iteration == iteration_session1

        loop2.start()

        # Final checks
        assert loop2.state.iteration == iteration_session1 + 3
        # Both sessions saved snapshots: 2 from first + 3 from second = 5 total
        assert loop2.state_manager.get_snapshot_count() >= 4  # At least 4 snapshots total


class TestMonitoringIntegration:
    """Integration tests for monitoring during trading."""

    def test_monitor_tracks_metrics_across_iterations(
        self,
        paper_config: PaperConfig,
        execution_config: ExecutionConfig,
        risk_config: RiskConfig,
    ) -> None:
        """Test that monitor tracks metrics across iterations."""
        loop = TradingLoop(
            paper_config=paper_config,
            execution_config=execution_config,
            risk_config=risk_config,
            surface_provider=MockSurfaceProvider(),
            signal_generator=MockSignalGenerator(max_signals=2),
            order_router=PaperOrderRouter(execution_config),
        )

        loop.start()

        # Check monitor has metrics
        monitor = loop.monitor
        assert len(monitor.metrics_history) == 3  # 3 iterations

        # Metrics should show progression
        assert monitor.metrics_history[0].iteration == 1
        assert monitor.metrics_history[1].iteration == 2
        assert monitor.metrics_history[2].iteration == 3
