"""Unit tests for TradingLoop."""

from datetime import date
from typing import Optional

import pandas as pd
import pytest

from src.config.schema import (
    ExecutionConfig,
    KillSwitchConfig,
    PaperConfig,
    PerTradeRiskConfig,
    PositionManagementConfig,
    RiskCapsConfig,
    RiskConfig,
)
from src.live.loop import LoopMetrics, LoopState, TradingLoop
from src.live.router import Fill
from src.strategy.orders import Order
from src.strategy.types import Signal, SignalType


class MockSurfaceProvider:
    """Mock surface provider for testing."""

    def __init__(self, surface: Optional[pd.DataFrame] = None) -> None:
        self._surface = surface
        self._call_count = 0

    def get_latest_surface(self) -> pd.DataFrame:
        self._call_count += 1
        if self._surface is not None:
            return self._surface
        # Return minimal surface
        return pd.DataFrame(
            [
                {
                    "option_symbol": "SPY240315C00450000",
                    "tenor_days": 30,
                    "delta_bucket": "ATM",
                    "strike": 450.0,
                    "expiry": date(2024, 3, 15),
                    "right": "C",
                    "bid": 5.00,
                    "ask": 5.10,
                    "delta": 0.50,
                    "gamma": 0.02,
                    "vega": 0.30,
                    "theta": -0.05,
                    "iv": 0.20,
                    "underlying_price": 450.0,
                }
            ]
        )


class MockSignalGenerator:
    """Mock signal generator for testing."""

    def __init__(self, signals: Optional[list[Signal]] = None) -> None:
        self._signals = signals or []
        self._call_count = 0

    def generate_signals(
        self,
        surface: pd.DataFrame,
        features: Optional[pd.DataFrame] = None,
    ) -> list[Signal]:
        self._call_count += 1
        return self._signals


class MockOrderRouter:
    """Mock order router for testing."""

    def __init__(self, fill: Optional[Fill] = None) -> None:
        self._fill = fill
        self._call_count = 0
        self._routed_orders: list[Order] = []

    def route_order(
        self,
        order: Order,
        surface: pd.DataFrame,
    ) -> Optional[Fill]:
        self._call_count += 1
        self._routed_orders.append(order)
        return self._fill

    def is_available(self) -> bool:
        return True


@pytest.fixture
def paper_config(tmp_path) -> PaperConfig:
    """Paper config for testing with isolated state directory."""
    return PaperConfig(
        loop_interval_seconds=1,
        halt_on_error=True,
        max_loop_iterations=3,
        lookback_minutes=5,
        save_state_interval=1,
        state_dir=str(tmp_path / "state"),  # Isolated state directory per test
    )


@pytest.fixture
def execution_config() -> ExecutionConfig:
    """Execution config for testing."""
    return ExecutionConfig()


@pytest.fixture
def risk_config() -> RiskConfig:
    """Risk config for testing."""
    return RiskConfig(
        per_trade=PerTradeRiskConfig(max_loss=2000.0, max_contracts=20),
        caps=RiskCapsConfig(max_abs_delta=500.0, max_abs_vega=5000.0),
        kill_switch=KillSwitchConfig(
            halt_on_daily_loss=True,
            max_daily_loss=1000.0,
        ),
        position_management=PositionManagementConfig(),
    )


@pytest.fixture
def mock_surface_provider() -> MockSurfaceProvider:
    """Create mock surface provider."""
    return MockSurfaceProvider()


@pytest.fixture
def mock_signal_generator() -> MockSignalGenerator:
    """Create mock signal generator with no signals."""
    return MockSignalGenerator([])


@pytest.fixture
def mock_order_router() -> MockOrderRouter:
    """Create mock order router."""
    return MockOrderRouter()


class TestTradingLoopInit:
    """Tests for TradingLoop initialization."""

    def test_init_with_required_args(
        self,
        paper_config: PaperConfig,
        execution_config: ExecutionConfig,
        risk_config: RiskConfig,
        mock_surface_provider: MockSurfaceProvider,
        mock_signal_generator: MockSignalGenerator,
        mock_order_router: MockOrderRouter,
    ) -> None:
        """Test initialization with required arguments."""
        loop = TradingLoop(
            paper_config=paper_config,
            execution_config=execution_config,
            risk_config=risk_config,
            surface_provider=mock_surface_provider,
            signal_generator=mock_signal_generator,
            order_router=mock_order_router,
        )

        assert loop is not None
        assert loop.state.is_running is False
        assert loop.state.iteration == 0
        assert loop.portfolio is not None

    def test_init_creates_default_components(
        self,
        paper_config: PaperConfig,
        execution_config: ExecutionConfig,
        risk_config: RiskConfig,
        mock_surface_provider: MockSurfaceProvider,
        mock_signal_generator: MockSignalGenerator,
        mock_order_router: MockOrderRouter,
    ) -> None:
        """Test that default components are created."""
        loop = TradingLoop(
            paper_config=paper_config,
            execution_config=execution_config,
            risk_config=risk_config,
            surface_provider=mock_surface_provider,
            signal_generator=mock_signal_generator,
            order_router=mock_order_router,
        )

        # Components should be created
        assert loop._risk_checker is not None
        assert loop._order_generator is not None
        assert loop._position_manager is not None
        assert loop._kill_switch is not None


class TestTradingLoopState:
    """Tests for TradingLoop state management."""

    def test_initial_state(
        self,
        paper_config: PaperConfig,
        execution_config: ExecutionConfig,
        risk_config: RiskConfig,
        mock_surface_provider: MockSurfaceProvider,
        mock_signal_generator: MockSignalGenerator,
        mock_order_router: MockOrderRouter,
    ) -> None:
        """Test initial loop state."""
        loop = TradingLoop(
            paper_config=paper_config,
            execution_config=execution_config,
            risk_config=risk_config,
            surface_provider=mock_surface_provider,
            signal_generator=mock_signal_generator,
            order_router=mock_order_router,
        )

        state = loop.state
        assert state.iteration == 0
        assert state.is_running is False
        assert state.last_surface_time is None
        assert state.last_signal_count == 0
        assert state.last_order_count == 0
        assert state.last_fill_count == 0
        assert state.errors_count == 0
        assert state.kill_switch_triggered is False

    def test_initial_metrics(
        self,
        paper_config: PaperConfig,
        execution_config: ExecutionConfig,
        risk_config: RiskConfig,
        mock_surface_provider: MockSurfaceProvider,
        mock_signal_generator: MockSignalGenerator,
        mock_order_router: MockOrderRouter,
    ) -> None:
        """Test initial loop metrics."""
        loop = TradingLoop(
            paper_config=paper_config,
            execution_config=execution_config,
            risk_config=risk_config,
            surface_provider=mock_surface_provider,
            signal_generator=mock_signal_generator,
            order_router=mock_order_router,
        )

        metrics = loop.metrics
        assert metrics.total_iterations == 0
        assert metrics.total_signals == 0
        assert metrics.total_orders == 0
        assert metrics.total_fills == 0
        assert metrics.total_rejections == 0
        assert metrics.start_time is None
        assert metrics.end_time is None
        assert metrics.daily_pnl == 0.0


class TestTradingLoopSingleIteration:
    """Tests for single iteration execution."""

    def test_run_single_iteration(
        self,
        paper_config: PaperConfig,
        execution_config: ExecutionConfig,
        risk_config: RiskConfig,
        mock_surface_provider: MockSurfaceProvider,
        mock_signal_generator: MockSignalGenerator,
        mock_order_router: MockOrderRouter,
    ) -> None:
        """Test running a single iteration."""
        loop = TradingLoop(
            paper_config=paper_config,
            execution_config=execution_config,
            risk_config=risk_config,
            surface_provider=mock_surface_provider,
            signal_generator=mock_signal_generator,
            order_router=mock_order_router,
        )

        loop.run_single_iteration()

        assert loop.state.iteration == 1
        assert mock_surface_provider._call_count == 1
        assert mock_signal_generator._call_count == 1

    def test_run_single_iteration_with_signals(
        self,
        paper_config: PaperConfig,
        execution_config: ExecutionConfig,
        risk_config: RiskConfig,
        mock_surface_provider: MockSurfaceProvider,
        mock_order_router: MockOrderRouter,
    ) -> None:
        """Test iteration with signals generated."""
        signals = [
            Signal(
                signal_type=SignalType.DIRECTIONAL_VOL,
                edge=0.05,
                confidence=0.8,
                tenor_days=30,
                delta_bucket="ATM",
            )
        ]
        signal_gen = MockSignalGenerator(signals)

        loop = TradingLoop(
            paper_config=paper_config,
            execution_config=execution_config,
            risk_config=risk_config,
            surface_provider=mock_surface_provider,
            signal_generator=signal_gen,
            order_router=mock_order_router,
        )

        loop.run_single_iteration()

        assert loop.state.last_signal_count == 1
        assert loop.metrics.total_signals == 1

    def test_run_single_iteration_empty_surface(
        self,
        paper_config: PaperConfig,
        execution_config: ExecutionConfig,
        risk_config: RiskConfig,
        mock_signal_generator: MockSignalGenerator,
        mock_order_router: MockOrderRouter,
    ) -> None:
        """Test iteration with empty surface."""
        empty_provider = MockSurfaceProvider(pd.DataFrame())

        loop = TradingLoop(
            paper_config=paper_config,
            execution_config=execution_config,
            risk_config=risk_config,
            surface_provider=empty_provider,
            signal_generator=mock_signal_generator,
            order_router=mock_order_router,
        )

        loop.run_single_iteration()

        # Should skip iteration but still increment
        assert loop.state.iteration == 1
        assert mock_signal_generator._call_count == 0  # Not called due to empty surface


class TestTradingLoopCallbacks:
    """Tests for callback functionality."""

    def test_on_iteration_callback(
        self,
        paper_config: PaperConfig,
        execution_config: ExecutionConfig,
        risk_config: RiskConfig,
        mock_surface_provider: MockSurfaceProvider,
        mock_signal_generator: MockSignalGenerator,
        mock_order_router: MockOrderRouter,
    ) -> None:
        """Test on_iteration callback is called."""
        callback_states: list[LoopState] = []

        def on_iteration(state: LoopState) -> None:
            callback_states.append(state)

        loop = TradingLoop(
            paper_config=paper_config,
            execution_config=execution_config,
            risk_config=risk_config,
            surface_provider=mock_surface_provider,
            signal_generator=mock_signal_generator,
            order_router=mock_order_router,
            on_iteration_callback=on_iteration,
        )

        loop.run_single_iteration()

        assert len(callback_states) == 1
        assert callback_states[0].iteration == 1


class TestTradingLoopStop:
    """Tests for stop functionality."""

    def test_stop_sets_flag(
        self,
        paper_config: PaperConfig,
        execution_config: ExecutionConfig,
        risk_config: RiskConfig,
        mock_surface_provider: MockSurfaceProvider,
        mock_signal_generator: MockSignalGenerator,
        mock_order_router: MockOrderRouter,
    ) -> None:
        """Test stop() sets the stop flag."""
        loop = TradingLoop(
            paper_config=paper_config,
            execution_config=execution_config,
            risk_config=risk_config,
            surface_provider=mock_surface_provider,
            signal_generator=mock_signal_generator,
            order_router=mock_order_router,
        )

        loop.stop()

        assert loop._stop_requested is True


class TestTradingLoopMaxIterations:
    """Tests for max iterations functionality."""

    def test_respects_max_iterations(
        self,
        execution_config: ExecutionConfig,
        risk_config: RiskConfig,
        mock_surface_provider: MockSurfaceProvider,
        mock_signal_generator: MockSignalGenerator,
        mock_order_router: MockOrderRouter,
        tmp_path,
    ) -> None:
        """Test loop respects max_loop_iterations."""
        paper_config = PaperConfig(
            loop_interval_seconds=0,  # No delay
            max_loop_iterations=5,
            halt_on_error=True,
            state_dir=str(tmp_path / "state"),
        )

        loop = TradingLoop(
            paper_config=paper_config,
            execution_config=execution_config,
            risk_config=risk_config,
            surface_provider=mock_surface_provider,
            signal_generator=mock_signal_generator,
            order_router=mock_order_router,
        )

        loop.start()

        assert loop.state.iteration == 5
        assert loop.metrics.total_iterations == 5
        assert loop.state.is_running is False


class TestTradingLoopResetDailyMetrics:
    """Tests for daily metrics reset."""

    def test_reset_daily_metrics(
        self,
        paper_config: PaperConfig,
        execution_config: ExecutionConfig,
        risk_config: RiskConfig,
        mock_surface_provider: MockSurfaceProvider,
        mock_signal_generator: MockSignalGenerator,
        mock_order_router: MockOrderRouter,
    ) -> None:
        """Test reset_daily_metrics clears daily P&L and kill switch."""
        loop = TradingLoop(
            paper_config=paper_config,
            execution_config=execution_config,
            risk_config=risk_config,
            surface_provider=mock_surface_provider,
            signal_generator=mock_signal_generator,
            order_router=mock_order_router,
        )

        # Simulate some P&L
        loop._portfolio.update_daily_pnl(-500.0)
        assert loop._portfolio.daily_pnl == -500.0

        loop.reset_daily_metrics()

        assert loop._portfolio.daily_pnl == 0.0


class TestLoopStateDataclass:
    """Tests for LoopState dataclass."""

    def test_loop_state_defaults(self) -> None:
        """Test LoopState default values."""
        state = LoopState()

        assert state.iteration == 0
        assert state.is_running is False
        assert state.last_surface_time is None
        assert state.last_signal_count == 0
        assert state.last_order_count == 0
        assert state.last_fill_count == 0
        assert state.errors_count == 0
        assert state.kill_switch_triggered is False


class TestLoopMetricsDataclass:
    """Tests for LoopMetrics dataclass."""

    def test_loop_metrics_defaults(self) -> None:
        """Test LoopMetrics default values."""
        metrics = LoopMetrics()

        assert metrics.total_iterations == 0
        assert metrics.total_signals == 0
        assert metrics.total_orders == 0
        assert metrics.total_fills == 0
        assert metrics.total_rejections == 0
        assert metrics.start_time is None
        assert metrics.end_time is None
        assert metrics.daily_pnl == 0.0


class TestTradingLoopKillSwitch:
    """Tests for kill switch integration."""

    def test_kill_switch_stops_loop(
        self,
        execution_config: ExecutionConfig,
        mock_surface_provider: MockSurfaceProvider,
        mock_signal_generator: MockSignalGenerator,
        mock_order_router: MockOrderRouter,
        tmp_path,
    ) -> None:
        """Test kill switch triggers loop stop."""
        paper_config = PaperConfig(
            loop_interval_seconds=0,
            max_loop_iterations=10,
            halt_on_error=True,
            state_dir=str(tmp_path / "state"),
        )

        risk_config = RiskConfig(
            kill_switch=KillSwitchConfig(
                halt_on_daily_loss=True,
                max_daily_loss=100.0,
            ),
        )

        loop = TradingLoop(
            paper_config=paper_config,
            execution_config=execution_config,
            risk_config=risk_config,
            surface_provider=mock_surface_provider,
            signal_generator=mock_signal_generator,
            order_router=mock_order_router,
        )

        # Trigger kill switch by setting large daily loss
        loop._portfolio.update_daily_pnl(-200.0)

        loop.start()

        # Should stop early due to kill switch
        assert loop.state.kill_switch_triggered is True
        assert loop.state.iteration < 10
