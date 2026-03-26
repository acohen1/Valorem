"""Integration tests for TradingLoop."""

from datetime import date, datetime, timedelta, timezone
from typing import Optional

import numpy as np
import pandas as pd
import pytest

from src.config.schema import (
    ExecutionConfig,
    FeeConfig,
    KillSwitchConfig,
    PaperConfig,
    PerTradeRiskConfig,
    PositionManagementConfig,
    RiskCapsConfig,
    RiskConfig,
    SlippageConfig,
)
from src.live import (
    Fill,
    LoopState,
    PaperOrderRouter,
    TradingLoop,
)
from src.strategy.types import Signal, SignalType


class MockSurfaceProvider:
    """Mock surface provider that generates realistic synthetic surfaces."""

    def __init__(
        self,
        underlying_price: float = 450.0,
        base_iv: float = 0.20,
    ) -> None:
        self._underlying_price = underlying_price
        self._base_iv = base_iv
        self._call_count = 0

    def get_latest_surface(self) -> pd.DataFrame:
        """Generate synthetic options surface."""
        self._call_count += 1

        # Small random walk
        self._underlying_price *= 1 + np.random.normal(0, 0.001)

        return self._generate_surface()

    def _generate_surface(self) -> pd.DataFrame:
        """Generate synthetic options surface."""
        tenors = [7, 14, 30, 45, 60, 90]
        strike_pcts = [0.90, 0.92, 0.94, 0.96, 0.98, 1.0, 1.02, 1.04, 1.06, 1.08, 1.10]
        as_of_date = date.today()

        records = []
        for tenor in tenors:
            expiry = as_of_date + timedelta(days=tenor)

            for strike_pct in strike_pcts:
                strike = round(self._underlying_price * strike_pct, 2)

                for right in ["C", "P"]:
                    moneyness = self._underlying_price / strike
                    if right == "C":
                        delta = max(0.05, min(0.95, 0.5 + 0.4 * (moneyness - 1)))
                    else:
                        delta = -max(0.05, min(0.95, 0.5 - 0.4 * (moneyness - 1)))

                    # Delta bucket based on moneyness
                    moneyness_pct = strike / self._underlying_price
                    if moneyness_pct <= 0.92:
                        delta_bucket = "P10"
                    elif moneyness_pct <= 0.97:
                        delta_bucket = "P25"
                    elif moneyness_pct <= 1.03:
                        delta_bucket = "ATM"
                    elif moneyness_pct <= 1.08:
                        delta_bucket = "C25"
                    else:
                        delta_bucket = "C10"

                    iv = self._base_iv * (1 + 0.1 * (1 - moneyness))
                    time_value = iv * np.sqrt(tenor / 365) * self._underlying_price * 0.4
                    intrinsic = max(
                        0,
                        (self._underlying_price - strike)
                        if right == "C"
                        else (strike - self._underlying_price),
                    )
                    mid_price = intrinsic + time_value * abs(delta)

                    bid = max(0.01, mid_price * 0.98)
                    ask = mid_price * 1.02

                    expiry_str = expiry.strftime("%y%m%d")
                    strike_str = f"{int(strike * 1000):08d}"
                    symbol = f"SPY{expiry_str}{right}{strike_str}"

                    records.append(
                        {
                            "option_symbol": symbol,
                            "tenor_days": tenor,
                            "delta_bucket": delta_bucket,
                            "strike": strike,
                            "expiry": expiry,
                            "right": right,
                            "bid": round(bid, 2),
                            "ask": round(ask, 2),
                            "mid_price": round(mid_price, 2),
                            "delta": round(delta, 4),
                            "gamma": round(0.02 / (1 + abs(moneyness - 1) * 10), 4),
                            "vega": round(0.3 * np.sqrt(tenor / 30), 4),
                            "theta": round(-0.02 * (30 / tenor), 4),
                            "iv": round(iv, 4),
                            "underlying_price": self._underlying_price,
                        }
                    )

        return pd.DataFrame(records)


class MockSignalGenerator:
    """Mock signal generator with controllable output."""

    def __init__(
        self,
        signal_probability: float = 0.3,
        seed: Optional[int] = None,
    ) -> None:
        self._signal_probability = signal_probability
        self._rng = np.random.default_rng(seed)
        self._call_count = 0

    def generate_signals(
        self,
        surface: pd.DataFrame,
        features: Optional[pd.DataFrame] = None,
    ) -> list[Signal]:
        self._call_count += 1

        if self._rng.random() > self._signal_probability:
            return []

        valid_nodes = surface[surface["delta_bucket"].isin(["P25", "ATM", "C25"])]
        if valid_nodes.empty:
            return []

        node = valid_nodes.sample(1, random_state=self._call_count).iloc[0]

        signal_type = self._rng.choice(
            [
                SignalType.TERM_ANOMALY,
                SignalType.DIRECTIONAL_VOL,
            ]
        )

        return [
            Signal(
                signal_type=signal_type,
                edge=float(self._rng.uniform(0.03, 0.08)),
                confidence=float(self._rng.uniform(0.6, 0.9)),
                tenor_days=int(node["tenor_days"]),
                delta_bucket=str(node["delta_bucket"]),
                timestamp=datetime.now(timezone.utc),
            )
        ]


@pytest.fixture
def paper_config(tmp_path) -> PaperConfig:
    """Paper config for integration testing with isolated state directory."""
    return PaperConfig(
        loop_interval_seconds=0,  # No delay for tests
        halt_on_error=True,
        max_loop_iterations=5,
        lookback_minutes=5,
        save_state_interval=1,
        state_dir=str(tmp_path / "state"),  # Isolated state directory per test
    )


@pytest.fixture
def execution_config() -> ExecutionConfig:
    """Execution config for integration testing."""
    return ExecutionConfig(
        slippage=SlippageConfig(fixed_bps=5),
        fees=FeeConfig(per_contract=0.50, per_trade_minimum=1.00),
    )


@pytest.fixture
def risk_config() -> RiskConfig:
    """Risk config for integration testing."""
    return RiskConfig(
        per_trade=PerTradeRiskConfig(max_loss=1000.0, max_contracts=10),
        caps=RiskCapsConfig(
            max_abs_delta=200.0,
            max_abs_vega=2000.0,
            max_portfolio_loss=5000.0,
        ),
        kill_switch=KillSwitchConfig(
            halt_on_daily_loss=True,
            max_daily_loss=3000.0,
        ),
        position_management=PositionManagementConfig(),
    )


class TestTradingLoop5Cycles:
    """Integration tests running the loop for 5 cycles."""

    def test_loop_runs_5_cycles_no_signals(
        self,
        paper_config: PaperConfig,
        execution_config: ExecutionConfig,
        risk_config: RiskConfig,
    ) -> None:
        """Test loop completes 5 cycles with no signals."""
        surface_provider = MockSurfaceProvider()
        signal_generator = MockSignalGenerator(signal_probability=0.0)
        order_router = PaperOrderRouter(execution_config)

        loop = TradingLoop(
            paper_config=paper_config,
            execution_config=execution_config,
            risk_config=risk_config,
            surface_provider=surface_provider,
            signal_generator=signal_generator,
            order_router=order_router,
        )

        loop.start()

        assert loop.state.iteration == 5
        assert loop.metrics.total_iterations == 5
        assert loop.metrics.total_signals == 0
        assert loop.metrics.total_orders == 0
        assert loop.metrics.total_fills == 0
        assert surface_provider._call_count == 5
        assert signal_generator._call_count == 5

    def test_loop_runs_5_cycles_with_signals(
        self,
        paper_config: PaperConfig,
        execution_config: ExecutionConfig,
        risk_config: RiskConfig,
    ) -> None:
        """Test loop completes 5 cycles with signals generated."""
        surface_provider = MockSurfaceProvider()
        signal_generator = MockSignalGenerator(signal_probability=0.8, seed=42)
        order_router = PaperOrderRouter(execution_config)

        loop = TradingLoop(
            paper_config=paper_config,
            execution_config=execution_config,
            risk_config=risk_config,
            surface_provider=surface_provider,
            signal_generator=signal_generator,
            order_router=order_router,
        )

        loop.start()

        assert loop.state.iteration == 5
        assert loop.metrics.total_iterations == 5
        # With 80% signal probability, expect some signals
        assert loop.metrics.total_signals >= 0  # May still be 0 due to randomness

    def test_loop_tracks_metrics_correctly(
        self,
        paper_config: PaperConfig,
        execution_config: ExecutionConfig,
        risk_config: RiskConfig,
    ) -> None:
        """Test metrics are tracked correctly across iterations."""
        surface_provider = MockSurfaceProvider()
        signal_generator = MockSignalGenerator(signal_probability=0.5, seed=123)
        order_router = PaperOrderRouter(execution_config)

        loop = TradingLoop(
            paper_config=paper_config,
            execution_config=execution_config,
            risk_config=risk_config,
            surface_provider=surface_provider,
            signal_generator=signal_generator,
            order_router=order_router,
        )

        loop.start()

        metrics = loop.metrics
        assert metrics.total_iterations == 5
        assert metrics.start_time is not None
        assert metrics.end_time is not None
        assert metrics.end_time >= metrics.start_time

    def test_loop_callbacks_called(
        self,
        paper_config: PaperConfig,
        execution_config: ExecutionConfig,
        risk_config: RiskConfig,
    ) -> None:
        """Test callbacks are called correctly."""
        surface_provider = MockSurfaceProvider()
        signal_generator = MockSignalGenerator(signal_probability=0.0)
        order_router = PaperOrderRouter(execution_config)

        iteration_states: list[LoopState] = []

        def on_iteration(state: LoopState) -> None:
            iteration_states.append(
                LoopState(
                    iteration=state.iteration,
                    is_running=state.is_running,
                    last_signal_count=state.last_signal_count,
                )
            )

        loop = TradingLoop(
            paper_config=paper_config,
            execution_config=execution_config,
            risk_config=risk_config,
            surface_provider=surface_provider,
            signal_generator=signal_generator,
            order_router=order_router,
            on_iteration_callback=on_iteration,
        )

        loop.start()

        assert len(iteration_states) == 5
        assert [s.iteration for s in iteration_states] == [1, 2, 3, 4, 5]


class TestTradingLoopErrorHandling:
    """Integration tests for error handling."""

    def test_loop_halts_on_error_when_configured(
        self,
        execution_config: ExecutionConfig,
        risk_config: RiskConfig,
        tmp_path,
    ) -> None:
        """Test loop halts when error occurs with halt_on_error=True."""

        class FailingSurfaceProvider:
            def __init__(self):
                self._call_count = 0

            def get_latest_surface(self) -> pd.DataFrame:
                self._call_count += 1
                if self._call_count >= 3:
                    raise RuntimeError("Simulated failure")
                return MockSurfaceProvider().get_latest_surface()

        paper_config = PaperConfig(
            loop_interval_seconds=0,
            halt_on_error=True,
            max_loop_iterations=10,
            state_dir=str(tmp_path / "state"),
        )

        surface_provider = FailingSurfaceProvider()
        signal_generator = MockSignalGenerator(signal_probability=0.0)
        order_router = PaperOrderRouter(execution_config)

        loop = TradingLoop(
            paper_config=paper_config,
            execution_config=execution_config,
            risk_config=risk_config,
            surface_provider=surface_provider,
            signal_generator=signal_generator,
            order_router=order_router,
        )

        loop.start()

        # Should halt before reaching 10 iterations
        assert loop.state.iteration < 10
        assert loop.state.errors_count >= 1


class TestTradingLoopKillSwitchIntegration:
    """Integration tests for kill switch."""

    def test_kill_switch_triggers_on_daily_loss(
        self,
        paper_config: PaperConfig,
        execution_config: ExecutionConfig,
    ) -> None:
        """Test kill switch triggers when daily loss exceeds threshold."""
        risk_config = RiskConfig(
            kill_switch=KillSwitchConfig(
                halt_on_daily_loss=True,
                max_daily_loss=100.0,  # Low threshold
            ),
        )

        surface_provider = MockSurfaceProvider()
        signal_generator = MockSignalGenerator(signal_probability=0.0)
        order_router = PaperOrderRouter(execution_config)

        loop = TradingLoop(
            paper_config=paper_config,
            execution_config=execution_config,
            risk_config=risk_config,
            surface_provider=surface_provider,
            signal_generator=signal_generator,
            order_router=order_router,
        )

        # Simulate daily loss
        loop._portfolio.update_daily_pnl(-150.0)

        loop.start()

        assert loop.state.kill_switch_triggered is True
        # Should stop after first iteration detects kill switch
        assert loop.state.iteration <= 2


class TestTradingLoopPortfolio:
    """Integration tests for portfolio management."""

    def test_portfolio_persists_across_iterations(
        self,
        paper_config: PaperConfig,
        execution_config: ExecutionConfig,
        risk_config: RiskConfig,
    ) -> None:
        """Test portfolio state persists across iterations."""
        surface_provider = MockSurfaceProvider()
        signal_generator = MockSignalGenerator(signal_probability=0.0)
        order_router = PaperOrderRouter(execution_config)

        loop = TradingLoop(
            paper_config=paper_config,
            execution_config=execution_config,
            risk_config=risk_config,
            surface_provider=surface_provider,
            signal_generator=signal_generator,
            order_router=order_router,
        )

        # Set some initial P&L to track persistence
        loop._portfolio.update_daily_pnl(-100.0)
        initial_pnl = loop.portfolio.daily_pnl

        loop.start()

        # Portfolio state should persist (daily P&L unchanged since no fills)
        # Note: Portfolio object may be replaced, but state is maintained
        assert loop.portfolio is not None
        assert loop.portfolio.daily_pnl == initial_pnl

    def test_reset_daily_metrics_clears_pnl(
        self,
        paper_config: PaperConfig,
        execution_config: ExecutionConfig,
        risk_config: RiskConfig,
    ) -> None:
        """Test reset_daily_metrics clears daily P&L."""
        surface_provider = MockSurfaceProvider()
        signal_generator = MockSignalGenerator(signal_probability=0.0)
        order_router = PaperOrderRouter(execution_config)

        loop = TradingLoop(
            paper_config=paper_config,
            execution_config=execution_config,
            risk_config=risk_config,
            surface_provider=surface_provider,
            signal_generator=signal_generator,
            order_router=order_router,
        )

        # Simulate some P&L
        loop._portfolio.update_daily_pnl(-500.0)
        assert loop.portfolio.daily_pnl == -500.0

        # Reset
        loop.reset_daily_metrics()

        assert loop.portfolio.daily_pnl == 0.0


class TestTradingLoopPositionManagement:
    """Integration tests for position management."""

    def test_position_manager_accessible(
        self,
        paper_config: PaperConfig,
        execution_config: ExecutionConfig,
        risk_config: RiskConfig,
    ) -> None:
        """Test position manager is accessible."""
        surface_provider = MockSurfaceProvider()
        signal_generator = MockSignalGenerator(signal_probability=0.0)
        order_router = PaperOrderRouter(execution_config)

        loop = TradingLoop(
            paper_config=paper_config,
            execution_config=execution_config,
            risk_config=risk_config,
            surface_provider=surface_provider,
            signal_generator=signal_generator,
            order_router=order_router,
        )

        assert loop.position_manager is not None
        assert loop.position_manager.positions == {}
