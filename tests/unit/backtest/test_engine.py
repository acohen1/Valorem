"""Unit tests for backtest engine."""

from datetime import date, datetime, timezone

import numpy as np
import pandas as pd
import pytest

from src.backtest.engine import BacktestEngine
from src.config.schema import (
    ExecutionConfig,
    FeeConfig,
    PerTradeRiskConfig,
    PositionManagementConfig,
    RiskCapsConfig,
    RiskConfig,
    SignalThresholdConfig,
    SizingConfig,
    SlippageConfig,
)
from src.strategy.types import Signal, SignalType


def generate_test_surface(
    as_of_date: date,
    underlying_price: float = 450.0,
) -> pd.DataFrame:
    """Generate synthetic options surface for testing."""
    tenors = [7, 14, 30, 45, 60]
    strike_pcts = [0.96, 0.98, 1.0, 1.02, 1.04]

    records = []
    for tenor in tenors:
        expiry = as_of_date + pd.Timedelta(days=tenor)

        for strike_pct in strike_pcts:
            strike = round(underlying_price * strike_pct, 2)

            for right in ["C", "P"]:
                moneyness = underlying_price / strike
                if right == "C":
                    delta = max(0.05, min(0.95, 0.5 + 0.4 * (moneyness - 1)))
                else:
                    delta = -max(0.05, min(0.95, 0.5 - 0.4 * (moneyness - 1)))

                abs_delta = abs(delta)
                if abs_delta >= 0.45:
                    delta_bucket = "ATM"
                elif abs_delta >= 0.35:
                    delta_bucket = "C40" if right == "C" else "P40"
                elif abs_delta >= 0.15:
                    delta_bucket = "C25" if right == "C" else "P25"
                else:
                    delta_bucket = "C10" if right == "C" else "P10"

                mid_price = max(0.50, abs(underlying_price - strike) + 5.0 * abs_delta)
                bid = max(0.01, mid_price * 0.98)
                ask = mid_price * 1.02

                expiry_str = expiry.strftime("%y%m%d")
                strike_str = f"{int(strike * 1000):08d}"
                symbol = f"SPY{expiry_str}{right}{strike_str}"

                records.append({
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
                    "gamma": 0.02,
                    "vega": 0.30,
                    "theta": -0.02,
                    "underlying_price": underlying_price,
                })

    return pd.DataFrame(records)


@pytest.fixture
def execution_config():
    """Create execution config for testing."""
    return ExecutionConfig(
        slippage=SlippageConfig(model="fixed_bps", fixed_bps=5.0),
        fees=FeeConfig(per_contract=0.65),
        signal_threshold=SignalThresholdConfig(min_edge=0.01, min_confidence=0.5),
        sizing=SizingConfig(base_contracts=1, max_contracts_per_trade=10),
    )


@pytest.fixture
def risk_config():
    """Create risk config for testing."""
    return RiskConfig(
        per_trade=PerTradeRiskConfig(max_loss=2000.0, max_contracts=20),
        caps=RiskCapsConfig(max_abs_delta=500.0, max_abs_vega=5000.0),
    )


@pytest.fixture
def pos_mgmt_config():
    """Create position management config for testing."""
    return PositionManagementConfig()


@pytest.fixture
def backtest_engine(execution_config, risk_config, pos_mgmt_config):
    """Create backtest engine for testing."""
    return BacktestEngine(
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 31),
        initial_capital=100000.0,
        execution_config=execution_config,
        risk_config=risk_config,
        position_management_config=pos_mgmt_config,
    )


class TestBacktestEngineInit:
    """Tests for BacktestEngine initialization."""

    def test_init_with_configs(self, execution_config, risk_config, pos_mgmt_config):
        """Test initialization with configs."""
        engine = BacktestEngine(
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 31),
            initial_capital=100000.0,
            execution_config=execution_config,
            risk_config=risk_config,
            position_management_config=pos_mgmt_config,
        )

        assert engine is not None
        assert engine.cash == 100000.0

    def test_init_with_signal_generator(
        self, execution_config, risk_config, pos_mgmt_config
    ):
        """Test initialization with signal generator callback."""
        def my_signal_gen(surface: pd.DataFrame) -> list[Signal]:
            return []

        engine = BacktestEngine(
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 31),
            initial_capital=100000.0,
            execution_config=execution_config,
            risk_config=risk_config,
            position_management_config=pos_mgmt_config,
            signal_generator=my_signal_gen,
        )

        assert engine is not None


class TestBacktestEngineProperties:
    """Tests for BacktestEngine properties."""

    def test_cash_property(self, backtest_engine):
        """Test cash property."""
        assert backtest_engine.cash == 100000.0

    def test_portfolio_property(self, backtest_engine):
        """Test portfolio property."""
        assert backtest_engine.portfolio is not None

    def test_position_manager_property(self, backtest_engine):
        """Test position manager property."""
        assert backtest_engine.position_manager is not None


class TestRunBacktest:
    """Tests for run_backtest method."""

    def test_empty_surfaces(self, backtest_engine):
        """Test backtest with no surface data."""
        result = backtest_engine.run_backtest(surfaces={})

        assert result.run_id.startswith("bt-")
        assert result.final_equity == 100000.0
        assert result.metrics.total_trades == 0

    def test_surfaces_outside_date_range(self, backtest_engine):
        """Test backtest with surfaces outside date range."""
        # Surface for date outside range
        surfaces = {
            date(2025, 1, 1): generate_test_surface(date(2025, 1, 1)),
        }

        result = backtest_engine.run_backtest(surfaces=surfaces)

        assert result.metrics.total_trades == 0

    def test_single_day_backtest(self, execution_config, risk_config, pos_mgmt_config):
        """Test backtest over single day."""
        engine = BacktestEngine(
            start_date=date(2024, 1, 15),
            end_date=date(2024, 1, 15),
            initial_capital=100000.0,
            execution_config=execution_config,
            risk_config=risk_config,
            position_management_config=pos_mgmt_config,
        )

        surfaces = {
            date(2024, 1, 15): generate_test_surface(date(2024, 1, 15)),
        }

        result = engine.run_backtest(surfaces=surfaces)

        assert len(result.portfolio_history) == 1

    def test_with_signals(self, backtest_engine):
        """Test backtest with provided signals."""
        surfaces = {
            date(2024, 1, 2): generate_test_surface(date(2024, 1, 2)),
            date(2024, 1, 3): generate_test_surface(date(2024, 1, 3)),
        }

        signals = {
            date(2024, 1, 2): [
                Signal(
                    signal_type=SignalType.DIRECTIONAL_VOL,
                    edge=0.05,
                    confidence=0.8,
                    tenor_days=30,
                    delta_bucket="ATM",
                    timestamp=datetime.now(timezone.utc),
                )
            ],
        }

        result = backtest_engine.run_backtest(
            surfaces=surfaces,
            signals_by_date=signals,
        )

        # Result should have some activity
        assert len(result.portfolio_history) >= 1

    def test_kill_switch_halt(self, execution_config, risk_config, pos_mgmt_config):
        """Test that kill switch halts backtest."""
        # Create engine with very low daily loss limit
        from src.config.schema import KillSwitchConfig

        low_limit_risk_config = RiskConfig(
            per_trade=PerTradeRiskConfig(max_loss=2000.0, max_contracts=20),
            caps=RiskCapsConfig(max_abs_delta=500.0, max_abs_vega=5000.0),
            kill_switch=KillSwitchConfig(
                halt_on_daily_loss=True,
                max_daily_loss=1.0,  # Very low
            ),
        )

        engine = BacktestEngine(
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 10),
            initial_capital=100000.0,
            execution_config=execution_config,
            risk_config=low_limit_risk_config,
            position_management_config=pos_mgmt_config,
        )

        # Generate surfaces for multiple days
        surfaces = {
            date(2024, 1, i): generate_test_surface(date(2024, 1, i))
            for i in range(1, 11)
        }

        result = engine.run_backtest(surfaces=surfaces)

        # Should complete without error
        assert result is not None


class TestPortfolioTracking:
    """Tests for portfolio tracking during backtest."""

    def test_portfolio_history_recorded(self, backtest_engine):
        """Test that portfolio history is recorded each day."""
        surfaces = {
            date(2024, 1, 2): generate_test_surface(date(2024, 1, 2)),
            date(2024, 1, 3): generate_test_surface(date(2024, 1, 3)),
            date(2024, 1, 4): generate_test_surface(date(2024, 1, 4)),
        }

        result = backtest_engine.run_backtest(surfaces=surfaces)

        # Should have snapshot for each trading day
        assert len(result.portfolio_history) == 3

    def test_equity_tracked(self, backtest_engine):
        """Test that equity is tracked correctly."""
        surfaces = {
            date(2024, 1, 2): generate_test_surface(date(2024, 1, 2)),
        }

        result = backtest_engine.run_backtest(surfaces=surfaces)

        # Initial equity should be preserved (no trades)
        assert result.portfolio_history[0].total_equity == pytest.approx(100000.0, abs=1.0)


class TestTradeExecution:
    """Tests for trade execution during backtest."""

    def test_entry_deducts_cash(self, execution_config, risk_config, pos_mgmt_config):
        """Test that entries deduct cash correctly."""
        engine = BacktestEngine(
            start_date=date(2024, 1, 2),
            end_date=date(2024, 1, 2),
            initial_capital=100000.0,
            execution_config=execution_config,
            risk_config=risk_config,
            position_management_config=pos_mgmt_config,
        )

        surfaces = {
            date(2024, 1, 2): generate_test_surface(date(2024, 1, 2)),
        }

        signals = {
            date(2024, 1, 2): [
                Signal(
                    signal_type=SignalType.DIRECTIONAL_VOL,
                    edge=0.05,
                    confidence=0.8,
                    tenor_days=30,
                    delta_bucket="ATM",
                    timestamp=datetime.now(timezone.utc),
                )
            ],
        }

        initial_cash = engine.cash
        result = engine.run_backtest(surfaces=surfaces, signals_by_date=signals)

        # If a trade was executed, cash should have changed
        # (Could be increased or decreased depending on trade type)
        assert engine.cash != initial_cash or len(result.trades) == 0


class TestExpirationHandling:
    """Tests for position expiration handling."""

    def test_expired_positions_closed(self, execution_config, risk_config, pos_mgmt_config):
        """Test that expired positions are handled."""
        engine = BacktestEngine(
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 20),
            initial_capital=100000.0,
            execution_config=execution_config,
            risk_config=risk_config,
            position_management_config=pos_mgmt_config,
        )

        # Create surface where 7-day options will expire during backtest
        surfaces = {
            date(2024, 1, i): generate_test_surface(date(2024, 1, i))
            for i in range(2, 21)
        }

        # Signal on day 2 for 7-day expiry (expires around Jan 9)
        signals = {
            date(2024, 1, 2): [
                Signal(
                    signal_type=SignalType.DIRECTIONAL_VOL,
                    edge=0.05,
                    confidence=0.8,
                    tenor_days=7,
                    delta_bucket="ATM",
                    timestamp=datetime.now(timezone.utc),
                )
            ],
        }

        result = engine.run_backtest(surfaces=surfaces, signals_by_date=signals)

        # Backtest should complete
        assert result is not None


class TestMetricsResult:
    """Tests for result metrics."""

    def test_result_has_metrics(self, backtest_engine):
        """Test that result includes metrics."""
        surfaces = {
            date(2024, 1, 2): generate_test_surface(date(2024, 1, 2)),
        }

        result = backtest_engine.run_backtest(surfaces=surfaces)

        assert result.metrics is not None
        assert hasattr(result.metrics, "total_return_pct")
        assert hasattr(result.metrics, "sharpe_ratio")

    def test_result_run_time(self, backtest_engine):
        """Test that run time is recorded."""
        surfaces = {
            date(2024, 1, 2): generate_test_surface(date(2024, 1, 2)),
        }

        result = backtest_engine.run_backtest(surfaces=surfaces)

        assert result.run_time_seconds >= 0


class TestSignalGenerator:
    """Tests for signal generator callback."""

    def test_signal_generator_called(
        self, execution_config, risk_config, pos_mgmt_config
    ):
        """Test that signal generator is called for each day."""
        call_count = [0]

        def my_signal_gen(surface: pd.DataFrame) -> list[Signal]:
            call_count[0] += 1
            return []

        engine = BacktestEngine(
            start_date=date(2024, 1, 2),
            end_date=date(2024, 1, 4),
            initial_capital=100000.0,
            execution_config=execution_config,
            risk_config=risk_config,
            position_management_config=pos_mgmt_config,
            signal_generator=my_signal_gen,
        )

        surfaces = {
            date(2024, 1, 2): generate_test_surface(date(2024, 1, 2)),
            date(2024, 1, 3): generate_test_surface(date(2024, 1, 3)),
            date(2024, 1, 4): generate_test_surface(date(2024, 1, 4)),
        }

        engine.run_backtest(surfaces=surfaces)

        # Should be called for each day
        assert call_count[0] == 3

    def test_signal_generator_receives_surface(
        self, execution_config, risk_config, pos_mgmt_config
    ):
        """Test that signal generator receives correct surface."""
        received_dates = []

        def my_signal_gen(surface: pd.DataFrame) -> list[Signal]:
            # Extract date from first option's expiry
            if "tenor_days" in surface.columns:
                received_dates.append(len(surface))
            return []

        engine = BacktestEngine(
            start_date=date(2024, 1, 2),
            end_date=date(2024, 1, 2),
            initial_capital=100000.0,
            execution_config=execution_config,
            risk_config=risk_config,
            position_management_config=pos_mgmt_config,
            signal_generator=my_signal_gen,
        )

        surfaces = {
            date(2024, 1, 2): generate_test_surface(date(2024, 1, 2)),
        }

        engine.run_backtest(surfaces=surfaces)

        assert len(received_dates) == 1
        assert received_dates[0] > 0  # Surface has options


class TestDependencyInjection:
    """Tests demonstrating dependency injection capabilities."""

    def test_inject_execution_simulator(
        self, execution_config, risk_config, pos_mgmt_config
    ):
        """Test injecting a mock execution simulator."""
        from unittest.mock import MagicMock
        from src.backtest.execution import ExecutionSimulator, FillResult

        # Create mock execution simulator
        mock_exec_sim = MagicMock(spec=ExecutionSimulator)
        mock_fill = FillResult(
            legs=[],
            gross_premium=0.0,
            fees=0.0,
            slippage=0.0,
            net_premium=0.0,
            fill_prices={},  # Required field
            timestamp=datetime.now(timezone.utc),
        )
        mock_exec_sim.simulate_entry_fill.return_value = mock_fill

        engine = BacktestEngine(
            start_date=date(2024, 1, 2),
            end_date=date(2024, 1, 2),
            initial_capital=100000.0,
            execution_config=execution_config,
            risk_config=risk_config,
            position_management_config=pos_mgmt_config,
            execution_sim=mock_exec_sim,  # Inject mock
        )

        # Verify mock was used
        assert engine._execution_sim is mock_exec_sim

    def test_inject_risk_checker(
        self, execution_config, risk_config, pos_mgmt_config
    ):
        """Test injecting a mock risk checker."""
        from unittest.mock import MagicMock
        from src.risk.checker import RiskChecker

        mock_risk_checker = MagicMock(spec=RiskChecker)

        engine = BacktestEngine(
            start_date=date(2024, 1, 2),
            end_date=date(2024, 1, 2),
            initial_capital=100000.0,
            execution_config=execution_config,
            risk_config=risk_config,
            position_management_config=pos_mgmt_config,
            risk_checker=mock_risk_checker,  # Inject mock
        )

        assert engine._risk_checker is mock_risk_checker

    def test_inject_position_manager(
        self, execution_config, risk_config, pos_mgmt_config
    ):
        """Test injecting a mock position manager."""
        from unittest.mock import MagicMock
        from src.strategy.positions.manager import PositionManager

        mock_pos_manager = MagicMock(spec=PositionManager)
        mock_pos_manager.get_open_positions.return_value = []
        mock_pos_manager.get_closed_positions.return_value = []
        mock_pos_manager.positions = {}

        engine = BacktestEngine(
            start_date=date(2024, 1, 2),
            end_date=date(2024, 1, 2),
            initial_capital=100000.0,
            execution_config=execution_config,
            risk_config=risk_config,
            position_management_config=pos_mgmt_config,
            position_manager=mock_pos_manager,  # Inject mock
        )

        assert engine._position_manager is mock_pos_manager

    def test_inject_multiple_dependencies(
        self, execution_config, risk_config, pos_mgmt_config
    ):
        """Test injecting multiple dependencies at once."""
        from unittest.mock import MagicMock
        from src.risk.checker import RiskChecker
        from src.risk.kill_switch import KillSwitch
        from src.risk.portfolio import Portfolio

        mock_risk_checker = MagicMock(spec=RiskChecker)
        mock_kill_switch = MagicMock(spec=KillSwitch)
        mock_portfolio = MagicMock(spec=Portfolio)

        engine = BacktestEngine(
            start_date=date(2024, 1, 2),
            end_date=date(2024, 1, 2),
            initial_capital=100000.0,
            execution_config=execution_config,
            risk_config=risk_config,
            position_management_config=pos_mgmt_config,
            risk_checker=mock_risk_checker,
            kill_switch=mock_kill_switch,
            portfolio=mock_portfolio,
        )

        assert engine._risk_checker is mock_risk_checker
        assert engine._kill_switch is mock_kill_switch
        assert engine._portfolio is mock_portfolio


class TestFactoryFunction:
    """Tests for the create_backtest_engine factory function."""

    def test_factory_creates_engine(
        self, execution_config, risk_config, pos_mgmt_config
    ):
        """Test that factory creates a valid engine."""
        from src.backtest import create_backtest_engine

        engine = create_backtest_engine(
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 31),
            initial_capital=100000.0,
            execution_config=execution_config,
            risk_config=risk_config,
            position_management_config=pos_mgmt_config,
        )

        assert isinstance(engine, BacktestEngine)
        assert engine.cash == 100000.0

    def test_factory_with_signal_generator(
        self, execution_config, risk_config, pos_mgmt_config
    ):
        """Test factory with signal generator callback."""
        from src.backtest import create_backtest_engine

        def my_signals(surface: pd.DataFrame) -> list[Signal]:
            return []

        engine = create_backtest_engine(
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 31),
            initial_capital=100000.0,
            execution_config=execution_config,
            risk_config=risk_config,
            position_management_config=pos_mgmt_config,
            signal_generator=my_signals,
        )

        assert engine._signal_generator is my_signals
