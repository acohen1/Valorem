"""Backtest engine for simulating strategy performance over historical data.

This module provides the BacktestEngine class that orchestrates:
- Historical data iteration
- Signal generation
- Order creation and execution simulation
- Position lifecycle management
- P&L tracking and attribution

Architecture:
    The BacktestEngine uses dependency injection for all components. Dependencies
    can be injected directly into the constructor for testing, or created via
    the `create_backtest_engine()` factory for production use.

Example:
    # Testing with mocks
    engine = BacktestEngine(
        start_date=start,
        end_date=end,
        initial_capital=100000,
        execution_config=exec_config,
        risk_config=risk_config,
        position_management_config=pos_config,
        execution_sim=mock_execution_sim,
        risk_checker=mock_risk_checker,
    )

    # Production with factory
    engine = create_backtest_engine(
        start_date=start,
        end_date=end,
        initial_capital=100000,
        execution_config=exec_config,
        risk_config=risk_config,
        position_management_config=pos_config,
    )
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Callable, Optional

import pandas as pd

from src.config.schema import (
    ExecutionConfig,
    PositionManagementConfig,
    RiskConfig,
)
from src.pricing import PositionPricer
from src.risk.checker import RiskChecker
from src.risk.kill_switch import KillSwitch
from src.risk.portfolio import Portfolio, Position, PositionState
from src.strategy.orders import Order, OrderGenerator
from src.strategy.positions.exit_orders import ExitOrder, ExitOrderGenerator
from src.strategy.positions.exit_signals import ExitSignalGenerator
from src.strategy.positions.lifecycle import ManagedPosition
from src.strategy.positions.manager import Fill, PositionManager
from src.strategy.positions.rebalance import RebalanceEngine
from src.strategy.selector import StructureSelector
from src.strategy.sizing import PositionSizer
from src.strategy.types import Greeks, Signal

from .execution import ExecutionSimulator, FillResult
from .results import (
    BacktestMetrics,
    BacktestResult,
    PortfolioSnapshot,
    PositionSnapshot,
    TradeRecord,
    calculate_metrics,
)

logger = logging.getLogger(__name__)


class BacktestEngine:
    """Engine for running backtests over historical data.

    The backtest loop follows the live trading loop structure:
    1. For each trading day:
       a. Load/build surface for the day
       b. Update positions with market data
       c. Evaluate exits for open positions
       d. Execute exit orders
       e. Generate entry signals
       f. Execute entry orders
       g. Record portfolio state
       h. Check kill switch

    All execution uses realistic bid/ask pricing with slippage and fees.
    """

    def __init__(
        self,
        start_date: date,
        end_date: date,
        initial_capital: float,
        execution_config: ExecutionConfig,
        risk_config: RiskConfig,
        position_management_config: PositionManagementConfig,
        signal_generator: Optional[Callable[[pd.DataFrame], list[Signal]]] = None,
        # Dependency injection parameters (all optional)
        pricer: Optional[PositionPricer] = None,
        execution_sim: Optional[ExecutionSimulator] = None,
        risk_checker: Optional[RiskChecker] = None,
        kill_switch: Optional[KillSwitch] = None,
        structure_selector: Optional[StructureSelector] = None,
        position_sizer: Optional[PositionSizer] = None,
        order_generator: Optional[OrderGenerator] = None,
        exit_signal_generator: Optional[ExitSignalGenerator] = None,
        exit_order_generator: Optional[ExitOrderGenerator] = None,
        rebalance_engine: Optional[RebalanceEngine] = None,
        position_manager: Optional[PositionManager] = None,
        portfolio: Optional[Portfolio] = None,
    ) -> None:
        """Initialize backtest engine.

        All component dependencies can be injected for testing. If not provided,
        they are created with default configurations.

        Args:
            start_date: Backtest start date
            end_date: Backtest end date
            initial_capital: Starting capital in USD
            execution_config: Execution rules (pricing, slippage, fees)
            risk_config: Risk limits and constraints
            position_management_config: Exit signals and rebalancing config
            signal_generator: Optional function to generate signals from surface.
                              If not provided, signals must be passed to run_backtest.
            pricer: Cascading pricer for option legs. If None, a
                surface-only pricer is created (backward compatible).
                Shared across execution simulator, exit order generator,
                and position manager.
            execution_sim: Execution simulator (inject for testing)
            risk_checker: Risk checker (inject for testing)
            kill_switch: Kill switch (inject for testing)
            structure_selector: Structure selector (inject for testing)
            position_sizer: Position sizer (inject for testing)
            order_generator: Order generator (inject for testing)
            exit_signal_generator: Exit signal generator (inject for testing)
            exit_order_generator: Exit order generator (inject for testing)
            rebalance_engine: Rebalance engine (inject for testing)
            position_manager: Position manager (inject for testing)
            portfolio: Portfolio (inject for testing)
        """
        self._start_date = start_date
        self._end_date = end_date
        self._initial_capital = initial_capital
        self._exec_config = execution_config
        self._risk_config = risk_config
        self._pos_mgmt_config = position_management_config
        self._signal_generator = signal_generator

        # Pricer is created first so it can be shared with sub-components
        self._pricer = pricer or PositionPricer()

        # Initialize components (use injected or create defaults)
        self._execution_sim = execution_sim or ExecutionSimulator(
            execution_config, pricer=self._pricer
        )
        self._risk_checker = risk_checker or RiskChecker(config=risk_config)
        self._kill_switch = kill_switch or KillSwitch(risk_config.kill_switch)

        self._structure_selector = structure_selector or StructureSelector()
        self._position_sizer = position_sizer or PositionSizer(config=execution_config.sizing)

        # Order generator depends on risk_checker, structure_selector, position_sizer
        self._order_generator = order_generator or OrderGenerator(
            config=execution_config,
            risk_checker=self._risk_checker,
            structure_selector=self._structure_selector,
            position_sizer=self._position_sizer,
        )

        # Exit and rebalance components
        self._exit_signal_gen = exit_signal_generator or ExitSignalGenerator(
            position_management_config.exit_signals
        )
        self._exit_order_gen = exit_order_generator or ExitOrderGenerator(
            execution_config, pricer=self._pricer
        )
        self._rebalance_engine = rebalance_engine or RebalanceEngine(
            position_management_config.rebalancing,
            position_management_config.drift_bands,
        )

        # Position manager depends on exit/rebalance components
        self._position_manager = position_manager or PositionManager(
            config=position_management_config,
            exit_signal_generator=self._exit_signal_gen,
            exit_order_generator=self._exit_order_gen,
            rebalance_engine=self._rebalance_engine,
            pricer=self._pricer,
        )

        # State tracking
        self._portfolio = portfolio or Portfolio()
        self._cash = initial_capital
        self._trades: list[TradeRecord] = []
        self._portfolio_history: list[PortfolioSnapshot] = []
        self._position_history: list[PositionSnapshot] = []

        # Daily tracking
        self._daily_pnl = 0.0
        self._entries_today = 0
        self._exits_today = 0
        self._current_date: Optional[date] = None

    def run_backtest(
        self,
        surfaces: dict[date, pd.DataFrame],
        signals_by_date: Optional[dict[date, list[Signal]]] = None,
        model_predictions: Optional[dict[date, pd.DataFrame]] = None,
    ) -> BacktestResult:
        """Run complete backtest over historical data.

        Args:
            surfaces: Dictionary mapping dates to surface DataFrames.
                     Each surface should have columns: option_symbol, bid, ask,
                     strike, expiry, right, delta, gamma, vega, theta, tenor_days,
                     delta_bucket
            signals_by_date: Optional pre-computed signals for each date.
                            If not provided, uses signal_generator callback.
            model_predictions: Optional model predictions for exit signals

        Returns:
            BacktestResult with complete trade history and metrics
        """
        run_id = f"bt-{uuid.uuid4().hex[:8]}"
        start_time = time.time()

        logger.info(
            f"Starting backtest {run_id}: "
            f"{self._start_date} to {self._end_date}"
        )

        # Sort dates chronologically
        dates = sorted([d for d in surfaces.keys()
                       if self._start_date <= d <= self._end_date])

        if not dates:
            logger.warning("No surface data in backtest date range")
            return self._create_empty_result(run_id, start_time)

        trading_days = 0
        for current_date in dates:
            if self._kill_switch.is_triggered:
                logger.warning(f"Kill switch triggered, stopping backtest at {current_date}")
                break

            self._run_single_day(
                current_date=current_date,
                surface=surfaces[current_date],
                signals=signals_by_date.get(current_date) if signals_by_date else None,
                model_predictions=model_predictions.get(current_date) if model_predictions else None,
            )
            trading_days += 1

        # Compute final metrics
        run_time = time.time() - start_time
        # Use the last processed surface for equity calculation
        last_surface = surfaces[current_date] if dates else pd.DataFrame()
        final_equity = self._compute_equity(last_surface)

        metrics = calculate_metrics(
            trades=self._trades,
            portfolio_history=self._portfolio_history,
            initial_capital=self._initial_capital,
            trading_days=trading_days,
        )

        result = BacktestResult(
            run_id=run_id,
            start_date=self._start_date,
            end_date=self._end_date,
            initial_capital=self._initial_capital,
            final_equity=final_equity,
            run_time_seconds=run_time,
            metrics=metrics,
            trades=self._trades,
            portfolio_history=self._portfolio_history,
            position_history=self._position_history,
        )

        logger.info(
            f"Backtest {run_id} complete: "
            f"{trading_days} days, "
            f"${final_equity:,.2f} final equity, "
            f"{metrics.total_return_pct:+.2f}% return"
        )

        return result

    def _run_single_day(
        self,
        current_date: date,
        surface: pd.DataFrame,
        signals: Optional[list[Signal]],
        model_predictions: Optional[pd.DataFrame],
    ) -> None:
        """Run backtest for a single trading day.

        Args:
            current_date: Current simulation date
            surface: Market surface for the day
            signals: Entry signals for the day (optional)
            model_predictions: Model predictions (optional)
        """
        # Reset daily counters
        if self._current_date != current_date:
            self._reset_daily_counters(current_date)

        timestamp = datetime.combine(current_date, datetime.min.time()).replace(
            tzinfo=timezone.utc
        )

        # 1. Check for expirations FIRST (before any pricing attempts)
        self._check_expirations(current_date)

        # 2. Preload quotes for open position symbols not in today's surface
        surface_syms = set(surface["option_symbol"]) if not surface.empty else set()
        missing_syms = [
            leg.symbol
            for pos in self._position_manager.get_open_positions()
            for leg in pos.legs
            if leg.symbol not in surface_syms
        ]
        if missing_syms:
            self._pricer.preload(current_date, missing_syms)

        # 3. Update positions with market data
        self._position_manager.update_positions(surface, as_of=current_date)

        # 4. Sync portfolio from position manager
        self._sync_portfolio()

        # 5. Evaluate and execute exits
        self._process_exits(surface, model_predictions, timestamp)

        # 6. Generate and execute entries
        if signals is None and self._signal_generator is not None:
            signals = self._signal_generator(surface)

        if signals:
            self._process_entries(signals, surface, timestamp)

        # 7. Record portfolio snapshot
        self._record_portfolio_snapshot(current_date, timestamp, surface)

        # 8. Record position snapshots
        self._record_position_snapshots(timestamp)

        # 9. Check kill switch
        self._check_kill_switch()

    def _process_entries(
        self,
        signals: list[Signal],
        surface: pd.DataFrame,
        timestamp: datetime,
    ) -> None:
        """Process entry signals and execute fills.

        Args:
            signals: Entry signals to process
            surface: Current market surface
            timestamp: Current timestamp
        """
        # Get current underlying price for risk checks
        underlying_price = self._get_underlying_price(surface)

        # Generate orders from signals
        result = self._order_generator.generate_orders(
            signals=signals,
            surface=surface,
            portfolio=self._portfolio,
            underlying_price=underlying_price,
        )

        if result.rejected_signals:
            logger.debug(f"Rejected {len(result.rejected_signals)} signals")

        # Execute each order
        for order in result.orders:
            # Simulate fill
            fill_result = self._execution_sim.simulate_entry_fill(
                legs=order.legs,
                surface=surface,
                timestamp=timestamp,
                as_of=self._current_date,
            )

            # Check if we have enough capital
            # Positive net_premium = debit (paid money), negative = credit (received)
            if fill_result.net_premium > 0:  # Debit order
                if fill_result.net_premium > self._cash:
                    logger.debug(f"Insufficient capital for order {order.order_id}")
                    continue

            # Record the fill
            self._record_entry(order, fill_result)
            self._entries_today += 1

    def _process_exits(
        self,
        surface: pd.DataFrame,
        model_predictions: Optional[pd.DataFrame],
        timestamp: datetime,
    ) -> None:
        """Process exit signals and execute fills.

        Args:
            surface: Current market surface
            model_predictions: Model predictions (optional)
            timestamp: Current timestamp
        """
        # Evaluate positions for exits
        exit_orders = self._position_manager.evaluate_exits(
            surface=surface,
            model_predictions=model_predictions,
            portfolio=self._portfolio,
            as_of=self._current_date,
        )

        for exit_order in exit_orders:
            # Simulate fill
            fill_result = self._execution_sim.simulate_exit_fill(
                legs=exit_order.legs,
                surface=surface,
                timestamp=timestamp,
                as_of=self._current_date,
            )

            # Record the exit
            self._record_exit(exit_order, fill_result)
            self._exits_today += 1

    def _check_expirations(self, current_date: date) -> None:
        """Check for and process expired positions.

        Args:
            current_date: Current simulation date
        """
        for position in self._position_manager.get_open_positions():
            # Only expire when ALL legs have expired (calendar spreads have
            # legs with different expiries — don't kill far leg early).
            # Use strict < so positions can still be traded on their expiry day;
            # they expire the day AFTER.
            all_expired = all(leg.expiry < current_date for leg in position.legs)
            if all_expired:
                self._position_manager.record_expiration(position.position_id)
                logger.debug(f"Position {position.position_id} expired")

    def _record_entry(self, order: Order, fill: FillResult) -> None:
        """Record an entry trade.

        Args:
            order: Original order
            fill: Fill result
        """
        # Add position to manager
        manager_fill = Fill(
            legs=fill.legs,
            fill_price=fill.net_premium,
            timestamp=fill.timestamp,
        )
        position = self._position_manager.add_position(order, manager_fill)

        # Update cash: positive net_premium means we paid, negative means we received
        # Subtract because paying reduces cash, receiving (negative) adds to cash
        self._cash -= fill.net_premium

        # Record trade
        trade = TradeRecord(
            trade_id=f"entry-{order.order_id}",
            timestamp=fill.timestamp,
            trade_type="ENTRY",
            position_id=position.position_id,
            structure_type=order.structure_type,
            legs=self._serialize_legs(fill.legs),
            gross_premium=fill.gross_premium,
            fees=fill.fees,
            slippage=fill.slippage,
            net_premium=fill.net_premium,
            signal_type=order.signal.signal_type.value,
        )
        self._trades.append(trade)

        logger.debug(
            f"Entry: {order.structure_type} @ ${fill.net_premium:,.2f}, "
            f"fees=${fill.fees:.2f}, slippage=${fill.slippage:.2f}"
        )

    def _record_exit(self, exit_order: ExitOrder, fill: FillResult) -> None:
        """Record an exit trade.

        Args:
            exit_order: Exit order
            fill: Fill result
        """
        position = self._position_manager.get_position(exit_order.position_id)
        if position is None:
            logger.warning(f"Position {exit_order.position_id} not found for exit")
            return

        # Get realized P&L before recording exit
        entry_price = position.entry_price

        # Record exit in position manager — negate net_premium so mark_closed
        # sees positive value for received money (matching its exit - entry formula).
        self._position_manager.record_exit(exit_order, -fill.net_premium, fill.timestamp)

        # Update cash: for exits, negative net_premium means we received money
        # Subtract negative = add to cash
        self._cash -= fill.net_premium

        # Calculate realized P&L for this trade
        # Entry: positive entry_price = we paid that amount
        # Exit: negative fill.net_premium = we received that amount
        # P&L = what we received - what we paid = -fill.net_premium - entry_price
        realized_pnl = -fill.net_premium - entry_price

        # Update daily P&L
        self._daily_pnl += realized_pnl

        # Record trade
        trade = TradeRecord(
            trade_id=f"exit-{exit_order.order_id}",
            timestamp=fill.timestamp,
            trade_type="EXIT",
            position_id=exit_order.position_id,
            structure_type=exit_order.structure_type,
            legs=self._serialize_legs(fill.legs),
            gross_premium=fill.gross_premium,
            fees=fill.fees,
            slippage=fill.slippage,
            net_premium=realized_pnl,  # Store realized P&L as net_premium for exits
            signal_type=exit_order.exit_signal.exit_type.value,
            exit_reason=exit_order.exit_signal.reason,
        )
        self._trades.append(trade)

        logger.debug(
            f"Exit: {exit_order.structure_type} @ ${fill.net_premium:,.2f}, "
            f"realized_pnl=${realized_pnl:,.2f}, reason={exit_order.exit_signal.exit_type.value}"
        )

    def _sync_portfolio(self) -> None:
        """Sync portfolio from position manager (single source of truth).

        Uses PositionManager.get_portfolio_state() to get an immutable snapshot
        and converts to Portfolio for backwards compatibility with risk checks.
        """
        state = self._position_manager.get_portfolio_state(
            daily_pnl=self._daily_pnl,
            max_acceptable_loss=self._risk_config.caps.max_daily_loss,
        )
        self._portfolio = Portfolio(
            positions=list(state.positions),
            daily_pnl=state.daily_pnl,
            max_acceptable_loss=state.max_acceptable_loss,
        )

    def _record_portfolio_snapshot(
        self,
        current_date: date,
        timestamp: datetime,
        surface: pd.DataFrame,
    ) -> None:
        """Record current portfolio state.

        Args:
            current_date: Current date
            timestamp: Current timestamp
            surface: Current surface for valuation
        """
        # Compute positions value (sum of MTM values)
        positions_value = self._compute_positions_value(surface)

        # Total equity = cash + positions value
        total_equity = self._cash + positions_value

        # Get unrealized P&L from open positions
        unrealized_pnl = sum(
            p.unrealized_pnl or 0.0
            for p in self._position_manager.get_open_positions()
        )

        # Get realized P&L from closed positions
        realized_pnl = sum(
            p.realized_pnl or 0.0
            for p in self._position_manager.get_closed_positions()
        )

        # Aggregate Greeks
        net_greeks = self._compute_net_greeks()

        snapshot = PortfolioSnapshot(
            timestamp=timestamp,
            date=current_date,
            cash=self._cash,
            positions_value=positions_value,
            total_equity=total_equity,
            unrealized_pnl=unrealized_pnl,
            realized_pnl=realized_pnl,
            daily_pnl=self._daily_pnl,
            net_delta=net_greeks.delta,
            net_gamma=net_greeks.gamma,
            net_vega=net_greeks.vega,
            net_theta=net_greeks.theta,
            open_positions=len(self._position_manager.get_open_positions()),
            entries_today=self._entries_today,
            exits_today=self._exits_today,
        )
        self._portfolio_history.append(snapshot)

    def _record_position_snapshots(self, timestamp: datetime) -> None:
        """Record snapshots of all positions.

        Args:
            timestamp: Current timestamp
        """
        for pos in self._position_manager.positions.values():
            snapshot = PositionSnapshot(
                timestamp=timestamp,
                position_id=pos.position_id,
                state=pos.state.value,
                structure_type=pos.structure_type,
                entry_price=pos.entry_price,
                current_price=pos.current_price or pos.entry_price,
                unrealized_pnl=pos.unrealized_pnl or 0.0,
                realized_pnl=pos.realized_pnl or 0.0,
                delta=pos.current_greeks.delta if pos.current_greeks else 0.0,
                gamma=pos.current_greeks.gamma if pos.current_greeks else 0.0,
                vega=pos.current_greeks.vega if pos.current_greeks else 0.0,
                theta=pos.current_greeks.theta if pos.current_greeks else 0.0,
                days_held=pos.days_held or 0,
                days_to_expiry=pos.days_to_expiry,
            )
            self._position_history.append(snapshot)

    def _compute_positions_value(self, surface: pd.DataFrame) -> float:
        """Compute total value of open positions.

        Args:
            surface: Current market surface

        Returns:
            Total positions value in USD
        """
        pricing_date = self._current_date or date.min
        total = 0.0

        for pos in self._position_manager.get_open_positions():
            total += self._pricer.price_position(pos.legs, surface, pricing_date)

        return total

    def _compute_net_greeks(self) -> Greeks:
        """Compute net Greeks across all open positions.

        Returns:
            Aggregate Greeks
        """
        total = Greeks(delta=0.0, gamma=0.0, vega=0.0, theta=0.0)

        for pos in self._position_manager.get_open_positions():
            if pos.current_greeks:
                total = total + pos.current_greeks

        return total

    def _compute_equity(self, surface: pd.DataFrame | None = None) -> float:
        """Compute current total equity.

        Args:
            surface: Optional market surface for position valuation.
                If not provided, uses entry prices for positions.

        Returns:
            Total equity (cash + market value of positions)
        """
        effective_surface = surface if surface is not None else pd.DataFrame()
        pricing_date = self._current_date or date.min

        positions_value = 0.0
        for pos in self._position_manager.get_open_positions():
            positions_value += self._pricer.price_position(
                pos.legs, effective_surface, pricing_date
            )

        return self._cash + positions_value

    def _get_underlying_price(self, surface: pd.DataFrame) -> float:
        """Get underlying price from surface.

        Args:
            surface: Market surface

        Returns:
            Underlying price (uses median ATM strike as proxy)
        """
        if surface.empty:
            return 0.0
        if "underlying_price" in surface.columns and len(surface) > 0:
            return float(surface["underlying_price"].iloc[0])

        # Fallback: use median strike as proxy
        return float(surface["strike"].median())

    def _check_kill_switch(self) -> None:
        """Check kill switch conditions."""
        # Update portfolio daily P&L
        self._portfolio.daily_pnl = self._daily_pnl

        result = self._kill_switch.check_daily_loss(self._portfolio)
        if result.triggered:
            logger.warning(
                f"Kill switch triggered: {result.reason}"
            )

    def _reset_daily_counters(self, new_date: date) -> None:
        """Reset daily tracking counters.

        Args:
            new_date: New trading date
        """
        self._current_date = new_date
        self._daily_pnl = 0.0
        self._entries_today = 0
        self._exits_today = 0
        self._portfolio.reset_daily_pnl()

    def _serialize_legs(self, legs: list) -> list[dict]:
        """Serialize option legs for storage.

        Args:
            legs: Option legs

        Returns:
            List of leg dictionaries
        """
        return [
            {
                "symbol": leg.symbol,
                "qty": leg.qty,
                "price": leg.entry_price,
                "strike": leg.strike,
                "expiry": str(leg.expiry),
                "right": leg.right.value,
            }
            for leg in legs
        ]

    def _create_empty_result(self, run_id: str, start_time: float) -> BacktestResult:
        """Create empty result when no data available.

        Args:
            run_id: Backtest run ID
            start_time: Start timestamp

        Returns:
            Empty BacktestResult
        """
        from .results import _empty_metrics

        return BacktestResult(
            run_id=run_id,
            start_date=self._start_date,
            end_date=self._end_date,
            initial_capital=self._initial_capital,
            final_equity=self._initial_capital,
            run_time_seconds=time.time() - start_time,
            metrics=_empty_metrics(),
            trades=[],
            portfolio_history=[],
            position_history=[],
        )

    @property
    def position_manager(self) -> PositionManager:
        """Get position manager for external access."""
        return self._position_manager

    @property
    def portfolio(self) -> Portfolio:
        """Get current portfolio state."""
        return self._portfolio

    @property
    def cash(self) -> float:
        """Get current cash balance."""
        return self._cash


def create_backtest_engine(
    start_date: date,
    end_date: date,
    initial_capital: float,
    execution_config: ExecutionConfig,
    risk_config: RiskConfig,
    position_management_config: PositionManagementConfig,
    signal_generator: Optional[Callable[[pd.DataFrame], list[Signal]]] = None,
) -> BacktestEngine:
    """Factory function for creating a BacktestEngine with default dependencies.

    This function creates all dependencies internally, which is the recommended
    approach for production use. For testing, inject dependencies directly
    into the BacktestEngine constructor.

    Args:
        start_date: Backtest start date
        end_date: Backtest end date
        initial_capital: Starting capital in USD
        execution_config: Execution rules (pricing, slippage, fees)
        risk_config: Risk limits and constraints
        position_management_config: Exit signals and rebalancing config
        signal_generator: Optional function to generate signals from surface

    Returns:
        Configured BacktestEngine instance

    Example:
        engine = create_backtest_engine(
            start_date=date(2024, 1, 1),
            end_date=date(2024, 12, 31),
            initial_capital=100000,
            execution_config=exec_config,
            risk_config=risk_config,
            position_management_config=pos_config,
        )
        result = engine.run_backtest(surfaces=surfaces)
    """
    return BacktestEngine(
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital,
        execution_config=execution_config,
        risk_config=risk_config,
        position_management_config=position_management_config,
        signal_generator=signal_generator,
    )
