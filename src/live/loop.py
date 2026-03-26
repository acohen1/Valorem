"""Main trading loop for paper and live trading.

This module provides the TradingLoop class that orchestrates the complete
trading cycle: data fetching, signal generation, order creation, execution,
and state management.
"""

import logging
import signal
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Optional

import pandas as pd

from ..config.schema import ExecutionConfig, PaperConfig, RiskConfig
from ..risk.checker import RiskChecker
from ..risk.kill_switch import KillSwitch, KillSwitchResult
from ..risk.portfolio import CONTRACT_MULTIPLIER, Portfolio, Position, PositionState
from ..strategy.orders import Order, OrderGenerator
from ..strategy.positions.exit_orders import ExitOrderGenerator
from ..strategy.positions.exit_signals import ExitSignalGenerator
from ..strategy.positions.manager import Fill as ManagerFill
from ..strategy.positions.manager import PositionManager
from ..strategy.positions.rebalance import RebalanceEngine
from ..strategy.types import Signal, SignalType
from ..utils.validation import validate_protocol
from .features import FeatureProvider
from .monitoring import TradingMonitor
from .positions import PositionTracker
from .router import Fill, OrderRouter
from .signal_generator import SignalGenerator
from .state import StateManager
from .surface_provider import SurfaceProvider

logger = logging.getLogger(__name__)


@dataclass
class LoopState:
    """Current state of the trading loop.

    Attributes:
        iteration: Current loop iteration number
        is_running: Whether the loop is running
        last_surface_time: When the last surface was built
        last_signal_count: Number of signals from last iteration
        last_order_count: Number of orders generated last iteration
        last_fill_count: Number of fills last iteration
        errors_count: Cumulative error count
        kill_switch_triggered: Whether kill switch is active
    """

    iteration: int = 0
    is_running: bool = False
    last_surface_time: Optional[datetime] = None
    last_signal_count: int = 0
    last_order_count: int = 0
    last_fill_count: int = 0
    errors_count: int = 0
    kill_switch_triggered: bool = False


@dataclass
class LoopMetrics:
    """Metrics collected during trading loop execution.

    Attributes:
        total_iterations: Total number of loop iterations
        total_signals: Total signals generated
        total_orders: Total orders created
        total_fills: Total successful fills
        total_rejections: Total rejected orders
        start_time: When the loop started
        end_time: When the loop stopped (if stopped)
        daily_pnl: Current day's P&L
    """

    total_iterations: int = 0
    total_signals: int = 0
    total_orders: int = 0
    total_fills: int = 0
    total_rejections: int = 0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    daily_pnl: float = 0.0


class TradingLoop:
    """Main event loop for paper/live trading.

    The trading loop executes the following cycle:
    1. Fetch latest market data and build surface
    2. Generate trading signals from model predictions
    3. Create orders from signals (with risk checks)
    4. Check kill switch conditions
    5. Route orders for execution
    6. Update positions with fills
    7. Evaluate exit signals and rebalancing
    8. Save state snapshot
    9. Sleep until next cycle

    The loop can be stopped gracefully via the stop() method or
    SIGINT/SIGTERM signals.

    Example:
        >>> loop = TradingLoop(
        ...     paper_config=paper_config,
        ...     execution_config=execution_config,
        ...     risk_config=risk_config,
        ...     surface_provider=surface_provider,
        ...     signal_generator=signal_generator,
        ...     order_router=order_router,
        ... )
        >>> loop.start()  # Blocks until stop() called or kill switch triggers
    """

    def __init__(
        self,
        paper_config: PaperConfig,
        execution_config: ExecutionConfig,
        risk_config: RiskConfig,
        surface_provider: SurfaceProvider,
        signal_generator: SignalGenerator,
        order_router: OrderRouter,
        feature_provider: Optional[FeatureProvider] = None,
        order_generator: Optional[OrderGenerator] = None,
        risk_checker: Optional[RiskChecker] = None,
        position_manager: Optional[PositionManager] = None,
        state_manager: Optional[StateManager] = None,
        position_tracker: Optional[PositionTracker] = None,
        monitor: Optional[TradingMonitor] = None,
        on_fill_callback: Optional[Callable[[Fill], None]] = None,
        on_iteration_callback: Optional[Callable[[LoopState], None]] = None,
    ) -> None:
        """Initialize trading loop.

        Args:
            paper_config: Paper trading configuration
            execution_config: Execution configuration
            risk_config: Risk management configuration
            surface_provider: Provider for latest volatility surface
            signal_generator: Generator for trading signals
            order_router: Router for order execution
            feature_provider: Optional provider for computing features
            order_generator: Order generator (created if not provided)
            risk_checker: Risk checker (created if not provided)
            position_manager: Position manager (created if not provided)
            state_manager: State manager for persistence (created if not provided)
            position_tracker: Position tracker for mark-to-market (created if not provided)
            monitor: Trading monitor for metrics/alerts (created if not provided)
            on_fill_callback: Optional callback when fills occur
            on_iteration_callback: Optional callback after each iteration
        """
        self._paper_config = paper_config
        self._execution_config = execution_config
        self._risk_config = risk_config

        # Validate required protocols at construction time
        validate_protocol(surface_provider, SurfaceProvider, "surface_provider")
        validate_protocol(signal_generator, SignalGenerator, "signal_generator")
        if feature_provider is not None:
            validate_protocol(feature_provider, FeatureProvider, "feature_provider")

        self._surface_provider = surface_provider
        self._signal_generator = signal_generator
        self._order_router = order_router
        self._feature_provider = feature_provider

        # Initialize components
        self._risk_checker = risk_checker or RiskChecker(risk_config)
        self._order_generator = order_generator or OrderGenerator(
            config=execution_config,
            risk_checker=self._risk_checker,
        )

        # Position management
        if position_manager is not None:
            self._position_manager = position_manager
        else:
            exit_signal_gen = ExitSignalGenerator(
                config=risk_config.position_management.exit_signals
            )
            exit_order_gen = ExitOrderGenerator(config=execution_config)
            rebalance_engine = RebalanceEngine(
                rebalancing_config=risk_config.position_management.rebalancing,
                drift_bands_config=risk_config.position_management.drift_bands,
            )
            self._position_manager = PositionManager(
                config=risk_config.position_management,
                exit_signal_generator=exit_signal_gen,
                exit_order_generator=exit_order_gen,
                rebalance_engine=rebalance_engine,
            )

        # Kill switch
        self._kill_switch = KillSwitch(risk_config.kill_switch)

        # State management (M28)
        self._state_manager = state_manager or StateManager(paper_config)

        # Load portfolio from state manager (enables crash recovery)
        self._portfolio = self._state_manager.portfolio
        if self._portfolio.max_acceptable_loss == 5000.0:  # Default value
            self._portfolio = Portfolio(
                positions=self._portfolio.positions,
                closed_positions=self._portfolio.closed_positions,
                daily_pnl=self._portfolio.daily_pnl,
                max_acceptable_loss=risk_config.caps.max_daily_loss,
            )

        # Position tracker for mark-to-market (M28)
        self._position_tracker = position_tracker or PositionTracker(self._portfolio)

        # Trading monitor (M28)
        self._monitor = monitor or TradingMonitor(
            log_to_stdout=True,
            log_to_file=False,
        )

        # Callbacks
        self._on_fill = on_fill_callback
        self._on_iteration = on_iteration_callback

        # State
        self._state = LoopState()
        # Resume from last iteration if loading from state (crash recovery)
        self._state.iteration = self._state_manager.last_iteration
        self._metrics = LoopMetrics()
        self._stop_requested = False

        # Setup signal handlers
        self._setup_signal_handlers()

    @property
    def state(self) -> LoopState:
        """Get current loop state."""
        return self._state

    @property
    def metrics(self) -> LoopMetrics:
        """Get loop metrics."""
        return self._metrics

    @property
    def portfolio(self) -> Portfolio:
        """Get current portfolio."""
        return self._portfolio

    @property
    def position_manager(self) -> PositionManager:
        """Get position manager."""
        return self._position_manager

    @property
    def state_manager(self) -> StateManager:
        """Get state manager."""
        return self._state_manager

    @property
    def position_tracker(self) -> PositionTracker:
        """Get position tracker."""
        return self._position_tracker

    @property
    def feature_provider(self) -> Optional[FeatureProvider]:
        """Get feature provider."""
        return self._feature_provider

    @property
    def monitor(self) -> TradingMonitor:
        """Get trading monitor."""
        return self._monitor

    def start(self) -> None:
        """Start the trading loop.

        This method blocks until stop() is called, kill switch triggers,
        or max_loop_iterations is reached.
        """
        if self._state.is_running:
            logger.warning("Trading loop is already running")
            return

        logger.info("Starting trading loop")
        self._state.is_running = True
        self._stop_requested = False
        self._metrics.start_time = datetime.now(timezone.utc)

        try:
            while self._should_continue():
                try:
                    self._run_iteration()
                except Exception as e:
                    self._handle_error(e)
                    if self._paper_config.halt_on_error:
                        logger.error("Halting due to error (halt_on_error=True)")
                        break

                # Sleep until next cycle
                if self._should_continue():
                    time.sleep(self._paper_config.loop_interval_seconds)

        finally:
            self._state.is_running = False
            self._metrics.end_time = datetime.now(timezone.utc)
            logger.info(
                f"Trading loop stopped after {self._state.iteration} iterations"
            )

    def stop(self) -> None:
        """Request graceful stop of the trading loop."""
        logger.info("Stop requested")
        self._stop_requested = True

    def run_single_iteration(self) -> None:
        """Run a single iteration of the trading loop.

        Useful for testing or manual step-through execution.
        """
        self._run_iteration()

    def _should_continue(self) -> bool:
        """Check if the loop should continue."""
        if self._stop_requested:
            return False

        if self._state.kill_switch_triggered:
            return False

        max_iter = self._paper_config.max_loop_iterations
        if max_iter > 0 and self._state.iteration >= max_iter:
            logger.info(f"Max iterations ({max_iter}) reached")
            return False

        return True

    def _run_iteration(self) -> None:
        """Execute a single trading loop iteration."""
        self._state.iteration += 1
        self._metrics.total_iterations += 1
        logger.info(f"=== Iteration {self._state.iteration} ===")

        # 1. Get latest surface
        surface = self._get_surface()
        if surface.empty:
            logger.warning("Empty surface - skipping iteration")
            return

        self._state.last_surface_time = datetime.now(timezone.utc)

        # 2. Check kill switch (early exit)
        kill_result = self._check_kill_switch(surface)
        if kill_result.triggered:
            self._state.kill_switch_triggered = True
            logger.critical(f"Kill switch triggered: {kill_result.reason}")
            return

        # 3. Update feature provider with current surface
        if self._feature_provider:
            self._feature_provider.update(surface)

        # 4. Update existing positions with current market data
        self._update_positions(surface)

        # 5. Evaluate exit signals
        self._process_exit_signals(surface)

        # 6. Generate new signals
        signals = self._generate_signals(surface)
        self._state.last_signal_count = len(signals)
        self._metrics.total_signals += len(signals)
        logger.debug(f"Generated {len(signals)} signals")

        # 7. Generate orders from signals
        orders = self._generate_orders(signals, surface)
        self._state.last_order_count = len(orders)
        self._metrics.total_orders += len(orders)
        logger.debug(f"Generated {len(orders)} orders")

        # 8. Route orders for execution
        fills = self._route_orders(orders, surface)
        self._state.last_fill_count = len(fills)
        self._metrics.total_fills += len(fills)
        logger.debug(f"Received {len(fills)} fills")

        # 9. Process fills
        for fill, order in zip(fills, orders):
            self._process_fill(fill, order)

        # 10. Save state (if interval reached)
        if self._state.iteration % self._paper_config.save_state_interval == 0:
            self._save_state()

        # 11. Log metrics via monitor
        self._monitor.log_metrics(
            portfolio=self._portfolio,
            iteration=self._state.iteration,
            signals_generated=self._state.last_signal_count,
            orders_created=self._state.last_order_count,
            fills_received=self._state.last_fill_count,
            errors_count=self._state.errors_count,
        )

        # 12. Callback
        if self._on_iteration:
            self._on_iteration(self._state)

        # Update daily P&L in metrics
        self._metrics.daily_pnl = self._portfolio.daily_pnl

    def _get_surface(self) -> pd.DataFrame:
        """Fetch latest volatility surface.

        Raises:
            Exception: Propagates any error from the surface provider.
        """
        return self._surface_provider.get_latest_surface()

    def _generate_signals(
        self, surface: pd.DataFrame
    ) -> list[Signal]:
        """Generate trading signals."""
        try:
            # Get features from provider if available
            features = None
            if self._feature_provider:
                features = self._feature_provider.get_features(surface)
                if features.empty:
                    logger.debug("No features available yet (building history)")

            return self._signal_generator.generate_signals(surface, features)
        except Exception as e:
            logger.error(f"Failed to generate signals: {e}")
            return []

    def _generate_orders(
        self,
        signals: list[Signal],
        surface: pd.DataFrame,
    ) -> list[Order]:
        """Generate orders from signals."""
        if not signals:
            return []

        underlying_price = self._get_underlying_price(surface)
        result = self._order_generator.generate_orders(
            signals=signals,
            surface=surface,
            portfolio=self._portfolio,
            underlying_price=underlying_price,
        )

        rejections = len(result.rejected_signals)
        if rejections > 0:
            self._metrics.total_rejections += rejections
            logger.info(f"{rejections} signals rejected")
            for idx, reason in result.rejection_reasons.items():
                logger.debug(f"  Signal {idx}: {reason}")

        return result.orders

    def _route_orders(
        self,
        orders: list[Order],
        surface: pd.DataFrame,
    ) -> list[Fill]:
        """Route orders for execution."""
        fills: list[Fill] = []

        for order in orders:
            try:
                fill = self._order_router.route_order(order, surface)
                if fill:
                    fills.append(fill)
            except Exception as e:
                logger.error(f"Failed to route order {order.order_id}: {e}")

        return fills

    def _process_fill(self, fill: Fill, order: Order) -> None:
        """Process a fill - add position and sync portfolio from manager."""
        # Convert router Fill to manager Fill
        manager_fill = ManagerFill(
            legs=fill.legs,
            fill_price=fill.net_premium,
            timestamp=fill.timestamp,
        )

        # Add position to the single source of truth (PositionManager)
        position = self._position_manager.add_position(order, manager_fill)

        # Sync portfolio from position manager
        self._sync_portfolio()

        # Entry premium is a cash outflow, not a realized loss.
        # Daily P&L should only update on exits and MTM, not entries.
        # (Backtest engine correctly omits this — see engine.py _record_entry.)

        # Record fill in state manager (M28)
        self._state_manager.record_order(order)
        self._state_manager.record_fill(fill)

        # Update position tracker's portfolio reference
        self._position_tracker.portfolio = self._portfolio

        # Callback
        if self._on_fill:
            self._on_fill(fill)

        logger.info(
            f"Fill processed: position={position.position_id}, "
            f"net_premium={fill.net_premium:.2f}"
        )

    def _sync_portfolio(self) -> None:
        """Sync portfolio from position manager (single source of truth).

        Uses PositionManager.get_portfolio_state() to get an immutable snapshot
        and converts to Portfolio for backwards compatibility. Closed positions
        are pulled from the PositionManager to preserve history for state saves.
        """
        state = self._position_manager.get_portfolio_state(
            daily_pnl=self._portfolio.daily_pnl,
            max_acceptable_loss=self._risk_config.caps.max_daily_loss,
        )

        # Build closed positions list from position manager
        closed = [
            Position(
                legs=mp.legs,
                entry_time=mp.entry_time,
                realized_pnl=mp.realized_pnl or 0.0,
                position_id=mp.position_id,
                structure_type=mp.structure_type,
                state=PositionState.CLOSED,
                max_loss=mp.max_loss,
            )
            for mp in self._position_manager.get_closed_positions()
        ]

        self._portfolio = Portfolio(
            positions=list(state.positions),
            closed_positions=closed,
            daily_pnl=state.daily_pnl,
            max_acceptable_loss=state.max_acceptable_loss,
        )

    def _update_positions(self, surface: pd.DataFrame) -> None:
        """Update positions with current market data using PositionTracker."""
        # Use position tracker for mark-to-market (M28)
        self._position_tracker.portfolio = self._portfolio
        self._portfolio = self._position_tracker.update_positions(surface)

        # Also update position manager's positions
        for position_id, position in self._position_manager.positions.items():
            snapshot = self._position_tracker.get_snapshot(position_id)
            if snapshot:
                # Compute days_to_expiry from actual leg expiry dates,
                # not by decrementing (which fires every loop iteration).
                min_dte = min(
                    (leg.expiry - datetime.now(timezone.utc).date()).days
                    for leg in position.legs
                ) if position.legs else 0
                position.update_market_data(
                    current_price=snapshot.current_value,
                    current_greeks=snapshot.current_greeks,
                    days_to_expiry=max(min_dte, 0),
                )

    def _process_exit_signals(self, surface: pd.DataFrame) -> None:
        """Evaluate exit signals and close positions if needed."""
        # Use position manager's evaluate_exits which checks all positions
        exit_orders = self._position_manager.evaluate_exits(
            surface=surface,
            model_predictions=None,  # Not using model predictions for exit signals
            portfolio=self._portfolio,
        )

        for exit_order in exit_orders:
            position_id = exit_order.position_id
            position = self._position_manager.positions.get(position_id)
            if position is None:
                continue

            logger.debug(
                f"Exit order for {position_id}: {exit_order.exit_signal.exit_type.value}"
            )

            # Execute the exit
            self._execute_exit_order(position, exit_order, surface)

    def _execute_exit_order(
        self,
        position,
        exit_order,
        surface: pd.DataFrame,
    ) -> None:
        """Execute an exit order for a position."""
        # Get exit signal and closing legs from the exit order
        exit_signal = exit_order.exit_signal
        closing_legs = exit_order.legs  # ExitOrder.legs contains the closing legs

        # Simulate exit fill
        fill_prices = {}
        net_premium = 0.0

        for leg in closing_legs:
            rows = surface[surface["option_symbol"] == leg.symbol]
            if not rows.empty:
                row = rows.iloc[0]
                if leg.qty > 0:
                    price = row["ask"]  # Buy to close
                else:
                    price = row["bid"]  # Sell to close
                fill_prices[leg.symbol] = price
                net_premium += leg.qty * price * CONTRACT_MULTIPLIER

        # Record exit in position manager — negate net_premium so mark_closed
        # sees positive value for received money (matching its exit - entry formula).
        self._position_manager.record_exit(
            exit_order=exit_order,
            fill_price=-net_premium,
            fill_time=datetime.now(timezone.utc),
        )

        # Compute realized P&L from actual fill, consistent with backtest engine:
        # P&L = what we received - what we paid = -net_premium - entry_price
        realized_pnl = -net_premium - position.entry_price

        # Sync portfolio from position manager
        self._sync_portfolio()

        # Update daily P&L with realized P&L
        self._portfolio.update_daily_pnl(realized_pnl)

        logger.info(
            f"Position {position.position_id} closed: "
            f"P&L={realized_pnl:.2f}"
        )

    def _check_kill_switch(self, surface: pd.DataFrame) -> KillSwitchResult:
        """Check kill switch conditions.

        Checks daily loss limits and liquidity conditions. Stress testing
        is handled separately during pre-trade risk checks.
        """
        # Compute max spread for liquidity check
        max_spread_pct = self._compute_max_spread(surface)

        return self._kill_switch.check(
            portfolio=self._portfolio,
            stress_result=None,  # Stress testing done in pre-trade checks
            max_spread_pct=max_spread_pct,
        )

    def _compute_max_spread(self, surface: pd.DataFrame) -> float:
        """Compute maximum spread percentage in market."""
        if surface.empty:
            return 0.0

        if "bid" not in surface.columns or "ask" not in surface.columns:
            return 0.0

        # Use mid-price as denominator to avoid inf when bid=0 (deep OTM)
        mid = (surface["ask"] + surface["bid"]) / 2
        spreads = (surface["ask"] - surface["bid"]) / mid.replace(0, float("nan"))
        return float(spreads.dropna().max()) if spreads.notna().any() else 0.0

    def _get_underlying_price(self, surface: pd.DataFrame) -> float:
        """Get underlying price from surface."""
        if "underlying_price" in surface.columns:
            return float(surface["underlying_price"].iloc[0])
        # Fallback: use ATM strike
        atm = surface[surface["delta_bucket"] == "ATM"]
        if not atm.empty:
            return float(atm["strike"].iloc[0])
        return float(surface["strike"].mean())

    def _save_state(self) -> None:
        """Save current state snapshot using StateManager."""
        self._state_manager.update_portfolio(self._portfolio)
        self._state_manager.save_snapshot(iteration=self._state.iteration)
        logger.debug(f"State snapshot saved (iteration {self._state.iteration})")

    def _handle_error(self, error: Exception) -> None:
        """Handle an error in the trading loop."""
        self._state.errors_count += 1
        logger.error(f"Error in trading loop: {error}", exc_info=True)

    def _setup_signal_handlers(self) -> None:
        """Setup SIGINT/SIGTERM handlers for graceful shutdown."""
        def handle_signal(signum, frame):
            logger.info(f"Received signal {signum}, requesting stop")
            self.stop()

        signal.signal(signal.SIGINT, handle_signal)
        signal.signal(signal.SIGTERM, handle_signal)

    def reset_daily_metrics(self) -> None:
        """Reset daily metrics (call at start of trading day)."""
        self._portfolio.reset_daily_pnl()
        self._kill_switch.reset()
        logger.info("Daily metrics reset")
