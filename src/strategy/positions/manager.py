"""Position lifecycle management orchestration.

This module provides the PositionManager class that orchestrates
position tracking, market data updates, exit signal evaluation,
and rebalancing.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Optional

import pandas as pd

from ...config.constants import TradingConstants
from ...config.schema import PositionManagementConfig
from ...pricing import PositionPricer
from ...risk.portfolio import (
    Portfolio,
    PortfolioState,
    Position,
    PositionState,
)
from ...utils.calculations import min_dte
from ..orders import Order
from ..types import ExitSignal, Greeks, OptionLeg, Signal
from .exit_orders import ExitOrder, ExitOrderGenerator
from .exit_signals import ExitSignalGenerator
from .lifecycle import ManagedPosition
from .rebalance import DriftResult, RebalanceEngine

logger = logging.getLogger(__name__)

CONTRACT_MULTIPLIER = TradingConstants.CONTRACT_MULTIPLIER


@dataclass
class Fill:
    """Order fill information.

    Attributes:
        legs: Filled option legs
        fill_price: Net fill price
        timestamp: Fill timestamp
    """

    legs: list[OptionLeg]
    fill_price: float
    timestamp: datetime


class PositionManager:
    """Orchestrate position lifecycle management.

    Manages positions through their full lifecycle:
    1. Track new positions from filled orders
    2. Update positions with market data
    3. Evaluate exit signals
    4. Check for rebalancing needs
    5. Generate exit orders
    6. Record exits

    Attributes:
        positions: Dictionary of position_id → ManagedPosition
    """

    def __init__(
        self,
        config: PositionManagementConfig,
        exit_signal_generator: ExitSignalGenerator,
        exit_order_generator: ExitOrderGenerator,
        rebalance_engine: RebalanceEngine,
        pricer: PositionPricer | None = None,
    ) -> None:
        """Initialize position manager.

        Args:
            config: Position management configuration
            exit_signal_generator: Exit signal generator
            exit_order_generator: Exit order generator
            rebalance_engine: Rebalance engine
            pricer: Cascading pricer for option legs. If None, a
                surface-only pricer is created (backward compatible).
        """
        self._config = config
        self._exit_gen = exit_signal_generator
        self._exit_order_gen = exit_order_generator
        self._rebalance = rebalance_engine
        self._pricer = pricer or PositionPricer()

        self._positions: dict[str, ManagedPosition] = {}

    @property
    def positions(self) -> dict[str, ManagedPosition]:
        """Get all managed positions."""
        return self._positions

    def add_position(self, order: Order, fill: Fill) -> ManagedPosition:
        """Add a new position from a filled order.

        Args:
            order: The original order
            fill: Fill information

        Returns:
            New ManagedPosition in OPEN state
        """
        position_id = self._generate_position_id()

        position = ManagedPosition.from_order_fill(
            position_id=position_id,
            legs=fill.legs,
            structure_type=order.structure_type,
            entry_signal=order.signal,
            entry_price=fill.fill_price,
            entry_greeks=order.greeks,
            max_loss=order.max_loss,
            days_to_expiry=min_dte(fill.legs),
            entry_time=fill.timestamp,
        )

        self._positions[position_id] = position

        logger.info(
            f"Added position {position_id}: {order.structure_type}, "
            f"entry_price={fill.fill_price:.2f}, max_loss={order.max_loss:.2f}"
        )

        return position

    def add_position_direct(
        self,
        legs: list[OptionLeg],
        structure_type: str,
        entry_signal: "Signal",
        entry_price: float,
        entry_greeks: Greeks,
        max_loss: float,
        entry_time: Optional[datetime] = None,
    ) -> ManagedPosition:
        """Add a position directly without an Order object.

        Convenience method for testing or manual position tracking.

        Args:
            legs: Option legs
            structure_type: Structure name
            entry_signal: Entry signal
            entry_price: Net entry premium
            entry_greeks: Entry Greeks
            max_loss: Maximum loss
            entry_time: Entry timestamp (defaults to now)

        Returns:
            New ManagedPosition
        """
        position_id = self._generate_position_id()

        position = ManagedPosition.from_order_fill(
            position_id=position_id,
            legs=legs,
            structure_type=structure_type,
            entry_signal=entry_signal,
            entry_price=entry_price,
            entry_greeks=entry_greeks,
            max_loss=max_loss,
            days_to_expiry=min_dte(legs),
            entry_time=entry_time,
        )

        self._positions[position_id] = position
        return position

    def update_positions(
        self, surface: pd.DataFrame, as_of: date | None = None
    ) -> None:
        """Update all open positions with current market data.

        Args:
            surface: Current surface snapshot with option prices and Greeks
            as_of: Trading date for quote lookups. Optional for
                backward compatibility; required when the pricer
                has a QuoteSource.
        """
        # date.min is a safe placeholder when as_of is not provided;
        # only the QuoteSource tier uses the date, and the default
        # pricer (no QuoteSource) never reaches it.
        pricing_date = as_of or date.min

        for position in self._positions.values():
            if position.state != PositionState.OPEN:
                continue

            # Mark-to-market
            current_price = self._pricer.price_position(
                position.legs, surface, pricing_date
            )

            # Update Greeks
            current_greeks = self._compute_current_greeks(
                position.legs, surface, pricing_date
            )

            # Update time metrics
            days_to_expiry = min_dte(position.legs)

            position.update_market_data(
                current_price=current_price,
                current_greeks=current_greeks,
                days_to_expiry=days_to_expiry,
            )

    def evaluate_exits(
        self,
        surface: pd.DataFrame,
        model_predictions: Optional[pd.DataFrame],
        portfolio: Portfolio,
        as_of: date | None = None,
    ) -> list[ExitOrder]:
        """Evaluate positions for exits and generate exit orders.

        This is the main position management loop:
        1. Evaluate exit signals for all open positions
        2. Check for rebalancing needs
        3. Combine signals (exit signals take priority)
        4. Generate exit orders
        5. Mark positions as CLOSING

        Args:
            surface: Current surface snapshot
            model_predictions: Current model predictions (optional)
            portfolio: Current portfolio state
            as_of: Trading date for quote lookups. Optional for
                backward compatibility; required when the pricer
                has a QuoteSource.

        Returns:
            List of exit orders to execute
        """
        open_positions = self.get_open_positions()

        if not open_positions:
            return []

        # 1. Generate exit signals from position evaluation
        exit_signals = self._exit_gen.evaluate_positions(
            positions=open_positions,
            surface=surface,
            model_predictions=model_predictions,
        )

        # 2. Check for rebalancing needs
        rebalance_signals: list[ExitSignal] = []
        if self._rebalance.enabled:
            drift_result = self._rebalance.check_drift(portfolio)
            rebalance_signals = self._rebalance.generate_rebalance_signals(
                drift_result=drift_result,
                positions=open_positions,
                portfolio=portfolio,
            )

        # 3. Combine signals (exit signals take priority over rebalance)
        exit_position_ids = {s.position_id for s in exit_signals}
        all_signals = exit_signals + [
            s for s in rebalance_signals if s.position_id not in exit_position_ids
        ]

        if not all_signals:
            return []

        # 4. Generate exit orders
        exit_orders = self._exit_order_gen.generate_exit_orders(
            exit_signals=all_signals,
            positions=self._positions,
            surface=surface,
            as_of=as_of,
        )

        # 5. Mark positions as closing
        for order in exit_orders:
            position = self._positions.get(order.position_id)
            if position:
                position.mark_closing(order.exit_signal)

        return exit_orders

    def record_exit(
        self,
        exit_order: ExitOrder,
        fill_price: float,
        fill_time: Optional[datetime] = None,
    ) -> None:
        """Record a completed exit.

        Args:
            exit_order: The exit order that was filled
            fill_price: Net fill price
            fill_time: Fill timestamp (defaults to now)
        """
        position = self._positions.get(exit_order.position_id)
        if position is None:
            logger.warning(f"Position {exit_order.position_id} not found for exit record")
            return

        position.mark_closed(
            exit_price=fill_price,
            exit_time=fill_time,
        )

        logger.info(
            f"Position {position.position_id} closed: "
            f"exit_price={fill_price:.2f}, "
            f"realized_pnl={position.realized_pnl:.2f}"
        )

    def record_expiration(self, position_id: str) -> None:
        """Record a position expiration.

        Args:
            position_id: ID of expired position
        """
        position = self._positions.get(position_id)
        if position is None:
            logger.warning(f"Position {position_id} not found for expiration record")
            return

        position.mark_expired()

        logger.info(
            f"Position {position_id} expired: "
            f"realized_pnl={position.realized_pnl:.2f}"
        )

    def get_open_positions(self) -> list[ManagedPosition]:
        """Get all positions in OPEN state.

        Returns:
            List of open positions
        """
        return [p for p in self._positions.values() if p.state == PositionState.OPEN]

    def get_closing_positions(self) -> list[ManagedPosition]:
        """Get all positions in CLOSING state.

        Returns:
            List of positions awaiting close
        """
        return [p for p in self._positions.values() if p.state == PositionState.CLOSING]

    def get_closed_positions(self) -> list[ManagedPosition]:
        """Get all closed/expired positions.

        Returns:
            List of closed positions
        """
        return [p for p in self._positions.values() if p.is_closed()]

    def get_position(self, position_id: str) -> Optional[ManagedPosition]:
        """Get a position by ID.

        Args:
            position_id: Position identifier

        Returns:
            Position if found, None otherwise
        """
        return self._positions.get(position_id)

    def check_rebalance(self, portfolio: Portfolio) -> DriftResult:
        """Check if portfolio needs rebalancing.

        Args:
            portfolio: Current portfolio state

        Returns:
            DriftResult with drift analysis
        """
        return self._rebalance.check_drift(portfolio)

    def create_manual_exit_signal(
        self,
        position_id: str,
        reason: str = "Manual exit requested",
    ) -> Optional[ExitSignal]:
        """Create a manual exit signal for a position.

        Args:
            position_id: Position to exit
            reason: Reason for manual exit

        Returns:
            ExitSignal if position exists, None otherwise
        """
        position = self._positions.get(position_id)
        if position is None or not position.is_open():
            return None

        return self._exit_gen.create_manual_exit(
            position_id=position_id,
            reason=reason,
        )

    def _compute_current_greeks(
        self,
        legs: list[OptionLeg],
        surface: pd.DataFrame,
        as_of: date,
    ) -> Greeks:
        """Compute current aggregate Greeks for position.

        Delegates per-leg pricing to the injected ``PositionPricer``
        which applies the three-tier cascade and NaN-safe Greeks
        extraction.

        Args:
            legs: Position legs
            surface: Current surface with Greeks
            as_of: Trading date for quote lookups

        Returns:
            Aggregate Greeks (contract-adjusted)
        """
        total = Greeks(delta=0.0, gamma=0.0, vega=0.0, theta=0.0)

        for leg in legs:
            priced = self._pricer.price_leg(leg, surface, as_of)
            scaled = priced.greeks.scale(leg.qty * CONTRACT_MULTIPLIER)
            total = total + scaled

        return total

    def _generate_position_id(self) -> str:
        """Generate unique position ID.

        Returns:
            UUID string prefixed with 'pos-'
        """
        return f"pos-{uuid.uuid4()}"

    def get_portfolio_summary(self) -> dict:
        """Get summary statistics for all positions.

        Returns:
            Dictionary with position counts and P&L summary
        """
        open_positions = self.get_open_positions()
        closed_positions = self.get_closed_positions()

        total_unrealized = sum(p.unrealized_pnl for p in open_positions)
        total_realized = sum(p.realized_pnl or 0.0 for p in closed_positions)

        return {
            "open_count": len(open_positions),
            "closing_count": len(self.get_closing_positions()),
            "closed_count": len(closed_positions),
            "total_unrealized_pnl": total_unrealized,
            "total_realized_pnl": total_realized,
            "total_pnl": total_unrealized + total_realized,
        }

    def get_portfolio_state(
        self,
        daily_pnl: float = 0.0,
        max_acceptable_loss: float = 5000.0,
    ) -> PortfolioState:
        """Get an immutable snapshot of the current portfolio state.

        This is the preferred way to access portfolio state for risk checks
        and monitoring. The snapshot is thread-safe and consistent.

        Args:
            daily_pnl: Current daily P&L value
            max_acceptable_loss: Maximum acceptable loss threshold

        Returns:
            Immutable PortfolioState with current positions and Greeks
        """
        open_positions = self.get_open_positions()
        closed_positions = self.get_closed_positions()

        # Convert ManagedPosition to Position for the snapshot
        positions: list[Position] = []
        total_greeks = Greeks(delta=0.0, gamma=0.0, vega=0.0, theta=0.0)
        total_unrealized = 0.0
        total_realized = 0.0

        for managed_pos in open_positions:
            # Create Position from ManagedPosition
            pos = Position(
                legs=managed_pos.legs,
                entry_time=managed_pos.entry_time,
                realized_pnl=0.0,
                unrealized_pnl=managed_pos.unrealized_pnl,
                position_id=managed_pos.position_id,
                structure_type=managed_pos.structure_type,
                state=PositionState.OPEN,
                max_loss=managed_pos.max_loss,
            )
            positions.append(pos)

            # Aggregate Greeks (use current Greeks from managed position)
            scaled = managed_pos.current_greeks
            total_greeks = total_greeks + scaled

            total_unrealized += managed_pos.unrealized_pnl

        # Sum realized P&L from closed positions
        for managed_pos in closed_positions:
            total_realized += managed_pos.realized_pnl or 0.0

        return PortfolioState(
            positions=tuple(positions),
            total_greeks=total_greeks,
            total_unrealized_pnl=total_unrealized,
            total_realized_pnl=total_realized,
            daily_pnl=daily_pnl,
            max_acceptable_loss=max_acceptable_loss,
        )
