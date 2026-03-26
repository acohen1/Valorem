"""Position tracking and mark-to-market valuation.

This module provides the PositionTracker class for tracking positions
and updating their valuations with current market data.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

from ..risk.portfolio import CONTRACT_MULTIPLIER, Portfolio, Position, PositionState
from ..strategy.types import Greeks

logger = logging.getLogger(__name__)


@dataclass
class PositionSnapshot:
    """Snapshot of a position at a point in time.

    Attributes:
        position_id: Unique identifier for the position
        entry_value: Total value at entry
        current_value: Current mark-to-market value
        unrealized_pnl: Current unrealized P&L
        current_greeks: Current Greeks from surface
        mark_prices: Symbol -> current mark price mapping
        timestamp: When this snapshot was taken
    """

    position_id: str
    entry_value: float
    current_value: float
    unrealized_pnl: float
    current_greeks: Greeks
    mark_prices: dict[str, float] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class PositionTracker:
    """Track positions and update mark-to-market valuations.

    The PositionTracker provides:
    - Mark-to-market valuation using current surface prices
    - Greek updates from surface data
    - P&L calculation (realized and unrealized)
    - Position state tracking

    Example:
        >>> tracker = PositionTracker(portfolio)
        >>> tracker.update_positions(surface)
        >>> for snapshot in tracker.get_position_snapshots():
        ...     print(f"{snapshot.position_id}: P&L={snapshot.unrealized_pnl:.2f}")
    """

    def __init__(self, portfolio: Optional[Portfolio] = None) -> None:
        """Initialize position tracker.

        Args:
            portfolio: Portfolio to track (or None to create empty)
        """
        self._portfolio = portfolio or Portfolio()
        self._snapshots: dict[str, PositionSnapshot] = {}
        self._last_update: Optional[datetime] = None

    @property
    def portfolio(self) -> Portfolio:
        """Get current portfolio."""
        return self._portfolio

    @portfolio.setter
    def portfolio(self, value: Portfolio) -> None:
        """Update portfolio."""
        self._portfolio = value

    @property
    def last_update(self) -> Optional[datetime]:
        """Get timestamp of last position update."""
        return self._last_update

    def update_positions(self, surface: pd.DataFrame) -> Portfolio:
        """Update all positions with current market data.

        Performs mark-to-market valuation and updates Greeks for all
        open positions using the latest surface data.

        Args:
            surface: Current volatility surface with columns:
                option_symbol, bid, ask, delta, gamma, vega, theta

        Returns:
            Updated portfolio with new valuations
        """
        self._last_update = datetime.now(timezone.utc)
        updated_positions: list[Position] = []

        for position in self._portfolio.positions:
            if position.state != PositionState.OPEN:
                updated_positions.append(position)
                continue

            # Calculate current value and Greeks
            snapshot = self._calculate_position_snapshot(position, surface)
            self._snapshots[position.position_id] = snapshot

            # Create updated position with new unrealized P&L
            updated_position = Position(
                legs=position.legs,
                entry_time=position.entry_time,
                realized_pnl=position.realized_pnl,
                unrealized_pnl=snapshot.unrealized_pnl,
                position_id=position.position_id,
                structure_type=position.structure_type,
                state=position.state,
                exit_signal=position.exit_signal,
                max_loss=position.max_loss,
            )
            updated_positions.append(updated_position)

            logger.debug(
                f"Position {position.position_id} updated: "
                f"value={snapshot.current_value:.2f}, "
                f"P&L={snapshot.unrealized_pnl:.2f}"
            )

        # Create updated portfolio
        self._portfolio = Portfolio(
            positions=updated_positions,
            closed_positions=self._portfolio.closed_positions,
            daily_pnl=self._portfolio.daily_pnl,
            max_acceptable_loss=self._portfolio.max_acceptable_loss,
        )

        logger.info(
            f"Updated {len(updated_positions)} positions, "
            f"total unrealized P&L: {self._portfolio.total_unrealized_pnl:.2f}"
        )

        return self._portfolio

    def _calculate_position_snapshot(
        self,
        position: Position,
        surface: pd.DataFrame,
    ) -> PositionSnapshot:
        """Calculate current valuation snapshot for a position.

        Args:
            position: Position to value
            surface: Current market surface

        Returns:
            PositionSnapshot with current valuations
        """
        entry_value = 0.0
        current_value = 0.0
        mark_prices: dict[str, float] = {}

        # Aggregate Greeks
        total_delta = 0.0
        total_gamma = 0.0
        total_vega = 0.0
        total_theta = 0.0

        for leg in position.legs:
            # Entry value
            entry_value += leg.qty * leg.entry_price * CONTRACT_MULTIPLIER

            # Get current market price
            rows = surface[surface["option_symbol"] == leg.symbol]
            if not rows.empty:
                row = rows.iloc[0]

                # Mark-to-market price
                # Long position (qty > 0): value at bid (what we'd get if we sold)
                # Short position (qty < 0): value at ask (what we'd pay to close)
                if leg.qty > 0:
                    mark_price = float(row["bid"])
                else:
                    mark_price = float(row["ask"])

                mark_prices[leg.symbol] = mark_price
                current_value += leg.qty * mark_price * CONTRACT_MULTIPLIER

                # Update Greeks if available
                if "delta" in row:
                    total_delta += float(row["delta"]) * leg.qty * CONTRACT_MULTIPLIER
                if "gamma" in row:
                    total_gamma += float(row["gamma"]) * leg.qty * CONTRACT_MULTIPLIER
                if "vega" in row:
                    total_vega += float(row["vega"]) * leg.qty * CONTRACT_MULTIPLIER
                if "theta" in row:
                    total_theta += float(row["theta"]) * leg.qty * CONTRACT_MULTIPLIER
            else:
                # Option not in surface - use entry price as fallback
                mark_prices[leg.symbol] = leg.entry_price
                current_value += leg.qty * leg.entry_price * CONTRACT_MULTIPLIER

                # Use entry Greeks as fallback
                total_delta += leg.greeks.delta * leg.qty * CONTRACT_MULTIPLIER
                total_gamma += leg.greeks.gamma * leg.qty * CONTRACT_MULTIPLIER
                total_vega += leg.greeks.vega * leg.qty * CONTRACT_MULTIPLIER
                total_theta += leg.greeks.theta * leg.qty * CONTRACT_MULTIPLIER

                logger.warning(
                    f"Quote not found for {leg.symbol}, using entry price"
                )

        # Calculate unrealized P&L
        # Entry value: negative for debit (bought), positive for credit (sold)
        # Current value: positive for long positions, negative for short
        # P&L = current_value - entry_value
        unrealized_pnl = current_value - entry_value

        return PositionSnapshot(
            position_id=position.position_id,
            entry_value=entry_value,
            current_value=current_value,
            unrealized_pnl=unrealized_pnl,
            current_greeks=Greeks(
                delta=total_delta,
                gamma=total_gamma,
                vega=total_vega,
                theta=total_theta,
            ),
            mark_prices=mark_prices,
            timestamp=datetime.now(timezone.utc),
        )

    def get_position_snapshots(self) -> list[PositionSnapshot]:
        """Get snapshots for all tracked positions.

        Returns:
            List of position snapshots from last update
        """
        return list(self._snapshots.values())

    def get_snapshot(self, position_id: str) -> Optional[PositionSnapshot]:
        """Get snapshot for a specific position.

        Args:
            position_id: Position identifier

        Returns:
            PositionSnapshot if found, None otherwise
        """
        return self._snapshots.get(position_id)

    def get_total_unrealized_pnl(self) -> float:
        """Get total unrealized P&L across all positions.

        Returns:
            Sum of unrealized P&L for all open positions
        """
        return sum(s.unrealized_pnl for s in self._snapshots.values())

    def get_portfolio_greeks(self) -> Greeks:
        """Get aggregated portfolio Greeks from latest snapshots.

        Returns:
            Aggregated Greeks across all positions
        """
        total = Greeks(delta=0.0, gamma=0.0, vega=0.0, theta=0.0)
        for snapshot in self._snapshots.values():
            total = total + snapshot.current_greeks
        return total

    def clear_snapshots(self) -> None:
        """Clear all position snapshots."""
        self._snapshots.clear()
        self._last_update = None
