"""Position lifecycle management.

This module provides the ManagedPosition dataclass for tracking
positions through their full lifecycle from entry to exit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from ...risk.portfolio import PositionState
from ..types import ExitSignal, Greeks, OptionLeg, Signal


@dataclass
class ManagedPosition:
    """Position with full lifecycle tracking.

    Extends the basic Position concept with entry/exit tracking,
    mark-to-market updates, and time metrics for exit signal evaluation.

    Attributes:
        position_id: Unique identifier for this position
        state: Current lifecycle state
        legs: Option legs comprising this position
        structure_type: Name of the structure (e.g., "CalendarSpread")
        entry_signal: Original signal that generated this position

        entry_time: When the position was opened
        entry_price: Net premium at entry (negative = debit, positive = credit)
        entry_greeks: Greeks at time of entry
        max_loss: Maximum possible loss computed at entry

        current_price: Mark-to-market value (updated each cycle)
        current_greeks: Current Greeks (updated each cycle)
        unrealized_pnl: Current unrealized P&L
        days_held: Number of days position has been open
        days_to_expiry: Minimum DTE across all legs

        exit_signal: Exit signal that triggered closing (if applicable)
        exit_time: When the exit order was filled
        exit_price: Net premium at exit
        realized_pnl: Final realized P&L after closing
    """

    # Core identification
    position_id: str
    state: PositionState
    legs: list[OptionLeg]
    structure_type: str
    entry_signal: Signal

    # Entry data
    entry_time: datetime
    entry_price: float
    entry_greeks: Greeks
    max_loss: float

    # Current state (updated each cycle)
    current_price: float = 0.0
    current_greeks: Greeks = field(
        default_factory=lambda: Greeks(delta=0.0, gamma=0.0, vega=0.0, theta=0.0)
    )
    unrealized_pnl: float = 0.0
    days_held: int = 0
    days_to_expiry: int = 0

    # Exit tracking
    exit_signal: Optional[ExitSignal] = None
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    realized_pnl: Optional[float] = None

    def is_open(self) -> bool:
        """Check if position is in an open state."""
        return self.state == PositionState.OPEN

    def is_closed(self) -> bool:
        """Check if position is fully closed."""
        return self.state in (PositionState.CLOSED, PositionState.EXPIRED)

    def pnl_pct_of_max_loss(self) -> float:
        """Get unrealized P&L as percentage of max loss.

        Returns:
            Positive value if losing money (e.g., 0.5 = lost 50% of max loss)
            Negative value if making money
        """
        if self.max_loss == 0:
            return 0.0
        return -self.unrealized_pnl / self.max_loss

    def mark_closing(self, exit_signal: ExitSignal) -> None:
        """Mark position as closing with the given exit signal.

        Args:
            exit_signal: The signal that triggered the exit
        """
        self.state = PositionState.CLOSING
        self.exit_signal = exit_signal

    def mark_closed(
        self,
        exit_price: float,
        exit_time: Optional[datetime] = None,
    ) -> None:
        """Mark position as closed with final P&L.

        Args:
            exit_price: Net premium received on exit
            exit_time: When the exit occurred (defaults to now)
        """
        self.state = PositionState.CLOSED
        self.exit_price = exit_price
        self.exit_time = exit_time or datetime.now(timezone.utc)
        # Realized P&L = exit_price - entry_price.
        # Callers must pass exit_price with consistent sign convention:
        # positive = value received on close (not raw fill.net_premium).
        self.realized_pnl = exit_price - self.entry_price

    def mark_expired(self) -> None:
        """Mark position as expired worthless."""
        self.state = PositionState.EXPIRED
        self.exit_time = datetime.now(timezone.utc)
        self.exit_price = 0.0
        # If entry was debit (negative), loss = abs(entry_price)
        # If entry was credit (positive), profit = entry_price
        self.realized_pnl = -self.entry_price

    def update_market_data(
        self,
        current_price: float,
        current_greeks: Greeks,
        days_to_expiry: int,
    ) -> None:
        """Update position with current market data.

        Args:
            current_price: Current mark-to-market value
            current_greeks: Current aggregate Greeks
            days_to_expiry: Minimum DTE across legs
        """
        self.current_price = current_price
        self.current_greeks = current_greeks
        self.days_to_expiry = days_to_expiry
        self.unrealized_pnl = current_price - self.entry_price

        # Update days held
        self.days_held = (datetime.now(timezone.utc) - self.entry_time).days

    @classmethod
    def from_order_fill(
        cls,
        position_id: str,
        legs: list[OptionLeg],
        structure_type: str,
        entry_signal: Signal,
        entry_price: float,
        entry_greeks: Greeks,
        max_loss: float,
        days_to_expiry: int,
        entry_time: Optional[datetime] = None,
    ) -> "ManagedPosition":
        """Create a ManagedPosition from a filled order.

        Args:
            position_id: Unique position identifier
            legs: Filled option legs
            structure_type: Name of the structure
            entry_signal: Original trading signal
            entry_price: Net premium at entry
            entry_greeks: Greeks at entry
            max_loss: Maximum possible loss
            days_to_expiry: Minimum DTE across legs
            entry_time: Fill timestamp (defaults to now)

        Returns:
            New ManagedPosition in OPEN state
        """
        now = entry_time or datetime.now(timezone.utc)

        return cls(
            position_id=position_id,
            state=PositionState.OPEN,
            legs=legs,
            structure_type=structure_type,
            entry_signal=entry_signal,
            entry_time=now,
            entry_price=entry_price,
            entry_greeks=entry_greeks,
            max_loss=max_loss,
            current_price=entry_price,
            current_greeks=entry_greeks,
            unrealized_pnl=0.0,
            days_held=0,
            days_to_expiry=days_to_expiry,
        )
