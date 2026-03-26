"""Portfolio state and Greek aggregation.

This module provides:
- PortfolioState: Immutable point-in-time view of portfolio state
- Portfolio: Mutable portfolio state (for backwards compatibility)
- Position: Individual position data
- PositionState: Position lifecycle states

Architecture:
    PortfolioState is the preferred way to work with portfolio state.
    It is immutable and thread-safe. Use PositionManager.get_portfolio_snapshot()
    to obtain snapshots. The Portfolio class is retained for backwards
    compatibility but its mutable state should not be relied upon.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional
import uuid

from ..config.constants import TradingConstants
from ..strategy.types import ExitSignal, ExitSignalType, Greeks, OptionLeg
from ..utils.calculations import aggregate_greeks

# Re-export for backwards compatibility
CONTRACT_MULTIPLIER = TradingConstants.CONTRACT_MULTIPLIER


class PositionState(str, Enum):
    """Lifecycle state of a position.

    Positions progress through these states:
    PENDING → FILLED → OPEN → CLOSING → CLOSED/EXPIRED
    """

    PENDING = "pending"  # Order submitted, awaiting fill
    FILLED = "filled"  # Order filled, position opening
    OPEN = "open"  # Active position
    CLOSING = "closing"  # Exit order submitted
    CLOSED = "closed"  # Position fully closed
    EXPIRED = "expired"  # Position expired worthless


@dataclass
class Position:
    """A position in the portfolio.

    Represents an active option position with its current Greeks and P&L.

    Attributes:
        legs: Option legs comprising this position
        entry_time: When the position was opened
        realized_pnl: Realized P&L from closed portions
        unrealized_pnl: Mark-to-market unrealized P&L
        position_id: Unique identifier for this position
        structure_type: Name of the structure (e.g., "CalendarSpread")
        state: Current lifecycle state
        exit_signal: Exit signal that triggered closing (if applicable)
        max_loss: Maximum possible loss for this position
    """

    legs: list[OptionLeg]
    entry_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    position_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    structure_type: Optional[str] = None
    state: PositionState = PositionState.OPEN
    exit_signal: Optional["ExitSignal"] = None
    max_loss: float = 0.0

    @property
    def greeks(self) -> Greeks:
        """Aggregate Greeks for this position (without contract multiplier).

        The contract multiplier is applied at the portfolio level.
        """
        total = Greeks(delta=0.0, gamma=0.0, vega=0.0, theta=0.0)
        for leg in self.legs:
            scaled = leg.greeks.scale(leg.qty)
            total = total + scaled
        return total

    @property
    def total_qty(self) -> int:
        """Total number of contracts in this position."""
        return sum(abs(leg.qty) for leg in self.legs)

    def to_dict(self) -> dict[str, Any]:
        """Serialize Position to dictionary."""
        return {
            "legs": [leg.to_dict() for leg in self.legs],
            "entry_time": self.entry_time.isoformat(),
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": self.unrealized_pnl,
            "position_id": self.position_id,
            "structure_type": self.structure_type,
            "state": self.state.value,
            "exit_signal": self.exit_signal.to_dict() if self.exit_signal else None,
            "max_loss": self.max_loss,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Position":
        """Deserialize Position from dictionary."""
        exit_signal = None
        if data.get("exit_signal"):
            exit_signal = ExitSignal.from_dict(data["exit_signal"])

        return cls(
            legs=[OptionLeg.from_dict(leg) for leg in data["legs"]],
            entry_time=datetime.fromisoformat(data["entry_time"]),
            realized_pnl=float(data["realized_pnl"]),
            unrealized_pnl=float(data["unrealized_pnl"]),
            position_id=data["position_id"],
            structure_type=data.get("structure_type"),
            state=PositionState(data["state"]),
            exit_signal=exit_signal,
            max_loss=float(data.get("max_loss", 0.0)),
        )


@dataclass(frozen=True)
class PortfolioState:
    """Immutable point-in-time view of portfolio state.

    This is the preferred way to access portfolio state. It provides:
    - Thread-safe read access (immutable)
    - Consistent state (no sync issues)
    - Functional methods for hypothetical scenarios

    Use PositionManager.get_portfolio_snapshot() to obtain snapshots.

    Attributes:
        positions: Tuple of active positions (immutable)
        total_greeks: Aggregated Greeks across all positions
        total_unrealized_pnl: Sum of unrealized P&L
        total_realized_pnl: Sum of realized P&L
        daily_pnl: Cumulative P&L for the trading day
        max_acceptable_loss: Maximum acceptable loss threshold
    """

    positions: tuple[Position, ...]
    total_greeks: Greeks
    total_unrealized_pnl: float
    total_realized_pnl: float
    daily_pnl: float = 0.0
    max_acceptable_loss: float = 5000.0

    @property
    def net_delta(self) -> float:
        """Net portfolio delta (contract-adjusted)."""
        return self.total_greeks.delta

    @property
    def net_gamma(self) -> float:
        """Net portfolio gamma (contract-adjusted)."""
        return self.total_greeks.gamma

    @property
    def net_vega(self) -> float:
        """Net portfolio vega (contract-adjusted)."""
        return self.total_greeks.vega

    @property
    def net_theta(self) -> float:
        """Net portfolio theta (contract-adjusted)."""
        return self.total_greeks.theta

    @property
    def total_pnl(self) -> float:
        """Total P&L (realized + unrealized)."""
        return self.total_unrealized_pnl + self.total_realized_pnl

    @property
    def total_contracts(self) -> int:
        """Total number of contracts across all positions."""
        return sum(pos.total_qty for pos in self.positions)

    def with_trade(self, legs: list[OptionLeg]) -> "PortfolioState":
        """Create hypothetical snapshot with an additional trade.

        This method is used for pre-trade risk checks to evaluate
        the portfolio state after a potential trade.

        Args:
            legs: Option legs to add as a new position

        Returns:
            New PortfolioState with the trade added
        """
        new_position = Position(legs=legs)

        # Compute new position's Greeks (contract-adjusted)
        new_greeks = aggregate_greeks(legs)

        return PortfolioState(
            positions=self.positions + (new_position,),
            total_greeks=self.total_greeks + new_greeks,
            total_unrealized_pnl=self.total_unrealized_pnl,
            total_realized_pnl=self.total_realized_pnl,
            daily_pnl=self.daily_pnl,
            max_acceptable_loss=self.max_acceptable_loss,
        )

    def without_position(self, position_id: str) -> "PortfolioState":
        """Create snapshot with a position removed.

        Used for hypothetical risk checks when evaluating exits.

        Args:
            position_id: ID of position to remove

        Returns:
            New PortfolioState without the position
        """
        position = self.get_position_by_id(position_id)
        if position is None:
            return self

        # Compute position's Greeks to subtract
        pos_greeks = aggregate_greeks(position.legs)

        # Subtract position Greeks from total
        new_greeks = Greeks(
            delta=self.total_greeks.delta - pos_greeks.delta,
            gamma=self.total_greeks.gamma - pos_greeks.gamma,
            vega=self.total_greeks.vega - pos_greeks.vega,
            theta=self.total_greeks.theta - pos_greeks.theta,
        )

        return PortfolioState(
            positions=tuple(p for p in self.positions if p.position_id != position_id),
            total_greeks=new_greeks,
            total_unrealized_pnl=self.total_unrealized_pnl - position.unrealized_pnl,
            total_realized_pnl=self.total_realized_pnl + position.unrealized_pnl,
            daily_pnl=self.daily_pnl,
            max_acceptable_loss=self.max_acceptable_loss,
        )

    def get_position_by_id(self, position_id: str) -> Optional[Position]:
        """Find a position by its ID.

        Args:
            position_id: Unique position identifier

        Returns:
            Position if found, None otherwise
        """
        for pos in self.positions:
            if pos.position_id == position_id:
                return pos
        return None

    def get_open_positions(self) -> tuple[Position, ...]:
        """Get all positions in OPEN state.

        Returns:
            Tuple of open positions
        """
        return tuple(pos for pos in self.positions if pos.state == PositionState.OPEN)

    def get_greeks(self) -> Greeks:
        """Get aggregated portfolio Greeks.

        Returns:
            Greeks instance with contract-adjusted values
        """
        return self.total_greeks

    @classmethod
    def empty(cls, max_acceptable_loss: float = 5000.0) -> "PortfolioState":
        """Create an empty portfolio snapshot.

        Args:
            max_acceptable_loss: Maximum acceptable loss threshold

        Returns:
            Empty PortfolioState
        """
        return cls(
            positions=(),
            total_greeks=Greeks(delta=0.0, gamma=0.0, vega=0.0, theta=0.0),
            total_unrealized_pnl=0.0,
            total_realized_pnl=0.0,
            daily_pnl=0.0,
            max_acceptable_loss=max_acceptable_loss,
        )

    @classmethod
    def from_positions(
        cls,
        positions: list[Position],
        daily_pnl: float = 0.0,
        max_acceptable_loss: float = 5000.0,
    ) -> "PortfolioState":
        """Create snapshot from a list of positions.

        Computes aggregated Greeks and P&L from the positions.

        Args:
            positions: List of positions
            daily_pnl: Daily P&L value
            max_acceptable_loss: Maximum acceptable loss threshold

        Returns:
            PortfolioState with computed aggregates
        """
        total_greeks = Greeks(delta=0.0, gamma=0.0, vega=0.0, theta=0.0)
        total_unrealized = 0.0
        total_realized = 0.0

        for pos in positions:
            # Aggregate Greeks (contract-adjusted)
            total_greeks = total_greeks + aggregate_greeks(pos.legs)

            total_unrealized += pos.unrealized_pnl
            total_realized += pos.realized_pnl

        return cls(
            positions=tuple(positions),
            total_greeks=total_greeks,
            total_unrealized_pnl=total_unrealized,
            total_realized_pnl=total_realized,
            daily_pnl=daily_pnl,
            max_acceptable_loss=max_acceptable_loss,
        )


@dataclass
class Portfolio:
    """Portfolio state tracking positions and risk metrics.

    Note:
        Consider using PortfolioState instead for read-only access to
        portfolio state. PortfolioState is immutable and thread-safe.
        Use PositionManager.get_portfolio_snapshot() to obtain snapshots.

    This class maintains a mutable collection of positions and provides
    aggregated risk metrics. It is retained for backwards compatibility.

    Attributes:
        positions: List of active positions
        closed_positions: List of closed/expired positions (for history)
        daily_pnl: Cumulative P&L for the current trading day
        max_acceptable_loss: Maximum acceptable loss threshold
    """

    positions: list[Position] = field(default_factory=list)
    closed_positions: list[Position] = field(default_factory=list)
    daily_pnl: float = 0.0
    max_acceptable_loss: float = 5000.0

    def to_snapshot(self) -> PortfolioState:
        """Create an immutable snapshot of the current portfolio state.

        Returns:
            PortfolioState with current positions and aggregated metrics
        """
        return PortfolioState.from_positions(
            positions=self.positions,
            daily_pnl=self.daily_pnl,
            max_acceptable_loss=self.max_acceptable_loss,
        )

    @property
    def net_delta(self) -> float:
        """Net portfolio delta (contract-adjusted)."""
        return sum(
            pos.greeks.delta * CONTRACT_MULTIPLIER for pos in self.positions
        )

    @property
    def net_gamma(self) -> float:
        """Net portfolio gamma (contract-adjusted)."""
        return sum(
            pos.greeks.gamma * CONTRACT_MULTIPLIER for pos in self.positions
        )

    @property
    def net_vega(self) -> float:
        """Net portfolio vega (contract-adjusted)."""
        return sum(
            pos.greeks.vega * CONTRACT_MULTIPLIER for pos in self.positions
        )

    @property
    def net_theta(self) -> float:
        """Net portfolio theta (contract-adjusted)."""
        return sum(
            pos.greeks.theta * CONTRACT_MULTIPLIER for pos in self.positions
        )

    @property
    def total_unrealized_pnl(self) -> float:
        """Total unrealized P&L across all positions."""
        return sum(pos.unrealized_pnl for pos in self.positions)

    @property
    def total_realized_pnl(self) -> float:
        """Total realized P&L across all positions."""
        return sum(pos.realized_pnl for pos in self.positions)

    @property
    def total_contracts(self) -> int:
        """Total number of contracts across all positions."""
        return sum(pos.total_qty for pos in self.positions)

    def add_trade(self, legs: list[OptionLeg]) -> "Portfolio":
        """Create a hypothetical portfolio with an additional trade.

        This method is used for pre-trade risk checks to evaluate
        the portfolio state after a potential trade.

        Args:
            legs: Option legs to add as a new position

        Returns:
            New Portfolio instance with the trade added
        """
        new_position = Position(legs=legs)
        return Portfolio(
            positions=self.positions + [new_position],
            daily_pnl=self.daily_pnl,
            max_acceptable_loss=self.max_acceptable_loss,
        )

    def get_greeks(self) -> Greeks:
        """Get aggregated portfolio Greeks.

        Returns:
            Greeks instance with contract-adjusted values
        """
        return Greeks(
            delta=self.net_delta,
            gamma=self.net_gamma,
            vega=self.net_vega,
            theta=self.net_theta,
        )

    def reset_daily_pnl(self) -> None:
        """Reset daily P&L (called at start of trading day)."""
        self.daily_pnl = 0.0

    def update_daily_pnl(self, pnl_change: float) -> None:
        """Update daily P&L with a change.

        Args:
            pnl_change: P&L change to add (positive or negative)
        """
        self.daily_pnl += pnl_change

    def get_position_by_id(self, position_id: str) -> Optional[Position]:
        """Find an active position by its ID.

        Args:
            position_id: Unique position identifier

        Returns:
            Position if found, None otherwise
        """
        for pos in self.positions:
            if pos.position_id == position_id:
                return pos
        return None

    def get_open_positions(self) -> list[Position]:
        """Get all positions in OPEN state.

        Returns:
            List of open positions
        """
        return [pos for pos in self.positions if pos.state == PositionState.OPEN]

    def close_position(
        self,
        position_id: str,
        exit_signal: Optional["ExitSignal"] = None,
        final_pnl: Optional[float] = None,
    ) -> "Portfolio":
        """Close a position and move it to closed_positions.

        Args:
            position_id: ID of position to close
            exit_signal: Exit signal that triggered the close
            final_pnl: Final realized P&L (if known)

        Returns:
            New Portfolio instance with position closed
        """
        position = self.get_position_by_id(position_id)
        if position is None:
            return self

        # Create closed position copy
        closed_pos = Position(
            legs=position.legs,
            entry_time=position.entry_time,
            realized_pnl=final_pnl if final_pnl is not None else position.unrealized_pnl,
            unrealized_pnl=0.0,
            position_id=position.position_id,
            structure_type=position.structure_type,
            state=PositionState.CLOSED,
            exit_signal=exit_signal,
            max_loss=position.max_loss,
        )

        # Remove from active, add to closed
        new_positions = [p for p in self.positions if p.position_id != position_id]

        return Portfolio(
            positions=new_positions,
            closed_positions=self.closed_positions + [closed_pos],
            daily_pnl=self.daily_pnl,
            max_acceptable_loss=self.max_acceptable_loss,
        )

    def add_position(
        self,
        legs: list[OptionLeg],
        position_id: Optional[str] = None,
        structure_type: Optional[str] = None,
        max_loss: float = 0.0,
    ) -> "Portfolio":
        """Create a portfolio with a new position added.

        Args:
            legs: Option legs for the new position
            position_id: Optional position ID (generated if not provided)
            structure_type: Name of the trade structure
            max_loss: Maximum possible loss for this position

        Returns:
            New Portfolio instance with position added
        """
        new_position = Position(
            legs=legs,
            position_id=position_id or str(uuid.uuid4()),
            structure_type=structure_type,
            state=PositionState.OPEN,
            max_loss=max_loss,
        )
        return Portfolio(
            positions=self.positions + [new_position],
            closed_positions=self.closed_positions,
            daily_pnl=self.daily_pnl,
            max_acceptable_loss=self.max_acceptable_loss,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize Portfolio to dictionary."""
        return {
            "positions": [pos.to_dict() for pos in self.positions],
            "closed_positions": [pos.to_dict() for pos in self.closed_positions],
            "daily_pnl": self.daily_pnl,
            "max_acceptable_loss": self.max_acceptable_loss,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Portfolio":
        """Deserialize Portfolio from dictionary."""
        return cls(
            positions=[Position.from_dict(pos) for pos in data.get("positions", [])],
            closed_positions=[
                Position.from_dict(pos) for pos in data.get("closed_positions", [])
            ],
            daily_pnl=float(data.get("daily_pnl", 0.0)),
            max_acceptable_loss=float(data.get("max_acceptable_loss", 5000.0)),
        )
