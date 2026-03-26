"""State management for paper and live trading.

This module provides the StateManager class for persisting and recovering
trading state including portfolio positions, orders, and fills.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from ..config.schema import PaperConfig
from ..risk.portfolio import Portfolio
from ..strategy.orders import Order
from .router import Fill

logger = logging.getLogger(__name__)


@dataclass
class TradingState:
    """Complete trading state snapshot.

    Attributes:
        portfolio: Current portfolio state
        order_history: List of all orders submitted
        fill_history: List of all fills received
        timestamp: When this snapshot was taken
        iteration: Loop iteration number at snapshot time
        daily_pnl: Daily P&L at snapshot time
    """

    portfolio: Portfolio
    order_history: list[dict[str, Any]] = field(default_factory=list)
    fill_history: list[dict[str, Any]] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    iteration: int = 0
    daily_pnl: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize state to dictionary."""
        return {
            "portfolio": self.portfolio.to_dict(),
            "order_history": self.order_history,
            "fill_history": self.fill_history,
            "timestamp": self.timestamp.isoformat(),
            "iteration": self.iteration,
            "daily_pnl": self.daily_pnl,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TradingState":
        """Deserialize state from dictionary."""
        return cls(
            portfolio=Portfolio.from_dict(data["portfolio"]),
            order_history=data.get("order_history", []),
            fill_history=data.get("fill_history", []),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            iteration=int(data.get("iteration", 0)),
            daily_pnl=float(data.get("daily_pnl", 0.0)),
        )


class StateManager:
    """Manage trading state persistence and recovery.

    The StateManager handles:
    - Saving state snapshots after each trading loop iteration
    - Loading the most recent state on startup (crash recovery)
    - Recording orders and fills for audit trail
    - Managing the state directory and file naming

    State is persisted as JSON files in the configured state directory.
    Files are named with timestamps for easy ordering and recovery.

    Example:
        >>> manager = StateManager(paper_config)
        >>> manager.record_fill(fill)
        >>> manager.save_snapshot(iteration=10)
        >>> # After crash/restart:
        >>> manager = StateManager(paper_config)  # Loads latest state
        >>> portfolio = manager.portfolio
    """

    def __init__(
        self,
        config: PaperConfig,
        portfolio: Optional[Portfolio] = None,
    ) -> None:
        """Initialize state manager.

        Args:
            config: Paper trading configuration with state_dir
            portfolio: Initial portfolio (or None to load from disk/create new)
        """
        self._config = config
        self._state_dir = Path(config.state_dir)
        self._state_dir.mkdir(parents=True, exist_ok=True)

        # Initialize state
        self._order_history: list[dict[str, Any]] = []
        self._fill_history: list[dict[str, Any]] = []
        self._last_iteration: int = 0
        self._daily_pnl: float = 0.0

        # Load existing state or use provided portfolio
        if portfolio is not None:
            self._portfolio = portfolio
        else:
            loaded_state = self._load_state()
            if loaded_state is not None:
                self._portfolio = loaded_state.portfolio
                self._order_history = loaded_state.order_history
                self._fill_history = loaded_state.fill_history
                self._last_iteration = loaded_state.iteration
                self._daily_pnl = loaded_state.daily_pnl
                logger.info(
                    f"Loaded state from iteration {self._last_iteration} "
                    f"with {len(self._portfolio.positions)} positions"
                )
            else:
                self._portfolio = Portfolio()
                logger.info("No existing state found, starting fresh")

    @property
    def portfolio(self) -> Portfolio:
        """Get current portfolio."""
        return self._portfolio

    @portfolio.setter
    def portfolio(self, value: Portfolio) -> None:
        """Update portfolio."""
        self._portfolio = value

    @property
    def order_history(self) -> list[dict[str, Any]]:
        """Get order history."""
        return self._order_history.copy()

    @property
    def fill_history(self) -> list[dict[str, Any]]:
        """Get fill history."""
        return self._fill_history.copy()

    @property
    def last_iteration(self) -> int:
        """Get the last saved iteration number."""
        return self._last_iteration

    def record_order(self, order: Order) -> None:
        """Record a submitted order.

        Args:
            order: Order that was submitted
        """
        order_record = {
            "order_id": order.order_id,
            "structure_type": order.structure_type,
            "max_loss": order.max_loss,
            "legs": [leg.to_dict() for leg in order.legs],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self._order_history.append(order_record)
        logger.debug(f"Recorded order {order.order_id}")

    def record_fill(self, fill: Fill) -> None:
        """Record an order fill.

        Args:
            fill: Fill from order execution
        """
        self._fill_history.append(fill.to_dict())
        logger.debug(f"Recorded fill {fill.fill_id} for order {fill.order_id}")

    def update_portfolio(self, portfolio: Portfolio) -> None:
        """Update the portfolio state.

        Args:
            portfolio: New portfolio state
        """
        self._portfolio = portfolio
        self._daily_pnl = portfolio.daily_pnl

    def save_snapshot(self, iteration: int = 0) -> Path:
        """Save current state to disk.

        Args:
            iteration: Current loop iteration number

        Returns:
            Path to the saved snapshot file
        """
        self._last_iteration = iteration

        state = TradingState(
            portfolio=self._portfolio,
            order_history=self._order_history,
            fill_history=self._fill_history,
            timestamp=datetime.now(timezone.utc),
            iteration=iteration,
            daily_pnl=self._daily_pnl,
        )

        # Create filename with timestamp and microseconds for uniqueness
        timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        snapshot_path = self._state_dir / f"state_{timestamp_str}.json"

        with open(snapshot_path, "w") as f:
            json.dump(state.to_dict(), f, indent=2)

        # Remove old snapshots, keeping only the 10 most recent
        snapshots = sorted(self._state_dir.glob("state_*.json"))
        for old_file in snapshots[:-10]:
            old_file.unlink()

        logger.info(f"Saved state snapshot to {snapshot_path}")
        return snapshot_path

    def load_state(self) -> Optional[TradingState]:
        """Load state from disk (public method).

        Returns:
            TradingState if found, None otherwise
        """
        return self._load_state()

    def _load_state(self) -> Optional[TradingState]:
        """Load the most recent state snapshot.

        Returns:
            TradingState if found, None otherwise
        """
        if not self._state_dir.exists():
            return None

        # Find most recent snapshot
        snapshot_files = sorted(self._state_dir.glob("state_*.json"))
        if not snapshot_files:
            return None

        latest = snapshot_files[-1]
        logger.info(f"Loading state from {latest}")

        try:
            with open(latest) as f:
                data = json.load(f)
            return TradingState.from_dict(data)
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to load state from {latest}: {e}")
            return None

    def clear_state(self) -> None:
        """Clear all state (for testing or reset).

        Removes all snapshot files and resets in-memory state.
        """
        # Remove snapshot files
        for f in self._state_dir.glob("state_*.json"):
            f.unlink()

        # Reset in-memory state
        self._portfolio = Portfolio()
        self._order_history = []
        self._fill_history = []
        self._last_iteration = 0
        self._daily_pnl = 0.0

        logger.info("State cleared")

    def get_latest_snapshot_path(self) -> Optional[Path]:
        """Get path to the most recent snapshot file.

        Returns:
            Path to latest snapshot, or None if no snapshots exist
        """
        snapshot_files = sorted(self._state_dir.glob("state_*.json"))
        return snapshot_files[-1] if snapshot_files else None

    def get_snapshot_count(self) -> int:
        """Get the number of saved snapshots.

        Returns:
            Number of snapshot files in state directory
        """
        return len(list(self._state_dir.glob("state_*.json")))
