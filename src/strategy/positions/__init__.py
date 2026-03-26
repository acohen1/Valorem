"""Position lifecycle and exit signal management.

This package provides position lifecycle tracking, exit signal generation,
exit order generation, rebalancing, and position management orchestration.
"""

from .exit_orders import ExitOrder, ExitOrderGenerator
from .exit_signals import ExitSignalGenerator
from .lifecycle import ManagedPosition
from .manager import Fill, PositionManager
from .rebalance import DriftResult, DriftStatus, RebalanceEngine

__all__ = [
    "ManagedPosition",
    "ExitSignalGenerator",
    "ExitOrder",
    "ExitOrderGenerator",
    "RebalanceEngine",
    "DriftResult",
    "DriftStatus",
    "PositionManager",
    "Fill",
]
