"""Strategy package for trade structure generation and order management.

This package provides:
- Core types: Greeks, OptionLeg, Signal, SignalType, OptionRight
- Trade structures: TradeStructure, CalendarSpread, VerticalSpread
- Structure selection: StructureSelector
- Position sizing: PositionSizer, SizingResult
- Order generation: Order, OrderGenerator, OrderGenerationResult
"""

from .orders import Order, OrderGenerationResult, OrderGenerator
from .selector import StructureSelector
from .sizing import PositionSizer, SizingResult
from .structures import (
    CONTRACT_MULTIPLIER,
    CalendarSpread,
    TradeStructure,
    VerticalSpread,
)
from .types import (
    ExitSignal,
    ExitSignalType,
    Greeks,
    OptionLeg,
    OptionRight,
    Signal,
    SignalType,
)

__all__ = [
    # Core types
    "Greeks",
    "OptionLeg",
    "OptionRight",
    "Signal",
    "SignalType",
    "ExitSignal",
    "ExitSignalType",
    # Structures
    "CONTRACT_MULTIPLIER",
    "TradeStructure",
    "CalendarSpread",
    "VerticalSpread",
    # Selection
    "StructureSelector",
    # Sizing
    "PositionSizer",
    "SizingResult",
    # Orders
    "Order",
    "OrderGenerator",
    "OrderGenerationResult",
]
