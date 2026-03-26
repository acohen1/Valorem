"""Trade structures package.

Provides bounded-risk option trade structures including calendar spreads,
vertical spreads, skew trades, and iron condors.
"""

from .base import CONTRACT_MULTIPLIER, TradeStructure
from .calendar import CalendarSpread
from .iron_condor import IronCondor
from .skew import SkewTrade
from .vertical import VerticalSpread

__all__ = [
    "CONTRACT_MULTIPLIER",
    "TradeStructure",
    "CalendarSpread",
    "IronCondor",
    "SkewTrade",
    "VerticalSpread",
]
