"""Global underlying feature generators.

This module provides feature generators for underlying price data:
- Returns: log and simple returns at various horizons
- Realized volatility: rolling variance, vol-of-vol, drawdown
"""

from src.features.global_.realized_vol import (
    RealizedVolConfig,
    RealizedVolGenerator,
)
from src.features.global_.returns import (
    ReturnsConfig,
    ReturnsGenerator,
)

__all__ = [
    "ReturnsGenerator",
    "ReturnsConfig",
    "RealizedVolGenerator",
    "RealizedVolConfig",
]
