"""
Light-weight statistical helpers used by Valorem's feature-engineering
pipelines.

Public surface
--------------
Import *only* what a caller needs, e.g.

    from valorem.features.compute import garman_klass, bipower_variation
"""

from __future__ import annotations

# ----------------------------------------------------------------------
# Realised-volatility estimators
# ----------------------------------------------------------------------
from .realised_vol import (
    garman_klass,              # OHLC-based estimator
    bipower_variation,         # robust to jumps
)

# ----------------------------------------------------------------------
# If you adding other compute modules (e.g. kurtosis, drawdown, etc.),
# expose their public symbols here, keeping the __all__ list in sync.
# Example:
#
# from .drawdown import max_drawdown
# __all__.extend(["max_drawdown"])
# ----------------------------------------------------------------------

__all__: list[str] = [
    "garman_klass",
    "bipower_variation",
]
