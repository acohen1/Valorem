"""Greeks and date calculation utilities.

This module consolidates duplicated calculation patterns across the codebase.
All position Greeks scaling and DTE calculations should use these utilities.

Usage:
    from src.utils.calculations import aggregate_greeks, compute_dte, min_dte

    # Aggregate Greeks across legs
    total_greeks = aggregate_greeks(legs)

    # Compute DTE for a single expiry
    dte = compute_dte(expiry_date)

    # Find minimum DTE across legs
    nearest_dte = min_dte(legs)
"""

import math
from datetime import date
from typing import TYPE_CHECKING, Iterable

import pandas as pd

from src.config.constants import TradingConstants
from src.strategy.types import Greeks

if TYPE_CHECKING:
    from src.strategy.types import OptionLeg


def aggregate_greeks(
    legs: Iterable["OptionLeg"],
    multiplier: int = TradingConstants.CONTRACT_MULTIPLIER,
) -> Greeks:
    """Aggregate Greeks across option legs with proper scaling.

    Each leg's Greeks are scaled by quantity and contract multiplier,
    then summed to produce total position Greeks.

    Args:
        legs: Iterable of option legs to aggregate
        multiplier: Contract multiplier (default: 100 for standard options)

    Returns:
        Aggregated Greeks scaled by qty * multiplier for each leg

    Example:
        >>> legs = [
        ...     OptionLeg(qty=1, greeks=Greeks(delta=0.5, gamma=0.01, vega=0.1, theta=-0.02)),
        ...     OptionLeg(qty=-1, greeks=Greeks(delta=0.3, gamma=0.02, vega=0.15, theta=-0.03)),
        ... ]
        >>> total = aggregate_greeks(legs)
        >>> # Result: delta = (1*0.5 - 1*0.3) * 100 = 20
    """
    total = Greeks(delta=0.0, gamma=0.0, vega=0.0, theta=0.0)
    for leg in legs:
        scaled = leg.greeks.scale(leg.qty * multiplier)
        total = total + scaled
    return total


def compute_dte(expiry: date, as_of: date | None = None) -> int:
    """Compute days to expiration.

    Args:
        expiry: Option expiration date
        as_of: Reference date (default: today)

    Returns:
        Days until expiration (can be negative if expired)

    Example:
        >>> from datetime import date
        >>> compute_dte(date(2024, 3, 15), as_of=date(2024, 3, 10))
        5
    """
    as_of = as_of or date.today()
    return (expiry - as_of).days


def min_dte(legs: Iterable["OptionLeg"], as_of: date | None = None) -> int:
    """Find minimum days to expiration across legs.

    Args:
        legs: Option legs to check
        as_of: Reference date (default: today)

    Returns:
        Minimum DTE across all legs (clamped to 0 minimum)

    Raises:
        ValueError: If legs is empty

    Example:
        >>> nearest = min_dte(position.legs)
        >>> if nearest <= 7:
        ...     print("Position nearing expiry")
    """
    as_of = as_of or date.today()
    min_value = float("inf")

    for leg in legs:
        dte = (leg.expiry - as_of).days
        if dte < min_value:
            min_value = dte

    if min_value == float("inf"):
        raise ValueError("Cannot compute min_dte for empty legs")

    return max(int(min_value), 0)


def greeks_from_row(row: pd.Series, fallback: Greeks) -> Greeks:
    """Extract Greeks from a surface row, NaN-safe.

    Returns the surface value when present and non-NaN, the fallback
    value otherwise. Needed because raw-quote rows lack Greeks columns,
    and ``pd.concat`` fills missing columns with NaN.

    ``pd.Series.get("delta", default)`` returns NaN (not default) when
    the key exists with a NaN value. This function handles that case
    explicitly.

    Args:
        row: Surface DataFrame row (pd.Series).
        fallback: Greeks to use when values are missing or NaN.

    Returns:
        Greeks with surface values preferred over fallback.
    """

    def _safe(key: str, default: float) -> float:
        val = row.get(key)
        if val is None or (isinstance(val, float) and math.isnan(val)):
            return default
        return float(val)

    return Greeks(
        delta=_safe("delta", fallback.delta),
        gamma=_safe("gamma", fallback.gamma),
        vega=_safe("vega", fallback.vega),
        theta=_safe("theta", fallback.theta),
    )
