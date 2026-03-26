"""Utility functions for the Valorem trading system.

This module provides reusable utilities:
- Greeks aggregation across positions/legs
- Date calculations (DTE computation)
- Protocol validation for dependency injection

Note:
    Imports are lazy to avoid circular dependencies.
    Import directly from submodules for best performance:
        from src.utils.calculations import aggregate_greeks
        from src.utils.validation import validate_protocol
"""

from typing import TYPE_CHECKING

__all__ = [
    # Calculations
    "aggregate_greeks",
    "compute_dte",
    "min_dte",
    # Validation
    "validate_protocol",
    "validate_not_none",
    "validate_positive",
    "validate_non_negative",
    "validate_in_range",
]


def __getattr__(name: str):
    """Lazy import to avoid circular dependencies."""
    if name in ("aggregate_greeks", "compute_dte", "min_dte"):
        from src.utils.calculations import aggregate_greeks, compute_dte, min_dte
        return {"aggregate_greeks": aggregate_greeks, "compute_dte": compute_dte, "min_dte": min_dte}[name]

    if name in ("validate_protocol", "validate_not_none", "validate_positive", "validate_non_negative", "validate_in_range"):
        from src.utils.validation import (
            validate_in_range,
            validate_non_negative,
            validate_not_none,
            validate_positive,
            validate_protocol,
        )
        return {
            "validate_protocol": validate_protocol,
            "validate_not_none": validate_not_none,
            "validate_positive": validate_positive,
            "validate_non_negative": validate_non_negative,
            "validate_in_range": validate_in_range,
        }[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
