"""Centralized constants for the Valorem trading system.

This module provides a single source of truth for all magic values
used throughout the codebase. Import from here instead of hardcoding.

Usage:
    from src.config.constants import SurfaceConstants, TradingConstants

    # Use constants
    for tenor in SurfaceConstants.TENOR_DAYS_DEFAULT:
        process_tenor(tenor)

    premium = qty * price * TradingConstants.CONTRACT_MULTIPLIER
"""

from typing import Final


class SurfaceConstants:
    """Constants for volatility surface construction and analysis.

    These constants define the standard node grid (tenors x delta buckets)
    used for surface interpolation, feature engineering, and signal generation.
    """

    # Tenor bins (days to expiration) — single canonical grid for all paths
    # (training, backtest, and live inference must all use the same grid)
    TENOR_DAYS_DEFAULT: Final[tuple[int, ...]] = (7, 14, 30, 60, 90, 120)

    # Delta bucket definitions
    # Live trading buckets (7 buckets: deep OTM puts to deep OTM calls)
    DELTA_BUCKETS_LIVE: Final[tuple[str, ...]] = (
        "P40",
        "P25",
        "P10",
        "ATM",
        "C10",
        "C25",
        "C40",
    )

    # Feature engineering bucket order (5 buckets, symmetric around ATM)
    DELTA_BUCKETS_FEATURE: Final[tuple[str, ...]] = (
        "P10",
        "P25",
        "ATM",
        "C25",
        "C10",
    )

    # Graph model bucket order (7 buckets, ordered by delta)
    DELTA_BUCKETS_GRAPH: Final[tuple[str, ...]] = (
        "P10",
        "P25",
        "P40",
        "ATM",
        "C40",
        "C25",
        "C10",
    )

    # Delta values for bucket mapping (moneyness)
    DELTA_VALUES: Final[dict[str, float]] = {
        "P10": -0.10,
        "P25": -0.25,
        "P40": -0.40,
        "ATM": 0.50,
        "C40": 0.40,
        "C25": 0.25,
        "C10": 0.10,
    }


class TradingConstants:
    """Constants for trading and execution.

    These constants are used for position sizing, Greeks scaling,
    and P&L calculations throughout the system.
    """

    # Standard options contract multiplier (100 shares per contract)
    CONTRACT_MULTIPLIER: Final[int] = 100


class MarketConstants:
    """Constants for market data and timing.

    These constants define market hours, default data sources,
    and other market-related configuration values.
    """

    # Default risk-free rate source (10-year Treasury yield)
    DEFAULT_RATE_SERIES: Final[str] = "DGS10"

    # Market close hour in UTC (4 PM ET = 21:00 UTC during EST, 20:00 during EDT).
    # Using 21 (EST) as fixed default — creates ~1hr TTE over-estimate during EDT
    # (Mar-Nov). Impact: ~0.6% for 7d options, ~4.2% for 1d options. Acceptable
    # for the current daily-snapshot pipeline; revisit if intraday TTE matters.
    CLOSE_HOUR_UTC: Final[int] = 21

    # Default FRED series for macro features
    DEFAULT_FRED_SERIES: Final[tuple[str, ...]] = ("DGS10", "DGS2", "VIXCLS")

    # Extended FRED series for comprehensive macro features
    EXTENDED_FRED_SERIES: Final[tuple[str, ...]] = (
        "DGS10",
        "DGS2",
        "DGS5",
        "DGS30",
        "FEDFUNDS",
        "T10Y2Y",
        "VIXCLS",
    )
