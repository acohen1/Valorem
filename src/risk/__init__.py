"""Risk management package.

This package provides pre-trade risk checks, stress testing, and
automated kill switch functionality for the trading system.

Classes:
    PortfolioState: Immutable point-in-time portfolio view (preferred)
    Portfolio: Mutable portfolio state (for backwards compatibility)
    Position: Individual position tracking
    RiskChecker: Pre-trade risk validation
    RiskCheckResult: Result of risk checks
    RiskCheckStatus: Approval status enum
    StressEngine: Portfolio stress testing
    StressResult: Aggregated stress test results
    StressScenario: Single stress scenario result
    KillSwitch: Automated trading halt logic
    KillSwitchResult: Result of kill switch check
    KillSwitchTrigger: Trigger type enum
"""

from .checker import RiskCheckResult, RiskCheckStatus, RiskChecker
from .covariance import CovarianceEstimator, CovarianceResult, filter_correlated_signals
from .kill_switch import KillSwitch, KillSwitchResult, KillSwitchTrigger
from .portfolio import (
    CONTRACT_MULTIPLIER,
    Portfolio,
    PortfolioState,
    Position,
    PositionState,
)
from .stress import StressEngine, StressResult, StressScenario

__all__ = [
    # Portfolio
    "PortfolioState",
    "Portfolio",
    "Position",
    "PositionState",
    "CONTRACT_MULTIPLIER",
    # Risk Checker
    "RiskChecker",
    "RiskCheckResult",
    "RiskCheckStatus",
    # Covariance
    "CovarianceEstimator",
    "CovarianceResult",
    "filter_correlated_signals",
    # Stress Testing
    "StressEngine",
    "StressResult",
    "StressScenario",
    # Kill Switch
    "KillSwitch",
    "KillSwitchResult",
    "KillSwitchTrigger",
]
