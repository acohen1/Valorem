"""Automated trading halt logic.

This module provides the KillSwitch class for automatically halting
trading when risk limits are breached.
"""

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import Optional

from ..config.schema import KillSwitchConfig
from .portfolio import Portfolio
from .stress import StressResult


logger = logging.getLogger(__name__)


class KillSwitchTrigger(str, Enum):
    """Reasons for kill switch activation."""

    DAILY_LOSS = "daily_loss"
    STRESS_BREACH = "stress_breach"
    LIQUIDITY_COLLAPSE = "liquidity_collapse"
    MANUAL = "manual"


@dataclass
class KillSwitchResult:
    """Result of kill switch check.

    Attributes:
        triggered: Whether the kill switch is active
        reason: Human-readable reason for trigger (if triggered)
        trigger_type: Enum type of trigger (if triggered)
        triggered_at: When the kill switch was triggered (if triggered)
    """

    triggered: bool
    reason: Optional[str] = None
    trigger_type: Optional[KillSwitchTrigger] = None
    triggered_at: Optional[datetime] = None


class KillSwitch:
    """Automated trading halt logic.

    Monitors portfolio and market conditions for breach events that
    should halt all trading activity. Once triggered, the kill switch
    requires manual reset to resume trading.

    Trigger conditions:
    - Daily loss exceeds threshold
    - Stress test worst-case exceeds acceptable loss
    - Liquidity collapse (spreads blow out)
    - Manual trigger
    """

    def __init__(self, config: KillSwitchConfig) -> None:
        """Initialize kill switch.

        Args:
            config: Kill switch configuration
        """
        self._config = config
        self._triggered = False
        self._trigger_reason: Optional[str] = None
        self._trigger_type: Optional[KillSwitchTrigger] = None
        self._triggered_at: Optional[datetime] = None

    @property
    def is_triggered(self) -> bool:
        """Check if kill switch is currently active."""
        return self._triggered

    @property
    def trigger_reason(self) -> Optional[str]:
        """Get the reason for trigger (if triggered)."""
        return self._trigger_reason

    def check(
        self,
        portfolio: Portfolio,
        stress_result: Optional[StressResult] = None,
        max_spread_pct: Optional[float] = None,
    ) -> KillSwitchResult:
        """Check if kill switch should trigger.

        Args:
            portfolio: Current portfolio state
            stress_result: Latest stress test result (optional)
            max_spread_pct: Maximum spread percentage in market (optional)

        Returns:
            KillSwitchResult indicating current state
        """
        # If already triggered, return current state
        if self._triggered:
            return KillSwitchResult(
                triggered=True,
                reason=self._trigger_reason,
                trigger_type=self._trigger_type,
                triggered_at=self._triggered_at,
            )

        # Check daily loss
        if self._config.halt_on_daily_loss:
            if portfolio.daily_pnl < -self._config.max_daily_loss:
                self._trigger(
                    f"Daily loss ${abs(portfolio.daily_pnl):.2f} exceeds limit "
                    f"${self._config.max_daily_loss:.2f}",
                    KillSwitchTrigger.DAILY_LOSS,
                )

        # Check stress breach
        if self._config.halt_on_stress_breach and stress_result is not None:
            if stress_result.worst_case_loss > portfolio.max_acceptable_loss:
                self._trigger(
                    f"Stress test worst-case ${stress_result.worst_case_loss:.2f} "
                    f"exceeds acceptable loss ${portfolio.max_acceptable_loss:.2f}",
                    KillSwitchTrigger.STRESS_BREACH,
                )

        # Check liquidity collapse
        if self._config.halt_on_liquidity_collapse and max_spread_pct is not None:
            if max_spread_pct > self._config.max_spread_pct:
                self._trigger(
                    f"Spread blowout: {max_spread_pct:.1%} exceeds "
                    f"{self._config.max_spread_pct:.1%}",
                    KillSwitchTrigger.LIQUIDITY_COLLAPSE,
                )

        return KillSwitchResult(
            triggered=self._triggered,
            reason=self._trigger_reason,
            trigger_type=self._trigger_type,
            triggered_at=self._triggered_at,
        )

    def check_daily_loss(self, portfolio: Portfolio) -> KillSwitchResult:
        """Check only daily loss condition.

        Args:
            portfolio: Current portfolio state

        Returns:
            KillSwitchResult indicating current state
        """
        if self._triggered:
            return KillSwitchResult(
                triggered=True,
                reason=self._trigger_reason,
                trigger_type=self._trigger_type,
                triggered_at=self._triggered_at,
            )

        if self._config.halt_on_daily_loss:
            if portfolio.daily_pnl < -self._config.max_daily_loss:
                self._trigger(
                    f"Daily loss ${abs(portfolio.daily_pnl):.2f} exceeds limit "
                    f"${self._config.max_daily_loss:.2f}",
                    KillSwitchTrigger.DAILY_LOSS,
                )

        return KillSwitchResult(
            triggered=self._triggered,
            reason=self._trigger_reason,
            trigger_type=self._trigger_type,
            triggered_at=self._triggered_at,
        )

    def check_liquidity(self, max_spread_pct: float) -> KillSwitchResult:
        """Check only liquidity condition.

        Args:
            max_spread_pct: Maximum spread percentage in market

        Returns:
            KillSwitchResult indicating current state
        """
        if self._triggered:
            return KillSwitchResult(
                triggered=True,
                reason=self._trigger_reason,
                trigger_type=self._trigger_type,
                triggered_at=self._triggered_at,
            )

        if self._config.halt_on_liquidity_collapse:
            if max_spread_pct > self._config.max_spread_pct:
                self._trigger(
                    f"Spread blowout: {max_spread_pct:.1%} exceeds "
                    f"{self._config.max_spread_pct:.1%}",
                    KillSwitchTrigger.LIQUIDITY_COLLAPSE,
                )

        return KillSwitchResult(
            triggered=self._triggered,
            reason=self._trigger_reason,
            trigger_type=self._trigger_type,
            triggered_at=self._triggered_at,
        )

    def trigger_manual(self, reason: str = "Manual halt") -> KillSwitchResult:
        """Manually trigger the kill switch.

        Args:
            reason: Reason for manual trigger

        Returns:
            KillSwitchResult indicating triggered state
        """
        self._trigger(reason, KillSwitchTrigger.MANUAL)
        return KillSwitchResult(
            triggered=True,
            reason=self._trigger_reason,
            trigger_type=self._trigger_type,
            triggered_at=self._triggered_at,
        )

    def reset(self) -> None:
        """Reset kill switch (requires manual intervention).

        This should only be called after the situation has been reviewed
        and it's safe to resume trading.
        """
        if self._triggered:
            logger.warning(
                f"Kill switch reset. Previous trigger: {self._trigger_reason} "
                f"at {self._triggered_at}"
            )

        self._triggered = False
        self._trigger_reason = None
        self._trigger_type = None
        self._triggered_at = None

    def _trigger(self, reason: str, trigger_type: KillSwitchTrigger) -> None:
        """Internal method to trigger the kill switch.

        Args:
            reason: Human-readable reason
            trigger_type: Type of trigger
        """
        if not self._triggered:
            self._triggered = True
            self._trigger_reason = reason
            self._trigger_type = trigger_type
            self._triggered_at = datetime.now(UTC)
            logger.critical(f"KILL SWITCH TRIGGERED: {reason}")
