"""Pre-trade risk validation.

This module provides the RiskChecker class for validating trades
against risk limits before execution.
"""

from dataclasses import dataclass, field
from enum import Enum

from typing import Optional

from ..config.schema import RiskConfig
from ..strategy.structures.base import CONTRACT_MULTIPLIER
from ..strategy.types import ExitSignal, ExitSignalType, OptionLeg, OptionRight
from .portfolio import Portfolio
from .stress import StressEngine


class RiskCheckStatus(str, Enum):
    """Status of a risk check."""

    APPROVED = "approved"
    REJECTED = "rejected"


@dataclass
class RiskCheckResult:
    """Result of pre-trade risk validation.

    Attributes:
        approved: Whether the trade passed all risk checks
        status: Enum status for programmatic access
        issues: List of risk check failures (empty if approved)
        hypothetical_greeks: Portfolio Greeks if trade were executed
    """

    approved: bool
    issues: list[str] = field(default_factory=list)
    hypothetical_delta: float = 0.0
    hypothetical_gamma: float = 0.0
    hypothetical_vega: float = 0.0
    stress_worst_case_loss: float = 0.0

    @property
    def status(self) -> RiskCheckStatus:
        """Get status as enum."""
        return RiskCheckStatus.APPROVED if self.approved else RiskCheckStatus.REJECTED


class RiskChecker:
    """Pre-trade risk validation.

    Validates proposed trades against risk limits including:
    - Per-trade max loss
    - Per-trade contract limits
    - Portfolio-level Greek caps (delta, gamma, vega)
    - Stress test worst-case loss limits

    All checks must pass for a trade to be approved.
    """

    def __init__(self, config: RiskConfig) -> None:
        """Initialize risk checker.

        Args:
            config: Risk management configuration
        """
        self._config = config
        self._stress_engine = StressEngine(config.stress)

    def check_trade(
        self,
        legs: list[OptionLeg],
        portfolio: Portfolio,
        underlying_price: float,
    ) -> RiskCheckResult:
        """Run all pre-trade risk checks.

        Args:
            legs: Proposed trade legs
            portfolio: Current portfolio state
            underlying_price: Current underlying price (for stress tests)

        Returns:
            RiskCheckResult with approval status and any issues
        """
        issues: list[str] = []

        # 1. Per-trade max loss check
        max_loss = self._compute_max_loss(legs)
        if max_loss > self._config.per_trade.max_loss:
            issues.append(
                f"Max loss ${max_loss:.2f} exceeds per-trade limit "
                f"${self._config.per_trade.max_loss:.2f}"
            )

        # 2. Per-trade contract limit check
        total_contracts = sum(abs(leg.qty) for leg in legs)
        if total_contracts > self._config.per_trade.max_contracts:
            issues.append(
                f"Total contracts {total_contracts} exceeds limit "
                f"{self._config.per_trade.max_contracts}"
            )

        # 3. Portfolio-level caps (check after adding trade)
        hypothetical_portfolio = portfolio.add_trade(legs)

        if abs(hypothetical_portfolio.net_delta) > self._config.caps.max_abs_delta:
            issues.append(
                f"Net delta {hypothetical_portfolio.net_delta:.1f} exceeds cap "
                f"±{self._config.caps.max_abs_delta:.1f}"
            )

        if abs(hypothetical_portfolio.net_gamma) > self._config.caps.max_abs_gamma:
            issues.append(
                f"Net gamma {hypothetical_portfolio.net_gamma:.1f} exceeds cap "
                f"±{self._config.caps.max_abs_gamma:.1f}"
            )

        if abs(hypothetical_portfolio.net_vega) > self._config.caps.max_abs_vega:
            issues.append(
                f"Net vega {hypothetical_portfolio.net_vega:.1f} exceeds cap "
                f"±{self._config.caps.max_abs_vega:.1f}"
            )

        # 4. Stress testing (if enabled)
        stress_worst_case = 0.0
        if self._config.stress.enabled:
            stress_result = self._stress_engine.run_stress_test(
                portfolio=hypothetical_portfolio,
                underlying_price=underlying_price,
            )
            stress_worst_case = stress_result.worst_case_loss

            if stress_result.worst_case_loss > self._config.caps.max_daily_loss:
                issues.append(
                    f"Stress test worst-case loss ${stress_result.worst_case_loss:.2f} "
                    f"exceeds limit ${self._config.caps.max_daily_loss:.2f} "
                    f"({stress_result.worst_scenario})"
                )

        return RiskCheckResult(
            approved=len(issues) == 0,
            issues=issues,
            hypothetical_delta=hypothetical_portfolio.net_delta,
            hypothetical_gamma=hypothetical_portfolio.net_gamma,
            hypothetical_vega=hypothetical_portfolio.net_vega,
            stress_worst_case_loss=stress_worst_case,
        )

    def _compute_max_loss(self, legs: list[OptionLeg]) -> float:
        """Compute maximum loss for a trade structure.

        Args:
            legs: Trade legs to evaluate

        Returns:
            Maximum possible loss as a positive number
        """
        # Calculate net premium (cash flow perspective)
        # Buying (qty > 0) = cash outflow (negative premium from our perspective)
        # Selling (qty < 0) = cash inflow (positive premium)
        net_premium = -sum(
            leg.qty * leg.entry_price * CONTRACT_MULTIPLIER for leg in legs
        )

        if net_premium < 0:
            # Debit spread: max loss = net debit paid
            return abs(net_premium)
        else:
            # Credit spread: max loss = spread width - net credit
            return self._compute_credit_spread_max_loss(legs, net_premium)

    def _compute_credit_spread_max_loss(
        self,
        legs: list[OptionLeg],
        net_credit: float,
    ) -> float:
        """Compute max loss for a credit spread.

        For vertical spreads: max loss = spread width * multiplier - credit
        For calendar spreads: max loss is more complex, use conservative estimate

        Args:
            legs: Trade legs
            net_credit: Net credit received (positive number)

        Returns:
            Maximum possible loss as a positive number
        """
        # Get unique strikes
        strikes = sorted(set(leg.strike for leg in legs))

        if len(strikes) == 2:
            # Vertical spread
            spread_width = (strikes[1] - strikes[0]) * CONTRACT_MULTIPLIER
            max_loss = spread_width - net_credit
            return max(0.0, max_loss)
        elif len(strikes) == 1:
            # Calendar spread (same strike, different expiries)
            # Conservative: max loss = premium of long leg
            long_legs = [leg for leg in legs if leg.qty > 0]
            if long_legs:
                long_premium = sum(
                    leg.entry_price * CONTRACT_MULTIPLIER * leg.qty
                    for leg in long_legs
                )
                return abs(long_premium)
            else:
                # Shouldn't happen for valid structures, return 0
                return 0.0
        else:
            # Complex structure (iron condor, skew trade, etc.)
            # Group legs by right and compute spread width per side.
            # Max loss = wider spread width - net credit (only one side
            # can be in-the-money at expiry).
            put_legs = [leg for leg in legs if leg.right == OptionRight.PUT]
            call_legs = [leg for leg in legs if leg.right == OptionRight.CALL]

            put_width = 0.0
            if len(put_legs) >= 2:
                put_strikes = sorted(leg.strike for leg in put_legs)
                put_width = (put_strikes[-1] - put_strikes[0]) * CONTRACT_MULTIPLIER

            call_width = 0.0
            if len(call_legs) >= 2:
                call_strikes = sorted(leg.strike for leg in call_legs)
                call_width = (call_strikes[-1] - call_strikes[0]) * CONTRACT_MULTIPLIER

            max_spread_width = max(put_width, call_width)
            return max(0.0, max_spread_width - net_credit)

    def check_portfolio_limits(self, portfolio: Portfolio) -> RiskCheckResult:
        """Check current portfolio against risk limits.

        This is used for continuous monitoring, not pre-trade validation.

        Args:
            portfolio: Current portfolio state

        Returns:
            RiskCheckResult indicating any limit breaches
        """
        issues: list[str] = []

        if abs(portfolio.net_delta) > self._config.caps.max_abs_delta:
            issues.append(
                f"Portfolio delta {portfolio.net_delta:.1f} exceeds cap "
                f"±{self._config.caps.max_abs_delta:.1f}"
            )

        if abs(portfolio.net_gamma) > self._config.caps.max_abs_gamma:
            issues.append(
                f"Portfolio gamma {portfolio.net_gamma:.1f} exceeds cap "
                f"±{self._config.caps.max_abs_gamma:.1f}"
            )

        if abs(portfolio.net_vega) > self._config.caps.max_abs_vega:
            issues.append(
                f"Portfolio vega {portfolio.net_vega:.1f} exceeds cap "
                f"±{self._config.caps.max_abs_vega:.1f}"
            )

        return RiskCheckResult(
            approved=len(issues) == 0,
            issues=issues,
            hypothetical_delta=portfolio.net_delta,
            hypothetical_gamma=portfolio.net_gamma,
            hypothetical_vega=portfolio.net_vega,
        )

    def check_exit_order(
        self,
        position_id: str,
        closing_legs: list[OptionLeg],
        portfolio: Portfolio,
        exit_signal: Optional[ExitSignal] = None,
    ) -> RiskCheckResult:
        """Validate an exit order for a position.

        Exit orders are more permissive than entry orders since closing
        positions generally reduces risk. However, we still validate:
        - The position exists and is open
        - The closing legs match the position (opposite quantities)
        - Stop-loss and emergency exits are always allowed

        Args:
            position_id: ID of position being closed
            closing_legs: Proposed closing legs (opposite quantities)
            portfolio: Current portfolio state
            exit_signal: Exit signal that triggered this order (if any)

        Returns:
            RiskCheckResult with approval status and any issues
        """
        issues: list[str] = []

        # 1. Find the position
        position = portfolio.get_position_by_id(position_id)
        if position is None:
            issues.append(f"Position {position_id} not found in portfolio")
            return RiskCheckResult(approved=False, issues=issues)

        # 2. Emergency exits (stop-loss, kill-switch) are always approved
        if exit_signal is not None and exit_signal.exit_type in (
            ExitSignalType.STOP_LOSS,
            ExitSignalType.MANUAL,
        ):
            return RiskCheckResult(
                approved=True,
                issues=[],
                hypothetical_delta=portfolio.net_delta,
                hypothetical_gamma=portfolio.net_gamma,
                hypothetical_vega=portfolio.net_vega,
            )

        # 3. Validate closing legs are inverse of position legs
        position_symbols = {leg.symbol: leg.qty for leg in position.legs}
        for closing_leg in closing_legs:
            expected_qty = position_symbols.get(closing_leg.symbol)
            if expected_qty is None:
                issues.append(
                    f"Closing leg {closing_leg.symbol} not found in position"
                )
            elif closing_leg.qty != -expected_qty:
                issues.append(
                    f"Closing leg qty {closing_leg.qty} should be {-expected_qty} "
                    f"for {closing_leg.symbol}"
                )

        # 4. Non-emergency exits: verify this reduces risk (portfolio Greeks)
        # Create hypothetical portfolio without this position
        hypothetical = portfolio.close_position(position_id)

        # Exit is approved if it reduces absolute Greeks or we have issues
        if issues:
            return RiskCheckResult(approved=False, issues=issues)

        return RiskCheckResult(
            approved=True,
            issues=[],
            hypothetical_delta=hypothetical.net_delta,
            hypothetical_gamma=hypothetical.net_gamma,
            hypothetical_vega=hypothetical.net_vega,
        )
