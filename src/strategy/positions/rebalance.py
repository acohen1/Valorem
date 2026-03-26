"""Greek drift detection and rebalancing for position management.

This module provides the RebalanceEngine class for detecting when portfolio
Greeks have drifted beyond acceptable bands and generating rebalancing signals.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from ...config.schema import DriftBandsConfig, RebalancingConfig
from ...risk.portfolio import Portfolio
from ..types import ExitSignal, ExitSignalType
from .lifecycle import ManagedPosition

logger = logging.getLogger(__name__)


@dataclass
class DriftStatus:
    """Status of a single Greek's drift from target.

    Attributes:
        greek_name: Name of the Greek (delta, gamma, vega)
        current: Current portfolio value
        target: Target value
        drift: Absolute drift from target
        max_allowed: Maximum allowed drift
        breached: Whether drift exceeds max_allowed
    """

    greek_name: str
    current: float
    target: float
    drift: float
    max_allowed: float
    breached: bool


@dataclass
class DriftResult:
    """Result of portfolio drift analysis.

    Attributes:
        needs_rebalance: Whether rebalancing is required
        drifts: Dictionary of Greek name → DriftStatus
        most_breached: Name of the most breached Greek (if any)
    """

    needs_rebalance: bool
    drifts: dict[str, DriftStatus]
    most_breached: Optional[str] = None


class RebalanceEngine:
    """Detect Greek drift and generate rebalancing signals.

    Monitors portfolio Greeks and generates exit signals for positions
    that contribute most to drift when bands are breached.
    """

    def __init__(
        self,
        rebalancing_config: RebalancingConfig,
        drift_bands_config: DriftBandsConfig,
    ) -> None:
        """Initialize rebalance engine.

        Args:
            rebalancing_config: Rebalancing strategy configuration
            drift_bands_config: Drift band thresholds
        """
        self._config = rebalancing_config
        self._bands = drift_bands_config

    @property
    def enabled(self) -> bool:
        """Check if rebalancing is enabled."""
        return self._config.enabled

    def check_drift(self, portfolio: Portfolio) -> DriftResult:
        """Check if portfolio Greeks have drifted beyond bands.

        Args:
            portfolio: Current portfolio state

        Returns:
            DriftResult with drift status for each Greek
        """
        drifts: dict[str, DriftStatus] = {}
        needs_rebalance = False
        most_breached: Optional[str] = None
        max_breach_ratio = 0.0

        # Delta drift
        delta_target = self._bands.delta_target
        delta_drift = abs(portfolio.net_delta - delta_target)
        delta_status = DriftStatus(
            greek_name="delta",
            current=portfolio.net_delta,
            target=delta_target,
            drift=delta_drift,
            max_allowed=self._bands.delta_max_drift,
            breached=delta_drift > self._bands.delta_max_drift,
        )
        drifts["delta"] = delta_status

        if delta_status.breached:
            needs_rebalance = True
            breach_ratio = delta_drift / self._bands.delta_max_drift
            if breach_ratio > max_breach_ratio:
                max_breach_ratio = breach_ratio
                most_breached = "delta"

        # Vega drift
        vega_target = self._bands.vega_target
        vega_drift = abs(portfolio.net_vega - vega_target)
        vega_status = DriftStatus(
            greek_name="vega",
            current=portfolio.net_vega,
            target=vega_target,
            drift=vega_drift,
            max_allowed=self._bands.vega_max_drift,
            breached=vega_drift > self._bands.vega_max_drift,
        )
        drifts["vega"] = vega_status

        if vega_status.breached:
            needs_rebalance = True
            breach_ratio = vega_drift / self._bands.vega_max_drift
            if breach_ratio > max_breach_ratio:
                max_breach_ratio = breach_ratio
                most_breached = "vega"

        # Gamma drift
        gamma_drift = abs(portfolio.net_gamma)
        gamma_status = DriftStatus(
            greek_name="gamma",
            current=portfolio.net_gamma,
            target=0.0,  # Gamma target is always 0
            drift=gamma_drift,
            max_allowed=self._bands.gamma_max_drift,
            breached=gamma_drift > self._bands.gamma_max_drift,
        )
        drifts["gamma"] = gamma_status

        if gamma_status.breached:
            needs_rebalance = True
            breach_ratio = gamma_drift / self._bands.gamma_max_drift
            if breach_ratio > max_breach_ratio:
                max_breach_ratio = breach_ratio
                most_breached = "gamma"

        result = DriftResult(
            needs_rebalance=needs_rebalance,
            drifts=drifts,
            most_breached=most_breached,
        )

        if needs_rebalance:
            logger.info(
                f"Drift detected: most_breached={most_breached}, "
                f"delta={delta_drift:.2f}, vega={vega_drift:.2f}, "
                f"gamma={gamma_drift:.2f}"
            )

        return result

    def generate_rebalance_signals(
        self,
        drift_result: DriftResult,
        positions: list[ManagedPosition],
        portfolio: Portfolio,
    ) -> list[ExitSignal]:
        """Generate exit signals to rebalance portfolio.

        Uses configured strategy:
        - close_first: Close positions contributing most to drift

        Args:
            drift_result: Result from check_drift()
            positions: List of managed positions
            portfolio: Current portfolio state

        Returns:
            List of exit signals for positions to close
        """
        if not self._config.enabled:
            return []

        if not drift_result.needs_rebalance:
            return []

        if self._config.strategy != "close_first":
            logger.warning(
                f"Rebalance strategy '{self._config.strategy}' not implemented, "
                "using close_first"
            )

        return self._close_first_rebalance(drift_result, positions, portfolio)

    def _close_first_rebalance(
        self,
        drift_result: DriftResult,
        positions: list[ManagedPosition],
        portfolio: Portfolio,
    ) -> list[ExitSignal]:
        """Rebalance by closing positions that contribute most to drift.

        Args:
            drift_result: Drift analysis result
            positions: Managed positions to evaluate
            portfolio: Current portfolio state

        Returns:
            Exit signals for positions to close
        """
        signals: list[ExitSignal] = []
        remaining_trades = self._config.max_trades_per_rebalance

        # Filter to only OPEN positions
        open_positions = [p for p in positions if p.is_open()]
        if not open_positions:
            return []

        # Track positions already signaled
        signaled_ids: set[str] = set()

        # Process breached Greeks in order of breach severity
        sorted_drifts = sorted(
            [(name, status) for name, status in drift_result.drifts.items() if status.breached],
            key=lambda x: x[1].drift / x[1].max_allowed,
            reverse=True,
        )

        for greek_name, drift_status in sorted_drifts:
            if remaining_trades <= 0:
                break

            # Get Greek value from position
            def get_greek(pos: ManagedPosition) -> float:
                return getattr(pos.current_greeks, greek_name, 0.0)

            # Determine direction: close positions with same-sign Greek as drift
            # If portfolio delta is positive (above target), close positions with positive delta
            drift_direction = 1 if drift_status.current > drift_status.target else -1

            # Sort positions by contribution to this Greek's drift
            # Highest same-sign Greek first
            sorted_positions = sorted(
                [p for p in open_positions if p.position_id not in signaled_ids],
                key=lambda p: get_greek(p) * drift_direction,
                reverse=True,
            )

            for position in sorted_positions:
                if remaining_trades <= 0:
                    break

                position_greek = get_greek(position)

                # Only close if position contributes to the drift
                # (same sign as drift direction)
                if position_greek * drift_direction <= 0:
                    continue

                # Calculate improvement from closing this position
                improvement_pct = abs(position_greek) / drift_status.drift

                # Only close if it would make meaningful improvement
                min_improvement = 0.1  # At least 10% improvement
                if improvement_pct < min_improvement:
                    continue

                signals.append(
                    ExitSignal(
                        exit_type=ExitSignalType.REBALANCE,
                        position_id=position.position_id,
                        urgency=0.7,  # Moderate urgency for rebalancing
                        reason=(
                            f"Rebalance {greek_name}: position {greek_name}="
                            f"{position_greek:.2f}, closing would reduce drift by "
                            f"{improvement_pct:.1%}"
                        ),
                        current_pnl_pct=position.pnl_pct_of_max_loss(),
                    )
                )

                signaled_ids.add(position.position_id)
                remaining_trades -= 1

                logger.info(
                    f"Rebalance signal for position {position.position_id}: "
                    f"{greek_name}={position_greek:.2f}, improvement={improvement_pct:.1%}"
                )

        return signals

    def get_current_drift(self, portfolio: Portfolio) -> dict[str, float]:
        """Get current drift values for each Greek.

        Convenience method for monitoring.

        Args:
            portfolio: Current portfolio state

        Returns:
            Dictionary of Greek name → drift value
        """
        return {
            "delta": abs(portfolio.net_delta - self._bands.delta_target),
            "vega": abs(portfolio.net_vega - self._bands.vega_target),
            "gamma": abs(portfolio.net_gamma),
        }

    def is_within_bands(self, portfolio: Portfolio) -> bool:
        """Check if portfolio is within all drift bands.

        Args:
            portfolio: Current portfolio state

        Returns:
            True if all Greeks are within bands
        """
        result = self.check_drift(portfolio)
        return not result.needs_rebalance
