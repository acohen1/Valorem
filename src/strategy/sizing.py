"""Position sizing logic.

This module provides the PositionSizer class for computing appropriate
position sizes based on signal characteristics, risk limits, and liquidity.
"""

from dataclasses import dataclass
from typing import Optional

from ..config.schema import SizingConfig
from .types import Signal


@dataclass
class SizingResult:
    """Result of position sizing calculation.

    Attributes:
        quantity_multiplier: Multiplier to apply to base leg quantities
        base_contracts: Base number of contracts before adjustment
        adjusted_contracts: Final number of contracts after all adjustments
        confidence_factor: Factor applied based on signal confidence
        liquidity_factor: Factor applied based on liquidity
        risk_factor: Factor applied based on risk limits
        reason: Description of how size was determined
    """

    quantity_multiplier: int
    base_contracts: int
    adjusted_contracts: int
    confidence_factor: float
    liquidity_factor: float
    risk_factor: float
    reason: str


class PositionSizer:
    """Compute position sizes based on signal, risk, and liquidity.

    Supports multiple sizing methods:
    - fixed: Use base_contracts from config
    - risk_parity: Size based on max_loss_per_trade / structure_max_loss
    - kelly_fraction: Use Kelly criterion with configured fraction

    Position sizes are further adjusted by:
    - Signal confidence (if scale_by_confidence is True)
    - Liquidity (if scale_by_liquidity is True)

    All sizes are capped at max_contracts_per_trade.
    """

    def __init__(self, config: Optional[SizingConfig] = None) -> None:
        """Initialize position sizer.

        Args:
            config: Sizing configuration (optional, uses defaults if not provided)
        """
        self._config = config or SizingConfig()

    def compute_size(
        self,
        signal: Signal,
        max_loss_per_contract: float,
        liquidity: Optional[float] = None,
    ) -> SizingResult:
        """Compute position size for a trade.

        Args:
            signal: Trading signal with edge and confidence
            max_loss_per_contract: Maximum loss per single contract for this structure
            liquidity: Optional liquidity measure (e.g., average daily volume)

        Returns:
            SizingResult with final size and breakdown
        """
        # Start with base size from config method
        base_contracts = self._compute_base_size(
            signal=signal,
            max_loss_per_contract=max_loss_per_contract,
        )

        # Apply confidence scaling
        confidence_factor = self._compute_confidence_factor(signal)

        # Apply liquidity scaling
        liquidity_factor = self._compute_liquidity_factor(liquidity)

        # Apply risk-based scaling
        risk_factor = self._compute_risk_factor(max_loss_per_contract)

        # Calculate adjusted size
        adjusted = base_contracts * confidence_factor * liquidity_factor * risk_factor

        # Round to nearest integer and cap
        final_contracts = max(1, min(int(round(adjusted)), self._config.max_contracts_per_trade))

        # Compute final multiplier
        quantity_multiplier = final_contracts

        # Build reason string
        reason = self._build_reason(
            method=self._config.method,
            base=base_contracts,
            confidence_factor=confidence_factor,
            liquidity_factor=liquidity_factor,
            risk_factor=risk_factor,
            final=final_contracts,
        )

        return SizingResult(
            quantity_multiplier=quantity_multiplier,
            base_contracts=base_contracts,
            adjusted_contracts=final_contracts,
            confidence_factor=confidence_factor,
            liquidity_factor=liquidity_factor,
            risk_factor=risk_factor,
            reason=reason,
        )

    def _compute_base_size(
        self,
        signal: Signal,
        max_loss_per_contract: float,
    ) -> int:
        """Compute base size from config method.

        Args:
            signal: Trading signal
            max_loss_per_contract: Max loss per contract

        Returns:
            Base number of contracts
        """
        if self._config.method == "fixed":
            return self._config.base_contracts

        if self._config.method == "risk_parity":
            # Size so that max loss equals configured limit
            if max_loss_per_contract <= 0:
                return self._config.base_contracts
            contracts = self._config.max_loss_per_trade / max_loss_per_contract
            return max(1, int(contracts))

        if self._config.method == "kelly_fraction":
            # Kelly: f* = (p*b - q) / b where b = edge, p = win rate (confidence proxy)
            # Simplified: use confidence * edge as sizing factor
            edge = abs(signal.edge)
            if edge <= 0:
                return self._config.base_contracts

            # Kelly fraction of the bankroll allocated to this trade
            kelly_size = signal.confidence * edge * self._config.kelly_fraction

            # Convert to contracts based on max loss
            if max_loss_per_contract <= 0:
                return self._config.base_contracts

            # Assume we're willing to risk kelly_size * max_loss_per_trade
            risk_amount = kelly_size * self._config.max_loss_per_trade * 10  # Scale up
            contracts = risk_amount / max_loss_per_contract
            return max(1, int(contracts))

        # Default to fixed
        return self._config.base_contracts

    def _compute_confidence_factor(self, signal: Signal) -> float:
        """Compute scaling factor based on signal confidence.

        Args:
            signal: Trading signal with confidence

        Returns:
            Scaling factor between 0.5 and 1.5
        """
        if not self._config.scale_by_confidence:
            return 1.0

        # Scale linearly: confidence 0.5 → factor 0.5, confidence 1.0 → factor 1.5
        # Minimum factor is 0.5 to avoid tiny positions
        return max(0.5, 0.5 + signal.confidence)

    def _compute_liquidity_factor(self, liquidity: Optional[float]) -> float:
        """Compute scaling factor based on liquidity.

        Args:
            liquidity: Liquidity measure (e.g., average daily volume)

        Returns:
            Scaling factor between 0.1 and 1.0
        """
        if not self._config.scale_by_liquidity:
            return 1.0

        if liquidity is None:
            return 1.0

        # Scale based on min_liquidity_contracts threshold
        # If liquidity >= threshold, factor = 1.0
        # If liquidity < threshold, scale down proportionally
        min_liq = self._config.min_liquidity_contracts
        if liquidity >= min_liq:
            return 1.0

        # Scale down, minimum factor is 0.1
        factor = liquidity / min_liq
        return max(0.1, factor)

    def _compute_risk_factor(self, max_loss_per_contract: float) -> float:
        """Compute scaling factor based on risk limits.

        Args:
            max_loss_per_contract: Max loss per contract

        Returns:
            Scaling factor (usually 1.0 unless risk is very high)
        """
        if max_loss_per_contract <= 0:
            return 1.0

        # If max loss per contract exceeds our per-trade limit, scale down
        if max_loss_per_contract > self._config.max_loss_per_trade:
            return self._config.max_loss_per_trade / max_loss_per_contract

        return 1.0

    def _build_reason(
        self,
        method: str,
        base: int,
        confidence_factor: float,
        liquidity_factor: float,
        risk_factor: float,
        final: int,
    ) -> str:
        """Build explanation string for sizing decision.

        Args:
            method: Sizing method used
            base: Base contracts
            confidence_factor: Confidence scaling factor
            liquidity_factor: Liquidity scaling factor
            risk_factor: Risk scaling factor
            final: Final contract count

        Returns:
            Human-readable explanation
        """
        parts = [f"Method: {method}, base={base}"]

        if confidence_factor != 1.0:
            parts.append(f"conf_factor={confidence_factor:.2f}")
        if liquidity_factor != 1.0:
            parts.append(f"liq_factor={liquidity_factor:.2f}")
        if risk_factor != 1.0:
            parts.append(f"risk_factor={risk_factor:.2f}")

        parts.append(f"final={final}")

        return ", ".join(parts)
