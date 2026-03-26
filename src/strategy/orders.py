"""Order generation from trading signals.

This module provides the Order dataclass and OrderGenerator class for
converting trading signals into risk-checked, sized orders.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional

import pandas as pd

from ..config.schema import ExecutionConfig
from ..risk.portfolio import Portfolio
from .selector import StructureSelector
from .sizing import PositionSizer, SizingResult
from .types import ExitSignal, Greeks, OptionLeg, Signal

if TYPE_CHECKING:
    from ..risk.checker import RiskChecker

logger = logging.getLogger(__name__)


@dataclass
class Order:
    """Executable order generated from a trading signal.

    Represents a fully-formed, risk-checked order ready for execution.

    Attributes:
        order_id: Unique identifier for this order
        legs: Option legs comprising the trade structure
        structure_type: Name of the structure (e.g., "CalendarSpread")
        signal: Original signal that generated this order
        max_loss: Maximum possible loss for this order
        greeks: Aggregate Greeks for the order
        sizing_result: Details of how the position was sized
        timestamp: When the order was generated
        is_exit: Whether this is an exit order (closing a position)
        exit_signal: Exit signal that triggered this order (if is_exit=True)
        position_id: ID of position being closed (if is_exit=True) or opened
    """

    order_id: str
    legs: list[OptionLeg]
    structure_type: str
    signal: Signal
    max_loss: float
    greeks: Greeks
    sizing_result: SizingResult
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    is_exit: bool = False
    exit_signal: Optional[ExitSignal] = None
    position_id: Optional[str] = None


@dataclass
class OrderGenerationResult:
    """Result of order generation for a batch of signals.

    Attributes:
        orders: Successfully generated orders
        rejected_signals: Signals that failed to generate orders
        rejection_reasons: Mapping of signal index to rejection reason
    """

    orders: list[Order]
    rejected_signals: list[Signal]
    rejection_reasons: dict[int, str]


class OrderGenerator:
    """Generate executable orders from trading signals.

    The order generation process:
    1. Filter signals by edge and confidence thresholds
    2. Select appropriate structure for each signal
    3. Create option legs from surface data
    4. Size the position based on risk and liquidity
    5. Run pre-trade risk checks
    6. Return approved orders

    Any signal that fails at any step is rejected with a reason.
    """

    def __init__(
        self,
        config: ExecutionConfig,
        risk_checker: RiskChecker,
        structure_selector: Optional[StructureSelector] = None,
        position_sizer: Optional[PositionSizer] = None,
    ) -> None:
        """Initialize order generator.

        Args:
            config: Execution configuration
            risk_checker: RiskChecker instance for pre-trade validation
            structure_selector: Optional StructureSelector (creates default if None)
            position_sizer: Optional PositionSizer (creates default if None)
        """
        self._config = config
        self._risk_checker = risk_checker
        self._structure_selector = structure_selector or StructureSelector(config)
        self._position_sizer = position_sizer or PositionSizer(config.sizing)

    def generate_orders(
        self,
        signals: list[Signal],
        surface: pd.DataFrame,
        portfolio: Portfolio,
        underlying_price: float,
    ) -> OrderGenerationResult:
        """Generate orders from a list of signals.

        Args:
            signals: List of trading signals to process
            surface: Surface snapshot DataFrame with option data
            portfolio: Current portfolio state for risk checks
            underlying_price: Current underlying price

        Returns:
            OrderGenerationResult with approved orders and rejected signals
        """
        orders: list[Order] = []
        rejected_signals: list[Signal] = []
        rejection_reasons: dict[int, str] = {}

        structure_failures: dict[str, int] = {}

        for idx, signal in enumerate(signals):
            try:
                order = self._process_signal(
                    signal=signal,
                    surface=surface,
                    portfolio=portfolio,
                    underlying_price=underlying_price,
                )

                if order is not None:
                    orders.append(order)
                    # Update portfolio for subsequent risk checks
                    portfolio = portfolio.add_trade(order.legs)
                else:
                    # Signal was filtered or rejected
                    if idx not in rejection_reasons:
                        rejection_reasons[idx] = "Unknown rejection"
                    rejected_signals.append(signal)

            except ValueError as e:
                rejection_reasons[idx] = f"Structure creation failed: {e}"
                rejected_signals.append(signal)
                reason = str(e)
                structure_failures[reason] = structure_failures.get(reason, 0) + 1
                logger.debug(f"Signal {idx} rejected: {e}")

            except Exception as e:
                rejection_reasons[idx] = f"Unexpected error: {e}"
                rejected_signals.append(signal)
                logger.error(f"Signal {idx} failed unexpectedly: {e}")

        # Log summary of structure creation failures (one line per reason)
        if structure_failures:
            summary = ", ".join(
                f"{reason} ({count}x)" for reason, count in structure_failures.items()
            )
            logger.info(f"Structure creation failures: {summary}")

        return OrderGenerationResult(
            orders=orders,
            rejected_signals=rejected_signals,
            rejection_reasons=rejection_reasons,
        )

    def _process_signal(
        self,
        signal: Signal,
        surface: pd.DataFrame,
        portfolio: Portfolio,
        underlying_price: float,
    ) -> Optional[Order]:
        """Process a single signal into an order.

        Args:
            signal: Trading signal to process
            surface: Surface snapshot DataFrame
            portfolio: Current portfolio state
            underlying_price: Current underlying price

        Returns:
            Order if approved, None if filtered or rejected
        """
        # 1. Check signal thresholds
        if not self._passes_thresholds(signal):
            return None

        # 2. Select structure
        structure = self._structure_selector.select_structure(signal, surface)
        if structure is None:
            logger.debug(f"No structure selected for signal type {signal.signal_type}")
            return None

        # 3. Create legs
        legs = structure.create_legs(signal, surface)

        # 4. Compute risk metrics for base size
        base_max_loss = structure.compute_max_loss(legs)
        base_greeks = structure.compute_greeks(legs)

        # 5. Get liquidity info
        liquidity = self._get_liquidity(legs, surface)

        # 6. Compute position size
        sizing_result = self._position_sizer.compute_size(
            signal=signal,
            max_loss_per_contract=base_max_loss,
            liquidity=liquidity,
        )

        # 7. Scale legs by quantity multiplier
        scaled_legs = self._scale_legs(legs, sizing_result.quantity_multiplier)

        # 8. Recompute risk metrics for scaled position
        scaled_max_loss = structure.compute_max_loss(scaled_legs)
        scaled_greeks = structure.compute_greeks(scaled_legs)

        # 9. Pre-trade risk checks
        risk_result = self._risk_checker.check_trade(
            legs=scaled_legs,
            portfolio=portfolio,
            underlying_price=underlying_price,
        )

        if not risk_result.approved:
            logger.info(f"Trade rejected by risk checker: {risk_result.issues}")
            return None

        # 10. Create order
        order = Order(
            order_id=self._generate_order_id(),
            legs=scaled_legs,
            structure_type=structure.__class__.__name__,
            signal=signal,
            max_loss=scaled_max_loss,
            greeks=scaled_greeks,
            sizing_result=sizing_result,
        )

        return order

    def _passes_thresholds(self, signal: Signal) -> bool:
        """Check if signal passes configured thresholds.

        Args:
            signal: Signal to check

        Returns:
            True if signal passes all thresholds
        """
        threshold = self._config.signal_threshold

        # Check minimum edge
        if abs(signal.edge) < threshold.min_edge:
            logger.debug(f"Signal edge {signal.edge} below threshold {threshold.min_edge}")
            return False

        # Check minimum confidence
        if signal.confidence < threshold.min_confidence:
            logger.debug(
                f"Signal confidence {signal.confidence} below threshold "
                f"{threshold.min_confidence}"
            )
            return False

        # Check maximum uncertainty (1 - confidence)
        uncertainty = 1.0 - signal.confidence
        if uncertainty > threshold.max_uncertainty:
            logger.debug(
                f"Signal uncertainty {uncertainty} above threshold {threshold.max_uncertainty}"
            )
            return False

        return True

    def _get_liquidity(
        self,
        legs: list[OptionLeg],
        surface: pd.DataFrame,
    ) -> Optional[float]:
        """Get liquidity measure for the trade legs.

        Looks up volume/OI from surface data for the leg symbols.

        Args:
            legs: Trade legs
            surface: Surface data with volume/OI columns

        Returns:
            Average daily volume across legs, or None if not available
        """
        if "volume" not in surface.columns:
            return None

        total_volume = 0.0
        count = 0

        for leg in legs:
            # Find the leg in surface by symbol
            matches = surface[surface["option_symbol"] == leg.symbol]
            if not matches.empty:
                vol = matches.iloc[0].get("volume", 0)
                if pd.notna(vol):
                    total_volume += vol
                    count += 1

        if count == 0:
            return None

        return total_volume / count

    def _scale_legs(
        self,
        legs: list[OptionLeg],
        multiplier: int,
    ) -> list[OptionLeg]:
        """Scale leg quantities by multiplier.

        Args:
            legs: Original legs
            multiplier: Quantity multiplier

        Returns:
            New list of legs with scaled quantities
        """
        return [
            OptionLeg(
                symbol=leg.symbol,
                qty=leg.qty * multiplier,
                entry_price=leg.entry_price,
                strike=leg.strike,
                expiry=leg.expiry,
                right=leg.right,
                greeks=leg.greeks,
            )
            for leg in legs
        ]

    def _generate_order_id(self) -> str:
        """Generate unique order ID.

        Returns:
            UUID string
        """
        return str(uuid.uuid4())

    def generate_single_order(
        self,
        signal: Signal,
        surface: pd.DataFrame,
        portfolio: Portfolio,
        underlying_price: float,
    ) -> Optional[Order]:
        """Convenience method to generate a single order.

        Args:
            signal: Single trading signal
            surface: Surface snapshot DataFrame
            portfolio: Current portfolio state
            underlying_price: Current underlying price

        Returns:
            Order if approved, None otherwise
        """
        result = self.generate_orders(
            signals=[signal],
            surface=surface,
            portfolio=portfolio,
            underlying_price=underlying_price,
        )

        if result.orders:
            return result.orders[0]
        return None
