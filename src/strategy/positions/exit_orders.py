"""Exit order generation for position closing.

This module provides the ExitOrderGenerator class for creating closing
orders from exit signals.
"""

from __future__ import annotations

import logging
import uuid
from datetime import date, datetime, timezone
from typing import Optional

import pandas as pd

from ...config.constants import TradingConstants
from ...config.schema import ExecutionConfig
from ...pricing import PositionPricer
from ...risk.portfolio import PositionState
from ...utils.calculations import aggregate_greeks
from ..types import ExitSignal, Greeks, OptionLeg
from .lifecycle import ManagedPosition

logger = logging.getLogger(__name__)

# Import from centralized constants
CONTRACT_MULTIPLIER = TradingConstants.CONTRACT_MULTIPLIER


class ExitOrderGenerator:
    """Generate orders to close positions.

    Creates closing orders from exit signals by reversing the position
    legs and pricing them at current market prices.
    """

    def __init__(
        self,
        config: ExecutionConfig,
        pricer: PositionPricer | None = None,
    ) -> None:
        """Initialize exit order generator.

        Args:
            config: Execution configuration for pricing
            pricer: Cascading pricer for option legs. If None, a
                surface-only pricer is created (backward compatible).
        """
        self._config = config
        self._pricer = pricer or PositionPricer()

    def generate_exit_orders(
        self,
        exit_signals: list[ExitSignal],
        positions: dict[str, ManagedPosition],
        surface: pd.DataFrame,
        as_of: date | None = None,
    ) -> list["ExitOrder"]:
        """Generate exit orders from exit signals.

        Args:
            exit_signals: Signals indicating positions to close
            positions: Map of position_id → ManagedPosition
            surface: Current surface for pricing
            as_of: Trading date for quote lookups. Optional for
                backward compatibility; required when the pricer
                has a QuoteSource.

        Returns:
            List of exit orders
        """
        orders: list[ExitOrder] = []

        for signal in exit_signals:
            position = positions.get(signal.position_id)
            if position is None:
                logger.warning(
                    f"Position {signal.position_id} not found for exit signal"
                )
                continue

            if position.state != PositionState.OPEN:
                logger.debug(
                    f"Position {signal.position_id} not in OPEN state "
                    f"(state={position.state}), skipping"
                )
                continue

            # Create closing legs (reverse of opening legs)
            closing_legs = self._create_closing_legs(position.legs, surface, as_of)

            # Compute Greeks for the closing order (opposite of position Greeks)
            closing_greeks = self._compute_closing_greeks(closing_legs)

            # Compute net premium for the exit
            net_premium = self._compute_net_premium(closing_legs)

            order = ExitOrder(
                order_id=self._generate_order_id(),
                legs=closing_legs,
                structure_type=position.structure_type,
                exit_signal=signal,
                position_id=position.position_id,
                greeks=closing_greeks,
                net_premium=net_premium,
                timestamp=datetime.now(timezone.utc),
            )

            orders.append(order)
            logger.info(
                f"Generated exit order {order.order_id} for position "
                f"{position.position_id} ({signal.exit_type.value})"
            )

        return orders

    def _create_closing_legs(
        self,
        open_legs: list[OptionLeg],
        surface: pd.DataFrame,
        as_of: date | None = None,
    ) -> list[OptionLeg]:
        """Create legs to close position (reverse quantities).

        Delegates pricing to the injected ``PositionPricer`` which applies
        the three-tier cascade (surface → QuoteSource → entry fallback).

        For closing:
        - Long positions (qty > 0) → sell at bid
        - Short positions (qty < 0) → buy at ask

        Args:
            open_legs: Open position legs
            surface: Current surface for pricing
            as_of: Trading date for quote lookups

        Returns:
            Closing legs with reversed quantities
        """
        closing_legs: list[OptionLeg] = []
        # date.min is a safe placeholder when as_of is not provided;
        # only the QuoteSource tier uses the date, and the default
        # pricer (no QuoteSource) never reaches it.
        pricing_date = as_of or date.min

        for leg in open_legs:
            priced = self._pricer.price_leg(leg, surface, pricing_date)

            closing_legs.append(
                OptionLeg(
                    symbol=leg.symbol,
                    qty=-leg.qty,  # Reverse quantity
                    entry_price=priced.price,
                    strike=leg.strike,
                    expiry=leg.expiry,
                    right=leg.right,
                    greeks=priced.greeks,
                )
            )

        return closing_legs

    def _compute_closing_greeks(self, closing_legs: list[OptionLeg]) -> Greeks:
        """Compute aggregate Greeks for closing order.

        Args:
            closing_legs: Closing legs

        Returns:
            Aggregate Greeks (scaled by qty and contract multiplier)
        """
        return aggregate_greeks(closing_legs)

    def _compute_net_premium(self, legs: list[OptionLeg]) -> float:
        """Compute net premium for the order.

        Args:
            legs: Order legs

        Returns:
            Net premium (negative = pay, positive = receive)
        """
        return sum(leg.qty * leg.entry_price * CONTRACT_MULTIPLIER for leg in legs)

    def _generate_order_id(self) -> str:
        """Generate unique order ID.

        Returns:
            UUID string prefixed with 'exit-'
        """
        return f"exit-{uuid.uuid4()}"


class ExitOrder:
    """Exit order generated from an exit signal.

    Represents a closing order ready for execution.

    Attributes:
        order_id: Unique identifier for this order
        legs: Option legs to close the position (reversed quantities)
        structure_type: Name of the original structure
        exit_signal: Exit signal that triggered this order
        position_id: ID of position being closed
        greeks: Aggregate Greeks for the closing order
        net_premium: Net premium for the exit (negative = pay, positive = receive)
        timestamp: When the order was generated
    """

    __slots__ = [
        "order_id",
        "legs",
        "structure_type",
        "exit_signal",
        "position_id",
        "greeks",
        "net_premium",
        "timestamp",
    ]

    def __init__(
        self,
        order_id: str,
        legs: list[OptionLeg],
        structure_type: str,
        exit_signal: ExitSignal,
        position_id: str,
        greeks: Greeks,
        net_premium: float,
        timestamp: datetime,
    ) -> None:
        """Initialize exit order.

        Args:
            order_id: Unique order identifier
            legs: Closing legs
            structure_type: Original structure name
            exit_signal: Triggering exit signal
            position_id: Position being closed
            greeks: Aggregate Greeks
            net_premium: Net premium for the exit
            timestamp: Order creation time
        """
        self.order_id = order_id
        self.legs = legs
        self.structure_type = structure_type
        self.exit_signal = exit_signal
        self.position_id = position_id
        self.greeks = greeks
        self.net_premium = net_premium
        self.timestamp = timestamp

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"ExitOrder(order_id={self.order_id!r}, "
            f"position_id={self.position_id!r}, "
            f"exit_type={self.exit_signal.exit_type.value}, "
            f"net_premium={self.net_premium:.2f})"
        )
