"""Order routing for paper and live trading.

This module provides the OrderRouter interface and PaperOrderRouter
implementation for simulating order execution in paper trading mode.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

import pandas as pd

from ..config.constants import TradingConstants
from ..config.schema import ExecutionConfig
from ..strategy.orders import Order
from ..strategy.types import OptionLeg

logger = logging.getLogger(__name__)

# Import from centralized constants
CONTRACT_MULTIPLIER = TradingConstants.CONTRACT_MULTIPLIER


@dataclass
class Fill:
    """Result of an order fill.

    Attributes:
        order_id: ID of the order that was filled
        legs: Filled option legs with execution prices
        gross_premium: Premium before fees/slippage (negative = debit)
        slippage: Total slippage cost
        fees: Total fees
        net_premium: Premium after fees/slippage
        fill_prices: Symbol -> fill price mapping
        timestamp: When the fill occurred
        fill_id: Unique identifier for this fill
    """

    order_id: str
    legs: list[OptionLeg]
    gross_premium: float
    slippage: float
    fees: float
    net_premium: float
    fill_prices: dict[str, float]
    timestamp: datetime
    fill_id: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize Fill to dictionary."""
        return {
            "order_id": self.order_id,
            "legs": [leg.to_dict() for leg in self.legs],
            "gross_premium": self.gross_premium,
            "slippage": self.slippage,
            "fees": self.fees,
            "net_premium": self.net_premium,
            "fill_prices": self.fill_prices,
            "timestamp": self.timestamp.isoformat(),
            "fill_id": self.fill_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Fill":
        """Deserialize Fill from dictionary."""
        return cls(
            order_id=data["order_id"],
            legs=[OptionLeg.from_dict(leg) for leg in data["legs"]],
            gross_premium=float(data["gross_premium"]),
            slippage=float(data["slippage"]),
            fees=float(data["fees"]),
            net_premium=float(data["net_premium"]),
            fill_prices=data["fill_prices"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            fill_id=data["fill_id"],
        )


class OrderRouter(ABC):
    """Abstract interface for order routing.

    Implementations handle routing orders to execution venues
    (paper trading simulation, broker APIs, etc.).
    """

    @abstractmethod
    def route_order(
        self,
        order: Order,
        surface: pd.DataFrame,
    ) -> Optional[Fill]:
        """Route an order for execution.

        Args:
            order: Order to execute
            surface: Current market surface with bid/ask quotes

        Returns:
            Fill if order executed successfully, None if rejected/failed
        """
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the router is available for order routing.

        Returns:
            True if the router can accept orders
        """
        ...


class PaperOrderRouter(OrderRouter):
    """Paper trading order router - simulates fills instantly.

    This router simulates order execution for paper trading:
    - Long positions (qty > 0): buy at ask + slippage
    - Short positions (qty < 0): sell at bid - slippage
    - Fees applied per contract
    """

    def __init__(self, config: ExecutionConfig) -> None:
        """Initialize paper order router.

        Args:
            config: Execution configuration with pricing, slippage, fees
        """
        self._config = config
        self._fill_counter = 0

    def route_order(
        self,
        order: Order,
        surface: pd.DataFrame,
    ) -> Optional[Fill]:
        """Route order - simulates instant fill in paper mode.

        Args:
            order: Order to execute
            surface: Current market surface

        Returns:
            Fill with simulated execution details
        """
        filled_legs: list[OptionLeg] = []
        fill_prices: dict[str, float] = {}
        gross_premium = 0.0
        total_slippage = 0.0
        total_contracts = 0

        for leg in order.legs:
            # Get market quote
            quote = self._get_quote(leg.symbol, surface)

            if quote is None:
                # Option not in surface - use entry price as fallback
                fill_price = leg.entry_price
                slippage = 0.0
                logger.warning(
                    f"Quote not found for {leg.symbol}, using entry price {fill_price}"
                )
            else:
                # Determine execution price based on side
                if leg.qty > 0:
                    # Buying: execute at ask + slippage
                    base_price = quote["ask"]
                    slippage_bps = self._config.slippage.fixed_bps
                    fill_price = base_price * (1 + slippage_bps / 10000)
                    slippage = (
                        (fill_price - base_price)
                        * abs(leg.qty)
                        * CONTRACT_MULTIPLIER
                    )
                else:
                    # Selling: execute at bid - slippage
                    base_price = quote["bid"]
                    slippage_bps = self._config.slippage.fixed_bps
                    fill_price = base_price * (1 - slippage_bps / 10000)
                    slippage = (
                        (base_price - fill_price)
                        * abs(leg.qty)
                        * CONTRACT_MULTIPLIER
                    )

            # Create filled leg with execution price
            filled_leg = OptionLeg(
                symbol=leg.symbol,
                qty=leg.qty,
                entry_price=fill_price,
                strike=leg.strike,
                expiry=leg.expiry,
                right=leg.right,
                greeks=leg.greeks,
            )
            filled_legs.append(filled_leg)
            fill_prices[leg.symbol] = fill_price

            # Accumulate totals
            # Premium: qty * price * multiplier
            # Negative qty (sell) -> receive money (positive flow)
            # Positive qty (buy) -> pay money (negative flow)
            leg_premium = leg.qty * fill_price * CONTRACT_MULTIPLIER
            gross_premium += leg_premium
            total_slippage += slippage
            total_contracts += abs(leg.qty)

        # Calculate fees
        fees = self._calculate_fees(total_contracts)

        # Net premium = gross - fees
        # Note: slippage already embedded in gross
        net_premium = gross_premium - fees

        # Generate fill ID
        self._fill_counter += 1
        fill_id = f"paper_fill_{self._fill_counter:06d}"

        fill = Fill(
            order_id=order.order_id,
            legs=filled_legs,
            gross_premium=gross_premium,
            slippage=total_slippage,
            fees=fees,
            net_premium=net_premium,
            fill_prices=fill_prices,
            timestamp=datetime.now(timezone.utc),
            fill_id=fill_id,
        )

        logger.info(
            f"Order {order.order_id} filled: "
            f"gross={gross_premium:.2f}, fees={fees:.2f}, net={net_premium:.2f}"
        )

        return fill

    def is_available(self) -> bool:
        """Paper router is always available."""
        return True

    def _get_quote(
        self, symbol: str, surface: pd.DataFrame
    ) -> Optional[dict[str, float]]:
        """Get bid/ask quote for an option.

        Args:
            symbol: Option symbol (OSI format)
            surface: Market surface DataFrame

        Returns:
            Dict with 'bid' and 'ask' keys, or None if not found
        """
        rows = surface[surface["option_symbol"] == symbol]
        if rows.empty:
            return None

        row = rows.iloc[0]
        return {
            "bid": float(row["bid"]),
            "ask": float(row["ask"]),
        }

    def _calculate_fees(self, total_contracts: int) -> float:
        """Calculate total fees for a trade.

        Args:
            total_contracts: Total number of contracts

        Returns:
            Total fee in USD
        """
        per_contract_fee = self._config.fees.per_contract * total_contracts
        return max(per_contract_fee, self._config.fees.per_trade_minimum)
