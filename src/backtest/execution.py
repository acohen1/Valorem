"""Execution simulation for backtesting.

This module simulates realistic trade execution including:
- Bid/ask pricing (no mid-price fills)
- Slippage modeling
- Fee calculation
"""

from dataclasses import dataclass
from datetime import date, datetime

import pandas as pd

from src.config.constants import TradingConstants
from src.config.schema import ExecutionConfig, FeeConfig, PricingConfig, SlippageConfig
from src.pricing import OptionQuote, PositionPricer
from src.strategy.types import OptionLeg

# Import from centralized constants
CONTRACT_MULTIPLIER = TradingConstants.CONTRACT_MULTIPLIER


@dataclass
class FillResult:
    """Result of simulating a trade execution."""

    legs: list[OptionLeg]
    gross_premium: float  # Premium before fees/slippage
    slippage: float  # Total slippage cost
    fees: float  # Total fees
    net_premium: float  # Premium after fees/slippage
    fill_prices: dict[str, float]  # Symbol -> fill price
    timestamp: datetime


class ExecutionSimulator:
    """Simulates trade execution with realistic pricing.

    Execution rules:
    - Buy orders execute at ask price (never mid)
    - Sell orders execute at bid price (never mid)
    - Slippage is applied based on configured model
    - Fees are applied per contract
    """

    def __init__(
        self,
        config: ExecutionConfig,
        pricer: PositionPricer | None = None,
    ) -> None:
        """Initialize execution simulator.

        Args:
            config: Execution configuration with pricing, slippage, and fees
            pricer: Cascading pricer for option quotes. If None, a
                surface-only pricer is created (backward compatible).
        """
        self._pricing_config: PricingConfig = config.pricing
        self._slippage_config: SlippageConfig = config.slippage
        self._fee_config: FeeConfig = config.fees
        self._pricer = pricer or PositionPricer()

    def simulate_entry_fill(
        self,
        legs: list[OptionLeg],
        surface: pd.DataFrame,
        timestamp: datetime,
        as_of: date | None = None,
    ) -> FillResult:
        """Simulate execution of entry order.

        For entries:
        - Long positions (qty > 0): buy at ask + slippage
        - Short positions (qty < 0): sell at bid - slippage

        Args:
            legs: Order legs to execute
            surface: Current market surface with bid/ask quotes
            timestamp: Execution timestamp
            as_of: Trading date for quote lookups. Optional for
                backward compatibility; required when the pricer
                has a QuoteSource.

        Returns:
            FillResult with executed prices and costs
        """
        return self._simulate_fill(legs, surface, timestamp, is_entry=True, as_of=as_of)

    def simulate_exit_fill(
        self,
        legs: list[OptionLeg],
        surface: pd.DataFrame,
        timestamp: datetime,
        as_of: date | None = None,
    ) -> FillResult:
        """Simulate execution of exit order.

        For exits (closing positions):
        - Closing longs (qty < 0 on exit leg): sell at bid - slippage
        - Closing shorts (qty > 0 on exit leg): buy at ask + slippage

        Args:
            legs: Closing legs to execute (already have reversed qty)
            surface: Current market surface with bid/ask quotes
            timestamp: Execution timestamp
            as_of: Trading date for quote lookups. Optional for
                backward compatibility; required when the pricer
                has a QuoteSource.

        Returns:
            FillResult with executed prices and costs
        """
        return self._simulate_fill(legs, surface, timestamp, is_entry=False, as_of=as_of)

    def _simulate_fill(
        self,
        legs: list[OptionLeg],
        surface: pd.DataFrame,
        timestamp: datetime,
        is_entry: bool,
        as_of: date | None = None,
    ) -> FillResult:
        """Internal method to simulate order fill.

        Args:
            legs: Order legs
            surface: Market surface
            timestamp: Execution time
            is_entry: True for entry orders, False for exits
            as_of: Trading date for quote lookups

        Returns:
            FillResult with all execution details
        """
        pricing_date = as_of or date.min

        filled_legs: list[OptionLeg] = []
        fill_prices: dict[str, float] = {}
        gross_premium = 0.0
        total_slippage = 0.0
        total_contracts = 0

        for leg in legs:
            # Get market quote for this option
            quote = self._get_quote(leg.symbol, surface, pricing_date)

            if quote is None:
                # Option not in surface or market data - use entry price as fallback
                fill_price = leg.entry_price
                slippage = 0.0
            else:
                # Determine execution price based on side
                # For entry: long buys at ask, short sells at bid
                # For exit: the leg qty is already reversed (closing)
                if leg.qty > 0:
                    # Buying: execute at ask + slippage
                    base_price = quote.ask
                    slippage_bps = self._slippage_config.fixed_bps
                    fill_price = base_price * (1 + slippage_bps / 10000)
                    slippage = (fill_price - base_price) * abs(leg.qty) * CONTRACT_MULTIPLIER
                else:
                    # Selling: execute at bid - slippage
                    base_price = quote.bid
                    slippage_bps = self._slippage_config.fixed_bps
                    fill_price = base_price * (1 - slippage_bps / 10000)
                    slippage = (base_price - fill_price) * abs(leg.qty) * CONTRACT_MULTIPLIER

            # Create filled leg with actual execution price
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
            # Negative qty (sell) -> negative premium (receive money)
            # Positive qty (buy) -> positive premium (pay money)
            leg_premium = leg.qty * fill_price * CONTRACT_MULTIPLIER
            gross_premium += leg_premium
            total_slippage += slippage
            total_contracts += abs(leg.qty)

        # Calculate fees
        fees = self._calculate_fees(total_contracts)

        # Net premium = gross + fees (fees increase the cost of any trade).
        # Slippage is already embedded in gross_premium via fill prices.
        # For debits (gross > 0): net > gross (pay more).
        # For credits (gross < 0): net closer to 0 (receive less).
        net_premium = gross_premium + fees

        return FillResult(
            legs=filled_legs,
            gross_premium=gross_premium,
            slippage=total_slippage,
            fees=fees,
            net_premium=net_premium,
            fill_prices=fill_prices,
            timestamp=timestamp,
        )

    def _get_quote(
        self, symbol: str, surface: pd.DataFrame, as_of: date
    ) -> OptionQuote | None:
        """Get bid/ask quote for an option via cascading lookup.

        Delegates to the injected PositionPricer which tries:
        1. Surface (exact option_symbol match)
        2. QuoteSource (raw quotes / market data)

        Args:
            symbol: Option symbol (OCC format)
            surface: Market surface DataFrame
            as_of: Trading date for quote lookups

        Returns:
            OptionQuote with bid/ask, or None if not found
        """
        return self._pricer.get_quote(symbol, surface, as_of)

    def _calculate_fees(self, total_contracts: int) -> float:
        """Calculate total fees for a trade.

        Args:
            total_contracts: Total number of contracts in the trade

        Returns:
            Total fee in USD
        """
        per_contract_fee = self._fee_config.per_contract * total_contracts
        return max(per_contract_fee, self._fee_config.per_trade_minimum)

    def calculate_slippage_cost(
        self,
        legs: list[OptionLeg],
        surface: pd.DataFrame,
        as_of: date | None = None,
    ) -> float:
        """Estimate slippage cost for a potential trade without executing.

        Args:
            legs: Proposed order legs
            surface: Current market surface
            as_of: Trading date for quote lookups

        Returns:
            Estimated slippage cost in USD
        """
        pricing_date = as_of or date.min
        total_slippage = 0.0

        for leg in legs:
            quote = self._get_quote(leg.symbol, surface, pricing_date)
            if quote is None:
                continue

            if leg.qty > 0:
                base_price = quote.ask
            else:
                base_price = quote.bid

            slippage_bps = self._slippage_config.fixed_bps
            slippage = base_price * (slippage_bps / 10000) * abs(leg.qty) * CONTRACT_MULTIPLIER
            total_slippage += slippage

        return total_slippage

    def calculate_fee_cost(self, legs: list[OptionLeg]) -> float:
        """Calculate fee cost for a potential trade.

        Args:
            legs: Proposed order legs

        Returns:
            Total fee in USD
        """
        total_contracts = sum(abs(leg.qty) for leg in legs)
        return self._calculate_fees(total_contracts)

    def estimate_execution_cost(
        self,
        legs: list[OptionLeg],
        surface: pd.DataFrame,
        as_of: date | None = None,
    ) -> dict[str, float]:
        """Estimate total execution costs for a trade.

        Args:
            legs: Proposed order legs
            surface: Current market surface
            as_of: Trading date for quote lookups

        Returns:
            Dict with 'slippage', 'fees', 'total' costs
        """
        slippage = self.calculate_slippage_cost(legs, surface, as_of=as_of)
        fees = self.calculate_fee_cost(legs)

        return {
            "slippage": slippage,
            "fees": fees,
            "total": slippage + fees,
        }
