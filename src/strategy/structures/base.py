"""Abstract base class for trade structures.

This module defines the TradeStructure abstract base class that all
option trade structures must implement.
"""

from abc import ABC, abstractmethod

import pandas as pd

from ...config.constants import TradingConstants
from ...utils.calculations import aggregate_greeks
from ..types import Greeks, OptionLeg, Signal

# Re-export for backwards compatibility
CONTRACT_MULTIPLIER = TradingConstants.CONTRACT_MULTIPLIER


class TradeStructure(ABC):
    """Abstract base class for bounded-risk option trade structures.

    All trade structures must implement:
    - create_legs: Create option legs from a signal and surface data
    - compute_max_loss: Calculate the maximum possible loss

    The base class provides default implementations for:
    - compute_greeks: Aggregate Greeks across all legs
    - net_premium: Calculate net premium paid/received
    - is_debit_spread: Check if the structure is a debit spread
    """

    @abstractmethod
    def create_legs(
        self,
        signal: Signal,
        surface: pd.DataFrame,
    ) -> list[OptionLeg]:
        """Create option legs for this structure.

        Args:
            signal: Trading signal with direction and node info
            surface: Surface snapshot DataFrame with columns:
                - option_symbol: OSI symbol
                - tenor_days: Days to expiration
                - delta_bucket: Bucket assignment ("P25", "ATM", "C25", etc.)
                - strike: Strike price
                - expiry: Expiration date
                - right: "C" or "P"
                - bid, ask: Quote prices
                - delta, gamma, vega, theta: Greeks

        Returns:
            List of OptionLeg objects for the trade

        Raises:
            ValueError: If required options cannot be found in surface
        """
        ...

    @abstractmethod
    def compute_max_loss(self, legs: list[OptionLeg]) -> float:
        """Compute maximum loss for this structure in dollars.

        All structures must have bounded risk - no naked positions allowed.

        Args:
            legs: List of option legs in the structure

        Returns:
            Maximum possible loss as a positive number
        """
        ...

    def compute_greeks(self, legs: list[OptionLeg]) -> Greeks:
        """Compute aggregate Greeks across all legs.

        Each leg's Greeks are scaled by quantity and contract multiplier.
        Subclasses can override for structure-specific logic.

        Args:
            legs: List of option legs in the structure

        Returns:
            Aggregate Greeks for the entire structure
        """
        return aggregate_greeks(legs)

    def net_premium(self, legs: list[OptionLeg]) -> float:
        """Compute net premium paid/received from cash flow perspective.

        Args:
            legs: List of option legs in the structure

        Returns:
            Net premium in dollars:
            - Negative = debit (paid to open, cash out)
            - Positive = credit (received to open, cash in)

        Note:
            Buying (qty > 0) means paying premium (negative cash flow).
            Selling (qty < 0) means receiving premium (positive cash flow).
        """
        # Negate because buying (positive qty) is cash outflow (negative)
        return -sum(leg.qty * leg.entry_price * CONTRACT_MULTIPLIER for leg in legs)

    def is_debit_spread(self, legs: list[OptionLeg]) -> bool:
        """Check if this is a debit spread (pay to open).

        Args:
            legs: List of option legs in the structure

        Returns:
            True if net premium is negative (debit spread)
        """
        return self.net_premium(legs) < 0
