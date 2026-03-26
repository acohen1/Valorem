"""Iron Condor trade structure.

An iron condor involves selling OTM call and put spreads to collect
premium when implied volatility is elevated.
"""

import pandas as pd

from ..types import Greeks, OptionLeg, OptionRight, Signal
from .base import CONTRACT_MULTIPLIER, TradeStructure


class IronCondor(TradeStructure):
    """Iron condor: sell OTM call spread + sell OTM put spread.

    Used when IV is elevated but directional confidence is low.
    Profits from IV contraction and time decay if price stays
    within the short strikes.

    Structure:
    - Sell OTM put spread: sell 25-delta put, buy further OTM put (wing)
    - Sell OTM call spread: sell 25-delta call, buy further OTM call (wing)

    This creates a credit spread on both sides, collecting premium.

    Max loss:
    - Wider wing spread width minus net credit received
    - Occurs if price moves beyond either wing

    Attributes:
        wing_width_pct: Width of each wing as % of underlying price
    """

    def __init__(self, wing_width_pct: float = 0.02) -> None:
        """Initialize iron condor builder.

        Args:
            wing_width_pct: Wing width as % of underlying (default 2%)
        """
        self._wing_width_pct = wing_width_pct

    def create_legs(
        self,
        signal: Signal,
        surface: pd.DataFrame,
    ) -> list[OptionLeg]:
        """Create iron condor legs.

        For iron condor (IV elevated, neutral view):
        - Sell 25-delta put, buy wing put (put credit spread)
        - Sell 25-delta call, buy wing call (call credit spread)

        Signal edge direction affects sizing but not structure:
        - Positive edge: standard iron condor (neutral)
        - Negative edge: also standard iron condor (we only enter on elevated IV)

        Args:
            signal: Trading signal indicating elevated IV
            surface: Surface snapshot with option data

        Returns:
            List of four OptionLeg objects (put spread + call spread)

        Raises:
            ValueError: If required options cannot be found
        """
        # Get options at signal tenor
        same_tenor = surface[surface["tenor_days"] == signal.tenor_days]
        if same_tenor.empty:
            raise ValueError(f"No options at tenor {signal.tenor_days}d")

        # Get underlying price
        if "underlying_price" in same_tenor.columns:
            underlying_price = same_tenor.iloc[0]["underlying_price"]
        else:
            # Approximate from ATM strike
            atm_options = same_tenor[same_tenor["delta_bucket"] == "ATM"]
            if not atm_options.empty:
                underlying_price = atm_options.iloc[0]["strike"]
            else:
                underlying_price = same_tenor.iloc[0]["strike"]

        wing_width = underlying_price * self._wing_width_pct

        # Find 25-delta put (short put)
        put_25 = same_tenor[
            (same_tenor["delta_bucket"] == "P25")
            & (same_tenor["right"] == "P")
        ]
        if put_25.empty:
            raise ValueError("No 25-delta put found")
        put_25_row = put_25.iloc[0]

        # Find 25-delta call (short call)
        call_25 = same_tenor[
            (same_tenor["delta_bucket"] == "C25")
            & (same_tenor["right"] == "C")
        ]
        if call_25.empty:
            raise ValueError("No 25-delta call found")
        call_25_row = call_25.iloc[0]

        # Find wing put (further OTM, lower strike) for protection
        wing_put_target = put_25_row["strike"] - wing_width
        wing_puts = same_tenor[
            (same_tenor["right"] == "P")
            & (same_tenor["strike"] < put_25_row["strike"])
        ]
        if wing_puts.empty:
            raise ValueError("No wing put found below 25-delta put")
        # Get nearest to target
        wing_put_row = wing_puts.iloc[
            (wing_puts["strike"] - wing_put_target).abs().argsort()
        ].iloc[0]

        # Find wing call (further OTM, higher strike) for protection
        wing_call_target = call_25_row["strike"] + wing_width
        wing_calls = same_tenor[
            (same_tenor["right"] == "C")
            & (same_tenor["strike"] > call_25_row["strike"])
        ]
        if wing_calls.empty:
            raise ValueError("No wing call found above 25-delta call")
        # Get nearest to target
        wing_call_row = wing_calls.iloc[
            (wing_calls["strike"] - wing_call_target).abs().argsort()
        ].iloc[0]

        # Iron condor: sell both spreads to collect premium
        legs = [
            # Put credit spread (sell higher strike put, buy lower strike put)
            self._create_leg(put_25_row, qty=-1),      # Sell 25P
            self._create_leg(wing_put_row, qty=1),     # Buy wing put

            # Call credit spread (sell lower strike call, buy higher strike call)
            self._create_leg(call_25_row, qty=-1),     # Sell 25C
            self._create_leg(wing_call_row, qty=1),    # Buy wing call
        ]

        return legs

    def compute_max_loss(self, legs: list[OptionLeg]) -> float:
        """Compute iron condor max loss.

        Max loss occurs when price moves beyond either wing:
        - Below wing put: lose on put spread
        - Above wing call: lose on call spread

        Since only one side can lose at a time, max loss is:
        - Wider spread width - net credit received

        Args:
            legs: Iron condor legs (4 legs: put spread + call spread)

        Returns:
            Maximum possible loss as a positive number
        """
        net = self.net_premium(legs)

        # Separate puts and calls
        put_legs = [leg for leg in legs if leg.right == OptionRight.PUT]
        call_legs = [leg for leg in legs if leg.right == OptionRight.CALL]

        # Calculate spread widths
        if len(put_legs) >= 2:
            put_strikes = sorted([leg.strike for leg in put_legs])
            put_width = (put_strikes[-1] - put_strikes[0]) * CONTRACT_MULTIPLIER
        else:
            put_width = 0.0

        if len(call_legs) >= 2:
            call_strikes = sorted([leg.strike for leg in call_legs])
            call_width = (call_strikes[-1] - call_strikes[0]) * CONTRACT_MULTIPLIER
        else:
            call_width = 0.0

        # Max loss is the wider spread width (only one side can lose)
        max_spread_width = max(put_width, call_width)

        # Iron condor should always be a net credit
        if net > 0:
            # Normal case: credit received, max loss = width - credit
            return max_spread_width - net
        else:
            # Unusual: debit paid (shouldn't happen with proper strikes)
            # Max loss = width + debit
            return max_spread_width + abs(net)

    def _create_leg(self, row: pd.Series, qty: int) -> OptionLeg:
        """Create OptionLeg from surface DataFrame row.

        Uses ask price for buys, bid price for sells.

        Args:
            row: Row from surface DataFrame
            qty: Position quantity (+1 or -1)

        Returns:
            OptionLeg with data from the row
        """
        price = row["ask"] if qty > 0 else row["bid"]
        return OptionLeg(
            symbol=row["option_symbol"],
            qty=qty,
            entry_price=price,
            strike=row["strike"],
            expiry=row["expiry"],
            right=OptionRight(row["right"]),
            greeks=Greeks(
                delta=row["delta"],
                gamma=row["gamma"],
                vega=row["vega"],
                theta=row["theta"],
            ),
        )
