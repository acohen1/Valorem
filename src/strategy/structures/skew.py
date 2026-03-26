"""Skew trade structure (Risk Reversal with bounded risk).

A skew trade exploits mispricing between OTM puts and OTM calls
by trading a risk reversal with protective wings.
"""

import pandas as pd

from ..types import Greeks, OptionLeg, OptionRight, Signal
from .base import CONTRACT_MULTIPLIER, TradeStructure


class SkewTrade(TradeStructure):
    """Risk reversal with wings: bounded-risk skew trade.

    Used for skew anomalies where OTM puts are mispriced relative to
    OTM calls (or vice versa).

    Structure (positive edge - puts overpriced, bullish skew bet):
    - Sell OTM put spread: sell 25-delta put, buy further OTM put
    - Buy OTM call spread: buy 25-delta call, sell further OTM call

    Structure (negative edge - calls overpriced, bearish skew bet):
    - Buy OTM put spread: buy 25-delta put, sell further OTM put
    - Sell OTM call spread: sell 25-delta call, buy further OTM call

    Max loss:
    - Limited to the net debit paid plus spread width on losing side

    Attributes:
        wing_width_pct: Width of protective wings as % of underlying
    """

    def __init__(self, wing_width_pct: float = 0.03) -> None:
        """Initialize skew trade builder.

        Args:
            wing_width_pct: Wing spread width as % of underlying (default 3%)
        """
        self._wing_width_pct = wing_width_pct

    def create_legs(
        self,
        signal: Signal,
        surface: pd.DataFrame,
    ) -> list[OptionLeg]:
        """Create skew trade legs (risk reversal with wings).

        For positive edge (puts overpriced, bullish view):
        - Sell 25P, buy wing put (put credit spread)
        - Buy 25C, sell wing call (call debit spread)

        For negative edge (calls overpriced, bearish view):
        - Buy 25P, sell wing put (put debit spread)
        - Sell 25C, buy wing call (call credit spread)

        Args:
            signal: Trading signal indicating skew mispricing
            surface: Surface snapshot with option data

        Returns:
            List of four OptionLeg objects (put spread + call spread)

        Raises:
            ValueError: If required options cannot be found
        """
        # Get underlying price for calculating wing strikes
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

        # Find 25-delta put
        put_25 = same_tenor[
            (same_tenor["delta_bucket"] == "P25")
            & (same_tenor["right"] == "P")
        ]
        if put_25.empty:
            raise ValueError("No 25-delta put found")
        put_25_row = put_25.iloc[0]

        # Find 25-delta call
        call_25 = same_tenor[
            (same_tenor["delta_bucket"] == "C25")
            & (same_tenor["right"] == "C")
        ]
        if call_25.empty:
            raise ValueError("No 25-delta call found")
        call_25_row = call_25.iloc[0]

        # Find wing put (further OTM, lower strike)
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

        # Find wing call (further OTM, higher strike)
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

        # Determine direction based on signal edge
        if signal.edge > 0:
            # Bullish skew: sell put spread, buy call spread
            # Puts overpriced -> sell puts, collect premium
            legs = [
                self._create_leg(put_25_row, qty=-1),      # Sell 25P
                self._create_leg(wing_put_row, qty=1),     # Buy wing put
                self._create_leg(call_25_row, qty=1),      # Buy 25C
                self._create_leg(wing_call_row, qty=-1),   # Sell wing call
            ]
        else:
            # Bearish skew: buy put spread, sell call spread
            # Calls overpriced -> sell calls, collect premium
            legs = [
                self._create_leg(put_25_row, qty=1),       # Buy 25P
                self._create_leg(wing_put_row, qty=-1),    # Sell wing put
                self._create_leg(call_25_row, qty=-1),     # Sell 25C
                self._create_leg(wing_call_row, qty=1),    # Buy wing call
            ]

        return legs

    def compute_max_loss(self, legs: list[OptionLeg]) -> float:
        """Compute skew trade max loss.

        Max loss occurs when price moves against the position:
        - For bullish skew: max loss if price drops below wing put
        - For bearish skew: max loss if price rises above wing call

        Max loss = wider spread width - net credit (if any)

        Args:
            legs: Skew trade legs (4 legs: put spread + call spread)

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

        # Max loss is the wider spread width minus any net credit
        max_spread_width = max(put_width, call_width)

        if net > 0:
            # Net credit: max loss = spread width - credit
            return max_spread_width - net
        else:
            # Net debit: max loss = spread width + debit
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
