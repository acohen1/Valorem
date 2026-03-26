"""Vertical spread trade structure.

A vertical spread involves buying and selling options at different
strikes but with the same expiration.
"""

import pandas as pd

from ..types import Greeks, OptionLeg, OptionRight, Signal
from .base import CONTRACT_MULTIPLIER, TradeStructure


class VerticalSpread(TradeStructure):
    """Vertical spread: buy and sell options at different strikes, same expiry.

    Used for directional volatility bets with defined risk.

    Structures:
    - Bull Call Spread: Buy lower strike call, sell higher strike call
    - Bear Put Spread: Buy higher strike put, sell lower strike put

    Max loss:
    - Debit spread: max loss = net debit paid
    - Credit spread: max loss = spread width - credit received

    Attributes:
        spread_width_pct: Strike width as percentage of underlying price
    """

    def __init__(self, spread_width_pct: float = 0.02) -> None:
        """Initialize vertical spread builder.

        Args:
            spread_width_pct: Strike width as % of underlying (default 2%)
        """
        self._spread_width_pct = spread_width_pct

    def create_legs(
        self,
        signal: Signal,
        surface: pd.DataFrame,
    ) -> list[OptionLeg]:
        """Create vertical spread legs.

        For positive edge (bullish vol):
        - Bull call spread: buy lower strike call, sell higher strike call

        For negative edge (bearish vol):
        - Bear put spread: buy higher strike put, sell lower strike put

        Args:
            signal: Trading signal indicating volatility direction
            surface: Surface snapshot with option data

        Returns:
            List of two OptionLeg objects (buy and sell)

        Raises:
            ValueError: If required strikes cannot be found
        """
        # Get options at signal node
        node_options = surface[
            (surface["tenor_days"] == signal.tenor_days)
            & (surface["delta_bucket"] == signal.delta_bucket)
        ]
        if node_options.empty:
            raise ValueError(
                f"No option at {signal.tenor_days}d {signal.delta_bucket}"
            )

        anchor = node_options.iloc[0]

        # Use underlying_price if available, otherwise approximate from strike
        if "underlying_price" in anchor.index:
            underlying_price = anchor["underlying_price"]
        else:
            underlying_price = anchor["strike"]

        spread_width = underlying_price * self._spread_width_pct

        # Determine spread direction based on signal
        if signal.edge > 0:
            # Bullish vol: buy call spread
            buy_strike = anchor["strike"]
            sell_strike = anchor["strike"] + spread_width
            option_right = OptionRight.CALL
        else:
            # Bearish vol: buy put spread
            buy_strike = anchor["strike"]
            sell_strike = anchor["strike"] - spread_width
            option_right = OptionRight.PUT

        # Find options at same expiry with correct type
        same_expiry = surface[
            (surface["tenor_days"] == signal.tenor_days)
            & (surface["right"] == option_right.value)
        ]

        # Find buy strike
        buy_options = same_expiry[same_expiry["strike"] == buy_strike]

        # Find sell strike - look for nearest strike in the correct direction
        # For call spread (positive edge): need strike > buy_strike
        # For put spread (negative edge): need strike < buy_strike
        if signal.edge > 0:
            # Call spread: find nearest strike above buy_strike
            sell_candidates = same_expiry[same_expiry["strike"] > buy_strike]
        else:
            # Put spread: find nearest strike below buy_strike
            sell_candidates = same_expiry[same_expiry["strike"] < buy_strike]

        if sell_candidates.empty:
            raise ValueError(
                f"No strikes {'above' if signal.edge > 0 else 'below'} {buy_strike}"
            )

        # Sort by distance from target sell strike and take the closest
        sell_options = sell_candidates.iloc[
            (sell_candidates["strike"] - sell_strike).abs().argsort()
        ]

        if buy_options.empty:
            raise ValueError(f"Cannot find buy strike {buy_strike}")
        if sell_options.empty:
            raise ValueError(f"Cannot find sell strike near {sell_strike}")

        buy_opt = buy_options.iloc[0]
        sell_opt = sell_options.iloc[0]

        return [
            self._create_leg(buy_opt, qty=1),
            self._create_leg(sell_opt, qty=-1),
        ]

    def compute_max_loss(self, legs: list[OptionLeg]) -> float:
        """Compute vertical spread max loss.

        For debit spreads: max loss = debit paid
        For credit spreads: max loss = spread width - credit received

        Args:
            legs: Vertical spread legs

        Returns:
            Maximum possible loss as a positive number
        """
        net = self.net_premium(legs)

        if net < 0:
            # Debit spread: max loss = debit paid
            return abs(net)
        else:
            # Credit spread: max loss = spread width - credit received
            strikes = sorted([leg.strike for leg in legs])
            spread_width = (strikes[1] - strikes[0]) * CONTRACT_MULTIPLIER
            return spread_width - net

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
