"""Calendar spread trade structure.

A calendar spread involves selling a near-term option and buying a
far-term option at the nearest available strike.
"""

import logging

import pandas as pd

from ..types import Greeks, OptionLeg, OptionRight, Signal
from .base import CONTRACT_MULTIPLIER, TradeStructure

logger = logging.getLogger(__name__)


class CalendarSpread(TradeStructure):
    """Calendar spread: sell near-term, buy far-term at nearest strike.

    Used for term structure anomalies where near-term IV is relatively
    higher than far-term IV.

    Structure:
    - Sell 1 near-term option
    - Buy 1 far-term option (nearest strike, same type)

    The far-tenor option is matched to the nearest available strike
    within a configurable tolerance of the near-tenor strike. In real
    market data, identical strikes across expirations are rare due to
    differing listed series and forward price shifts.

    Max loss:
    - Debit spread: max loss = net debit paid
    - Credit spread (rare): max loss = far leg premium

    Attributes:
        tenor_gap_days: Minimum days between near and far expiries
        strike_tolerance_pct: Maximum allowed strike deviation as a
            fraction of underlying price (default 1%)
    """

    def __init__(
        self,
        tenor_gap_days: int = 30,
        strike_tolerance_pct: float = 0.01,
    ) -> None:
        """Initialize calendar spread builder.

        Args:
            tenor_gap_days: Minimum days between near and far expiries
            strike_tolerance_pct: Maximum strike deviation as fraction of
                underlying price (e.g. 0.01 = 1%). Far-tenor strikes
                beyond this tolerance are rejected.
        """
        self._tenor_gap_days = tenor_gap_days
        self._strike_tolerance_pct = strike_tolerance_pct

    def create_legs(
        self,
        signal: Signal,
        surface: pd.DataFrame,
    ) -> list[OptionLeg]:
        """Create calendar spread legs.

        For positive edge (sell near vol, buy far vol):
        - Sell 1 near-term option
        - Buy 1 far-term option at nearest strike

        For negative edge (buy near vol, sell far vol):
        - Buy 1 near-term option
        - Sell 1 far-term option at nearest strike

        The far-tenor option is selected as the closest available strike
        to the near-tenor strike, within ``strike_tolerance_pct`` of the
        underlying price.

        Args:
            signal: Trading signal indicating term structure mispricing
            surface: Surface snapshot with option data

        Returns:
            List of two OptionLeg objects (near and far)

        Raises:
            ValueError: If near or far option cannot be found, or if the
                nearest far-tenor strike exceeds the tolerance
        """
        # Find near-tenor option at signal node
        near_options = surface[
            (surface["tenor_days"] == signal.tenor_days)
            & (surface["delta_bucket"] == signal.delta_bucket)
        ]
        if near_options.empty:
            raise ValueError(
                f"No near option found for {signal.tenor_days}d {signal.delta_bucket}"
            )

        near = near_options.iloc[0]
        near_strike = near["strike"]

        # Determine underlying price for tolerance calculation
        if "underlying_price" in near.index and pd.notna(near["underlying_price"]):
            underlying_price = near["underlying_price"]
        else:
            underlying_price = near_strike

        strike_tolerance = underlying_price * self._strike_tolerance_pct

        # Find far-tenor candidates (same delta bucket, later expiry)
        far_candidates = surface[
            (surface["tenor_days"] > signal.tenor_days + self._tenor_gap_days)
            & (surface["delta_bucket"] == signal.delta_bucket)
        ]

        if far_candidates.empty:
            raise ValueError(f"No far option found for {signal.delta_bucket}")

        # Select the nearest strike to the near-tenor strike,
        # preferring earlier tenors when multiple candidates tie
        far_candidates = far_candidates.copy()
        far_candidates["_strike_dist"] = (
            far_candidates["strike"] - near_strike
        ).abs()
        far_candidates = far_candidates.sort_values(
            ["_strike_dist", "tenor_days"]
        )
        far = far_candidates.iloc[0]

        # Enforce tolerance
        strike_deviation = abs(far["strike"] - near_strike)
        if strike_deviation > strike_tolerance:
            raise ValueError(
                f"No far option within strike tolerance for {signal.delta_bucket} "
                f"(nearest strike {far['strike']:.2f} deviates "
                f"{strike_deviation:.2f} from {near_strike:.2f}, "
                f"tolerance {strike_tolerance:.2f})"
            )

        if strike_deviation > 0:
            logger.debug(
                f"Calendar spread: near strike {near_strike:.2f}, "
                f"far strike {far['strike']:.2f} "
                f"(deviation {strike_deviation:.2f})"
            )

        # Direction: positive edge = sell near vol, buy far vol
        if signal.edge > 0:
            near_qty, far_qty = -1, 1  # Sell near, buy far (standard calendar)
        else:
            near_qty, far_qty = 1, -1  # Buy near, sell far (reverse calendar)

        return [
            self._create_leg(near, near_qty),
            self._create_leg(far, far_qty),
        ]

    def compute_max_loss(self, legs: list[OptionLeg]) -> float:
        """Compute calendar spread max loss.

        For debit spreads: max loss = debit paid
        For credit spreads: max loss = far leg premium (conservative estimate)

        Args:
            legs: Calendar spread legs (near and far)

        Returns:
            Maximum possible loss as a positive number
        """
        net = self.net_premium(legs)

        if net < 0:
            # Debit spread: max loss = debit paid
            return abs(net)
        else:
            # Credit calendar (rare): max loss is the far leg cost
            # This is a conservative estimate
            far_leg = max(legs, key=lambda leg: leg.expiry)
            return abs(far_leg.entry_price * CONTRACT_MULTIPLIER)

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
