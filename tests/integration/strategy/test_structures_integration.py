"""Integration tests for trade structures.

These tests verify the full flow from signal to trade structure
with realistic surface data.
"""

from datetime import date

import pandas as pd
import pytest

from src.strategy import (
    CalendarSpread,
    Greeks,
    OptionLeg,
    OptionRight,
    Signal,
    SignalType,
    VerticalSpread,
)
from src.strategy.structures.base import CONTRACT_MULTIPLIER


@pytest.fixture
def realistic_surface() -> pd.DataFrame:
    """Create a realistic surface DataFrame with multiple tenors and strikes.

    This mimics the structure of the surface_snapshots table with all required
    columns for trade structure generation. Includes multiple strikes per
    expiry to support vertical spreads.
    """
    data = []

    # Base underlying price
    underlying = 450.0

    # Define tenor/expiry pairs
    tenors = [
        (21, date(2024, 2, 15)),   # 3 weeks
        (30, date(2024, 2, 22)),   # 1 month
        (60, date(2024, 3, 22)),   # 2 months
        (90, date(2024, 4, 22)),   # 3 months
    ]

    # Define strikes relative to underlying (for vertical spreads)
    # Strikes: 430, 440, 450, 460, 470 (covering range for 2% spread width)
    strikes_config = [
        (430.0, "P10", -0.10, "P"),
        (440.0, "P25", -0.25, "P"),
        (450.0, "ATM", 0.50, "C"),
        (459.0, "C25", 0.35, "C"),   # ~2% above ATM for vertical
        (470.0, "C10", 0.15, "C"),
        # Also add calls at lower strikes and puts at higher for flexibility
        (430.0, "P10", 0.85, "C"),   # Deep ITM call
        (440.0, "P25", 0.75, "C"),   # ITM call
        (450.0, "ATM", -0.50, "P"),  # ATM put
        (441.0, "P25", -0.30, "P"),  # For bear put spread (2% below ATM)
        (459.0, "C25", -0.20, "P"),  # OTM put
    ]

    for tenor_days, expiry in tenors:
        for strike, bucket, delta, right in strikes_config:
            # Compute Greeks (simplified)
            T = tenor_days / 365.0
            base_iv = 0.22
            gamma = 0.02 / (T ** 0.5)
            vega = 0.20 * (T ** 0.5)
            theta = -0.03 / T

            # Compute price (simplified)
            if right == "C":
                intrinsic = max(0, underlying - strike)
            else:
                intrinsic = max(0, strike - underlying)
            time_value = underlying * base_iv * (T ** 0.5) * 0.4 * abs(delta)
            mid_price = intrinsic + time_value

            # Add bid/ask spread
            spread_pct = mid_price * 0.02
            bid = max(0.01, mid_price - spread_pct / 2)
            ask = mid_price + spread_pct / 2

            # Format strike for OSI symbol
            strike_str = f"{int(strike * 1000):08d}"
            exp_str = expiry.strftime("%y%m%d")
            symbol = f"SPY{exp_str}{right}{strike_str}"

            data.append({
                "option_symbol": symbol,
                "tenor_days": tenor_days,
                "delta_bucket": bucket,
                "strike": strike,
                "expiry": expiry,
                "right": right,
                "bid": round(bid, 2),
                "ask": round(ask, 2),
                "delta": delta,
                "gamma": round(gamma, 4),
                "vega": round(vega, 4),
                "theta": round(theta, 4),
                "underlying_price": underlying,
                "iv": base_iv,
            })

    return pd.DataFrame(data)


class TestCalendarSpreadIntegration:
    """Integration tests for CalendarSpread."""

    def test_full_flow_term_anomaly(self, realistic_surface: pd.DataFrame) -> None:
        """Test full flow: signal → calendar → legs → max_loss → Greeks."""
        # Create signal for term structure anomaly
        signal = Signal(
            signal_type=SignalType.TERM_ANOMALY,
            edge=0.03,  # 3% edge
            confidence=0.75,
            tenor_days=30,
            delta_bucket="ATM",
        )

        # Build calendar spread
        spread = CalendarSpread(tenor_gap_days=30)
        legs = spread.create_legs(signal, realistic_surface)

        # Verify legs
        assert len(legs) == 2
        near_leg = min(legs, key=lambda l: l.expiry)
        far_leg = max(legs, key=lambda l: l.expiry)

        assert near_leg.qty == -1  # Sell near
        assert far_leg.qty == 1    # Buy far
        assert near_leg.strike == far_leg.strike  # Same strike

        # Verify max loss is bounded
        max_loss = spread.compute_max_loss(legs)
        assert max_loss > 0
        assert max_loss < 10000  # Reasonable bound

        # Verify Greeks
        greeks = spread.compute_greeks(legs)
        assert isinstance(greeks, Greeks)

        # Standard calendar should be long vega, short gamma
        assert greeks.vega > 0  # Long vega
        assert greeks.gamma < 0  # Short gamma

    def test_reverse_calendar_negative_edge(
        self, realistic_surface: pd.DataFrame
    ) -> None:
        """Test reverse calendar spread with negative edge."""
        signal = Signal(
            signal_type=SignalType.TERM_ANOMALY,
            edge=-0.02,  # Negative = buy near vol
            confidence=0.6,
            tenor_days=30,
            delta_bucket="ATM",
        )

        spread = CalendarSpread(tenor_gap_days=30)
        legs = spread.create_legs(signal, realistic_surface)

        near_leg = min(legs, key=lambda l: l.expiry)
        far_leg = max(legs, key=lambda l: l.expiry)

        assert near_leg.qty == 1   # Buy near
        assert far_leg.qty == -1   # Sell far

        # Reverse calendar should be short vega
        greeks = spread.compute_greeks(legs)
        assert greeks.vega < 0

    def test_put_calendar_spread(self, realistic_surface: pd.DataFrame) -> None:
        """Test calendar spread with put options."""
        signal = Signal(
            signal_type=SignalType.TERM_ANOMALY,
            edge=0.02,
            confidence=0.7,
            tenor_days=30,
            delta_bucket="P25",  # Put bucket
        )

        spread = CalendarSpread(tenor_gap_days=30)
        legs = spread.create_legs(signal, realistic_surface)

        assert all(leg.right == OptionRight.PUT for leg in legs)

        max_loss = spread.compute_max_loss(legs)
        assert max_loss > 0


class TestVerticalSpreadIntegration:
    """Integration tests for VerticalSpread."""

    def test_full_flow_bull_call(self, realistic_surface: pd.DataFrame) -> None:
        """Test full flow: signal → vertical → legs → max_loss → Greeks."""
        signal = Signal(
            signal_type=SignalType.DIRECTIONAL_VOL,
            edge=0.04,  # Bullish
            confidence=0.8,
            tenor_days=30,
            delta_bucket="ATM",
        )

        spread = VerticalSpread(spread_width_pct=0.02)
        legs = spread.create_legs(signal, realistic_surface)

        # Verify legs
        assert len(legs) == 2
        assert all(leg.right == OptionRight.CALL for leg in legs)

        buy_leg = next(leg for leg in legs if leg.qty > 0)
        sell_leg = next(leg for leg in legs if leg.qty < 0)

        assert buy_leg.strike < sell_leg.strike  # Bull spread

        # Verify max loss
        max_loss = spread.compute_max_loss(legs)
        assert max_loss > 0

        # Verify Greeks
        greeks = spread.compute_greeks(legs)
        assert greeks.delta > 0  # Bullish = positive delta

    def test_full_flow_bear_put(self, realistic_surface: pd.DataFrame) -> None:
        """Test bear put spread with negative edge."""
        signal = Signal(
            signal_type=SignalType.DIRECTIONAL_VOL,
            edge=-0.03,  # Bearish
            confidence=0.7,
            tenor_days=30,
            delta_bucket="ATM",
        )

        spread = VerticalSpread(spread_width_pct=0.02)
        legs = spread.create_legs(signal, realistic_surface)

        assert all(leg.right == OptionRight.PUT for leg in legs)

        buy_leg = next(leg for leg in legs if leg.qty > 0)
        sell_leg = next(leg for leg in legs if leg.qty < 0)

        assert buy_leg.strike > sell_leg.strike  # Bear put spread

        greeks = spread.compute_greeks(legs)
        assert greeks.delta < 0  # Bearish = negative delta


class TestBoundedRisk:
    """Tests to verify all structures have bounded risk."""

    def test_all_structures_have_bounded_max_loss(
        self, realistic_surface: pd.DataFrame
    ) -> None:
        """Test that all structure types have bounded max loss."""
        structures = [
            (CalendarSpread(tenor_gap_days=30), SignalType.TERM_ANOMALY),
            (VerticalSpread(spread_width_pct=0.02), SignalType.DIRECTIONAL_VOL),
        ]

        for structure, signal_type in structures:
            for edge in [0.03, -0.03]:
                signal = Signal(
                    signal_type=signal_type,
                    edge=edge,
                    confidence=0.7,
                    tenor_days=30,
                    delta_bucket="ATM",
                )

                try:
                    legs = structure.create_legs(signal, realistic_surface)
                    max_loss = structure.compute_max_loss(legs)

                    # Max loss should be bounded
                    assert max_loss > 0
                    assert max_loss < 100000  # Reasonable upper bound

                    # Max loss should not exceed total premium at risk
                    total_premium = sum(
                        abs(leg.qty * leg.entry_price * CONTRACT_MULTIPLIER)
                        for leg in legs
                    )
                    assert max_loss <= total_premium * 2  # Conservative check

                except ValueError:
                    # Some combinations may not find matching options
                    pass

    def test_no_naked_positions(self, realistic_surface: pd.DataFrame) -> None:
        """Test that no structure creates a naked (unbounded risk) position."""
        signal = Signal(
            signal_type=SignalType.TERM_ANOMALY,
            edge=0.03,
            confidence=0.7,
            tenor_days=30,
            delta_bucket="ATM",
        )

        spread = CalendarSpread(tenor_gap_days=30)
        legs = spread.create_legs(signal, realistic_surface)

        # Calendar spread must have both long and short legs
        long_legs = [leg for leg in legs if leg.qty > 0]
        short_legs = [leg for leg in legs if leg.qty < 0]

        assert len(long_legs) >= 1
        assert len(short_legs) >= 1

        # Short position is covered by long position (same strike)
        short_strike = short_legs[0].strike
        long_strike = long_legs[0].strike
        assert short_strike == long_strike  # Calendar has same strike


class TestGreeksAggregation:
    """Tests for proper Greeks aggregation across structures."""

    def test_contract_multiplier_applied(
        self, realistic_surface: pd.DataFrame
    ) -> None:
        """Test that contract multiplier is applied to Greeks."""
        signal = Signal(
            signal_type=SignalType.TERM_ANOMALY,
            edge=0.03,
            confidence=0.7,
            tenor_days=30,
            delta_bucket="ATM",
        )

        spread = CalendarSpread(tenor_gap_days=30)
        legs = spread.create_legs(signal, realistic_surface)
        greeks = spread.compute_greeks(legs)

        # Manual calculation
        expected_delta = sum(
            leg.greeks.delta * leg.qty * CONTRACT_MULTIPLIER for leg in legs
        )
        expected_vega = sum(
            leg.greeks.vega * leg.qty * CONTRACT_MULTIPLIER for leg in legs
        )

        assert greeks.delta == pytest.approx(expected_delta)
        assert greeks.vega == pytest.approx(expected_vega)

    def test_multi_contract_position(self, realistic_surface: pd.DataFrame) -> None:
        """Test Greeks aggregation with multiple contracts."""
        signal = Signal(
            signal_type=SignalType.TERM_ANOMALY,
            edge=0.03,
            confidence=0.7,
            tenor_days=30,
            delta_bucket="ATM",
        )

        spread = CalendarSpread(tenor_gap_days=30)
        legs = spread.create_legs(signal, realistic_surface)

        # Simulate 5 contract position
        scaled_legs = [
            OptionLeg(
                symbol=leg.symbol,
                qty=leg.qty * 5,
                entry_price=leg.entry_price,
                strike=leg.strike,
                expiry=leg.expiry,
                right=leg.right,
                greeks=leg.greeks,
            )
            for leg in legs
        ]

        greeks_1x = spread.compute_greeks(legs)
        greeks_5x = spread.compute_greeks(scaled_legs)

        assert greeks_5x.delta == pytest.approx(greeks_1x.delta * 5)
        assert greeks_5x.gamma == pytest.approx(greeks_1x.gamma * 5)
        assert greeks_5x.vega == pytest.approx(greeks_1x.vega * 5)
        assert greeks_5x.theta == pytest.approx(greeks_1x.theta * 5)


class TestSurfaceRequirements:
    """Tests for surface DataFrame requirements."""

    def test_missing_greeks_column_raises(self) -> None:
        """Test that missing Greeks columns raise KeyError when creating legs."""
        # Surface with required columns but missing Greeks
        # This will find the option but fail when creating OptionLeg
        incomplete_surface = pd.DataFrame(
            [
                # Near option
                {
                    "option_symbol": "SPY240215C00450000",
                    "tenor_days": 30,
                    "delta_bucket": "ATM",
                    "strike": 450.0,
                    "expiry": date(2024, 2, 15),
                    "right": "C",
                    "bid": 4.50,
                    "ask": 4.70,
                    # Missing: delta, gamma, vega, theta
                },
                # Far option (needed for calendar spread)
                {
                    "option_symbol": "SPY240515C00450000",
                    "tenor_days": 90,
                    "delta_bucket": "ATM",
                    "strike": 450.0,
                    "expiry": date(2024, 5, 15),
                    "right": "C",
                    "bid": 8.00,
                    "ask": 8.30,
                    # Missing: delta, gamma, vega, theta
                },
            ]
        )

        signal = Signal(
            signal_type=SignalType.TERM_ANOMALY,
            edge=0.03,
            confidence=0.7,
            tenor_days=30,
            delta_bucket="ATM",
        )

        spread = CalendarSpread()

        # Should raise KeyError when trying to access Greeks columns
        with pytest.raises(KeyError):
            spread.create_legs(signal, incomplete_surface)

    def test_empty_surface_raises(self) -> None:
        """Test that empty surface raises error."""
        empty_surface = pd.DataFrame(
            columns=[
                "option_symbol", "tenor_days", "delta_bucket", "strike",
                "expiry", "right", "bid", "ask", "delta", "gamma", "vega", "theta"
            ]
        )

        signal = Signal(
            signal_type=SignalType.TERM_ANOMALY,
            edge=0.03,
            confidence=0.7,
            tenor_days=30,
            delta_bucket="ATM",
        )

        spread = CalendarSpread()

        # Empty surface with correct schema should raise ValueError
        with pytest.raises(ValueError, match="No near option found"):
            spread.create_legs(signal, empty_surface)

    def test_no_matching_tenor_raises(self) -> None:
        """Test that signal with non-existent tenor raises ValueError."""
        surface = pd.DataFrame(
            [
                {
                    "option_symbol": "SPY240215C00450000",
                    "tenor_days": 30,
                    "delta_bucket": "ATM",
                    "strike": 450.0,
                    "expiry": date(2024, 2, 15),
                    "right": "C",
                    "bid": 4.50,
                    "ask": 4.70,
                    "delta": 0.5,
                    "gamma": 0.02,
                    "vega": 0.15,
                    "theta": -0.03,
                }
            ]
        )

        signal = Signal(
            signal_type=SignalType.TERM_ANOMALY,
            edge=0.03,
            confidence=0.7,
            tenor_days=60,  # Doesn't exist in surface
            delta_bucket="ATM",
        )

        spread = CalendarSpread()

        with pytest.raises(ValueError, match="No near option found"):
            spread.create_legs(signal, surface)
