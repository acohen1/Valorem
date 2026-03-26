"""Unit tests for calculation utilities."""

from datetime import date

import pytest

from src.config.constants import TradingConstants
from src.strategy.types import Greeks, OptionLeg, OptionRight
from src.utils.calculations import aggregate_greeks, compute_dte, min_dte


class TestAggregateGreeks:
    """Tests for aggregate_greeks function."""

    def test_single_long_leg(self):
        """Test aggregating a single long leg."""
        leg = OptionLeg(
            symbol="SPY240315C00450000",
            qty=1,
            entry_price=5.0,
            strike=450.0,
            expiry=date(2024, 3, 15),
            right=OptionRight.CALL,
            greeks=Greeks(delta=0.5, gamma=0.02, vega=0.3, theta=-0.05),
        )

        result = aggregate_greeks([leg])

        # qty (1) * CONTRACT_MULTIPLIER (100) * greek
        assert result.delta == pytest.approx(50.0)
        assert result.gamma == pytest.approx(2.0)
        assert result.vega == pytest.approx(30.0)
        assert result.theta == pytest.approx(-5.0)

    def test_single_short_leg(self):
        """Test aggregating a single short leg."""
        leg = OptionLeg(
            symbol="SPY240315P00440000",
            qty=-1,
            entry_price=3.0,
            strike=440.0,
            expiry=date(2024, 3, 15),
            right=OptionRight.PUT,
            greeks=Greeks(delta=-0.3, gamma=0.01, vega=0.2, theta=-0.03),
        )

        result = aggregate_greeks([leg])

        # qty (-1) * CONTRACT_MULTIPLIER (100) * greek
        assert result.delta == pytest.approx(30.0)  # -1 * -0.3 * 100 = 30
        assert result.gamma == pytest.approx(-1.0)  # -1 * 0.01 * 100 = -1
        assert result.vega == pytest.approx(-20.0)
        assert result.theta == pytest.approx(3.0)

    def test_spread_aggregation(self):
        """Test aggregating a vertical spread (long + short)."""
        legs = [
            OptionLeg(
                symbol="SPY240315C00450000",
                qty=1,  # Long
                entry_price=5.0,
                strike=450.0,
                expiry=date(2024, 3, 15),
                right=OptionRight.CALL,
                greeks=Greeks(delta=0.5, gamma=0.02, vega=0.3, theta=-0.05),
            ),
            OptionLeg(
                symbol="SPY240315C00460000",
                qty=-1,  # Short
                entry_price=2.0,
                strike=460.0,
                expiry=date(2024, 3, 15),
                right=OptionRight.CALL,
                greeks=Greeks(delta=0.3, gamma=0.015, vega=0.2, theta=-0.03),
            ),
        ]

        result = aggregate_greeks(legs)

        # Net delta: (1 * 0.5 - 1 * 0.3) * 100 = 20
        assert result.delta == pytest.approx(20.0)
        # Net gamma: (1 * 0.02 - 1 * 0.015) * 100 = 0.5
        assert result.gamma == pytest.approx(0.5)
        # Net vega: (1 * 0.3 - 1 * 0.2) * 100 = 10
        assert result.vega == pytest.approx(10.0)
        # Net theta: (1 * -0.05 - 1 * -0.03) * 100 = -2
        assert result.theta == pytest.approx(-2.0)

    def test_empty_legs(self):
        """Test aggregating empty list returns zero Greeks."""
        result = aggregate_greeks([])

        assert result.delta == 0.0
        assert result.gamma == 0.0
        assert result.vega == 0.0
        assert result.theta == 0.0

    def test_custom_multiplier(self):
        """Test using a custom multiplier."""
        leg = OptionLeg(
            symbol="SPY240315C00450000",
            qty=1,
            entry_price=5.0,
            strike=450.0,
            expiry=date(2024, 3, 15),
            right=OptionRight.CALL,
            greeks=Greeks(delta=0.5, gamma=0.02, vega=0.3, theta=-0.05),
        )

        result = aggregate_greeks([leg], multiplier=50)

        # qty (1) * multiplier (50) * greek
        assert result.delta == pytest.approx(25.0)
        assert result.gamma == pytest.approx(1.0)

    def test_uses_contract_multiplier_by_default(self):
        """Verify default multiplier matches TradingConstants."""
        assert TradingConstants.CONTRACT_MULTIPLIER == 100


class TestComputeDte:
    """Tests for compute_dte function."""

    def test_future_expiry(self):
        """Test DTE for future expiration."""
        expiry = date(2024, 3, 15)
        as_of = date(2024, 3, 10)

        result = compute_dte(expiry, as_of=as_of)

        assert result == 5

    def test_same_day(self):
        """Test DTE when expiry equals as_of date."""
        expiry = date(2024, 3, 15)
        as_of = date(2024, 3, 15)

        result = compute_dte(expiry, as_of=as_of)

        assert result == 0

    def test_expired(self):
        """Test DTE for expired option (negative)."""
        expiry = date(2024, 3, 10)
        as_of = date(2024, 3, 15)

        result = compute_dte(expiry, as_of=as_of)

        assert result == -5

    def test_default_as_of_is_today(self):
        """Test that default as_of is today."""
        # Using a far future date to ensure positive DTE
        expiry = date(2099, 12, 31)

        result = compute_dte(expiry)

        # Should be positive and large
        assert result > 0


class TestMinDte:
    """Tests for min_dte function."""

    def test_single_leg(self):
        """Test min_dte with single leg."""
        as_of = date(2024, 3, 10)
        legs = [
            OptionLeg(
                symbol="SPY240315C00450000",
                qty=1,
                entry_price=5.0,
                strike=450.0,
                expiry=date(2024, 3, 15),
                right=OptionRight.CALL,
                greeks=Greeks(delta=0.5, gamma=0.02, vega=0.3, theta=-0.05),
            ),
        ]

        result = min_dte(legs, as_of=as_of)

        assert result == 5

    def test_multiple_legs_different_expiries(self):
        """Test min_dte returns minimum across legs."""
        as_of = date(2024, 3, 10)
        legs = [
            OptionLeg(
                symbol="SPY240315C00450000",
                qty=1,
                entry_price=5.0,
                strike=450.0,
                expiry=date(2024, 3, 15),  # 5 DTE
                right=OptionRight.CALL,
                greeks=Greeks(delta=0.5, gamma=0.02, vega=0.3, theta=-0.05),
            ),
            OptionLeg(
                symbol="SPY240322C00460000",
                qty=-1,
                entry_price=2.0,
                strike=460.0,
                expiry=date(2024, 3, 22),  # 12 DTE
                right=OptionRight.CALL,
                greeks=Greeks(delta=0.3, gamma=0.015, vega=0.2, theta=-0.03),
            ),
        ]

        result = min_dte(legs, as_of=as_of)

        assert result == 5  # Minimum of 5 and 12

    def test_clamps_to_zero(self):
        """Test that negative DTE is clamped to 0."""
        as_of = date(2024, 3, 20)
        legs = [
            OptionLeg(
                symbol="SPY240315C00450000",
                qty=1,
                entry_price=5.0,
                strike=450.0,
                expiry=date(2024, 3, 15),  # -5 DTE (expired)
                right=OptionRight.CALL,
                greeks=Greeks(delta=0.5, gamma=0.02, vega=0.3, theta=-0.05),
            ),
        ]

        result = min_dte(legs, as_of=as_of)

        assert result == 0  # Clamped to 0

    def test_empty_legs_raises(self):
        """Test that empty legs raises ValueError."""
        with pytest.raises(ValueError, match="Cannot compute min_dte for empty legs"):
            min_dte([])

    def test_calendar_spread_near_expiry(self):
        """Test min_dte for calendar spread (same strike, different expiries)."""
        as_of = date(2024, 3, 10)
        legs = [
            OptionLeg(
                symbol="SPY240315C00450000",
                qty=-1,  # Short near-term
                entry_price=3.0,
                strike=450.0,
                expiry=date(2024, 3, 15),  # 5 DTE
                right=OptionRight.CALL,
                greeks=Greeks(delta=0.5, gamma=0.02, vega=0.3, theta=-0.08),
            ),
            OptionLeg(
                symbol="SPY240415C00450000",
                qty=1,  # Long far-term
                entry_price=6.0,
                strike=450.0,
                expiry=date(2024, 4, 15),  # 36 DTE
                right=OptionRight.CALL,
                greeks=Greeks(delta=0.5, gamma=0.015, vega=0.4, theta=-0.04),
            ),
        ]

        result = min_dte(legs, as_of=as_of)

        assert result == 5  # Near-term expiry
