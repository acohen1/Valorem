"""Unit tests for IronCondor."""

from datetime import date

import pandas as pd
import pytest

from src.strategy.structures.base import CONTRACT_MULTIPLIER
from src.strategy.structures.iron_condor import IronCondor
from src.strategy.types import OptionRight, Signal, SignalType


@pytest.fixture
def sample_surface() -> pd.DataFrame:
    """Create a sample surface DataFrame for testing iron condors.

    Includes 25P, ATM, 25C options and wing options for bounded structure.
    """
    return pd.DataFrame(
        [
            # 25-delta put (strike 440)
            {
                "option_symbol": "SPY240315P00440000",
                "tenor_days": 30,
                "delta_bucket": "P25",
                "strike": 440.0,
                "expiry": date(2024, 3, 15),
                "right": "P",
                "bid": 3.00,
                "ask": 3.20,
                "delta": -0.25,
                "gamma": 0.015,
                "vega": 0.12,
                "theta": -0.03,
                "underlying_price": 450.0,
            },
            # Wing put (strike 430, further OTM)
            {
                "option_symbol": "SPY240315P00430000",
                "tenor_days": 30,
                "delta_bucket": "P10",
                "strike": 430.0,
                "expiry": date(2024, 3, 15),
                "right": "P",
                "bid": 1.50,
                "ask": 1.70,
                "delta": -0.10,
                "gamma": 0.008,
                "vega": 0.08,
                "theta": -0.02,
                "underlying_price": 450.0,
            },
            # ATM option (strike 450)
            {
                "option_symbol": "SPY240315C00450000",
                "tenor_days": 30,
                "delta_bucket": "ATM",
                "strike": 450.0,
                "expiry": date(2024, 3, 15),
                "right": "C",
                "bid": 5.00,
                "ask": 5.20,
                "delta": 0.50,
                "gamma": 0.020,
                "vega": 0.15,
                "theta": -0.05,
                "underlying_price": 450.0,
            },
            # 25-delta call (strike 460)
            {
                "option_symbol": "SPY240315C00460000",
                "tenor_days": 30,
                "delta_bucket": "C25",
                "strike": 460.0,
                "expiry": date(2024, 3, 15),
                "right": "C",
                "bid": 2.50,
                "ask": 2.70,
                "delta": 0.25,
                "gamma": 0.015,
                "vega": 0.12,
                "theta": -0.03,
                "underlying_price": 450.0,
            },
            # Wing call (strike 470, further OTM)
            {
                "option_symbol": "SPY240315C00470000",
                "tenor_days": 30,
                "delta_bucket": "C10",
                "strike": 470.0,
                "expiry": date(2024, 3, 15),
                "right": "C",
                "bid": 1.00,
                "ask": 1.20,
                "delta": 0.10,
                "gamma": 0.008,
                "vega": 0.08,
                "theta": -0.02,
                "underlying_price": 450.0,
            },
        ]
    )


class TestIronCondorInit:
    """Tests for IronCondor initialization."""

    def test_default_wing_width(self) -> None:
        """Test default wing width is 2%."""
        condor = IronCondor()
        assert condor._wing_width_pct == 0.02

    def test_custom_wing_width(self) -> None:
        """Test custom wing width."""
        condor = IronCondor(wing_width_pct=0.05)
        assert condor._wing_width_pct == 0.05


class TestIronCondorCreateLegs:
    """Tests for create_legs method."""

    def test_creates_four_legs(self, sample_surface: pd.DataFrame) -> None:
        """Test iron condor creates exactly 4 legs."""
        condor = IronCondor()
        signal = Signal(
            signal_type=SignalType.ELEVATED_IV,
            edge=0.05,
            confidence=0.3,  # Low confidence = neutral view
            tenor_days=30,
            delta_bucket="ATM",
        )

        legs = condor.create_legs(signal, sample_surface)

        assert len(legs) == 4

    def test_sell_both_spreads(self, sample_surface: pd.DataFrame) -> None:
        """Test iron condor sells both put and call spreads (credit)."""
        condor = IronCondor()
        signal = Signal(
            signal_type=SignalType.ELEVATED_IV,
            edge=0.05,
            confidence=0.3,
            tenor_days=30,
            delta_bucket="ATM",
        )

        legs = condor.create_legs(signal, sample_surface)

        # Put spread: sell 25P, buy wing put
        put_legs = [leg for leg in legs if leg.right == OptionRight.PUT]
        assert len(put_legs) == 2

        # 25P should be short (sell)
        put_25 = next(leg for leg in put_legs if leg.strike == 440.0)
        assert put_25.qty == -1
        assert put_25.entry_price == 3.00  # Bid for sell

        # Wing put should be long (buy for protection)
        wing_put = next(leg for leg in put_legs if leg.strike == 430.0)
        assert wing_put.qty == 1
        assert wing_put.entry_price == 1.70  # Ask for buy

        # Call spread: sell 25C, buy wing call
        call_legs = [leg for leg in legs if leg.right == OptionRight.CALL]
        assert len(call_legs) == 2

        # 25C should be short (sell)
        call_25 = next(leg for leg in call_legs if leg.strike == 460.0)
        assert call_25.qty == -1
        assert call_25.entry_price == 2.50  # Bid for sell

        # Wing call should be long (buy for protection)
        wing_call = next(leg for leg in call_legs if leg.strike == 470.0)
        assert wing_call.qty == 1
        assert wing_call.entry_price == 1.20  # Ask for buy

    def test_is_net_credit(self, sample_surface: pd.DataFrame) -> None:
        """Test iron condor receives net credit."""
        condor = IronCondor()
        signal = Signal(
            signal_type=SignalType.ELEVATED_IV,
            edge=0.05,
            confidence=0.3,
            tenor_days=30,
            delta_bucket="ATM",
        )

        legs = condor.create_legs(signal, sample_surface)
        net = condor.net_premium(legs)

        # Should be a net credit (positive)
        # Put spread credit: sell 3.00, buy 1.70 = 1.30
        # Call spread credit: sell 2.50, buy 1.20 = 1.30
        # Total credit = 2.60 * 100 = 260
        assert net > 0

    def test_greeks_populated(self, sample_surface: pd.DataFrame) -> None:
        """Test Greeks are populated from surface data."""
        condor = IronCondor()
        signal = Signal(
            signal_type=SignalType.ELEVATED_IV,
            edge=0.05,
            confidence=0.3,
            tenor_days=30,
            delta_bucket="ATM",
        )

        legs = condor.create_legs(signal, sample_surface)

        for leg in legs:
            assert leg.greeks.delta != 0.0
            assert leg.greeks.gamma != 0.0
            assert leg.greeks.vega != 0.0
            assert leg.greeks.theta != 0.0

    def test_missing_25p_raises(self, sample_surface: pd.DataFrame) -> None:
        """Test ValueError when 25P not found."""
        surface = sample_surface[sample_surface["delta_bucket"] != "P25"]

        condor = IronCondor()
        signal = Signal(
            signal_type=SignalType.ELEVATED_IV,
            edge=0.05,
            confidence=0.3,
            tenor_days=30,
            delta_bucket="ATM",
        )

        with pytest.raises(ValueError, match="No 25-delta put found"):
            condor.create_legs(signal, surface)

    def test_missing_25c_raises(self, sample_surface: pd.DataFrame) -> None:
        """Test ValueError when 25C not found."""
        surface = sample_surface[sample_surface["delta_bucket"] != "C25"]

        condor = IronCondor()
        signal = Signal(
            signal_type=SignalType.ELEVATED_IV,
            edge=0.05,
            confidence=0.3,
            tenor_days=30,
            delta_bucket="ATM",
        )

        with pytest.raises(ValueError, match="No 25-delta call found"):
            condor.create_legs(signal, surface)

    def test_missing_wing_put_raises(self, sample_surface: pd.DataFrame) -> None:
        """Test ValueError when wing put not found."""
        surface = sample_surface[sample_surface["delta_bucket"] != "P10"]

        condor = IronCondor()
        signal = Signal(
            signal_type=SignalType.ELEVATED_IV,
            edge=0.05,
            confidence=0.3,
            tenor_days=30,
            delta_bucket="ATM",
        )

        with pytest.raises(ValueError, match="No wing put found"):
            condor.create_legs(signal, surface)

    def test_missing_wing_call_raises(self, sample_surface: pd.DataFrame) -> None:
        """Test ValueError when wing call not found."""
        surface = sample_surface[sample_surface["delta_bucket"] != "C10"]

        condor = IronCondor()
        signal = Signal(
            signal_type=SignalType.ELEVATED_IV,
            edge=0.05,
            confidence=0.3,
            tenor_days=30,
            delta_bucket="ATM",
        )

        with pytest.raises(ValueError, match="No wing call found"):
            condor.create_legs(signal, surface)

    def test_no_options_at_tenor_raises(self, sample_surface: pd.DataFrame) -> None:
        """Test ValueError when no options at signal tenor."""
        condor = IronCondor()
        signal = Signal(
            signal_type=SignalType.ELEVATED_IV,
            edge=0.05,
            confidence=0.3,
            tenor_days=60,  # No options at 60 days
            delta_bucket="ATM",
        )

        with pytest.raises(ValueError, match="No options at tenor"):
            condor.create_legs(signal, sample_surface)


class TestIronCondorComputeMaxLoss:
    """Tests for compute_max_loss method."""

    def test_max_loss_is_spread_width_minus_credit(
        self, sample_surface: pd.DataFrame
    ) -> None:
        """Test max loss = spread width - net credit."""
        condor = IronCondor()
        signal = Signal(
            signal_type=SignalType.ELEVATED_IV,
            edge=0.05,
            confidence=0.3,
            tenor_days=30,
            delta_bucket="ATM",
        )

        legs = condor.create_legs(signal, sample_surface)
        max_loss = condor.compute_max_loss(legs)
        net_credit = condor.net_premium(legs)

        # Put spread width = 440 - 430 = 10
        # Call spread width = 470 - 460 = 10
        # Max spread width = 10 * 100 = 1000
        # Max loss = 1000 - credit
        expected_max_loss = 10 * CONTRACT_MULTIPLIER - net_credit

        assert max_loss == pytest.approx(expected_max_loss)

    def test_max_loss_is_positive(self, sample_surface: pd.DataFrame) -> None:
        """Test max loss is always positive."""
        condor = IronCondor()
        signal = Signal(
            signal_type=SignalType.ELEVATED_IV,
            edge=0.05,
            confidence=0.3,
            tenor_days=30,
            delta_bucket="ATM",
        )

        legs = condor.create_legs(signal, sample_surface)
        max_loss = condor.compute_max_loss(legs)

        assert max_loss > 0

    def test_max_loss_less_than_spread_width(
        self, sample_surface: pd.DataFrame
    ) -> None:
        """Test max loss is less than spread width (due to credit received)."""
        condor = IronCondor()
        signal = Signal(
            signal_type=SignalType.ELEVATED_IV,
            edge=0.05,
            confidence=0.3,
            tenor_days=30,
            delta_bucket="ATM",
        )

        legs = condor.create_legs(signal, sample_surface)
        max_loss = condor.compute_max_loss(legs)

        # Max loss should be less than the wider spread width
        spread_width = 10 * CONTRACT_MULTIPLIER  # Both spreads are $10 wide
        assert max_loss < spread_width


class TestIronCondorComputeGreeks:
    """Tests for compute_greeks method."""

    def test_aggregate_greeks(self, sample_surface: pd.DataFrame) -> None:
        """Test Greeks are correctly aggregated across all 4 legs."""
        condor = IronCondor()
        signal = Signal(
            signal_type=SignalType.ELEVATED_IV,
            edge=0.05,
            confidence=0.3,
            tenor_days=30,
            delta_bucket="ATM",
        )

        legs = condor.create_legs(signal, sample_surface)
        greeks = condor.compute_greeks(legs)

        # Should have float Greeks
        assert isinstance(greeks.delta, float)
        assert isinstance(greeks.gamma, float)
        assert isinstance(greeks.vega, float)
        assert isinstance(greeks.theta, float)

    def test_iron_condor_is_short_vega(self, sample_surface: pd.DataFrame) -> None:
        """Test iron condor is short vega (profits from IV crush)."""
        condor = IronCondor()
        signal = Signal(
            signal_type=SignalType.ELEVATED_IV,
            edge=0.05,
            confidence=0.3,
            tenor_days=30,
            delta_bucket="ATM",
        )

        legs = condor.create_legs(signal, sample_surface)
        greeks = condor.compute_greeks(legs)

        # Iron condor (sell both spreads) should be short vega
        # Selling options = negative vega exposure
        assert greeks.vega < 0

    def test_iron_condor_is_short_gamma(self, sample_surface: pd.DataFrame) -> None:
        """Test iron condor is short gamma."""
        condor = IronCondor()
        signal = Signal(
            signal_type=SignalType.ELEVATED_IV,
            edge=0.05,
            confidence=0.3,
            tenor_days=30,
            delta_bucket="ATM",
        )

        legs = condor.create_legs(signal, sample_surface)
        greeks = condor.compute_greeks(legs)

        # Iron condor (sell both spreads) should be short gamma
        assert greeks.gamma < 0

    def test_iron_condor_collects_theta(self, sample_surface: pd.DataFrame) -> None:
        """Test iron condor has positive theta (collects time decay)."""
        condor = IronCondor()
        signal = Signal(
            signal_type=SignalType.ELEVATED_IV,
            edge=0.05,
            confidence=0.3,
            tenor_days=30,
            delta_bucket="ATM",
        )

        legs = condor.create_legs(signal, sample_surface)
        greeks = condor.compute_greeks(legs)

        # Iron condor (sell both spreads) should collect theta
        # Selling options = positive theta (time works in our favor)
        assert greeks.theta > 0

    def test_iron_condor_near_delta_neutral(
        self, sample_surface: pd.DataFrame
    ) -> None:
        """Test iron condor is approximately delta neutral."""
        condor = IronCondor()
        signal = Signal(
            signal_type=SignalType.ELEVATED_IV,
            edge=0.05,
            confidence=0.3,
            tenor_days=30,
            delta_bucket="ATM",
        )

        legs = condor.create_legs(signal, sample_surface)
        greeks = condor.compute_greeks(legs)

        # Iron condor should be roughly delta neutral
        # (short put spread has positive delta, short call spread has negative delta)
        # With symmetric strikes, they should roughly cancel
        assert abs(greeks.delta) < 10  # Within 10 delta of neutral
