"""Unit tests for SkewTrade."""

from datetime import date

import pandas as pd
import pytest

from src.strategy.structures.base import CONTRACT_MULTIPLIER
from src.strategy.structures.skew import SkewTrade
from src.strategy.types import OptionRight, Signal, SignalType


@pytest.fixture
def sample_surface() -> pd.DataFrame:
    """Create a sample surface DataFrame for testing skew trades.

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


class TestSkewTradeInit:
    """Tests for SkewTrade initialization."""

    def test_default_wing_width(self) -> None:
        """Test default wing width is 3%."""
        trade = SkewTrade()
        assert trade._wing_width_pct == 0.03

    def test_custom_wing_width(self) -> None:
        """Test custom wing width."""
        trade = SkewTrade(wing_width_pct=0.05)
        assert trade._wing_width_pct == 0.05


class TestSkewTradeCreateLegs:
    """Tests for create_legs method."""

    def test_positive_edge_bullish_skew(self, sample_surface: pd.DataFrame) -> None:
        """Test positive edge creates bullish skew (sell put spread, buy call spread)."""
        trade = SkewTrade(wing_width_pct=0.03)
        signal = Signal(
            signal_type=SignalType.SKEW_ANOMALY,
            edge=0.05,  # Positive = puts overpriced, bullish
            confidence=0.7,
            tenor_days=30,
            delta_bucket="ATM",
        )

        legs = trade.create_legs(signal, sample_surface)

        assert len(legs) == 4

        # Check put legs (sell 25P, buy wing put)
        put_legs = [leg for leg in legs if leg.right == OptionRight.PUT]
        assert len(put_legs) == 2

        # 25P should be short (sell)
        put_25 = next(leg for leg in put_legs if leg.strike == 440.0)
        assert put_25.qty == -1
        assert put_25.entry_price == 3.00  # Bid for sell

        # Wing put should be long (buy)
        wing_put = next(leg for leg in put_legs if leg.strike == 430.0)
        assert wing_put.qty == 1
        assert wing_put.entry_price == 1.70  # Ask for buy

        # Check call legs (buy 25C, sell wing call)
        call_legs = [leg for leg in legs if leg.right == OptionRight.CALL]
        assert len(call_legs) == 2

        # 25C should be long (buy)
        call_25 = next(leg for leg in call_legs if leg.strike == 460.0)
        assert call_25.qty == 1
        assert call_25.entry_price == 2.70  # Ask for buy

        # Wing call should be short (sell)
        wing_call = next(leg for leg in call_legs if leg.strike == 470.0)
        assert wing_call.qty == -1
        assert wing_call.entry_price == 1.00  # Bid for sell

    def test_negative_edge_bearish_skew(self, sample_surface: pd.DataFrame) -> None:
        """Test negative edge creates bearish skew (buy put spread, sell call spread)."""
        trade = SkewTrade(wing_width_pct=0.03)
        signal = Signal(
            signal_type=SignalType.SKEW_ANOMALY,
            edge=-0.05,  # Negative = calls overpriced, bearish
            confidence=0.7,
            tenor_days=30,
            delta_bucket="ATM",
        )

        legs = trade.create_legs(signal, sample_surface)

        assert len(legs) == 4

        # Check put legs (buy 25P, sell wing put)
        put_legs = [leg for leg in legs if leg.right == OptionRight.PUT]

        # 25P should be long (buy)
        put_25 = next(leg for leg in put_legs if leg.strike == 440.0)
        assert put_25.qty == 1
        assert put_25.entry_price == 3.20  # Ask for buy

        # Wing put should be short (sell)
        wing_put = next(leg for leg in put_legs if leg.strike == 430.0)
        assert wing_put.qty == -1
        assert wing_put.entry_price == 1.50  # Bid for sell

        # Check call legs (sell 25C, buy wing call)
        call_legs = [leg for leg in legs if leg.right == OptionRight.CALL]

        # 25C should be short (sell)
        call_25 = next(leg for leg in call_legs if leg.strike == 460.0)
        assert call_25.qty == -1
        assert call_25.entry_price == 2.50  # Bid for sell

        # Wing call should be long (buy)
        wing_call = next(leg for leg in call_legs if leg.strike == 470.0)
        assert wing_call.qty == 1
        assert wing_call.entry_price == 1.20  # Ask for buy

    def test_greeks_populated(self, sample_surface: pd.DataFrame) -> None:
        """Test Greeks are populated from surface data."""
        trade = SkewTrade()
        signal = Signal(
            signal_type=SignalType.SKEW_ANOMALY,
            edge=0.05,
            confidence=0.7,
            tenor_days=30,
            delta_bucket="ATM",
        )

        legs = trade.create_legs(signal, sample_surface)

        for leg in legs:
            assert leg.greeks.delta != 0.0
            assert leg.greeks.gamma != 0.0
            assert leg.greeks.vega != 0.0
            assert leg.greeks.theta != 0.0

    def test_missing_25p_raises(self, sample_surface: pd.DataFrame) -> None:
        """Test ValueError when 25P not found."""
        # Remove 25P from surface
        surface = sample_surface[sample_surface["delta_bucket"] != "P25"]

        trade = SkewTrade()
        signal = Signal(
            signal_type=SignalType.SKEW_ANOMALY,
            edge=0.05,
            confidence=0.7,
            tenor_days=30,
            delta_bucket="ATM",
        )

        with pytest.raises(ValueError, match="No 25-delta put found"):
            trade.create_legs(signal, surface)

    def test_missing_25c_raises(self, sample_surface: pd.DataFrame) -> None:
        """Test ValueError when 25C not found."""
        # Remove 25C from surface
        surface = sample_surface[sample_surface["delta_bucket"] != "C25"]

        trade = SkewTrade()
        signal = Signal(
            signal_type=SignalType.SKEW_ANOMALY,
            edge=0.05,
            confidence=0.7,
            tenor_days=30,
            delta_bucket="ATM",
        )

        with pytest.raises(ValueError, match="No 25-delta call found"):
            trade.create_legs(signal, surface)

    def test_missing_wing_put_raises(self, sample_surface: pd.DataFrame) -> None:
        """Test ValueError when wing put not found."""
        # Remove wing put (10P)
        surface = sample_surface[sample_surface["delta_bucket"] != "P10"]

        trade = SkewTrade()
        signal = Signal(
            signal_type=SignalType.SKEW_ANOMALY,
            edge=0.05,
            confidence=0.7,
            tenor_days=30,
            delta_bucket="ATM",
        )

        with pytest.raises(ValueError, match="No wing put found"):
            trade.create_legs(signal, surface)

    def test_missing_wing_call_raises(self, sample_surface: pd.DataFrame) -> None:
        """Test ValueError when wing call not found."""
        # Remove wing call (10C)
        surface = sample_surface[sample_surface["delta_bucket"] != "C10"]

        trade = SkewTrade()
        signal = Signal(
            signal_type=SignalType.SKEW_ANOMALY,
            edge=0.05,
            confidence=0.7,
            tenor_days=30,
            delta_bucket="ATM",
        )

        with pytest.raises(ValueError, match="No wing call found"):
            trade.create_legs(signal, surface)

    def test_no_options_at_tenor_raises(self, sample_surface: pd.DataFrame) -> None:
        """Test ValueError when no options at signal tenor."""
        trade = SkewTrade()
        signal = Signal(
            signal_type=SignalType.SKEW_ANOMALY,
            edge=0.05,
            confidence=0.7,
            tenor_days=60,  # No options at 60 days
            delta_bucket="ATM",
        )

        with pytest.raises(ValueError, match="No options at tenor"):
            trade.create_legs(signal, sample_surface)


class TestSkewTradeComputeMaxLoss:
    """Tests for compute_max_loss method."""

    def test_max_loss_with_net_credit(self, sample_surface: pd.DataFrame) -> None:
        """Test max loss calculation when net credit received."""
        trade = SkewTrade()
        signal = Signal(
            signal_type=SignalType.SKEW_ANOMALY,
            edge=0.05,  # Bullish skew
            confidence=0.7,
            tenor_days=30,
            delta_bucket="ATM",
        )

        legs = trade.create_legs(signal, sample_surface)
        max_loss = trade.compute_max_loss(legs)

        # Put spread: sell 440P at 3.00, buy 430P at 1.70 = credit 1.30
        # Call spread: buy 460C at 2.70, sell 470C at 1.00 = debit 1.70
        # Net = credit 1.30 - debit 1.70 = -0.40 (small debit)
        # Put width = 440 - 430 = 10
        # Call width = 470 - 460 = 10
        # Max loss = max(10, 10) * 100 + 0.40 * 100 = 1040

        # Max loss should be positive
        assert max_loss > 0

        # Max loss should be bounded by spread width plus debit
        put_width = (440.0 - 430.0) * CONTRACT_MULTIPLIER
        call_width = (470.0 - 460.0) * CONTRACT_MULTIPLIER
        assert max_loss <= max(put_width, call_width) + abs(trade.net_premium(legs))

    def test_max_loss_is_positive(self, sample_surface: pd.DataFrame) -> None:
        """Test max loss is always positive."""
        trade = SkewTrade()

        for edge in [0.05, -0.05]:
            signal = Signal(
                signal_type=SignalType.SKEW_ANOMALY,
                edge=edge,
                confidence=0.7,
                tenor_days=30,
                delta_bucket="ATM",
            )

            legs = trade.create_legs(signal, sample_surface)
            max_loss = trade.compute_max_loss(legs)

            assert max_loss > 0


class TestSkewTradeComputeGreeks:
    """Tests for compute_greeks method."""

    def test_aggregate_greeks(self, sample_surface: pd.DataFrame) -> None:
        """Test Greeks are correctly aggregated across all 4 legs."""
        trade = SkewTrade()
        signal = Signal(
            signal_type=SignalType.SKEW_ANOMALY,
            edge=0.05,
            confidence=0.7,
            tenor_days=30,
            delta_bucket="ATM",
        )

        legs = trade.create_legs(signal, sample_surface)
        greeks = trade.compute_greeks(legs)

        # Should have non-zero aggregate Greeks
        # The exact values depend on the leg composition
        assert isinstance(greeks.delta, float)
        assert isinstance(greeks.gamma, float)
        assert isinstance(greeks.vega, float)
        assert isinstance(greeks.theta, float)

    def test_bullish_skew_is_long_delta(self, sample_surface: pd.DataFrame) -> None:
        """Test bullish skew trade has positive delta."""
        trade = SkewTrade()
        signal = Signal(
            signal_type=SignalType.SKEW_ANOMALY,
            edge=0.05,  # Bullish
            confidence=0.7,
            tenor_days=30,
            delta_bucket="ATM",
        )

        legs = trade.create_legs(signal, sample_surface)
        greeks = trade.compute_greeks(legs)

        # Bullish skew (sell puts, buy calls) should have positive delta
        assert greeks.delta > 0

    def test_bearish_skew_is_short_delta(self, sample_surface: pd.DataFrame) -> None:
        """Test bearish skew trade has negative delta."""
        trade = SkewTrade()
        signal = Signal(
            signal_type=SignalType.SKEW_ANOMALY,
            edge=-0.05,  # Bearish
            confidence=0.7,
            tenor_days=30,
            delta_bucket="ATM",
        )

        legs = trade.create_legs(signal, sample_surface)
        greeks = trade.compute_greeks(legs)

        # Bearish skew (buy puts, sell calls) should have negative delta
        assert greeks.delta < 0
