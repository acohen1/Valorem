"""Unit tests for strategy types."""

from datetime import UTC, date, datetime

import pytest

from src.strategy.types import (
    Greeks,
    OptionLeg,
    OptionRight,
    Signal,
    SignalType,
)


class TestGreeks:
    """Tests for Greeks dataclass."""

    def test_creation(self) -> None:
        """Test Greeks creation with all values."""
        greeks = Greeks(delta=0.5, gamma=0.01, vega=0.15, theta=-0.02)
        assert greeks.delta == 0.5
        assert greeks.gamma == 0.01
        assert greeks.vega == 0.15
        assert greeks.theta == -0.02

    def test_immutability(self) -> None:
        """Test that Greeks is immutable (frozen)."""
        greeks = Greeks(delta=0.5, gamma=0.01, vega=0.15, theta=-0.02)
        with pytest.raises(AttributeError):
            greeks.delta = 0.6  # type: ignore[misc]

    def test_scale_positive(self) -> None:
        """Test scaling Greeks by positive multiplier."""
        greeks = Greeks(delta=0.5, gamma=0.01, vega=0.15, theta=-0.02)
        scaled = greeks.scale(100)

        assert scaled.delta == 50.0
        assert scaled.gamma == 1.0
        assert scaled.vega == 15.0
        assert scaled.theta == -2.0

    def test_scale_negative(self) -> None:
        """Test scaling Greeks by negative multiplier (short position)."""
        greeks = Greeks(delta=0.5, gamma=0.01, vega=0.15, theta=-0.02)
        scaled = greeks.scale(-100)

        assert scaled.delta == -50.0
        assert scaled.gamma == -1.0
        assert scaled.vega == -15.0
        assert scaled.theta == 2.0

    def test_scale_zero(self) -> None:
        """Test scaling Greeks by zero."""
        greeks = Greeks(delta=0.5, gamma=0.01, vega=0.15, theta=-0.02)
        scaled = greeks.scale(0)

        assert scaled.delta == 0.0
        assert scaled.gamma == 0.0
        assert scaled.vega == 0.0
        assert scaled.theta == 0.0

    def test_add_two_greeks(self) -> None:
        """Test adding two Greeks together."""
        g1 = Greeks(delta=0.5, gamma=0.01, vega=0.15, theta=-0.02)
        g2 = Greeks(delta=-0.3, gamma=0.02, vega=0.10, theta=-0.01)
        result = g1 + g2

        assert result.delta == pytest.approx(0.2)
        assert result.gamma == pytest.approx(0.03)
        assert result.vega == pytest.approx(0.25)
        assert result.theta == pytest.approx(-0.03)

    def test_add_returns_new_instance(self) -> None:
        """Test that addition returns a new Greeks instance."""
        g1 = Greeks(delta=0.5, gamma=0.01, vega=0.15, theta=-0.02)
        g2 = Greeks(delta=0.3, gamma=0.01, vega=0.10, theta=-0.01)
        result = g1 + g2

        assert result is not g1
        assert result is not g2

    def test_scale_returns_new_instance(self) -> None:
        """Test that scale returns a new Greeks instance."""
        greeks = Greeks(delta=0.5, gamma=0.01, vega=0.15, theta=-0.02)
        scaled = greeks.scale(100)

        assert scaled is not greeks

    def test_chained_operations(self) -> None:
        """Test chaining scale and add operations."""
        g1 = Greeks(delta=0.5, gamma=0.01, vega=0.15, theta=-0.02)
        g2 = Greeks(delta=0.3, gamma=0.02, vega=0.10, theta=-0.01)

        # Simulate aggregating two scaled positions
        result = g1.scale(100) + g2.scale(-100)

        assert result.delta == pytest.approx(20.0)
        assert result.gamma == pytest.approx(-1.0)
        assert result.vega == pytest.approx(5.0)
        assert result.theta == pytest.approx(-1.0)


class TestOptionRight:
    """Tests for OptionRight enum."""

    def test_call_value(self) -> None:
        """Test CALL has value 'C'."""
        assert OptionRight.CALL.value == "C"

    def test_put_value(self) -> None:
        """Test PUT has value 'P'."""
        assert OptionRight.PUT.value == "P"

    def test_from_string(self) -> None:
        """Test creating from string value."""
        assert OptionRight("C") == OptionRight.CALL
        assert OptionRight("P") == OptionRight.PUT

    def test_is_string_enum(self) -> None:
        """Test that OptionRight inherits from str."""
        assert isinstance(OptionRight.CALL, str)
        assert OptionRight.CALL == "C"


class TestSignalType:
    """Tests for SignalType enum."""

    def test_all_signal_types(self) -> None:
        """Test all signal types exist and have expected values."""
        assert SignalType.TERM_ANOMALY.value == "term_anomaly"
        assert SignalType.DIRECTIONAL_VOL.value == "directional_vol"
        assert SignalType.SKEW_ANOMALY.value == "skew_anomaly"
        assert SignalType.ELEVATED_IV.value == "elevated_iv"

    def test_is_string_enum(self) -> None:
        """Test that SignalType inherits from str."""
        assert isinstance(SignalType.TERM_ANOMALY, str)


class TestOptionLeg:
    """Tests for OptionLeg dataclass."""

    @pytest.fixture
    def sample_greeks(self) -> Greeks:
        """Create sample Greeks for testing."""
        return Greeks(delta=0.5, gamma=0.01, vega=0.15, theta=-0.02)

    def test_creation(self, sample_greeks: Greeks) -> None:
        """Test OptionLeg creation with all values."""
        leg = OptionLeg(
            symbol="SPY240315C00450000",
            qty=1,
            entry_price=5.50,
            strike=450.0,
            expiry=date(2024, 3, 15),
            right=OptionRight.CALL,
            greeks=sample_greeks,
        )

        assert leg.symbol == "SPY240315C00450000"
        assert leg.qty == 1
        assert leg.entry_price == 5.50
        assert leg.strike == 450.0
        assert leg.expiry == date(2024, 3, 15)
        assert leg.right == OptionRight.CALL
        assert leg.greeks == sample_greeks

    def test_long_position(self, sample_greeks: Greeks) -> None:
        """Test long position has positive qty."""
        leg = OptionLeg(
            symbol="SPY240315C00450000",
            qty=1,
            entry_price=5.50,
            strike=450.0,
            expiry=date(2024, 3, 15),
            right=OptionRight.CALL,
            greeks=sample_greeks,
        )
        assert leg.qty > 0

    def test_short_position(self, sample_greeks: Greeks) -> None:
        """Test short position has negative qty."""
        leg = OptionLeg(
            symbol="SPY240315P00440000",
            qty=-1,
            entry_price=4.00,
            strike=440.0,
            expiry=date(2024, 3, 15),
            right=OptionRight.PUT,
            greeks=sample_greeks,
        )
        assert leg.qty < 0

    def test_put_option(self, sample_greeks: Greeks) -> None:
        """Test put option creation."""
        leg = OptionLeg(
            symbol="SPY240315P00440000",
            qty=1,
            entry_price=3.50,
            strike=440.0,
            expiry=date(2024, 3, 15),
            right=OptionRight.PUT,
            greeks=sample_greeks,
        )
        assert leg.right == OptionRight.PUT


class TestSignal:
    """Tests for Signal dataclass."""

    def test_creation_with_all_fields(self) -> None:
        """Test Signal creation with all fields."""
        ts = datetime(2024, 1, 15, 10, 30, 0)
        signal = Signal(
            signal_type=SignalType.TERM_ANOMALY,
            edge=0.05,
            confidence=0.8,
            tenor_days=30,
            delta_bucket="ATM",
            timestamp=ts,
        )

        assert signal.signal_type == SignalType.TERM_ANOMALY
        assert signal.edge == 0.05
        assert signal.confidence == 0.8
        assert signal.tenor_days == 30
        assert signal.delta_bucket == "ATM"
        assert signal.timestamp == ts

    def test_creation_with_default_timestamp(self) -> None:
        """Test Signal creation uses current time as default timestamp."""
        before = datetime.now(UTC)
        signal = Signal(
            signal_type=SignalType.DIRECTIONAL_VOL,
            edge=-0.03,
            confidence=0.6,
            tenor_days=60,
            delta_bucket="P25",
        )
        after = datetime.now(UTC)

        assert before <= signal.timestamp <= after

    def test_positive_edge(self) -> None:
        """Test positive edge signal (buy vol)."""
        signal = Signal(
            signal_type=SignalType.TERM_ANOMALY,
            edge=0.05,
            confidence=0.7,
            tenor_days=30,
            delta_bucket="ATM",
        )
        assert signal.edge > 0

    def test_negative_edge(self) -> None:
        """Test negative edge signal (sell vol)."""
        signal = Signal(
            signal_type=SignalType.ELEVATED_IV,
            edge=-0.04,
            confidence=0.5,
            tenor_days=21,
            delta_bucket="C25",
        )
        assert signal.edge < 0

    def test_different_signal_types(self) -> None:
        """Test signals with different types."""
        for signal_type in SignalType:
            signal = Signal(
                signal_type=signal_type,
                edge=0.01,
                confidence=0.5,
                tenor_days=30,
                delta_bucket="ATM",
            )
            assert signal.signal_type == signal_type

    def test_delta_bucket_values(self) -> None:
        """Test various delta bucket values."""
        buckets = ["P25", "P10", "ATM", "C10", "C25"]
        for bucket in buckets:
            signal = Signal(
                signal_type=SignalType.TERM_ANOMALY,
                edge=0.01,
                confidence=0.5,
                tenor_days=30,
                delta_bucket=bucket,
            )
            assert signal.delta_bucket == bucket
