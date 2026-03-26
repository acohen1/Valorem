"""Unit tests for TradeStructure base class."""

from datetime import date

import pandas as pd
import pytest

from src.strategy.structures.base import CONTRACT_MULTIPLIER, TradeStructure
from src.strategy.types import Greeks, OptionLeg, OptionRight, Signal, SignalType


class ConcreteStructure(TradeStructure):
    """Concrete implementation for testing base class methods."""

    def create_legs(
        self,
        signal: Signal,
        surface: pd.DataFrame,
    ) -> list[OptionLeg]:
        """Not used in base class tests."""
        return []

    def compute_max_loss(self, legs: list[OptionLeg]) -> float:
        """Not used in base class tests."""
        return 0.0


class TestContractMultiplier:
    """Tests for CONTRACT_MULTIPLIER constant."""

    def test_contract_multiplier_value(self) -> None:
        """Test contract multiplier is 100."""
        assert CONTRACT_MULTIPLIER == 100


class TestComputeGreeks:
    """Tests for compute_greeks method."""

    @pytest.fixture
    def structure(self) -> TradeStructure:
        """Create concrete structure for testing."""
        return ConcreteStructure()

    @pytest.fixture
    def long_call_leg(self) -> OptionLeg:
        """Create a long call leg."""
        return OptionLeg(
            symbol="SPY240315C00450000",
            qty=1,
            entry_price=5.00,
            strike=450.0,
            expiry=date(2024, 3, 15),
            right=OptionRight.CALL,
            greeks=Greeks(delta=0.5, gamma=0.02, vega=0.20, theta=-0.03),
        )

    @pytest.fixture
    def short_call_leg(self) -> OptionLeg:
        """Create a short call leg."""
        return OptionLeg(
            symbol="SPY240315C00460000",
            qty=-1,
            entry_price=3.00,
            strike=460.0,
            expiry=date(2024, 3, 15),
            right=OptionRight.CALL,
            greeks=Greeks(delta=0.3, gamma=0.015, vega=0.15, theta=-0.02),
        )

    def test_single_long_leg(
        self, structure: TradeStructure, long_call_leg: OptionLeg
    ) -> None:
        """Test Greeks computation for single long leg."""
        greeks = structure.compute_greeks([long_call_leg])

        # qty=1, multiplier=100
        assert greeks.delta == pytest.approx(50.0)
        assert greeks.gamma == pytest.approx(2.0)
        assert greeks.vega == pytest.approx(20.0)
        assert greeks.theta == pytest.approx(-3.0)

    def test_single_short_leg(
        self, structure: TradeStructure, short_call_leg: OptionLeg
    ) -> None:
        """Test Greeks computation for single short leg."""
        greeks = structure.compute_greeks([short_call_leg])

        # qty=-1, multiplier=100
        assert greeks.delta == pytest.approx(-30.0)
        assert greeks.gamma == pytest.approx(-1.5)
        assert greeks.vega == pytest.approx(-15.0)
        assert greeks.theta == pytest.approx(2.0)

    def test_two_leg_spread(
        self,
        structure: TradeStructure,
        long_call_leg: OptionLeg,
        short_call_leg: OptionLeg,
    ) -> None:
        """Test Greeks computation for two-leg spread."""
        greeks = structure.compute_greeks([long_call_leg, short_call_leg])

        # Long: delta=50, gamma=2, vega=20, theta=-3
        # Short: delta=-30, gamma=-1.5, vega=-15, theta=2
        assert greeks.delta == pytest.approx(20.0)
        assert greeks.gamma == pytest.approx(0.5)
        assert greeks.vega == pytest.approx(5.0)
        assert greeks.theta == pytest.approx(-1.0)

    def test_empty_legs_list(self, structure: TradeStructure) -> None:
        """Test Greeks computation for empty legs list."""
        greeks = structure.compute_greeks([])

        assert greeks.delta == 0.0
        assert greeks.gamma == 0.0
        assert greeks.vega == 0.0
        assert greeks.theta == 0.0

    def test_multiple_contracts(self, structure: TradeStructure) -> None:
        """Test Greeks computation with multiple contracts."""
        leg = OptionLeg(
            symbol="SPY240315C00450000",
            qty=3,  # 3 contracts
            entry_price=5.00,
            strike=450.0,
            expiry=date(2024, 3, 15),
            right=OptionRight.CALL,
            greeks=Greeks(delta=0.5, gamma=0.02, vega=0.20, theta=-0.03),
        )
        greeks = structure.compute_greeks([leg])

        # qty=3, multiplier=100 → 300
        assert greeks.delta == pytest.approx(150.0)
        assert greeks.gamma == pytest.approx(6.0)
        assert greeks.vega == pytest.approx(60.0)
        assert greeks.theta == pytest.approx(-9.0)


class TestNetPremium:
    """Tests for net_premium method."""

    @pytest.fixture
    def structure(self) -> TradeStructure:
        """Create concrete structure for testing."""
        return ConcreteStructure()

    def test_debit_spread(self, structure: TradeStructure) -> None:
        """Test net premium for debit spread (pay to open)."""
        legs = [
            OptionLeg(
                symbol="SPY240315C00450000",
                qty=1,  # Buy at ask
                entry_price=5.00,
                strike=450.0,
                expiry=date(2024, 3, 15),
                right=OptionRight.CALL,
                greeks=Greeks(delta=0.5, gamma=0.02, vega=0.20, theta=-0.03),
            ),
            OptionLeg(
                symbol="SPY240315C00460000",
                qty=-1,  # Sell at bid
                entry_price=3.00,
                strike=460.0,
                expiry=date(2024, 3, 15),
                right=OptionRight.CALL,
                greeks=Greeks(delta=0.3, gamma=0.015, vega=0.15, theta=-0.02),
            ),
        ]
        net = structure.net_premium(legs)

        # Buy for $5, sell for $3 → pay $2 * 100 = -$200
        assert net == pytest.approx(-200.0)

    def test_credit_spread(self, structure: TradeStructure) -> None:
        """Test net premium for credit spread (receive to open)."""
        legs = [
            OptionLeg(
                symbol="SPY240315C00450000",
                qty=-1,  # Sell at bid
                entry_price=5.00,
                strike=450.0,
                expiry=date(2024, 3, 15),
                right=OptionRight.CALL,
                greeks=Greeks(delta=0.5, gamma=0.02, vega=0.20, theta=-0.03),
            ),
            OptionLeg(
                symbol="SPY240315C00460000",
                qty=1,  # Buy at ask
                entry_price=3.00,
                strike=460.0,
                expiry=date(2024, 3, 15),
                right=OptionRight.CALL,
                greeks=Greeks(delta=0.3, gamma=0.015, vega=0.15, theta=-0.02),
            ),
        ]
        net = structure.net_premium(legs)

        # Sell for $5, buy for $3 → receive $2 * 100 = $200
        assert net == pytest.approx(200.0)

    def test_empty_legs(self, structure: TradeStructure) -> None:
        """Test net premium for empty legs list."""
        net = structure.net_premium([])
        assert net == 0.0


class TestIsDebitSpread:
    """Tests for is_debit_spread method."""

    @pytest.fixture
    def structure(self) -> TradeStructure:
        """Create concrete structure for testing."""
        return ConcreteStructure()

    def test_debit_spread_returns_true(self, structure: TradeStructure) -> None:
        """Test is_debit_spread returns True for debit spread."""
        legs = [
            OptionLeg(
                symbol="SPY240315C00450000",
                qty=1,
                entry_price=5.00,
                strike=450.0,
                expiry=date(2024, 3, 15),
                right=OptionRight.CALL,
                greeks=Greeks(delta=0.5, gamma=0.02, vega=0.20, theta=-0.03),
            ),
            OptionLeg(
                symbol="SPY240315C00460000",
                qty=-1,
                entry_price=3.00,
                strike=460.0,
                expiry=date(2024, 3, 15),
                right=OptionRight.CALL,
                greeks=Greeks(delta=0.3, gamma=0.015, vega=0.15, theta=-0.02),
            ),
        ]
        assert structure.is_debit_spread(legs) is True

    def test_credit_spread_returns_false(self, structure: TradeStructure) -> None:
        """Test is_debit_spread returns False for credit spread."""
        legs = [
            OptionLeg(
                symbol="SPY240315C00450000",
                qty=-1,
                entry_price=5.00,
                strike=450.0,
                expiry=date(2024, 3, 15),
                right=OptionRight.CALL,
                greeks=Greeks(delta=0.5, gamma=0.02, vega=0.20, theta=-0.03),
            ),
            OptionLeg(
                symbol="SPY240315C00460000",
                qty=1,
                entry_price=3.00,
                strike=460.0,
                expiry=date(2024, 3, 15),
                right=OptionRight.CALL,
                greeks=Greeks(delta=0.3, gamma=0.015, vega=0.15, theta=-0.02),
            ),
        ]
        assert structure.is_debit_spread(legs) is False

    def test_zero_net_returns_false(self, structure: TradeStructure) -> None:
        """Test is_debit_spread returns False for zero net premium."""
        legs = [
            OptionLeg(
                symbol="SPY240315C00450000",
                qty=1,
                entry_price=3.00,
                strike=450.0,
                expiry=date(2024, 3, 15),
                right=OptionRight.CALL,
                greeks=Greeks(delta=0.5, gamma=0.02, vega=0.20, theta=-0.03),
            ),
            OptionLeg(
                symbol="SPY240315C00460000",
                qty=-1,
                entry_price=3.00,
                strike=460.0,
                expiry=date(2024, 3, 15),
                right=OptionRight.CALL,
                greeks=Greeks(delta=0.3, gamma=0.015, vega=0.15, theta=-0.02),
            ),
        ]
        assert structure.is_debit_spread(legs) is False
