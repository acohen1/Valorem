"""Unit tests for portfolio module."""

from datetime import date, datetime

import pytest

from src.risk.portfolio import CONTRACT_MULTIPLIER, Portfolio, Position
from src.strategy.types import Greeks, OptionLeg, OptionRight


@pytest.fixture
def sample_greeks() -> Greeks:
    """Sample Greeks for testing."""
    return Greeks(delta=0.5, gamma=0.02, vega=0.15, theta=-0.03)


@pytest.fixture
def sample_leg(sample_greeks: Greeks) -> OptionLeg:
    """Sample option leg for testing."""
    return OptionLeg(
        symbol="SPY240315C00450000",
        qty=1,
        entry_price=5.00,
        strike=450.0,
        expiry=date(2024, 3, 15),
        right=OptionRight.CALL,
        greeks=sample_greeks,
    )


@pytest.fixture
def sample_position(sample_leg: OptionLeg) -> Position:
    """Sample position for testing."""
    return Position(legs=[sample_leg])


class TestPosition:
    """Tests for Position class."""

    def test_position_creation(self, sample_leg: OptionLeg) -> None:
        """Test basic position creation."""
        position = Position(legs=[sample_leg])
        assert len(position.legs) == 1
        assert position.realized_pnl == 0.0
        assert position.unrealized_pnl == 0.0

    def test_position_greeks(self, sample_leg: OptionLeg) -> None:
        """Test Greeks aggregation for position."""
        position = Position(legs=[sample_leg])
        greeks = position.greeks

        # Single leg with qty=1, no multiplier at position level
        assert greeks.delta == 0.5
        assert greeks.gamma == 0.02
        assert greeks.vega == 0.15
        assert greeks.theta == -0.03

    def test_position_greeks_multiple_legs(self, sample_greeks: Greeks) -> None:
        """Test Greeks aggregation with multiple legs."""
        leg1 = OptionLeg(
            symbol="SPY240315C00450000",
            qty=1,
            entry_price=5.00,
            strike=450.0,
            expiry=date(2024, 3, 15),
            right=OptionRight.CALL,
            greeks=sample_greeks,
        )
        leg2 = OptionLeg(
            symbol="SPY240315C00460000",
            qty=-1,  # Short
            entry_price=3.00,
            strike=460.0,
            expiry=date(2024, 3, 15),
            right=OptionRight.CALL,
            greeks=Greeks(delta=0.3, gamma=0.015, vega=0.10, theta=-0.02),
        )
        position = Position(legs=[leg1, leg2])
        greeks = position.greeks

        # Aggregation with qty: 1*0.5 + (-1)*0.3 = 0.2
        assert greeks.delta == pytest.approx(0.2)
        assert greeks.gamma == pytest.approx(0.005)  # 0.02 - 0.015
        assert greeks.vega == pytest.approx(0.05)    # 0.15 - 0.10
        assert greeks.theta == pytest.approx(-0.01) # -0.03 - (-0.02)

    def test_position_total_qty(self, sample_greeks: Greeks) -> None:
        """Test total contract count."""
        leg1 = OptionLeg(
            symbol="SPY240315C00450000",
            qty=2,
            entry_price=5.00,
            strike=450.0,
            expiry=date(2024, 3, 15),
            right=OptionRight.CALL,
            greeks=sample_greeks,
        )
        leg2 = OptionLeg(
            symbol="SPY240315C00460000",
            qty=-1,
            entry_price=3.00,
            strike=460.0,
            expiry=date(2024, 3, 15),
            right=OptionRight.CALL,
            greeks=sample_greeks,
        )
        position = Position(legs=[leg1, leg2])
        assert position.total_qty == 3  # |2| + |-1|

    def test_position_entry_time(self, sample_leg: OptionLeg) -> None:
        """Test entry time is set."""
        position = Position(legs=[sample_leg])
        assert isinstance(position.entry_time, datetime)


class TestPortfolio:
    """Tests for Portfolio class."""

    def test_empty_portfolio(self) -> None:
        """Test empty portfolio initialization."""
        portfolio = Portfolio()
        assert len(portfolio.positions) == 0
        assert portfolio.daily_pnl == 0.0
        assert portfolio.net_delta == 0.0
        assert portfolio.net_gamma == 0.0
        assert portfolio.net_vega == 0.0
        assert portfolio.net_theta == 0.0

    def test_portfolio_with_position(self, sample_position: Position) -> None:
        """Test portfolio with one position."""
        portfolio = Portfolio(positions=[sample_position])
        assert len(portfolio.positions) == 1

        # Greeks should be contract-adjusted (multiplied by 100)
        assert portfolio.net_delta == pytest.approx(0.5 * CONTRACT_MULTIPLIER)
        assert portfolio.net_gamma == pytest.approx(0.02 * CONTRACT_MULTIPLIER)
        assert portfolio.net_vega == pytest.approx(0.15 * CONTRACT_MULTIPLIER)
        assert portfolio.net_theta == pytest.approx(-0.03 * CONTRACT_MULTIPLIER)

    def test_portfolio_multiple_positions(self, sample_greeks: Greeks) -> None:
        """Test portfolio with multiple positions."""
        leg1 = OptionLeg(
            symbol="SPY240315C00450000",
            qty=1,
            entry_price=5.00,
            strike=450.0,
            expiry=date(2024, 3, 15),
            right=OptionRight.CALL,
            greeks=sample_greeks,
        )
        leg2 = OptionLeg(
            symbol="SPY240315P00440000",
            qty=2,
            entry_price=4.00,
            strike=440.0,
            expiry=date(2024, 3, 15),
            right=OptionRight.PUT,
            greeks=Greeks(delta=-0.4, gamma=0.03, vega=0.20, theta=-0.04),
        )

        pos1 = Position(legs=[leg1])
        pos2 = Position(legs=[leg2])
        portfolio = Portfolio(positions=[pos1, pos2])

        # Delta: (1*0.5 + 2*(-0.4)) * 100 = (0.5 - 0.8) * 100 = -30
        assert portfolio.net_delta == pytest.approx(-30.0)
        # Gamma: (1*0.02 + 2*0.03) * 100 = 0.08 * 100 = 8
        assert portfolio.net_gamma == pytest.approx(8.0)

    def test_add_trade(self, sample_leg: OptionLeg) -> None:
        """Test adding a trade creates new portfolio."""
        portfolio = Portfolio()
        new_portfolio = portfolio.add_trade([sample_leg])

        # Original should be unchanged
        assert len(portfolio.positions) == 0

        # New portfolio should have the trade
        assert len(new_portfolio.positions) == 1
        assert new_portfolio.net_delta == pytest.approx(50.0)  # 0.5 * 100

    def test_add_trade_preserves_daily_pnl(self, sample_leg: OptionLeg) -> None:
        """Test that add_trade preserves daily_pnl."""
        portfolio = Portfolio(daily_pnl=-100.0)
        new_portfolio = portfolio.add_trade([sample_leg])

        assert new_portfolio.daily_pnl == -100.0

    def test_get_greeks(self, sample_position: Position) -> None:
        """Test get_greeks returns aggregated Greeks."""
        portfolio = Portfolio(positions=[sample_position])
        greeks = portfolio.get_greeks()

        assert isinstance(greeks, Greeks)
        assert greeks.delta == pytest.approx(50.0)
        assert greeks.gamma == pytest.approx(2.0)
        assert greeks.vega == pytest.approx(15.0)
        assert greeks.theta == pytest.approx(-3.0)

    def test_total_unrealized_pnl(self, sample_leg: OptionLeg) -> None:
        """Test total unrealized P&L calculation."""
        pos1 = Position(legs=[sample_leg], unrealized_pnl=100.0)
        pos2 = Position(legs=[sample_leg], unrealized_pnl=-50.0)
        portfolio = Portfolio(positions=[pos1, pos2])

        assert portfolio.total_unrealized_pnl == 50.0

    def test_total_realized_pnl(self, sample_leg: OptionLeg) -> None:
        """Test total realized P&L calculation."""
        pos1 = Position(legs=[sample_leg], realized_pnl=200.0)
        pos2 = Position(legs=[sample_leg], realized_pnl=-75.0)
        portfolio = Portfolio(positions=[pos1, pos2])

        assert portfolio.total_realized_pnl == 125.0

    def test_total_contracts(self, sample_greeks: Greeks) -> None:
        """Test total contract count across positions."""
        leg1 = OptionLeg(
            symbol="SPY240315C00450000",
            qty=2,
            entry_price=5.00,
            strike=450.0,
            expiry=date(2024, 3, 15),
            right=OptionRight.CALL,
            greeks=sample_greeks,
        )
        leg2 = OptionLeg(
            symbol="SPY240315P00440000",
            qty=-3,
            entry_price=4.00,
            strike=440.0,
            expiry=date(2024, 3, 15),
            right=OptionRight.PUT,
            greeks=sample_greeks,
        )

        pos1 = Position(legs=[leg1])
        pos2 = Position(legs=[leg2])
        portfolio = Portfolio(positions=[pos1, pos2])

        assert portfolio.total_contracts == 5  # |2| + |-3|

    def test_reset_daily_pnl(self) -> None:
        """Test daily P&L reset."""
        portfolio = Portfolio(daily_pnl=-500.0)
        portfolio.reset_daily_pnl()
        assert portfolio.daily_pnl == 0.0

    def test_update_daily_pnl(self) -> None:
        """Test daily P&L update."""
        portfolio = Portfolio(daily_pnl=100.0)
        portfolio.update_daily_pnl(-150.0)
        assert portfolio.daily_pnl == -50.0

        portfolio.update_daily_pnl(200.0)
        assert portfolio.daily_pnl == 150.0


class TestContractMultiplier:
    """Tests for CONTRACT_MULTIPLIER constant."""

    def test_contract_multiplier_value(self) -> None:
        """Test CONTRACT_MULTIPLIER is 100."""
        assert CONTRACT_MULTIPLIER == 100
