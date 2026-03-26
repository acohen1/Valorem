"""Unit tests for PositionTracker."""

from datetime import date, datetime, timezone

import pandas as pd
import pytest

from src.live.positions import PositionSnapshot, PositionTracker
from src.risk.portfolio import Portfolio
from src.strategy.types import Greeks, OptionLeg, OptionRight


@pytest.fixture
def sample_surface() -> pd.DataFrame:
    """Create sample options surface."""
    return pd.DataFrame(
        [
            {
                "option_symbol": "SPY240315C00450000",
                "strike": 450.0,
                "expiry": date(2024, 3, 15),
                "right": "C",
                "bid": 5.20,
                "ask": 5.30,
                "delta": 0.46,
                "gamma": 0.021,
                "vega": 0.31,
                "theta": -0.052,
            },
            {
                "option_symbol": "SPY240315C00455000",
                "strike": 455.0,
                "expiry": date(2024, 3, 15),
                "right": "C",
                "bid": 3.10,
                "ask": 3.18,
                "delta": 0.36,
                "gamma": 0.019,
                "vega": 0.26,
                "theta": -0.042,
            },
            {
                "option_symbol": "SPY240315P00445000",
                "strike": 445.0,
                "expiry": date(2024, 3, 15),
                "right": "P",
                "bid": 4.55,
                "ask": 4.65,
                "delta": -0.41,
                "gamma": 0.021,
                "vega": 0.29,
                "theta": -0.041,
            },
        ]
    )


@pytest.fixture
def sample_portfolio() -> Portfolio:
    """Create sample portfolio with positions."""
    legs = [
        OptionLeg(
            symbol="SPY240315C00450000",
            qty=1,
            entry_price=5.00,
            strike=450.0,
            expiry=date(2024, 3, 15),
            right=OptionRight.CALL,
            greeks=Greeks(delta=0.45, gamma=0.02, vega=0.30, theta=-0.05),
        ),
        OptionLeg(
            symbol="SPY240315C00455000",
            qty=-1,
            entry_price=3.00,
            strike=455.0,
            expiry=date(2024, 3, 15),
            right=OptionRight.CALL,
            greeks=Greeks(delta=0.35, gamma=0.018, vega=0.25, theta=-0.04),
        ),
    ]

    portfolio = Portfolio(daily_pnl=0.0, max_acceptable_loss=5000.0)
    return portfolio.add_position(
        legs=legs,
        position_id="test_pos_001",
        structure_type="VerticalSpread",
        max_loss=200.0,
    )


class TestPositionTrackerInit:
    """Tests for PositionTracker initialization."""

    def test_init_with_portfolio(self, sample_portfolio: Portfolio) -> None:
        """Test initialization with portfolio."""
        tracker = PositionTracker(sample_portfolio)
        assert tracker.portfolio == sample_portfolio
        assert tracker.last_update is None

    def test_init_without_portfolio(self) -> None:
        """Test initialization without portfolio creates empty one."""
        tracker = PositionTracker()
        assert tracker.portfolio is not None
        assert len(tracker.portfolio.positions) == 0


class TestPositionTrackerUpdatePositions:
    """Tests for PositionTracker.update_positions."""

    def test_update_positions_calculates_mtm(
        self,
        sample_portfolio: Portfolio,
        sample_surface: pd.DataFrame,
    ) -> None:
        """Test mark-to-market valuation is calculated."""
        tracker = PositionTracker(sample_portfolio)
        updated = tracker.update_positions(sample_surface)

        # Should have updated the portfolio
        assert tracker.last_update is not None
        assert len(updated.positions) == 1

        # Get snapshot
        snapshot = tracker.get_snapshot("test_pos_001")
        assert snapshot is not None

        # Entry: buy at 5.00, sell at 3.00 -> net debit 2.00 * 100 = 200
        # Current: sell at 5.20 bid, buy back at 3.18 ask -> net 2.02 * 100 = 202
        # P&L = 202 - 200 = 2.00
        # Actually: entry_value = 1*5.00*100 + (-1)*3.00*100 = 500 - 300 = 200
        # Current: long at bid (5.20), short at ask (3.18)
        # current_value = 1*5.20*100 + (-1)*3.18*100 = 520 - 318 = 202
        # P&L = 202 - 200 = 2.00
        assert snapshot.entry_value == pytest.approx(200.0, rel=0.01)
        assert snapshot.current_value == pytest.approx(202.0, rel=0.01)
        assert snapshot.unrealized_pnl == pytest.approx(2.0, rel=0.1)

    def test_update_positions_updates_greeks(
        self,
        sample_portfolio: Portfolio,
        sample_surface: pd.DataFrame,
    ) -> None:
        """Test that Greeks are updated from surface."""
        tracker = PositionTracker(sample_portfolio)
        tracker.update_positions(sample_surface)

        snapshot = tracker.get_snapshot("test_pos_001")
        assert snapshot is not None

        # Greeks should be updated from surface
        # Long leg: delta=0.46, Short leg: delta=0.36
        # Net delta = 0.46*1*100 + 0.36*(-1)*100 = 46 - 36 = 10
        expected_delta = (0.46 * 1 - 0.36 * 1) * 100
        assert snapshot.current_greeks.delta == pytest.approx(expected_delta, rel=0.01)

    def test_update_positions_handles_missing_quote(
        self,
        sample_portfolio: Portfolio,
    ) -> None:
        """Test handling of missing quotes - uses entry price."""
        # Empty surface
        empty_surface = pd.DataFrame(columns=["option_symbol", "bid", "ask"])

        tracker = PositionTracker(sample_portfolio)
        updated = tracker.update_positions(empty_surface)

        snapshot = tracker.get_snapshot("test_pos_001")
        assert snapshot is not None
        # Should use entry prices as fallback
        assert snapshot.current_value == snapshot.entry_value


class TestPositionTrackerSnapshots:
    """Tests for position snapshot management."""

    def test_get_position_snapshots(
        self,
        sample_surface: pd.DataFrame,
    ) -> None:
        """Test getting all position snapshots."""
        # Create portfolio with multiple positions
        portfolio = Portfolio()
        portfolio = portfolio.add_position(
            legs=[
                OptionLeg(
                    symbol="SPY240315C00450000",
                    qty=1,
                    entry_price=5.00,
                    strike=450.0,
                    expiry=date(2024, 3, 15),
                    right=OptionRight.CALL,
                    greeks=Greeks(delta=0.45, gamma=0.02, vega=0.30, theta=-0.05),
                )
            ],
            position_id="pos_1",
        )
        portfolio = portfolio.add_position(
            legs=[
                OptionLeg(
                    symbol="SPY240315C00455000",
                    qty=-1,
                    entry_price=3.00,
                    strike=455.0,
                    expiry=date(2024, 3, 15),
                    right=OptionRight.CALL,
                    greeks=Greeks(delta=0.35, gamma=0.018, vega=0.25, theta=-0.04),
                )
            ],
            position_id="pos_2",
        )

        tracker = PositionTracker(portfolio)
        tracker.update_positions(sample_surface)

        snapshots = tracker.get_position_snapshots()
        assert len(snapshots) == 2

    def test_get_total_unrealized_pnl(
        self,
        sample_portfolio: Portfolio,
        sample_surface: pd.DataFrame,
    ) -> None:
        """Test getting total unrealized P&L."""
        tracker = PositionTracker(sample_portfolio)
        tracker.update_positions(sample_surface)

        total = tracker.get_total_unrealized_pnl()
        assert isinstance(total, float)

    def test_get_portfolio_greeks(
        self,
        sample_portfolio: Portfolio,
        sample_surface: pd.DataFrame,
    ) -> None:
        """Test getting aggregated portfolio Greeks."""
        tracker = PositionTracker(sample_portfolio)
        tracker.update_positions(sample_surface)

        greeks = tracker.get_portfolio_greeks()
        assert isinstance(greeks, Greeks)

    def test_clear_snapshots(
        self,
        sample_portfolio: Portfolio,
        sample_surface: pd.DataFrame,
    ) -> None:
        """Test clearing snapshots."""
        tracker = PositionTracker(sample_portfolio)
        tracker.update_positions(sample_surface)

        assert len(tracker.get_position_snapshots()) == 1

        tracker.clear_snapshots()

        assert len(tracker.get_position_snapshots()) == 0
        assert tracker.last_update is None


class TestPositionSnapshot:
    """Tests for PositionSnapshot dataclass."""

    def test_snapshot_attributes(self) -> None:
        """Test snapshot has all required attributes."""
        snapshot = PositionSnapshot(
            position_id="test_001",
            entry_value=500.0,
            current_value=520.0,
            unrealized_pnl=20.0,
            current_greeks=Greeks(delta=10.0, gamma=0.5, vega=15.0, theta=-2.0),
            mark_prices={"SPY240315C00450000": 5.20},
        )

        assert snapshot.position_id == "test_001"
        assert snapshot.entry_value == 500.0
        assert snapshot.current_value == 520.0
        assert snapshot.unrealized_pnl == 20.0
        assert snapshot.current_greeks.delta == 10.0
        assert "SPY240315C00450000" in snapshot.mark_prices
        assert snapshot.timestamp is not None
