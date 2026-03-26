"""Unit tests for backtest results and metrics."""

from datetime import date, datetime, timezone

import pandas as pd
import pytest

from src.backtest.results import (
    BacktestMetrics,
    BacktestResult,
    PortfolioSnapshot,
    PositionSnapshot,
    TradeRecord,
    calculate_metrics,
    _empty_metrics,
)


@pytest.fixture
def sample_trades():
    """Create sample trades for testing."""
    return [
        TradeRecord(
            trade_id="entry-001",
            timestamp=datetime(2024, 1, 2, 10, 0, 0, tzinfo=timezone.utc),
            trade_type="ENTRY",
            position_id="pos-001",
            structure_type="VerticalSpread",
            legs=[{"symbol": "SPY", "qty": 1}],
            gross_premium=-200.0,
            fees=1.30,
            slippage=0.50,
            net_premium=-201.80,
            signal_type="directional_vol",
        ),
        TradeRecord(
            trade_id="exit-001",
            timestamp=datetime(2024, 1, 10, 10, 0, 0, tzinfo=timezone.utc),
            trade_type="EXIT",
            position_id="pos-001",
            structure_type="VerticalSpread",
            legs=[{"symbol": "SPY", "qty": -1}],
            gross_premium=250.0,
            fees=1.30,
            slippage=0.50,
            net_premium=50.0,  # Realized P&L for exits
            signal_type="take_profit",
            exit_reason="Take profit",
        ),
        TradeRecord(
            trade_id="entry-002",
            timestamp=datetime(2024, 1, 5, 10, 0, 0, tzinfo=timezone.utc),
            trade_type="ENTRY",
            position_id="pos-002",
            structure_type="CalendarSpread",
            legs=[{"symbol": "SPY", "qty": 1}],
            gross_premium=-150.0,
            fees=1.30,
            slippage=0.50,
            net_premium=-151.80,
            signal_type="term_anomaly",
        ),
        TradeRecord(
            trade_id="exit-002",
            timestamp=datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc),
            trade_type="EXIT",
            position_id="pos-002",
            structure_type="CalendarSpread",
            legs=[{"symbol": "SPY", "qty": -1}],
            gross_premium=100.0,
            fees=1.30,
            slippage=0.50,
            net_premium=-50.0,  # Loss
            signal_type="stop_loss",
            exit_reason="Stop loss",
        ),
    ]


@pytest.fixture
def sample_portfolio_history():
    """Create sample portfolio history for testing."""
    snapshots = []
    base_equity = 100000.0
    daily_pnls = [0.0, 20.0, -10.0, 30.0, -25.0, 50.0]

    for i, pnl in enumerate(daily_pnls):
        base_equity += pnl
        snapshots.append(
            PortfolioSnapshot(
                timestamp=datetime(2024, 1, 2 + i, 16, 0, 0, tzinfo=timezone.utc),
                date=date(2024, 1, 2 + i),
                cash=base_equity - 500,  # Some in positions
                positions_value=500.0,
                total_equity=base_equity,
                unrealized_pnl=10.0,
                realized_pnl=0.0,
                daily_pnl=pnl,
                net_delta=50.0,
                net_gamma=5.0,
                net_vega=100.0,
                net_theta=-10.0,
                open_positions=1,
                entries_today=0,
                exits_today=0,
            )
        )

    return snapshots


class TestTradeRecord:
    """Tests for TradeRecord dataclass."""

    def test_entry_trade_creation(self):
        """Test creating an entry trade record."""
        trade = TradeRecord(
            trade_id="entry-001",
            timestamp=datetime.now(timezone.utc),
            trade_type="ENTRY",
            position_id="pos-001",
            structure_type="VerticalSpread",
            legs=[{"symbol": "SPY240315C00450000", "qty": 1}],
            gross_premium=-200.0,
            fees=1.30,
            slippage=0.50,
            net_premium=-201.80,
            signal_type="directional_vol",
        )

        assert trade.trade_type == "ENTRY"
        assert trade.exit_reason is None

    def test_exit_trade_creation(self):
        """Test creating an exit trade record."""
        trade = TradeRecord(
            trade_id="exit-001",
            timestamp=datetime.now(timezone.utc),
            trade_type="EXIT",
            position_id="pos-001",
            structure_type="VerticalSpread",
            legs=[{"symbol": "SPY240315C00450000", "qty": -1}],
            gross_premium=250.0,
            fees=1.30,
            slippage=0.50,
            net_premium=50.0,
            signal_type="take_profit",
            exit_reason="Take profit at 50%",
        )

        assert trade.trade_type == "EXIT"
        assert trade.exit_reason == "Take profit at 50%"


class TestPortfolioSnapshot:
    """Tests for PortfolioSnapshot dataclass."""

    def test_snapshot_creation(self):
        """Test creating a portfolio snapshot."""
        snapshot = PortfolioSnapshot(
            timestamp=datetime.now(timezone.utc),
            date=date(2024, 1, 15),
            cash=99000.0,
            positions_value=1000.0,
            total_equity=100000.0,
            unrealized_pnl=50.0,
            realized_pnl=100.0,
            daily_pnl=25.0,
            net_delta=50.0,
            net_gamma=5.0,
            net_vega=100.0,
            net_theta=-10.0,
            open_positions=2,
            entries_today=1,
            exits_today=0,
        )

        assert snapshot.total_equity == 100000.0
        assert snapshot.open_positions == 2


class TestPositionSnapshot:
    """Tests for PositionSnapshot dataclass."""

    def test_snapshot_creation(self):
        """Test creating a position snapshot."""
        snapshot = PositionSnapshot(
            timestamp=datetime.now(timezone.utc),
            position_id="pos-001",
            state="OPEN",
            structure_type="VerticalSpread",
            entry_price=-200.0,
            current_price=-180.0,
            unrealized_pnl=20.0,
            realized_pnl=0.0,
            delta=50.0,
            gamma=5.0,
            vega=100.0,
            theta=-10.0,
            days_held=5,
            days_to_expiry=25,
        )

        assert snapshot.unrealized_pnl == 20.0
        assert snapshot.days_held == 5


class TestCalculateMetrics:
    """Tests for calculate_metrics function."""

    def test_basic_metrics(self, sample_trades, sample_portfolio_history):
        """Test basic metrics calculation."""
        metrics = calculate_metrics(
            trades=sample_trades,
            portfolio_history=sample_portfolio_history,
            initial_capital=100000.0,
            trading_days=6,
        )

        assert metrics.total_trades == 2  # Only EXIT trades count
        assert metrics.winning_trades == 1
        assert metrics.losing_trades == 1
        assert metrics.win_rate == pytest.approx(50.0, abs=1.0)

    def test_profit_factor(self, sample_trades, sample_portfolio_history):
        """Test profit factor calculation."""
        metrics = calculate_metrics(
            trades=sample_trades,
            portfolio_history=sample_portfolio_history,
            initial_capital=100000.0,
            trading_days=6,
        )

        # Win: +50, Loss: -50
        # Profit factor = 50 / 50 = 1.0
        assert metrics.profit_factor == pytest.approx(1.0, abs=0.1)

    def test_returns_calculation(self, sample_portfolio_history):
        """Test return calculations."""
        # Final equity is 100065
        metrics = calculate_metrics(
            trades=[],
            portfolio_history=sample_portfolio_history,
            initial_capital=100000.0,
            trading_days=6,
        )

        # Total return = (100065 / 100000 - 1) * 100 = 0.065%
        assert metrics.total_return_pct == pytest.approx(0.065, abs=0.01)

    def test_sharpe_ratio(self, sample_portfolio_history):
        """Test Sharpe ratio calculation."""
        metrics = calculate_metrics(
            trades=[],
            portfolio_history=sample_portfolio_history,
            initial_capital=100000.0,
            trading_days=6,
        )

        # Should be calculated (may be small given the data)
        assert isinstance(metrics.sharpe_ratio, float)

    def test_drawdown_calculation(self, sample_portfolio_history):
        """Test drawdown calculation."""
        metrics = calculate_metrics(
            trades=[],
            portfolio_history=sample_portfolio_history,
            initial_capital=100000.0,
            trading_days=6,
        )

        assert metrics.max_drawdown_pct >= 0
        assert metrics.max_drawdown_duration_days >= 0

    def test_empty_trades(self, sample_portfolio_history):
        """Test metrics with no trades."""
        metrics = calculate_metrics(
            trades=[],
            portfolio_history=sample_portfolio_history,
            initial_capital=100000.0,
            trading_days=6,
        )

        assert metrics.total_trades == 0
        assert metrics.win_rate == 0.0

    def test_empty_portfolio_history(self):
        """Test metrics with no portfolio history."""
        metrics = calculate_metrics(
            trades=[],
            portfolio_history=[],
            initial_capital=100000.0,
            trading_days=0,
        )

        assert metrics.total_return_pct == 0.0
        assert metrics.sharpe_ratio == 0.0

    def test_fees_and_slippage_totals(self, sample_trades, sample_portfolio_history):
        """Test total fees and slippage calculation."""
        metrics = calculate_metrics(
            trades=sample_trades,
            portfolio_history=sample_portfolio_history,
            initial_capital=100000.0,
            trading_days=6,
        )

        # 4 trades * $1.30 fees = $5.20
        assert metrics.total_fees == pytest.approx(5.20, abs=0.1)

        # 4 trades * $0.50 slippage = $2.00
        assert metrics.total_slippage == pytest.approx(2.00, abs=0.1)


class TestEmptyMetrics:
    """Tests for _empty_metrics function."""

    def test_empty_metrics_zeroed(self):
        """Test that empty metrics returns zeros."""
        metrics = _empty_metrics()

        assert metrics.total_return_pct == 0.0
        assert metrics.sharpe_ratio == 0.0
        assert metrics.total_trades == 0
        assert metrics.win_rate == 0.0
        assert metrics.total_fees == 0.0


class TestBacktestResult:
    """Tests for BacktestResult dataclass."""

    def test_result_creation(self, sample_trades, sample_portfolio_history):
        """Test creating a backtest result."""
        metrics = calculate_metrics(
            trades=sample_trades,
            portfolio_history=sample_portfolio_history,
            initial_capital=100000.0,
            trading_days=6,
        )

        result = BacktestResult(
            run_id="bt-test",
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 31),
            initial_capital=100000.0,
            final_equity=100065.0,
            run_time_seconds=1.5,
            metrics=metrics,
            trades=sample_trades,
            portfolio_history=sample_portfolio_history,
        )

        assert result.run_id == "bt-test"
        assert result.final_equity == 100065.0

    def test_to_trades_df(self, sample_trades, sample_portfolio_history):
        """Test converting trades to DataFrame."""
        metrics = _empty_metrics()
        result = BacktestResult(
            run_id="bt-test",
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 31),
            initial_capital=100000.0,
            final_equity=100000.0,
            run_time_seconds=1.0,
            metrics=metrics,
            trades=sample_trades,
            portfolio_history=sample_portfolio_history,
        )

        df = result.to_trades_df()

        assert len(df) == 4
        assert "trade_id" in df.columns
        assert "trade_type" in df.columns

    def test_to_portfolio_df(self, sample_portfolio_history):
        """Test converting portfolio history to DataFrame."""
        metrics = _empty_metrics()
        result = BacktestResult(
            run_id="bt-test",
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 31),
            initial_capital=100000.0,
            final_equity=100000.0,
            run_time_seconds=1.0,
            metrics=metrics,
            trades=[],
            portfolio_history=sample_portfolio_history,
        )

        df = result.to_portfolio_df()

        assert len(df) == len(sample_portfolio_history)
        assert "total_equity" in df.columns
        assert "net_delta" in df.columns

    def test_to_greeks_df(self, sample_portfolio_history):
        """Test extracting Greeks timeseries."""
        metrics = _empty_metrics()
        result = BacktestResult(
            run_id="bt-test",
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 31),
            initial_capital=100000.0,
            final_equity=100000.0,
            run_time_seconds=1.0,
            metrics=metrics,
            trades=[],
            portfolio_history=sample_portfolio_history,
        )

        df = result.to_greeks_df()

        assert "net_delta" in df.columns
        assert "net_vega" in df.columns

    def test_to_pnl_df(self, sample_portfolio_history):
        """Test extracting P&L timeseries."""
        metrics = _empty_metrics()
        result = BacktestResult(
            run_id="bt-test",
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 31),
            initial_capital=100000.0,
            final_equity=100000.0,
            run_time_seconds=1.0,
            metrics=metrics,
            trades=[],
            portfolio_history=sample_portfolio_history,
        )

        df = result.to_pnl_df()

        assert "total_equity" in df.columns
        assert "daily_pnl" in df.columns

    def test_summary(self, sample_trades, sample_portfolio_history):
        """Test summary generation."""
        metrics = calculate_metrics(
            trades=sample_trades,
            portfolio_history=sample_portfolio_history,
            initial_capital=100000.0,
            trading_days=6,
        )

        result = BacktestResult(
            run_id="bt-test",
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 31),
            initial_capital=100000.0,
            final_equity=100065.0,
            run_time_seconds=1.5,
            metrics=metrics,
            trades=sample_trades,
            portfolio_history=sample_portfolio_history,
        )

        summary = result.summary()

        assert "bt-test" in summary
        assert "Total Return" in summary
        assert "Sharpe Ratio" in summary
        assert "Win Rate" in summary

    def test_empty_result(self):
        """Test empty result creation."""
        metrics = _empty_metrics()
        result = BacktestResult(
            run_id="bt-empty",
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 31),
            initial_capital=100000.0,
            final_equity=100000.0,
            run_time_seconds=0.1,
            metrics=metrics,
        )

        assert result.trades == []
        assert result.portfolio_history == []
        assert result.to_trades_df().empty
        assert result.to_portfolio_df().empty


class TestBacktestMetrics:
    """Tests for BacktestMetrics dataclass."""

    def test_metrics_creation(self):
        """Test creating metrics directly."""
        metrics = BacktestMetrics(
            total_return_pct=10.5,
            annualized_return_pct=42.0,
            sharpe_ratio=1.5,
            sortino_ratio=2.0,
            calmar_ratio=3.0,
            max_drawdown_pct=5.0,
            max_drawdown_duration_days=10,
            avg_drawdown_pct=2.0,
            total_trades=50,
            winning_trades=30,
            losing_trades=20,
            win_rate=60.0,
            avg_win=100.0,
            avg_loss=50.0,
            profit_factor=2.0,
            avg_trade_pnl=40.0,
            max_win=500.0,
            max_loss=-200.0,
            avg_holding_period_days=5.0,
            max_concurrent_positions=3,
            avg_concurrent_positions=1.5,
            avg_daily_pnl=50.0,
            daily_pnl_std=100.0,
            max_daily_gain=200.0,
            max_daily_loss=-150.0,
            var_95=-100.0,
            cvar_95=-125.0,
            total_fees=65.0,
            total_slippage=25.0,
            total_commission_pct=0.9,
        )

        assert metrics.win_rate == 60.0
        assert metrics.profit_factor == 2.0
        assert metrics.sharpe_ratio == 1.5
