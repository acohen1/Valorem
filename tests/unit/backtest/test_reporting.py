"""Unit tests for backtest reporting module."""

import tempfile
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from src.backtest.reporting import (
    BacktestReporter,
    ReportConfig,
    generate_comparison_report,
)
from src.backtest.results import (
    BacktestMetrics,
    BacktestResult,
    PortfolioSnapshot,
    TradeRecord,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_metrics() -> BacktestMetrics:
    """Create sample backtest metrics."""
    return BacktestMetrics(
        total_return_pct=15.5,
        annualized_return_pct=45.2,
        sharpe_ratio=1.8,
        sortino_ratio=2.5,
        calmar_ratio=1.2,
        max_drawdown_pct=12.5,
        max_drawdown_duration_days=15,
        avg_drawdown_pct=5.2,
        total_trades=25,
        winning_trades=15,
        losing_trades=10,
        win_rate=60.0,
        avg_win=500.0,
        avg_loss=300.0,
        profit_factor=2.5,
        avg_trade_pnl=180.0,
        max_win=1200.0,
        max_loss=-800.0,
        avg_holding_period_days=5.2,
        max_concurrent_positions=3,
        avg_concurrent_positions=1.5,
        avg_daily_pnl=150.0,
        daily_pnl_std=200.0,
        max_daily_gain=800.0,
        max_daily_loss=-500.0,
        var_95=-350.0,
        cvar_95=-450.0,
        total_fees=125.0,
        total_slippage=75.0,
        total_commission_pct=1.5,
    )


@pytest.fixture
def sample_portfolio_history() -> list[PortfolioSnapshot]:
    """Create sample portfolio history."""
    base_date = date(2024, 1, 1)
    snapshots = []
    for i in range(10):
        d = date(2024, 1, i + 1)
        snapshots.append(
            PortfolioSnapshot(
                timestamp=datetime(2024, 1, i + 1, 16, 0, tzinfo=timezone.utc),
                date=d,
                cash=100000.0 + i * 100,
                positions_value=5000.0 + i * 50,
                total_equity=105000.0 + i * 150,
                unrealized_pnl=i * 50,
                realized_pnl=i * 100,
                daily_pnl=150.0 if i > 0 else 0.0,
                net_delta=10.0 * (i % 3 - 1),
                net_gamma=0.5 * (i % 2),
                net_vega=100.0 * (1 - i % 2),
                net_theta=-5.0 * i,
                open_positions=i % 3,
                entries_today=1 if i % 3 == 0 else 0,
                exits_today=1 if i % 4 == 0 else 0,
            )
        )
    return snapshots


@pytest.fixture
def sample_trades() -> list[TradeRecord]:
    """Create sample trade records."""
    return [
        TradeRecord(
            trade_id="entry-1",
            timestamp=datetime(2024, 1, 2, 10, 0, tzinfo=timezone.utc),
            trade_type="ENTRY",
            position_id="pos-1",
            structure_type="CalendarSpread",
            legs=[{"symbol": "SPY240115C450", "qty": 1}],
            gross_premium=500.0,
            fees=2.60,
            slippage=5.0,
            net_premium=492.4,
            signal_type="term_anomaly",
            exit_reason=None,
        ),
        TradeRecord(
            trade_id="exit-1",
            timestamp=datetime(2024, 1, 5, 10, 0, tzinfo=timezone.utc),
            trade_type="EXIT",
            position_id="pos-1",
            structure_type="CalendarSpread",
            legs=[{"symbol": "SPY240115C450", "qty": -1}],
            gross_premium=-450.0,
            fees=2.60,
            slippage=0.0,
            net_premium=50.0,
            signal_type="time_decay",
            exit_reason="Time decay exit",
        ),
        TradeRecord(
            trade_id="entry-2",
            timestamp=datetime(2024, 1, 6, 10, 0, tzinfo=timezone.utc),
            trade_type="ENTRY",
            position_id="pos-2",
            structure_type="VerticalSpread",
            legs=[{"symbol": "SPY240120C455", "qty": 1}],
            gross_premium=800.0,
            fees=2.60,
            slippage=8.0,
            net_premium=789.4,
            signal_type="directional_vol",
            exit_reason=None,
        ),
        TradeRecord(
            trade_id="exit-2",
            timestamp=datetime(2024, 1, 9, 10, 0, tzinfo=timezone.utc),
            trade_type="EXIT",
            position_id="pos-2",
            structure_type="VerticalSpread",
            legs=[{"symbol": "SPY240120C455", "qty": -1}],
            gross_premium=-750.0,
            fees=2.60,
            slippage=0.0,
            net_premium=-50.0,
            signal_type="stop_loss",
            exit_reason="Stop loss triggered",
        ),
    ]


@pytest.fixture
def sample_result(
    sample_metrics: BacktestMetrics,
    sample_portfolio_history: list[PortfolioSnapshot],
    sample_trades: list[TradeRecord],
) -> BacktestResult:
    """Create sample backtest result."""
    return BacktestResult(
        run_id="bt-test123",
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 10),
        initial_capital=100000.0,
        final_equity=106350.0,
        run_time_seconds=1.5,
        metrics=sample_metrics,
        trades=sample_trades,
        portfolio_history=sample_portfolio_history,
        position_history=[],
    )


@pytest.fixture
def temp_output_dir():
    """Create temporary output directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# =============================================================================
# ReportConfig Tests
# =============================================================================


class TestReportConfig:
    """Tests for ReportConfig."""

    def test_default_values(self, temp_output_dir: Path):
        """Test default configuration values."""
        config = ReportConfig(output_dir=temp_output_dir)

        assert config.output_dir == temp_output_dir
        assert config.include_plots is True
        assert config.plot_format == "png"
        assert config.plot_dpi == 150
        assert config.figsize == (12, 8)

    def test_custom_values(self, temp_output_dir: Path):
        """Test custom configuration values."""
        config = ReportConfig(
            output_dir=temp_output_dir,
            include_plots=False,
            plot_format="pdf",
            plot_dpi=300,
            figsize=(10, 6),
        )

        assert config.include_plots is False
        assert config.plot_format == "pdf"
        assert config.plot_dpi == 300
        assert config.figsize == (10, 6)


# =============================================================================
# BacktestReporter Tests
# =============================================================================


class TestBacktestReporter:
    """Tests for BacktestReporter."""

    def test_init_creates_output_dir(self, temp_output_dir: Path):
        """Test that initialization creates output directory."""
        nested_dir = temp_output_dir / "nested" / "reports"
        config = ReportConfig(output_dir=nested_dir)

        reporter = BacktestReporter(config)

        assert nested_dir.exists()

    def test_export_trades(self, temp_output_dir: Path, sample_result: BacktestResult):
        """Test trades CSV export."""
        config = ReportConfig(output_dir=temp_output_dir, include_plots=False)
        reporter = BacktestReporter(config)

        path = reporter._export_trades(sample_result)

        assert path.exists()
        assert path.name == f"{sample_result.run_id}_trades.csv"

        # Verify content
        import pandas as pd

        df = pd.read_csv(path)
        assert len(df) == 4  # 2 entries + 2 exits

    def test_export_portfolio(self, temp_output_dir: Path, sample_result: BacktestResult):
        """Test portfolio CSV export."""
        config = ReportConfig(output_dir=temp_output_dir, include_plots=False)
        reporter = BacktestReporter(config)

        path = reporter._export_portfolio(sample_result)

        assert path.exists()
        assert path.name == f"{sample_result.run_id}_portfolio.csv"

        import pandas as pd

        df = pd.read_csv(path)
        assert len(df) == 10  # 10 days

    def test_export_greeks(self, temp_output_dir: Path, sample_result: BacktestResult):
        """Test Greeks CSV export."""
        config = ReportConfig(output_dir=temp_output_dir, include_plots=False)
        reporter = BacktestReporter(config)

        path = reporter._export_greeks(sample_result)

        assert path.exists()
        assert path.name == f"{sample_result.run_id}_greeks.csv"

        import pandas as pd

        df = pd.read_csv(path)
        assert "net_delta" in df.columns
        assert "net_gamma" in df.columns
        assert "net_vega" in df.columns
        assert "net_theta" in df.columns

    def test_export_summary(self, temp_output_dir: Path, sample_result: BacktestResult):
        """Test summary text export."""
        config = ReportConfig(output_dir=temp_output_dir, include_plots=False)
        reporter = BacktestReporter(config)

        path = reporter._export_summary(sample_result)

        assert path.exists()
        assert path.name == f"{sample_result.run_id}_summary.txt"

        content = path.read_text()
        assert "BACKTEST REPORT" in content
        assert sample_result.run_id in content
        assert "PERFORMANCE METRICS" in content
        assert "TRADE STATISTICS" in content

    def test_generate_detailed_summary(self, temp_output_dir: Path, sample_result: BacktestResult):
        """Test detailed summary generation."""
        config = ReportConfig(output_dir=temp_output_dir, include_plots=False)
        reporter = BacktestReporter(config)

        summary = reporter._generate_detailed_summary(sample_result)

        # Check key sections
        assert "OVERVIEW" in summary
        assert "PERFORMANCE METRICS" in summary
        assert "DRAWDOWN ANALYSIS" in summary
        assert "TRADE STATISTICS" in summary
        assert "POSITION METRICS" in summary
        assert "RISK METRICS" in summary
        assert "EXECUTION COSTS" in summary

        # Check values are present
        assert "15.50%" in summary  # total return
        assert "1.800" in summary  # sharpe ratio
        assert "25" in summary  # total trades

    def test_generate_report_without_plots(
        self, temp_output_dir: Path, sample_result: BacktestResult
    ):
        """Test full report generation without plots."""
        config = ReportConfig(output_dir=temp_output_dir, include_plots=False)
        reporter = BacktestReporter(config)

        outputs = reporter.generate_report(sample_result)

        assert "trades_csv" in outputs
        assert "portfolio_csv" in outputs
        assert "greeks_csv" in outputs
        assert "summary_txt" in outputs

        # Plots should not be included
        assert "equity_plot" not in outputs
        assert "drawdown_plot" not in outputs

    def test_generate_report_with_plots(
        self, temp_output_dir: Path, sample_result: BacktestResult
    ):
        """Test full report generation with plots."""
        pytest.importorskip("matplotlib")

        config = ReportConfig(output_dir=temp_output_dir, include_plots=True)
        reporter = BacktestReporter(config)

        outputs = reporter.generate_report(sample_result)

        # All outputs should be present
        assert "trades_csv" in outputs
        assert "portfolio_csv" in outputs
        assert "greeks_csv" in outputs
        assert "summary_txt" in outputs
        assert "equity_plot" in outputs
        assert "drawdown_plot" in outputs
        assert "greeks_plot" in outputs
        assert "trades_plot" in outputs

        # Verify plot files exist
        assert outputs["equity_plot"].exists()
        assert outputs["drawdown_plot"].exists()

    def test_plot_equity_curve(self, temp_output_dir: Path, sample_result: BacktestResult):
        """Test equity curve plot generation."""
        plt = pytest.importorskip("matplotlib.pyplot")

        config = ReportConfig(output_dir=temp_output_dir)
        reporter = BacktestReporter(config)

        path = reporter._plot_equity_curve(sample_result)

        assert path.exists()
        assert path.suffix == ".png"
        plt.close("all")

    def test_plot_drawdown(self, temp_output_dir: Path, sample_result: BacktestResult):
        """Test drawdown plot generation."""
        plt = pytest.importorskip("matplotlib.pyplot")

        config = ReportConfig(output_dir=temp_output_dir)
        reporter = BacktestReporter(config)

        path = reporter._plot_drawdown(sample_result)

        assert path.exists()
        assert path.suffix == ".png"
        plt.close("all")

    def test_plot_greeks(self, temp_output_dir: Path, sample_result: BacktestResult):
        """Test Greeks plot generation."""
        plt = pytest.importorskip("matplotlib.pyplot")

        config = ReportConfig(output_dir=temp_output_dir)
        reporter = BacktestReporter(config)

        path = reporter._plot_greeks(sample_result)

        assert path.exists()
        assert path.suffix == ".png"
        plt.close("all")

    def test_plot_trade_analysis(self, temp_output_dir: Path, sample_result: BacktestResult):
        """Test trade analysis plot generation."""
        plt = pytest.importorskip("matplotlib.pyplot")

        config = ReportConfig(output_dir=temp_output_dir)
        reporter = BacktestReporter(config)

        path = reporter._plot_trade_analysis(sample_result)

        assert path.exists()
        assert path.suffix == ".png"
        plt.close("all")

    def test_plot_format_pdf(self, temp_output_dir: Path, sample_result: BacktestResult):
        """Test plot generation with PDF format."""
        plt = pytest.importorskip("matplotlib.pyplot")

        config = ReportConfig(output_dir=temp_output_dir, plot_format="pdf")
        reporter = BacktestReporter(config)

        path = reporter._plot_equity_curve(sample_result)

        assert path.suffix == ".pdf"
        plt.close("all")

    def test_empty_portfolio_history_raises(self, temp_output_dir: Path):
        """Test that empty portfolio history raises error for plots."""
        plt = pytest.importorskip("matplotlib.pyplot")

        empty_result = BacktestResult(
            run_id="bt-empty",
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 10),
            initial_capital=100000.0,
            final_equity=100000.0,
            run_time_seconds=0.1,
            metrics=BacktestMetrics(
                total_return_pct=0.0,
                annualized_return_pct=0.0,
                sharpe_ratio=0.0,
                sortino_ratio=0.0,
                calmar_ratio=0.0,
                max_drawdown_pct=0.0,
                max_drawdown_duration_days=0,
                avg_drawdown_pct=0.0,
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate=0.0,
                avg_win=0.0,
                avg_loss=0.0,
                profit_factor=0.0,
                avg_trade_pnl=0.0,
                max_win=0.0,
                max_loss=0.0,
                avg_holding_period_days=0.0,
                max_concurrent_positions=0,
                avg_concurrent_positions=0.0,
                avg_daily_pnl=0.0,
                daily_pnl_std=0.0,
                max_daily_gain=0.0,
                max_daily_loss=0.0,
                var_95=0.0,
                cvar_95=0.0,
                total_fees=0.0,
                total_slippage=0.0,
                total_commission_pct=0.0,
            ),
            trades=[],
            portfolio_history=[],
            position_history=[],
        )

        config = ReportConfig(output_dir=temp_output_dir)
        reporter = BacktestReporter(config)

        with pytest.raises(ValueError, match="No portfolio data"):
            reporter._plot_equity_curve(empty_result)

        plt.close("all")


# =============================================================================
# Comparison Report Tests
# =============================================================================


class TestComparisonReport:
    """Tests for comparison report generation."""

    def test_generate_comparison_report(
        self, temp_output_dir: Path, sample_result: BacktestResult
    ):
        """Test comparison report generation."""
        # Create a second result with different metrics
        result2 = BacktestResult(
            run_id="bt-test456",
            start_date=date(2024, 2, 1),
            end_date=date(2024, 2, 10),
            initial_capital=100000.0,
            final_equity=108000.0,
            run_time_seconds=2.0,
            metrics=BacktestMetrics(
                total_return_pct=8.0,
                annualized_return_pct=25.0,
                sharpe_ratio=1.2,
                sortino_ratio=1.5,
                calmar_ratio=0.8,
                max_drawdown_pct=10.0,
                max_drawdown_duration_days=8,
                avg_drawdown_pct=4.0,
                total_trades=20,
                winning_trades=12,
                losing_trades=8,
                win_rate=60.0,
                avg_win=400.0,
                avg_loss=250.0,
                profit_factor=2.0,
                avg_trade_pnl=120.0,
                max_win=900.0,
                max_loss=-600.0,
                avg_holding_period_days=4.0,
                max_concurrent_positions=2,
                avg_concurrent_positions=1.2,
                avg_daily_pnl=100.0,
                daily_pnl_std=150.0,
                max_daily_gain=500.0,
                max_daily_loss=-300.0,
                var_95=-250.0,
                cvar_95=-350.0,
                total_fees=100.0,
                total_slippage=50.0,
                total_commission_pct=1.2,
            ),
            trades=[],
            portfolio_history=[],
            position_history=[],
        )

        path = generate_comparison_report([sample_result, result2], temp_output_dir)

        assert path.exists()
        assert path.name == "backtest_comparison.csv"

        import pandas as pd

        df = pd.read_csv(path)
        assert len(df) == 2
        assert "run_id" in df.columns
        assert "total_return_pct" in df.columns
        assert "sharpe_ratio" in df.columns

    def test_comparison_report_single_result(
        self, temp_output_dir: Path, sample_result: BacktestResult
    ):
        """Test comparison with single result."""
        path = generate_comparison_report([sample_result], temp_output_dir)

        import pandas as pd

        df = pd.read_csv(path)
        assert len(df) == 1

    def test_comparison_report_creates_dir(
        self, temp_output_dir: Path, sample_result: BacktestResult
    ):
        """Test that comparison report creates output directory."""
        nested_dir = temp_output_dir / "comparisons"

        path = generate_comparison_report([sample_result], nested_dir)

        assert nested_dir.exists()
        assert path.exists()
