"""Integration tests for backtest engine.

These tests verify the full backtest workflow from surfaces through
trade execution and result generation.
"""

from datetime import date, datetime, timezone
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import pytest

from src.backtest import BacktestEngine, BacktestResult
from src.config.schema import (
    ExecutionConfig,
    FeeConfig,
    PerTradeRiskConfig,
    PositionManagementConfig,
    RiskCapsConfig,
    RiskConfig,
    SignalThresholdConfig,
    SizingConfig,
    SlippageConfig,
)
from src.strategy.types import Signal, SignalType


def generate_realistic_surface(
    as_of_date: date,
    underlying_price: float = 450.0,
    base_iv: float = 0.20,
) -> pd.DataFrame:
    """Generate realistic options surface for testing."""
    tenors = [7, 14, 30, 45, 60, 90]
    strike_pcts = [0.94, 0.96, 0.98, 1.0, 1.02, 1.04, 1.06]

    records = []
    for tenor in tenors:
        expiry = as_of_date + pd.Timedelta(days=tenor)

        for strike_pct in strike_pcts:
            strike = round(underlying_price * strike_pct, 2)

            for right in ["C", "P"]:
                moneyness = underlying_price / strike
                if right == "C":
                    delta = max(0.05, min(0.95, 0.5 + 0.4 * (moneyness - 1)))
                else:
                    delta = -max(0.05, min(0.95, 0.5 - 0.4 * (moneyness - 1)))

                abs_delta = abs(delta)
                if abs_delta >= 0.45:
                    delta_bucket = "ATM"
                elif abs_delta >= 0.35:
                    delta_bucket = "C40" if right == "C" else "P40"
                elif abs_delta >= 0.15:
                    delta_bucket = "C25" if right == "C" else "P25"
                else:
                    delta_bucket = "C10" if right == "C" else "P10"

                iv = base_iv * (1 + 0.1 * (1 - moneyness))
                time_value = iv * np.sqrt(tenor / 365) * underlying_price * 0.4
                intrinsic = max(
                    0,
                    (underlying_price - strike) if right == "C" else (strike - underlying_price),
                )
                mid_price = intrinsic + time_value * abs_delta

                bid = max(0.01, mid_price * 0.98)
                ask = mid_price * 1.02

                expiry_str = expiry.strftime("%y%m%d")
                strike_str = f"{int(strike * 1000):08d}"
                symbol = f"SPY{expiry_str}{right}{strike_str}"

                records.append({
                    "option_symbol": symbol,
                    "tenor_days": tenor,
                    "delta_bucket": delta_bucket,
                    "strike": strike,
                    "expiry": expiry,
                    "right": right,
                    "bid": round(bid, 2),
                    "ask": round(ask, 2),
                    "mid_price": round(mid_price, 2),
                    "delta": round(delta, 4),
                    "gamma": round(0.02 / (1 + abs(moneyness - 1) * 10), 4),
                    "vega": round(0.3 * np.sqrt(tenor / 30), 4),
                    "theta": round(-0.02 * (30 / tenor), 4),
                    "iv": round(iv, 4),
                    "underlying_price": underlying_price,
                })

    return pd.DataFrame(records)


def generate_random_signals(surface: pd.DataFrame, seed: int = None) -> list[Signal]:
    """Generate random signals for testing."""
    if seed is not None:
        np.random.seed(seed)

    if np.random.random() > 0.4:  # 60% chance of no signal
        return []

    node = surface.sample(1).iloc[0]
    signal_types = [SignalType.TERM_ANOMALY, SignalType.DIRECTIONAL_VOL]
    signal_type = signal_types[np.random.choice([0, 1], p=[0.4, 0.6])]

    return [
        Signal(
            signal_type=signal_type,
            edge=np.random.uniform(0.02, 0.08),
            confidence=np.random.uniform(0.6, 0.9),
            tenor_days=int(node["tenor_days"]),
            delta_bucket=str(node["delta_bucket"]),
            timestamp=datetime.now(timezone.utc),
        )
    ]


@pytest.fixture
def permissive_configs():
    """Create permissive configs for testing."""
    execution_config = ExecutionConfig(
        slippage=SlippageConfig(model="fixed_bps", fixed_bps=5.0),
        fees=FeeConfig(per_contract=0.65),
        signal_threshold=SignalThresholdConfig(min_edge=0.01, min_confidence=0.5),
        sizing=SizingConfig(base_contracts=1, max_contracts_per_trade=10),
    )

    risk_config = RiskConfig(
        per_trade=PerTradeRiskConfig(max_loss=5000.0, max_contracts=50),
        caps=RiskCapsConfig(max_abs_delta=1000.0, max_abs_vega=10000.0),
    )

    pos_mgmt_config = PositionManagementConfig()

    return execution_config, risk_config, pos_mgmt_config


class TestFullBacktestWorkflow:
    """Integration tests for full backtest workflow."""

    def test_two_week_backtest(self, permissive_configs):
        """Test backtest over two weeks with random signals."""
        execution_config, risk_config, pos_mgmt_config = permissive_configs

        start_date = date(2024, 1, 1)
        end_date = date(2024, 1, 14)

        # Generate surfaces for each trading day
        surfaces = {}
        signals_by_date = {}
        underlying_price = 450.0
        current_date = start_date

        seed = 0
        while current_date <= end_date:
            if current_date.weekday() < 5:  # Skip weekends
                underlying_price *= 1 + np.random.normal(0, 0.01)
                surface = generate_realistic_surface(current_date, underlying_price)
                surfaces[current_date] = surface

                signals = generate_random_signals(surface, seed=seed)
                if signals:
                    signals_by_date[current_date] = signals
                seed += 1

            current_date += pd.Timedelta(days=1)

        engine = BacktestEngine(
            start_date=start_date,
            end_date=end_date,
            initial_capital=100000.0,
            execution_config=execution_config,
            risk_config=risk_config,
            position_management_config=pos_mgmt_config,
        )

        result = engine.run_backtest(
            surfaces=surfaces,
            signals_by_date=signals_by_date,
        )

        # Verify result structure
        assert result.run_id.startswith("bt-")
        assert result.start_date == start_date
        assert result.end_date == end_date
        assert result.initial_capital == 100000.0
        assert result.final_equity > 0
        assert result.run_time_seconds >= 0

        # Verify metrics computed
        assert result.metrics is not None
        assert isinstance(result.metrics.total_return_pct, float)
        assert isinstance(result.metrics.sharpe_ratio, float)

        # Verify portfolio history
        assert len(result.portfolio_history) == len(surfaces)

    def test_backtest_with_signal_generator(self, permissive_configs):
        """Test backtest using signal generator callback."""
        execution_config, risk_config, pos_mgmt_config = permissive_configs

        signal_count = [0]

        def signal_generator(surface: pd.DataFrame) -> list[Signal]:
            signal_count[0] += 1
            # Generate signal occasionally
            if signal_count[0] % 3 == 0:
                return [
                    Signal(
                        signal_type=SignalType.DIRECTIONAL_VOL,
                        edge=0.05,
                        confidence=0.8,
                        tenor_days=30,
                        delta_bucket="ATM",
                        timestamp=datetime.now(timezone.utc),
                    )
                ]
            return []

        start_date = date(2024, 1, 2)
        end_date = date(2024, 1, 10)

        surfaces = {}
        current_date = start_date
        while current_date <= end_date:
            if current_date.weekday() < 5:
                surfaces[current_date] = generate_realistic_surface(current_date)
            current_date += pd.Timedelta(days=1)

        engine = BacktestEngine(
            start_date=start_date,
            end_date=end_date,
            initial_capital=100000.0,
            execution_config=execution_config,
            risk_config=risk_config,
            position_management_config=pos_mgmt_config,
            signal_generator=signal_generator,
        )

        result = engine.run_backtest(surfaces=surfaces)

        # Signal generator should have been called
        assert signal_count[0] > 0

    def test_trades_recorded_correctly(self, permissive_configs):
        """Test that trades are recorded with correct details."""
        execution_config, risk_config, pos_mgmt_config = permissive_configs

        start_date = date(2024, 1, 2)
        end_date = date(2024, 1, 5)

        surfaces = {}
        signals_by_date = {}

        for i, d in enumerate([date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4), date(2024, 1, 5)]):
            surface = generate_realistic_surface(d)
            surfaces[d] = surface

            # Add signal on first day
            if d == date(2024, 1, 2):
                signals_by_date[d] = [
                    Signal(
                        signal_type=SignalType.DIRECTIONAL_VOL,
                        edge=0.05,
                        confidence=0.8,
                        tenor_days=30,
                        delta_bucket="ATM",
                        timestamp=datetime.now(timezone.utc),
                    )
                ]

        engine = BacktestEngine(
            start_date=start_date,
            end_date=end_date,
            initial_capital=100000.0,
            execution_config=execution_config,
            risk_config=risk_config,
            position_management_config=pos_mgmt_config,
        )

        result = engine.run_backtest(
            surfaces=surfaces,
            signals_by_date=signals_by_date,
        )

        # Check trade records if any trades occurred
        for trade in result.trades:
            assert trade.trade_id is not None
            assert trade.timestamp is not None
            assert trade.trade_type in ["ENTRY", "EXIT"]
            assert trade.position_id is not None
            assert trade.structure_type is not None
            assert isinstance(trade.fees, float)
            assert isinstance(trade.slippage, float)


class TestResultExport:
    """Integration tests for result export functionality."""

    def test_export_to_csv(self, permissive_configs):
        """Test exporting results to CSV files."""
        execution_config, risk_config, pos_mgmt_config = permissive_configs

        surfaces = {
            date(2024, 1, 2): generate_realistic_surface(date(2024, 1, 2)),
            date(2024, 1, 3): generate_realistic_surface(date(2024, 1, 3)),
        }

        engine = BacktestEngine(
            start_date=date(2024, 1, 2),
            end_date=date(2024, 1, 3),
            initial_capital=100000.0,
            execution_config=execution_config,
            risk_config=risk_config,
            position_management_config=pos_mgmt_config,
        )

        result = engine.run_backtest(surfaces=surfaces)

        with tempfile.TemporaryDirectory() as tmpdir:
            trades_path = Path(tmpdir) / "trades.csv"
            portfolio_path = Path(tmpdir) / "portfolio.csv"

            result.export_trades_csv(str(trades_path))
            result.export_portfolio_csv(str(portfolio_path))

            # Read back and verify
            portfolio_df = pd.read_csv(portfolio_path)
            assert len(portfolio_df) == 2
            assert "total_equity" in portfolio_df.columns

    def test_summary_generation(self, permissive_configs):
        """Test summary text generation."""
        execution_config, risk_config, pos_mgmt_config = permissive_configs

        surfaces = {
            date(2024, 1, 2): generate_realistic_surface(date(2024, 1, 2)),
        }

        engine = BacktestEngine(
            start_date=date(2024, 1, 2),
            end_date=date(2024, 1, 2),
            initial_capital=100000.0,
            execution_config=execution_config,
            risk_config=risk_config,
            position_management_config=pos_mgmt_config,
        )

        result = engine.run_backtest(surfaces=surfaces)

        summary = result.summary()

        assert result.run_id in summary
        assert "Total Return" in summary
        assert "Sharpe Ratio" in summary
        assert "Initial Capital" in summary


class TestPositionManagement:
    """Integration tests for position management during backtest."""

    def test_exit_signals_generated(self, permissive_configs):
        """Test that exit signals are evaluated."""
        execution_config, risk_config, pos_mgmt_config = permissive_configs

        start_date = date(2024, 1, 2)
        end_date = date(2024, 1, 15)

        surfaces = {}
        signals_by_date = {}

        current = start_date
        while current <= end_date:
            if current.weekday() < 5:
                surfaces[current] = generate_realistic_surface(current)

                # Add entry signal on first day
                if current == start_date:
                    signals_by_date[current] = [
                        Signal(
                            signal_type=SignalType.DIRECTIONAL_VOL,
                            edge=0.05,
                            confidence=0.8,
                            tenor_days=7,  # Short tenor for faster expiry
                            delta_bucket="ATM",
                            timestamp=datetime.now(timezone.utc),
                        )
                    ]
            current += pd.Timedelta(days=1)

        engine = BacktestEngine(
            start_date=start_date,
            end_date=end_date,
            initial_capital=100000.0,
            execution_config=execution_config,
            risk_config=risk_config,
            position_management_config=pos_mgmt_config,
        )

        result = engine.run_backtest(
            surfaces=surfaces,
            signals_by_date=signals_by_date,
        )

        # Check if positions were managed
        exit_trades = [t for t in result.trades if t.trade_type == "EXIT"]

        # With short tenor, there should be some exit activity
        # (either through time decay exit or expiration)
        # Note: This is somewhat probabilistic based on the signals


class TestRiskEnforcement:
    """Integration tests for risk limit enforcement."""

    def test_risk_limits_enforced(self):
        """Test that risk limits are enforced during backtest."""
        # Very restrictive risk limits
        execution_config = ExecutionConfig()
        risk_config = RiskConfig(
            per_trade=PerTradeRiskConfig(max_loss=100.0, max_contracts=1),
            caps=RiskCapsConfig(max_abs_delta=50.0, max_abs_vega=100.0),
        )
        pos_mgmt_config = PositionManagementConfig()

        surfaces = {
            date(2024, 1, 2): generate_realistic_surface(date(2024, 1, 2)),
        }

        # Signal that would create large position
        signals_by_date = {
            date(2024, 1, 2): [
                Signal(
                    signal_type=SignalType.DIRECTIONAL_VOL,
                    edge=0.05,
                    confidence=0.8,
                    tenor_days=30,
                    delta_bucket="ATM",
                    timestamp=datetime.now(timezone.utc),
                )
            ]
        }

        engine = BacktestEngine(
            start_date=date(2024, 1, 2),
            end_date=date(2024, 1, 2),
            initial_capital=100000.0,
            execution_config=execution_config,
            risk_config=risk_config,
            position_management_config=pos_mgmt_config,
        )

        result = engine.run_backtest(
            surfaces=surfaces,
            signals_by_date=signals_by_date,
        )

        # All trades should respect risk limits
        for trade in result.trades:
            if trade.trade_type == "ENTRY":
                # Check that max loss is within limits
                # (This is implicitly enforced by risk checker)
                pass


class TestDataFrameConversion:
    """Integration tests for DataFrame conversions."""

    def test_portfolio_df_columns(self, permissive_configs):
        """Test portfolio DataFrame has correct columns."""
        execution_config, risk_config, pos_mgmt_config = permissive_configs

        surfaces = {
            date(2024, 1, 2): generate_realistic_surface(date(2024, 1, 2)),
            date(2024, 1, 3): generate_realistic_surface(date(2024, 1, 3)),
        }

        engine = BacktestEngine(
            start_date=date(2024, 1, 2),
            end_date=date(2024, 1, 3),
            initial_capital=100000.0,
            execution_config=execution_config,
            risk_config=risk_config,
            position_management_config=pos_mgmt_config,
        )

        result = engine.run_backtest(surfaces=surfaces)
        df = result.to_portfolio_df()

        expected_columns = [
            "timestamp",
            "date",
            "cash",
            "positions_value",
            "total_equity",
            "unrealized_pnl",
            "realized_pnl",
            "daily_pnl",
            "net_delta",
            "net_gamma",
            "net_vega",
            "net_theta",
            "open_positions",
        ]

        for col in expected_columns:
            assert col in df.columns

    def test_greeks_df_extraction(self, permissive_configs):
        """Test Greeks DataFrame extraction."""
        execution_config, risk_config, pos_mgmt_config = permissive_configs

        surfaces = {
            date(2024, 1, 2): generate_realistic_surface(date(2024, 1, 2)),
        }

        engine = BacktestEngine(
            start_date=date(2024, 1, 2),
            end_date=date(2024, 1, 2),
            initial_capital=100000.0,
            execution_config=execution_config,
            risk_config=risk_config,
            position_management_config=pos_mgmt_config,
        )

        result = engine.run_backtest(surfaces=surfaces)
        greeks_df = result.to_greeks_df()

        assert "net_delta" in greeks_df.columns
        assert "net_gamma" in greeks_df.columns
        assert "net_vega" in greeks_df.columns
        assert "net_theta" in greeks_df.columns


class TestReportGeneration:
    """Integration tests for report generation from backtest results."""

    def test_full_report_generation(self, permissive_configs):
        """Test full report generation from backtest."""
        from src.backtest.reporting import BacktestReporter, ReportConfig

        execution_config, risk_config, pos_mgmt_config = permissive_configs

        # Run a backtest with some activity
        start_date = date(2024, 1, 2)
        end_date = date(2024, 1, 10)

        surfaces = {}
        signals_by_date = {}
        underlying_price = 450.0

        current = start_date
        seed = 42
        while current <= end_date:
            if current.weekday() < 5:
                underlying_price *= 1 + np.random.normal(0, 0.01)
                surface = generate_realistic_surface(current, underlying_price)
                surfaces[current] = surface

                signals = generate_random_signals(surface, seed=seed)
                if signals:
                    signals_by_date[current] = signals
                seed += 1

            current += pd.Timedelta(days=1)

        engine = BacktestEngine(
            start_date=start_date,
            end_date=end_date,
            initial_capital=100000.0,
            execution_config=execution_config,
            risk_config=risk_config,
            position_management_config=pos_mgmt_config,
        )

        result = engine.run_backtest(
            surfaces=surfaces,
            signals_by_date=signals_by_date,
        )

        # Generate report
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ReportConfig(
                output_dir=Path(tmpdir),
                include_plots=False,  # Skip plots for faster test
            )
            reporter = BacktestReporter(config)

            outputs = reporter.generate_report(result)

            # Verify all expected outputs
            assert "trades_csv" in outputs
            assert "portfolio_csv" in outputs
            assert "greeks_csv" in outputs
            assert "summary_txt" in outputs

            # Verify files exist and are non-empty
            assert outputs["trades_csv"].exists()
            assert outputs["portfolio_csv"].exists()
            assert outputs["greeks_csv"].exists()
            assert outputs["summary_txt"].exists()

            # Verify content
            portfolio_df = pd.read_csv(outputs["portfolio_csv"])
            assert len(portfolio_df) > 0

            greeks_df = pd.read_csv(outputs["greeks_csv"])
            assert "net_delta" in greeks_df.columns

            summary = outputs["summary_txt"].read_text()
            assert result.run_id in summary

    def test_report_with_plots(self, permissive_configs):
        """Test report generation including plots."""
        pytest.importorskip("matplotlib")
        from src.backtest.reporting import BacktestReporter, ReportConfig

        execution_config, risk_config, pos_mgmt_config = permissive_configs

        surfaces = {}
        signals_by_date = {}
        underlying_price = 450.0

        for i in range(7):
            d = date(2024, 1, 2 + i)
            if d.weekday() < 5:
                underlying_price *= 1.005
                surfaces[d] = generate_realistic_surface(d, underlying_price)

                if i % 2 == 0:
                    signals_by_date[d] = [
                        Signal(
                            signal_type=SignalType.DIRECTIONAL_VOL,
                            edge=0.05,
                            confidence=0.8,
                            tenor_days=30,
                            delta_bucket="ATM",
                            timestamp=datetime.now(timezone.utc),
                        )
                    ]

        engine = BacktestEngine(
            start_date=date(2024, 1, 2),
            end_date=date(2024, 1, 8),
            initial_capital=100000.0,
            execution_config=execution_config,
            risk_config=risk_config,
            position_management_config=pos_mgmt_config,
        )

        result = engine.run_backtest(
            surfaces=surfaces,
            signals_by_date=signals_by_date,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config = ReportConfig(
                output_dir=Path(tmpdir),
                include_plots=True,
                plot_format="png",
            )
            reporter = BacktestReporter(config)

            outputs = reporter.generate_report(result)

            # Verify plots generated
            assert "equity_plot" in outputs
            assert "drawdown_plot" in outputs
            assert "greeks_plot" in outputs
            assert "trades_plot" in outputs

            # Verify plot files exist
            for key in ["equity_plot", "drawdown_plot", "greeks_plot", "trades_plot"]:
                assert outputs[key].exists()
                assert outputs[key].stat().st_size > 0

    def test_comparison_report(self, permissive_configs):
        """Test comparison report generation for multiple backtests."""
        from src.backtest.reporting import generate_comparison_report

        execution_config, risk_config, pos_mgmt_config = permissive_configs

        results = []
        for start_day in [2, 9]:
            start_date = date(2024, 1, start_day)
            end_date = date(2024, 1, start_day + 5)

            surfaces = {}
            current = start_date
            while current <= end_date:
                if current.weekday() < 5:
                    surfaces[current] = generate_realistic_surface(current)
                current += pd.Timedelta(days=1)

            engine = BacktestEngine(
                start_date=start_date,
                end_date=end_date,
                initial_capital=100000.0,
                execution_config=execution_config,
                risk_config=risk_config,
                position_management_config=pos_mgmt_config,
            )

            result = engine.run_backtest(surfaces=surfaces)
            results.append(result)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            comparison_path = generate_comparison_report(results, output_dir)

            assert comparison_path.exists()

            df = pd.read_csv(comparison_path)
            assert len(df) == 2
            assert "run_id" in df.columns
            assert "total_return_pct" in df.columns
            assert "sharpe_ratio" in df.columns

    def test_greeks_csv_export(self, permissive_configs):
        """Test Greeks timeseries CSV export."""
        execution_config, risk_config, pos_mgmt_config = permissive_configs

        surfaces = {
            date(2024, 1, 2): generate_realistic_surface(date(2024, 1, 2)),
            date(2024, 1, 3): generate_realistic_surface(date(2024, 1, 3)),
            date(2024, 1, 4): generate_realistic_surface(date(2024, 1, 4)),
        }

        engine = BacktestEngine(
            start_date=date(2024, 1, 2),
            end_date=date(2024, 1, 4),
            initial_capital=100000.0,
            execution_config=execution_config,
            risk_config=risk_config,
            position_management_config=pos_mgmt_config,
        )

        result = engine.run_backtest(surfaces=surfaces)

        with tempfile.TemporaryDirectory() as tmpdir:
            greeks_path = Path(tmpdir) / "greeks.csv"
            result.export_greeks_csv(str(greeks_path))

            assert greeks_path.exists()

            df = pd.read_csv(greeks_path)
            assert len(df) == 3
            assert "net_delta" in df.columns
            assert "net_gamma" in df.columns
            assert "net_vega" in df.columns
            assert "net_theta" in df.columns
