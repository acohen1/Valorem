"""Backtest reporting and visualization."""

import logging
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from matplotlib.figure import Figure

from .results import BacktestResult

logger = logging.getLogger(__name__)


@dataclass
class ReportConfig:
    """Configuration for report generation."""

    output_dir: Path
    include_plots: bool = True
    plot_format: str = "png"  # png, pdf, svg
    plot_dpi: int = 150
    figsize: tuple[int, int] = (12, 8)


class BacktestReporter:
    """Generate comprehensive backtest reports with visualizations."""

    def __init__(self, config: ReportConfig) -> None:
        """Initialize reporter.

        Args:
            config: Report configuration
        """
        self._config = config
        self._config.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_report(self, result: BacktestResult) -> dict[str, Path]:
        """Generate complete backtest report with all outputs.

        Args:
            result: Backtest result to report on

        Returns:
            Dictionary mapping output type to file path
        """
        outputs: dict[str, Path] = {}

        # Export data files
        outputs["trades_csv"] = self._export_trades(result)
        outputs["portfolio_csv"] = self._export_portfolio(result)
        outputs["greeks_csv"] = self._export_greeks(result)
        outputs["summary_txt"] = self._export_summary(result)

        # Generate plots if enabled
        if self._config.include_plots:
            try:
                outputs["equity_plot"] = self._plot_equity_curve(result)
                outputs["drawdown_plot"] = self._plot_drawdown(result)
                outputs["greeks_plot"] = self._plot_greeks(result)
                outputs["trades_plot"] = self._plot_trade_analysis(result)
            except ImportError:
                logger.warning("matplotlib not available, skipping plots")

        logger.info(f"Generated {len(outputs)} report files in {self._config.output_dir}")
        return outputs

    def _export_trades(self, result: BacktestResult) -> Path:
        """Export trades to CSV."""
        path = self._config.output_dir / f"{result.run_id}_trades.csv"
        result.export_trades_csv(str(path))
        return path

    def _export_portfolio(self, result: BacktestResult) -> Path:
        """Export portfolio history to CSV."""
        path = self._config.output_dir / f"{result.run_id}_portfolio.csv"
        result.export_portfolio_csv(str(path))
        return path

    def _export_greeks(self, result: BacktestResult) -> Path:
        """Export Greeks timeseries to CSV."""
        path = self._config.output_dir / f"{result.run_id}_greeks.csv"
        greeks_df = result.to_greeks_df()
        greeks_df.to_csv(path, index=False)
        return path

    def _export_summary(self, result: BacktestResult) -> Path:
        """Export detailed summary report."""
        path = self._config.output_dir / f"{result.run_id}_summary.txt"

        # Extended summary with more details
        summary = self._generate_detailed_summary(result)

        with open(path, "w") as f:
            f.write(summary)
        return path

    def _generate_detailed_summary(self, result: BacktestResult) -> str:
        """Generate detailed text summary."""
        m = result.metrics

        return f"""
================================================================================
                         BACKTEST REPORT: {result.run_id}
================================================================================

OVERVIEW
--------
Period:              {result.start_date} to {result.end_date}
Trading Days:        {len(result.portfolio_history)}
Initial Capital:     ${result.initial_capital:,.2f}
Final Equity:        ${result.final_equity:,.2f}
Net P&L:             ${result.final_equity - result.initial_capital:,.2f}
Run Time:            {result.run_time_seconds:.2f}s

PERFORMANCE METRICS
-------------------
Total Return:        {m.total_return_pct:+.2f}%
Annualized Return:   {m.annualized_return_pct:+.2f}%
Sharpe Ratio:        {m.sharpe_ratio:.3f}
Sortino Ratio:       {m.sortino_ratio:.3f}
Calmar Ratio:        {m.calmar_ratio:.3f}

DRAWDOWN ANALYSIS
-----------------
Max Drawdown:        {m.max_drawdown_pct:.2f}%
Max DD Duration:     {m.max_drawdown_duration_days} days
Avg Drawdown:        {m.avg_drawdown_pct:.2f}%

TRADE STATISTICS
----------------
Total Trades:        {m.total_trades}
Winning Trades:      {m.winning_trades}
Losing Trades:       {m.losing_trades}
Win Rate:            {m.win_rate:.1f}%
Profit Factor:       {m.profit_factor:.2f}

Avg Trade P&L:       ${m.avg_trade_pnl:,.2f}
Avg Win:             ${m.avg_win:,.2f}
Avg Loss:            ${m.avg_loss:,.2f}
Max Win:             ${m.max_win:,.2f}
Max Loss:            ${m.max_loss:,.2f}

POSITION METRICS
----------------
Avg Holding Period:  {m.avg_holding_period_days:.1f} days
Max Concurrent:      {m.max_concurrent_positions}
Avg Concurrent:      {m.avg_concurrent_positions:.1f}

RISK METRICS
------------
Daily P&L (Avg):     ${m.avg_daily_pnl:,.2f}
Daily P&L (Std):     ${m.daily_pnl_std:,.2f}
Max Daily Gain:      ${m.max_daily_gain:,.2f}
Max Daily Loss:      ${m.max_daily_loss:,.2f}
VaR (95%):           ${m.var_95:,.2f}
CVaR (95%):          ${m.cvar_95:,.2f}

EXECUTION COSTS
---------------
Total Fees:          ${m.total_fees:,.2f}
Total Slippage:      ${m.total_slippage:,.2f}
Cost as % of P&L:    {m.total_commission_pct:.2f}%

================================================================================
"""

    def _plot_equity_curve(self, result: BacktestResult) -> Path:
        """Plot equity curve with key metrics."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=self._config.figsize)

        portfolio_df = result.to_portfolio_df()
        if portfolio_df.empty:
            plt.close(fig)
            raise ValueError("No portfolio data to plot")

        dates = pd.to_datetime(portfolio_df["date"])
        equity = portfolio_df["total_equity"]

        # Plot equity curve
        ax.plot(dates, equity, "b-", linewidth=1.5, label="Portfolio Equity")

        # Add initial capital line
        ax.axhline(
            y=result.initial_capital,
            color="gray",
            linestyle="--",
            alpha=0.7,
            label="Initial Capital",
        )

        # Highlight high watermark
        running_max = equity.cummax()
        ax.fill_between(
            dates,
            equity,
            running_max,
            alpha=0.3,
            color="red",
            label="Drawdown",
        )

        ax.set_xlabel("Date")
        ax.set_ylabel("Equity ($)")
        ax.set_title(f"Equity Curve - {result.run_id}")
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)

        # Add metrics annotation
        m = result.metrics
        textstr = (
            f"Return: {m.total_return_pct:+.2f}%\n"
            f"Sharpe: {m.sharpe_ratio:.2f}\n"
            f"Max DD: {m.max_drawdown_pct:.2f}%"
        )
        props = dict(boxstyle="round", facecolor="wheat", alpha=0.8)
        ax.text(
            0.02,
            0.98,
            textstr,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=props,
        )

        fig.tight_layout()

        path = self._config.output_dir / f"{result.run_id}_equity.{self._config.plot_format}"
        fig.savefig(path, dpi=self._config.plot_dpi, bbox_inches="tight")
        plt.close(fig)

        return path

    def _plot_drawdown(self, result: BacktestResult) -> Path:
        """Plot drawdown chart."""
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self._config.figsize, sharex=True)

        portfolio_df = result.to_portfolio_df()
        if portfolio_df.empty:
            plt.close(fig)
            raise ValueError("No portfolio data to plot")

        dates = pd.to_datetime(portfolio_df["date"])
        equity = portfolio_df["total_equity"]

        # Calculate drawdown
        running_max = equity.cummax()
        drawdown = (equity - running_max) / running_max * 100

        # Top panel: equity
        ax1.plot(dates, equity, "b-", linewidth=1.5)
        ax1.plot(dates, running_max, "g--", alpha=0.7, label="High Water Mark")
        ax1.set_ylabel("Equity ($)")
        ax1.set_title(f"Equity & Drawdown - {result.run_id}")
        ax1.legend(loc="upper left")
        ax1.grid(True, alpha=0.3)

        # Bottom panel: drawdown
        ax2.fill_between(dates, drawdown, 0, color="red", alpha=0.5)
        ax2.axhline(y=-result.metrics.max_drawdown_pct, color="darkred", linestyle="--", alpha=0.7)
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Drawdown (%)")
        ax2.grid(True, alpha=0.3)

        # Annotate max drawdown
        max_dd_idx = drawdown.idxmin()
        ax2.annotate(
            f"Max: {result.metrics.max_drawdown_pct:.1f}%",
            xy=(dates.iloc[max_dd_idx], drawdown.iloc[max_dd_idx]),
            xytext=(10, -20),
            textcoords="offset points",
            fontsize=9,
            arrowprops=dict(arrowstyle="->", color="darkred"),
        )

        fig.tight_layout()

        path = self._config.output_dir / f"{result.run_id}_drawdown.{self._config.plot_format}"
        fig.savefig(path, dpi=self._config.plot_dpi, bbox_inches="tight")
        plt.close(fig)

        return path

    def _plot_greeks(self, result: BacktestResult) -> Path:
        """Plot portfolio Greeks over time."""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=self._config.figsize)

        greeks_df = result.to_greeks_df()
        if greeks_df.empty:
            plt.close(fig)
            raise ValueError("No Greeks data to plot")

        dates = pd.to_datetime(greeks_df["date"])

        # Delta
        ax = axes[0, 0]
        ax.plot(dates, greeks_df["net_delta"], "b-", linewidth=1.2)
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax.set_ylabel("Net Delta")
        ax.set_title("Portfolio Delta")
        ax.grid(True, alpha=0.3)

        # Gamma
        ax = axes[0, 1]
        ax.plot(dates, greeks_df["net_gamma"], "g-", linewidth=1.2)
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax.set_ylabel("Net Gamma")
        ax.set_title("Portfolio Gamma")
        ax.grid(True, alpha=0.3)

        # Vega
        ax = axes[1, 0]
        ax.plot(dates, greeks_df["net_vega"], "r-", linewidth=1.2)
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel("Date")
        ax.set_ylabel("Net Vega")
        ax.set_title("Portfolio Vega")
        ax.grid(True, alpha=0.3)

        # Theta
        ax = axes[1, 1]
        ax.plot(dates, greeks_df["net_theta"], "purple", linewidth=1.2)
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel("Date")
        ax.set_ylabel("Net Theta")
        ax.set_title("Portfolio Theta")
        ax.grid(True, alpha=0.3)

        fig.suptitle(f"Portfolio Greeks - {result.run_id}", fontsize=14)
        fig.tight_layout()

        path = self._config.output_dir / f"{result.run_id}_greeks.{self._config.plot_format}"
        fig.savefig(path, dpi=self._config.plot_dpi, bbox_inches="tight")
        plt.close(fig)

        return path

    def _plot_trade_analysis(self, result: BacktestResult) -> Path:
        """Plot trade analysis charts."""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=self._config.figsize)

        trades_df = result.to_trades_df()
        exit_trades = trades_df[trades_df["trade_type"] == "EXIT"] if not trades_df.empty else pd.DataFrame()

        # Trade P&L distribution
        ax = axes[0, 0]
        if not exit_trades.empty:
            pnls = exit_trades["net_premium"]
            ax.hist(pnls, bins=20, edgecolor="black", alpha=0.7)
            ax.axvline(x=0, color="red", linestyle="--", alpha=0.7)
            ax.axvline(x=pnls.mean(), color="blue", linestyle="--", alpha=0.7, label=f"Mean: ${pnls.mean():.2f}")
        ax.set_xlabel("Trade P&L ($)")
        ax.set_ylabel("Frequency")
        ax.set_title("Trade P&L Distribution")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Cumulative P&L
        ax = axes[0, 1]
        if not exit_trades.empty:
            cumulative_pnl = exit_trades["net_premium"].cumsum()
            ax.plot(range(len(cumulative_pnl)), cumulative_pnl, "b-", linewidth=1.2)
            ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel("Trade Number")
        ax.set_ylabel("Cumulative P&L ($)")
        ax.set_title("Cumulative Trade P&L")
        ax.grid(True, alpha=0.3)

        # Win/Loss by structure type
        ax = axes[1, 0]
        if not exit_trades.empty:
            by_structure = exit_trades.groupby("structure_type")["net_premium"].agg(["sum", "count"])
            if not by_structure.empty:
                colors = ["green" if v > 0 else "red" for v in by_structure["sum"]]
                ax.bar(by_structure.index, by_structure["sum"], color=colors, alpha=0.7, edgecolor="black")
        ax.set_xlabel("Structure Type")
        ax.set_ylabel("Total P&L ($)")
        ax.set_title("P&L by Structure Type")
        ax.grid(True, alpha=0.3, axis="y")

        # Trade outcomes over time
        ax = axes[1, 1]
        if not exit_trades.empty:
            wins = exit_trades["net_premium"] > 0
            dates = pd.to_datetime(exit_trades["timestamp"])
            rolling_win_rate = wins.rolling(window=min(10, len(wins)), min_periods=1).mean() * 100
            ax.plot(dates, rolling_win_rate, "b-", linewidth=1.2)
            ax.axhline(y=50, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel("Date")
        ax.set_ylabel("Win Rate (%)")
        ax.set_title("Rolling Win Rate (10 trades)")
        ax.grid(True, alpha=0.3)

        fig.suptitle(f"Trade Analysis - {result.run_id}", fontsize=14)
        fig.tight_layout()

        path = self._config.output_dir / f"{result.run_id}_trades.{self._config.plot_format}"
        fig.savefig(path, dpi=self._config.plot_dpi, bbox_inches="tight")
        plt.close(fig)

        return path


def generate_comparison_report(
    results: list[BacktestResult],
    output_dir: Path,
) -> Path:
    """Generate comparison report for multiple backtest runs.

    Args:
        results: List of backtest results to compare
        output_dir: Output directory for report

    Returns:
        Path to comparison report
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build comparison DataFrame
    records = []
    for r in results:
        records.append({
            "run_id": r.run_id,
            "start_date": r.start_date,
            "end_date": r.end_date,
            "total_return_pct": r.metrics.total_return_pct,
            "sharpe_ratio": r.metrics.sharpe_ratio,
            "sortino_ratio": r.metrics.sortino_ratio,
            "max_drawdown_pct": r.metrics.max_drawdown_pct,
            "total_trades": r.metrics.total_trades,
            "win_rate": r.metrics.win_rate,
            "profit_factor": r.metrics.profit_factor,
            "avg_trade_pnl": r.metrics.avg_trade_pnl,
            "total_fees": r.metrics.total_fees,
        })

    comparison_df = pd.DataFrame(records)
    path = output_dir / "backtest_comparison.csv"
    comparison_df.to_csv(path, index=False)

    logger.info(f"Generated comparison report: {path}")
    return path
