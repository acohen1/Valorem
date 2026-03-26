"""Backtest result types and data structures."""

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

import pandas as pd


@dataclass
class TradeRecord:
    """Record of a single trade (entry or exit)."""

    trade_id: str
    timestamp: datetime
    trade_type: str  # "ENTRY" or "EXIT"
    position_id: str
    structure_type: str
    legs: list[dict[str, Any]]  # Simplified leg data for recording
    gross_premium: float  # Premium before fees
    fees: float
    slippage: float
    net_premium: float  # Premium after fees and slippage
    signal_type: str | None = None  # Entry signal type or exit signal type
    exit_reason: str | None = None  # For exit trades


@dataclass
class PositionSnapshot:
    """Snapshot of a position at a point in time."""

    timestamp: datetime
    position_id: str
    state: str
    structure_type: str
    entry_price: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float
    delta: float
    gamma: float
    vega: float
    theta: float
    days_held: int
    days_to_expiry: int | None


@dataclass
class PortfolioSnapshot:
    """Snapshot of portfolio state at a point in time."""

    timestamp: datetime
    date: date
    cash: float
    positions_value: float
    total_equity: float
    unrealized_pnl: float
    realized_pnl: float
    daily_pnl: float
    net_delta: float
    net_gamma: float
    net_vega: float
    net_theta: float
    open_positions: int
    entries_today: int
    exits_today: int


@dataclass
class BacktestMetrics:
    """Summary metrics for a backtest run."""

    # Returns metrics
    total_return_pct: float
    annualized_return_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float

    # Drawdown metrics
    max_drawdown_pct: float
    max_drawdown_duration_days: int
    avg_drawdown_pct: float

    # Trade metrics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    avg_trade_pnl: float
    max_win: float
    max_loss: float

    # Position metrics
    avg_holding_period_days: float
    max_concurrent_positions: int
    avg_concurrent_positions: float

    # Risk metrics
    avg_daily_pnl: float
    daily_pnl_std: float
    max_daily_gain: float
    max_daily_loss: float
    var_95: float  # 95% Value at Risk
    cvar_95: float  # 95% Conditional VaR (Expected Shortfall)

    # Execution metrics
    total_fees: float
    total_slippage: float
    total_commission_pct: float  # Fees + slippage as % of gross P&L


@dataclass
class BacktestResult:
    """Complete result of a backtest run."""

    # Metadata
    run_id: str
    start_date: date
    end_date: date
    initial_capital: float
    final_equity: float
    run_time_seconds: float

    # Summary metrics
    metrics: BacktestMetrics

    # Time series data
    trades: list[TradeRecord] = field(default_factory=list)
    portfolio_history: list[PortfolioSnapshot] = field(default_factory=list)
    position_history: list[PositionSnapshot] = field(default_factory=list)

    def to_trades_df(self) -> pd.DataFrame:
        """Convert trades to DataFrame."""
        if not self.trades:
            return pd.DataFrame()

        records = []
        for trade in self.trades:
            records.append({
                "trade_id": trade.trade_id,
                "timestamp": trade.timestamp,
                "trade_type": trade.trade_type,
                "position_id": trade.position_id,
                "structure_type": trade.structure_type,
                "gross_premium": trade.gross_premium,
                "fees": trade.fees,
                "slippage": trade.slippage,
                "net_premium": trade.net_premium,
                "signal_type": trade.signal_type,
                "exit_reason": trade.exit_reason,
            })
        return pd.DataFrame(records)

    def to_portfolio_df(self) -> pd.DataFrame:
        """Convert portfolio history to DataFrame."""
        if not self.portfolio_history:
            return pd.DataFrame()

        records = []
        for snap in self.portfolio_history:
            records.append({
                "timestamp": snap.timestamp,
                "date": snap.date,
                "cash": snap.cash,
                "positions_value": snap.positions_value,
                "total_equity": snap.total_equity,
                "unrealized_pnl": snap.unrealized_pnl,
                "realized_pnl": snap.realized_pnl,
                "daily_pnl": snap.daily_pnl,
                "net_delta": snap.net_delta,
                "net_gamma": snap.net_gamma,
                "net_vega": snap.net_vega,
                "net_theta": snap.net_theta,
                "open_positions": snap.open_positions,
                "entries_today": snap.entries_today,
                "exits_today": snap.exits_today,
            })
        return pd.DataFrame(records)

    def to_greeks_df(self) -> pd.DataFrame:
        """Extract Greeks timeseries from portfolio history."""
        df = self.to_portfolio_df()
        if df.empty:
            return df
        return df[["timestamp", "date", "net_delta", "net_gamma", "net_vega", "net_theta"]]

    def to_pnl_df(self) -> pd.DataFrame:
        """Extract P&L timeseries from portfolio history."""
        df = self.to_portfolio_df()
        if df.empty:
            return df
        return df[["timestamp", "date", "total_equity", "unrealized_pnl", "realized_pnl", "daily_pnl"]]

    def export_trades_csv(self, path: str) -> None:
        """Export trades to CSV file."""
        self.to_trades_df().to_csv(path, index=False)

    def export_portfolio_csv(self, path: str) -> None:
        """Export portfolio history to CSV file."""
        self.to_portfolio_df().to_csv(path, index=False)

    def export_greeks_csv(self, path: str) -> None:
        """Export Greeks timeseries to CSV file."""
        self.to_greeks_df().to_csv(path, index=False)

    def summary(self) -> str:
        """Generate human-readable summary."""
        m = self.metrics
        return f"""
Backtest Summary: {self.run_id}
{'=' * 60}
Period: {self.start_date} to {self.end_date}
Initial Capital: ${self.initial_capital:,.2f}
Final Equity: ${self.final_equity:,.2f}
Run Time: {self.run_time_seconds:.2f}s

Performance
-----------
Total Return: {m.total_return_pct:+.2f}%
Annualized Return: {m.annualized_return_pct:+.2f}%
Sharpe Ratio: {m.sharpe_ratio:.2f}
Sortino Ratio: {m.sortino_ratio:.2f}
Max Drawdown: {m.max_drawdown_pct:.2f}%

Trades
------
Total Trades: {m.total_trades}
Win Rate: {m.win_rate:.1f}%
Avg Trade P&L: ${m.avg_trade_pnl:,.2f}
Profit Factor: {m.profit_factor:.2f}
Avg Holding Period: {m.avg_holding_period_days:.1f} days

Execution
---------
Total Fees: ${m.total_fees:,.2f}
Total Slippage: ${m.total_slippage:,.2f}
"""


def calculate_metrics(
    trades: list[TradeRecord],
    portfolio_history: list[PortfolioSnapshot],
    initial_capital: float,
    trading_days: int,
) -> BacktestMetrics:
    """Calculate all backtest metrics from trade and portfolio data."""
    if not portfolio_history:
        return _empty_metrics()

    # Build DataFrames for analysis
    portfolio_df = pd.DataFrame([
        {
            "date": s.date,
            "equity": s.total_equity,
            "daily_pnl": s.daily_pnl,
            "unrealized_pnl": s.unrealized_pnl,
            "realized_pnl": s.realized_pnl,
            "open_positions": s.open_positions,
        }
        for s in portfolio_history
    ])

    # Calculate returns metrics
    final_equity = portfolio_df["equity"].iloc[-1]
    total_return_pct = (final_equity / initial_capital - 1) * 100

    # Annualized return (assume 252 trading days per year)
    years = trading_days / 252.0
    # Require at least ~21 trading days to annualize meaningfully
    if years >= 1 / 12:
        annualized_return_pct = ((final_equity / initial_capital) ** (1 / years) - 1) * 100
    else:
        annualized_return_pct = total_return_pct

    # Daily returns for Sharpe/Sortino
    daily_pnl = portfolio_df["daily_pnl"]
    avg_daily_pnl = daily_pnl.mean()
    daily_pnl_std = daily_pnl.std() if len(daily_pnl) > 1 else 0.0

    # Sharpe ratio (assume 0% risk-free rate)
    sharpe_ratio = (avg_daily_pnl / daily_pnl_std * (252 ** 0.5)) if daily_pnl_std > 0 else 0.0

    # Sortino ratio (downside deviation only)
    downside_returns = daily_pnl[daily_pnl < 0]
    downside_std = downside_returns.std() if len(downside_returns) > 0 else 0.0
    sortino_ratio = (avg_daily_pnl / downside_std * (252 ** 0.5)) if downside_std > 0 else 0.0

    # Drawdown analysis
    equity = portfolio_df["equity"]
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max.replace(0, 1e-10) * 100
    max_drawdown_pct = abs(drawdown.min())

    # Max drawdown duration
    in_drawdown = drawdown < 0
    if in_drawdown.any():
        drawdown_groups = (~in_drawdown).cumsum()
        drawdown_lengths = in_drawdown.groupby(drawdown_groups).sum()
        max_drawdown_duration_days = int(drawdown_lengths.max())
    else:
        max_drawdown_duration_days = 0

    avg_drawdown_pct = abs(drawdown[drawdown < 0].mean()) if (drawdown < 0).any() else 0.0

    # Calmar ratio
    calmar_ratio = annualized_return_pct / max_drawdown_pct if max_drawdown_pct > 0 else 0.0

    # Trade analysis
    exit_trades = [t for t in trades if t.trade_type == "EXIT"]
    total_trades = len(exit_trades)

    if total_trades > 0:
        trade_pnls = [t.net_premium for t in exit_trades]  # For exits, net_premium is the realized P&L
        winning_trades = len([p for p in trade_pnls if p > 0])
        losing_trades = len([p for p in trade_pnls if p < 0])
        win_rate = winning_trades / total_trades * 100

        wins = [p for p in trade_pnls if p > 0]
        losses = [p for p in trade_pnls if p < 0]
        avg_win = sum(wins) / len(wins) if wins else 0.0
        avg_loss = abs(sum(losses) / len(losses)) if losses else 0.0

        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        avg_trade_pnl = sum(trade_pnls) / total_trades
        max_win = max(trade_pnls) if trade_pnls else 0.0
        max_loss = min(trade_pnls) if trade_pnls else 0.0
    else:
        winning_trades = 0
        losing_trades = 0
        win_rate = 0.0
        avg_win = 0.0
        avg_loss = 0.0
        profit_factor = 0.0
        avg_trade_pnl = 0.0
        max_win = 0.0
        max_loss = 0.0

    # Position metrics
    avg_concurrent_positions = portfolio_df["open_positions"].mean()
    max_concurrent_positions = int(portfolio_df["open_positions"].max())

    # Holding period (rough estimate from trade count / avg positions)
    avg_holding_period_days = trading_days / total_trades if total_trades > 0 else 0.0

    # Daily P&L extremes
    max_daily_gain = daily_pnl.max()
    max_daily_loss = daily_pnl.min()

    # VaR and CVaR (95%)
    var_95 = daily_pnl.quantile(0.05)  # 5th percentile for losses
    cvar_95 = daily_pnl[daily_pnl <= var_95].mean() if (daily_pnl <= var_95).any() else var_95

    # Execution costs
    total_fees = sum(t.fees for t in trades)
    total_slippage = sum(t.slippage for t in trades)
    gross_pnl = final_equity - initial_capital + total_fees + total_slippage
    total_commission_pct = (total_fees + total_slippage) / abs(gross_pnl) * 100 if gross_pnl != 0 else 0.0

    return BacktestMetrics(
        total_return_pct=total_return_pct,
        annualized_return_pct=annualized_return_pct,
        sharpe_ratio=sharpe_ratio,
        sortino_ratio=sortino_ratio,
        calmar_ratio=calmar_ratio,
        max_drawdown_pct=max_drawdown_pct,
        max_drawdown_duration_days=max_drawdown_duration_days,
        avg_drawdown_pct=avg_drawdown_pct,
        total_trades=total_trades,
        winning_trades=winning_trades,
        losing_trades=losing_trades,
        win_rate=win_rate,
        avg_win=avg_win,
        avg_loss=avg_loss,
        profit_factor=profit_factor,
        avg_trade_pnl=avg_trade_pnl,
        max_win=max_win,
        max_loss=max_loss,
        avg_holding_period_days=avg_holding_period_days,
        max_concurrent_positions=max_concurrent_positions,
        avg_concurrent_positions=avg_concurrent_positions,
        avg_daily_pnl=avg_daily_pnl,
        daily_pnl_std=daily_pnl_std,
        max_daily_gain=max_daily_gain,
        max_daily_loss=max_daily_loss,
        var_95=var_95,
        cvar_95=cvar_95,
        total_fees=total_fees,
        total_slippage=total_slippage,
        total_commission_pct=total_commission_pct,
    )


def _empty_metrics() -> BacktestMetrics:
    """Return empty metrics when no data available."""
    return BacktestMetrics(
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
    )
