"""Backtesting module for strategy simulation.

This module provides:
- BacktestEngine: Main orchestrator for backtests
- create_backtest_engine: Factory function for production use
- ExecutionSimulator: Realistic trade execution simulation
- BacktestResult: Complete result container with metrics
- BacktestReporter: Report generation with visualizations
- BacktestDataPipeline: Real data loading and model inference for backtests

Architecture:
    BacktestEngine uses dependency injection for all components.
    Use create_backtest_engine() for production, or inject mocks
    directly into BacktestEngine for testing.
"""

from .data_pipeline import BacktestData, BacktestDataConfig, BacktestDataPipeline
from .engine import BacktestEngine, create_backtest_engine
from .execution import ExecutionSimulator, FillResult
from .reporting import BacktestReporter, ReportConfig, generate_comparison_report
from .results import (
    BacktestMetrics,
    BacktestResult,
    PortfolioSnapshot,
    PositionSnapshot,
    TradeRecord,
    calculate_metrics,
)

__all__ = [
    # Engine
    "BacktestEngine",
    "create_backtest_engine",
    # Data pipeline
    "BacktestDataPipeline",
    "BacktestDataConfig",
    "BacktestData",
    # Execution
    "ExecutionSimulator",
    "FillResult",
    # Results
    "BacktestMetrics",
    "BacktestResult",
    "PortfolioSnapshot",
    "PositionSnapshot",
    "TradeRecord",
    "calculate_metrics",
    # Reporting
    "BacktestReporter",
    "ReportConfig",
    "generate_comparison_report",
]
