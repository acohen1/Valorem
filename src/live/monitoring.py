"""Monitoring and alerting for live trading.

This module provides metrics logging and alerting functionality for
monitoring the trading loop and detecting issues.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

from ..risk.portfolio import Portfolio
from ..strategy.types import Greeks

logger = logging.getLogger(__name__)


class AlertLevel(str, Enum):
    """Severity level for alerts."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class Alert:
    """Trading alert.

    Attributes:
        level: Severity level
        message: Alert message
        source: Source component that generated the alert
        timestamp: When the alert was generated
        data: Additional data associated with the alert
    """

    level: AlertLevel
    message: str
    source: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize alert to dictionary."""
        return {
            "level": self.level.value,
            "message": self.message,
            "source": self.source,
            "timestamp": self.timestamp.isoformat(),
            "data": self.data,
        }


@dataclass
class TradingMetrics:
    """Trading metrics snapshot.

    Attributes:
        timestamp: When metrics were captured
        iteration: Current loop iteration
        portfolio_value: Total portfolio value
        daily_pnl: Daily P&L
        total_unrealized_pnl: Total unrealized P&L
        position_count: Number of open positions
        net_delta: Portfolio net delta
        net_gamma: Portfolio net gamma
        net_vega: Portfolio net vega
        net_theta: Portfolio net theta
        signals_generated: Signals in last iteration
        orders_created: Orders in last iteration
        fills_received: Fills in last iteration
        errors_count: Cumulative errors
    """

    timestamp: datetime
    iteration: int
    portfolio_value: float
    daily_pnl: float
    total_unrealized_pnl: float
    position_count: int
    net_delta: float
    net_gamma: float
    net_vega: float
    net_theta: float
    signals_generated: int = 0
    orders_created: int = 0
    fills_received: int = 0
    errors_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize metrics to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "iteration": self.iteration,
            "portfolio_value": self.portfolio_value,
            "daily_pnl": self.daily_pnl,
            "total_unrealized_pnl": self.total_unrealized_pnl,
            "position_count": self.position_count,
            "net_delta": self.net_delta,
            "net_gamma": self.net_gamma,
            "net_vega": self.net_vega,
            "net_theta": self.net_theta,
            "signals_generated": self.signals_generated,
            "orders_created": self.orders_created,
            "fills_received": self.fills_received,
            "errors_count": self.errors_count,
        }


class TradingMonitor:
    """Monitor trading loop and generate alerts.

    The TradingMonitor:
    - Logs trading metrics after each iteration
    - Generates alerts when thresholds are breached
    - Writes metrics and alerts to configured destinations

    Destinations:
    - stdout: Log to standard output via Python logging
    - file: Write JSON lines to metrics file

    Example:
        >>> monitor = TradingMonitor(log_to_file=True, file_path="metrics.jsonl")
        >>> monitor.log_metrics(portfolio, iteration=1)
        >>> monitor.send_alert(AlertLevel.WARNING, "High delta exposure", "risk")
    """

    def __init__(
        self,
        log_to_stdout: bool = True,
        log_to_file: bool = False,
        file_path: Optional[str] = None,
        alert_callback: Optional[Callable[[Alert], None]] = None,
        # Threshold settings
        delta_threshold: float = 500.0,
        pnl_threshold: float = -1000.0,
    ) -> None:
        """Initialize trading monitor.

        Args:
            log_to_stdout: Log metrics to stdout via logging
            log_to_file: Write metrics to file (JSON lines format)
            file_path: Path for metrics file (required if log_to_file=True)
            alert_callback: Optional callback for alerts
            delta_threshold: Absolute delta threshold for warning
            pnl_threshold: Daily P&L threshold for warning (negative = loss)
        """
        self._log_to_stdout = log_to_stdout
        self._log_to_file = log_to_file
        self._file_path = Path(file_path) if file_path else None
        self._alert_callback = alert_callback

        self._delta_threshold = delta_threshold
        self._pnl_threshold = pnl_threshold

        # History
        self._metrics_history: list[TradingMetrics] = []
        self._alert_history: list[Alert] = []

        # Ensure file directory exists
        if self._log_to_file and self._file_path:
            self._file_path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def metrics_history(self) -> list[TradingMetrics]:
        """Get metrics history."""
        return self._metrics_history.copy()

    @property
    def alert_history(self) -> list[Alert]:
        """Get alert history."""
        return self._alert_history.copy()

    def log_metrics(
        self,
        portfolio: Portfolio,
        iteration: int,
        signals_generated: int = 0,
        orders_created: int = 0,
        fills_received: int = 0,
        errors_count: int = 0,
    ) -> TradingMetrics:
        """Log trading metrics.

        Args:
            portfolio: Current portfolio state
            iteration: Current loop iteration
            signals_generated: Signals generated this iteration
            orders_created: Orders created this iteration
            fills_received: Fills received this iteration
            errors_count: Cumulative error count

        Returns:
            TradingMetrics snapshot
        """
        greeks = portfolio.get_greeks()

        metrics = TradingMetrics(
            timestamp=datetime.now(timezone.utc),
            iteration=iteration,
            portfolio_value=portfolio.total_unrealized_pnl + portfolio.total_realized_pnl,
            daily_pnl=portfolio.daily_pnl,
            total_unrealized_pnl=portfolio.total_unrealized_pnl,
            position_count=len(portfolio.positions),
            net_delta=greeks.delta,
            net_gamma=greeks.gamma,
            net_vega=greeks.vega,
            net_theta=greeks.theta,
            signals_generated=signals_generated,
            orders_created=orders_created,
            fills_received=fills_received,
            errors_count=errors_count,
        )

        self._metrics_history.append(metrics)
        # Cap history to prevent unbounded memory growth in long sessions
        if len(self._metrics_history) > 10000:
            self._metrics_history = self._metrics_history[-10000:]

        # Log to stdout
        if self._log_to_stdout:
            logger.info(
                f"[Iter {iteration}] "
                f"P&L: ${metrics.daily_pnl:.2f}, "
                f"Positions: {metrics.position_count}, "
                f"Delta: {metrics.net_delta:.1f}, "
                f"Signals: {signals_generated}, "
                f"Orders: {orders_created}, "
                f"Fills: {fills_received}"
            )

        # Log to file
        if self._log_to_file and self._file_path:
            with open(self._file_path, "a") as f:
                f.write(json.dumps(metrics.to_dict()) + "\n")

        # Check thresholds and generate alerts
        self._check_thresholds(metrics)

        return metrics

    def _check_thresholds(self, metrics: TradingMetrics) -> None:
        """Check metrics against thresholds and generate alerts."""
        # Check delta threshold
        if abs(metrics.net_delta) > self._delta_threshold:
            self.send_alert(
                level=AlertLevel.WARNING,
                message=f"High delta exposure: {metrics.net_delta:.1f} "
                f"(threshold: ±{self._delta_threshold})",
                source="risk_monitor",
                data={"net_delta": metrics.net_delta},
            )

        # Check P&L threshold
        if metrics.daily_pnl < self._pnl_threshold:
            self.send_alert(
                level=AlertLevel.WARNING,
                message=f"Daily P&L below threshold: ${metrics.daily_pnl:.2f} "
                f"(threshold: ${self._pnl_threshold:.2f})",
                source="risk_monitor",
                data={"daily_pnl": metrics.daily_pnl},
            )

    def send_alert(
        self,
        level: AlertLevel,
        message: str,
        source: str,
        data: Optional[dict[str, Any]] = None,
    ) -> Alert:
        """Send an alert.

        Args:
            level: Alert severity level
            message: Alert message
            source: Source component
            data: Additional data

        Returns:
            Generated Alert object
        """
        alert = Alert(
            level=level,
            message=message,
            source=source,
            data=data or {},
        )

        self._alert_history.append(alert)
        if len(self._alert_history) > 5000:
            self._alert_history = self._alert_history[-5000:]

        # Log alert
        log_level = {
            AlertLevel.INFO: logging.INFO,
            AlertLevel.WARNING: logging.WARNING,
            AlertLevel.ERROR: logging.ERROR,
            AlertLevel.CRITICAL: logging.CRITICAL,
        }[level]

        logger.log(log_level, f"ALERT [{source}]: {message}")

        # Callback
        if self._alert_callback:
            self._alert_callback(alert)

        # Write to file
        if self._log_to_file and self._file_path:
            alert_path = self._file_path.parent / "alerts.jsonl"
            with open(alert_path, "a") as f:
                f.write(json.dumps(alert.to_dict()) + "\n")

        return alert

    def get_latest_metrics(self) -> Optional[TradingMetrics]:
        """Get the most recent metrics snapshot.

        Returns:
            Latest metrics, or None if no metrics recorded
        """
        return self._metrics_history[-1] if self._metrics_history else None

    def get_alerts_by_level(self, level: AlertLevel) -> list[Alert]:
        """Get alerts filtered by severity level.

        Args:
            level: Alert level to filter by

        Returns:
            List of alerts at specified level
        """
        return [a for a in self._alert_history if a.level == level]

    def clear_history(self) -> None:
        """Clear metrics and alert history."""
        self._metrics_history.clear()
        self._alert_history.clear()

    def get_summary(self) -> dict[str, Any]:
        """Get monitoring summary.

        Returns:
            Dictionary with monitoring statistics
        """
        return {
            "total_iterations": len(self._metrics_history),
            "total_alerts": len(self._alert_history),
            "alerts_by_level": {
                level.value: len(self.get_alerts_by_level(level))
                for level in AlertLevel
            },
            "latest_metrics": (
                self.get_latest_metrics().to_dict()
                if self.get_latest_metrics()
                else None
            ),
        }
