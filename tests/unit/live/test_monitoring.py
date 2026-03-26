"""Unit tests for TradingMonitor."""

import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pytest

from src.live.monitoring import Alert, AlertLevel, TradingMetrics, TradingMonitor
from src.risk.portfolio import Portfolio


@pytest.fixture
def sample_portfolio() -> Portfolio:
    """Create sample portfolio for testing."""
    return Portfolio(daily_pnl=100.0, max_acceptable_loss=5000.0)


@pytest.fixture
def monitor_with_file(tmp_path: Path) -> TradingMonitor:
    """Create monitor with file logging."""
    return TradingMonitor(
        log_to_stdout=False,
        log_to_file=True,
        file_path=str(tmp_path / "metrics.jsonl"),
    )


class TestTradingMonitorInit:
    """Tests for TradingMonitor initialization."""

    def test_init_default(self) -> None:
        """Test default initialization."""
        monitor = TradingMonitor()
        assert len(monitor.metrics_history) == 0
        assert len(monitor.alert_history) == 0

    def test_init_with_thresholds(self) -> None:
        """Test initialization with custom thresholds."""
        monitor = TradingMonitor(
            delta_threshold=1000.0,
            pnl_threshold=-5000.0,
        )
        assert monitor._delta_threshold == 1000.0
        assert monitor._pnl_threshold == -5000.0


class TestTradingMonitorLogMetrics:
    """Tests for TradingMonitor.log_metrics."""

    def test_log_metrics_basic(self, sample_portfolio: Portfolio) -> None:
        """Test basic metrics logging."""
        monitor = TradingMonitor(log_to_stdout=False)
        metrics = monitor.log_metrics(
            portfolio=sample_portfolio,
            iteration=1,
            signals_generated=5,
            orders_created=2,
            fills_received=2,
        )

        assert metrics.iteration == 1
        assert metrics.signals_generated == 5
        assert metrics.orders_created == 2
        assert metrics.fills_received == 2
        assert metrics.daily_pnl == 100.0

        # Check history
        assert len(monitor.metrics_history) == 1

    def test_log_metrics_captures_greeks(self, sample_portfolio: Portfolio) -> None:
        """Test that Greeks are captured."""
        monitor = TradingMonitor(log_to_stdout=False)
        metrics = monitor.log_metrics(portfolio=sample_portfolio, iteration=1)

        # Empty portfolio has zero Greeks
        assert metrics.net_delta == 0.0
        assert metrics.net_gamma == 0.0
        assert metrics.net_vega == 0.0
        assert metrics.net_theta == 0.0

    def test_log_metrics_to_file(
        self, sample_portfolio: Portfolio, monitor_with_file: TradingMonitor
    ) -> None:
        """Test logging metrics to file."""
        monitor_with_file.log_metrics(portfolio=sample_portfolio, iteration=1)
        monitor_with_file.log_metrics(portfolio=sample_portfolio, iteration=2)

        # Check file exists and has content
        file_path = Path(monitor_with_file._file_path)
        assert file_path.exists()

        lines = file_path.read_text().strip().split("\n")
        assert len(lines) == 2


class TestTradingMonitorAlerts:
    """Tests for TradingMonitor alerting."""

    def test_send_alert(self) -> None:
        """Test sending an alert."""
        monitor = TradingMonitor(log_to_stdout=False)
        alert = monitor.send_alert(
            level=AlertLevel.WARNING,
            message="Test warning",
            source="test",
            data={"key": "value"},
        )

        assert alert.level == AlertLevel.WARNING
        assert alert.message == "Test warning"
        assert alert.source == "test"
        assert alert.data["key"] == "value"

        # Check history
        assert len(monitor.alert_history) == 1

    def test_alert_callback(self) -> None:
        """Test alert callback is called."""
        received_alerts: list[Alert] = []

        def callback(alert: Alert) -> None:
            received_alerts.append(alert)

        monitor = TradingMonitor(log_to_stdout=False, alert_callback=callback)
        monitor.send_alert(AlertLevel.INFO, "Test", "test")

        assert len(received_alerts) == 1
        assert received_alerts[0].message == "Test"

    def test_delta_threshold_alert(self) -> None:
        """Test that delta threshold triggers alert."""
        monitor = TradingMonitor(
            log_to_stdout=False,
            delta_threshold=100.0,
        )

        # Portfolio with high delta would trigger alert
        # We need to mock a portfolio with high delta
        # For simplicity, we'll check the alert logic directly
        class MockPortfolio:
            daily_pnl = 0.0
            total_unrealized_pnl = 0.0
            total_realized_pnl = 0.0
            positions = []

            def get_greeks(self):
                from src.strategy.types import Greeks
                return Greeks(delta=500.0, gamma=0.0, vega=0.0, theta=0.0)

        mock_portfolio = MockPortfolio()
        monitor.log_metrics(portfolio=mock_portfolio, iteration=1)  # type: ignore

        # Should have triggered an alert
        warnings = monitor.get_alerts_by_level(AlertLevel.WARNING)
        assert len(warnings) == 1
        assert "delta" in warnings[0].message.lower()

    def test_pnl_threshold_alert(self) -> None:
        """Test that P&L threshold triggers alert."""
        monitor = TradingMonitor(
            log_to_stdout=False,
            pnl_threshold=-500.0,
        )

        # Portfolio with large loss
        portfolio = Portfolio(daily_pnl=-1000.0)
        monitor.log_metrics(portfolio=portfolio, iteration=1)

        # Should have triggered an alert
        warnings = monitor.get_alerts_by_level(AlertLevel.WARNING)
        assert len(warnings) == 1
        assert "p&l" in warnings[0].message.lower()


class TestTradingMonitorUtilities:
    """Tests for TradingMonitor utility methods."""

    def test_get_latest_metrics(self, sample_portfolio: Portfolio) -> None:
        """Test getting latest metrics."""
        monitor = TradingMonitor(log_to_stdout=False)

        # No metrics yet
        assert monitor.get_latest_metrics() is None

        # Add metrics
        monitor.log_metrics(portfolio=sample_portfolio, iteration=1)
        monitor.log_metrics(portfolio=sample_portfolio, iteration=2)

        latest = monitor.get_latest_metrics()
        assert latest is not None
        assert latest.iteration == 2

    def test_get_alerts_by_level(self) -> None:
        """Test filtering alerts by level."""
        monitor = TradingMonitor(log_to_stdout=False)

        monitor.send_alert(AlertLevel.INFO, "Info 1", "test")
        monitor.send_alert(AlertLevel.WARNING, "Warning 1", "test")
        monitor.send_alert(AlertLevel.INFO, "Info 2", "test")
        monitor.send_alert(AlertLevel.ERROR, "Error 1", "test")

        info_alerts = monitor.get_alerts_by_level(AlertLevel.INFO)
        warning_alerts = monitor.get_alerts_by_level(AlertLevel.WARNING)
        error_alerts = monitor.get_alerts_by_level(AlertLevel.ERROR)

        assert len(info_alerts) == 2
        assert len(warning_alerts) == 1
        assert len(error_alerts) == 1

    def test_clear_history(self, sample_portfolio: Portfolio) -> None:
        """Test clearing history."""
        monitor = TradingMonitor(log_to_stdout=False)

        monitor.log_metrics(portfolio=sample_portfolio, iteration=1)
        monitor.send_alert(AlertLevel.INFO, "Test", "test")

        assert len(monitor.metrics_history) == 1
        assert len(monitor.alert_history) == 1

        monitor.clear_history()

        assert len(monitor.metrics_history) == 0
        assert len(monitor.alert_history) == 0

    def test_get_summary(self, sample_portfolio: Portfolio) -> None:
        """Test getting monitoring summary."""
        monitor = TradingMonitor(log_to_stdout=False)

        monitor.log_metrics(portfolio=sample_portfolio, iteration=1)
        monitor.log_metrics(portfolio=sample_portfolio, iteration=2)
        monitor.send_alert(AlertLevel.WARNING, "Test", "test")

        summary = monitor.get_summary()

        assert summary["total_iterations"] == 2
        assert summary["total_alerts"] == 1
        assert summary["alerts_by_level"]["warning"] == 1
        assert summary["latest_metrics"]["iteration"] == 2


class TestTradingMetrics:
    """Tests for TradingMetrics dataclass."""

    def test_to_dict(self) -> None:
        """Test metrics serialization."""
        metrics = TradingMetrics(
            timestamp=datetime.now(timezone.utc),
            iteration=5,
            portfolio_value=10000.0,
            daily_pnl=500.0,
            total_unrealized_pnl=300.0,
            position_count=3,
            net_delta=50.0,
            net_gamma=1.0,
            net_vega=100.0,
            net_theta=-20.0,
            signals_generated=10,
            orders_created=5,
            fills_received=4,
            errors_count=1,
        )

        data = metrics.to_dict()

        assert data["iteration"] == 5
        assert data["portfolio_value"] == 10000.0
        assert data["daily_pnl"] == 500.0
        assert data["signals_generated"] == 10


class TestAlert:
    """Tests for Alert dataclass."""

    def test_to_dict(self) -> None:
        """Test alert serialization."""
        alert = Alert(
            level=AlertLevel.ERROR,
            message="Something went wrong",
            source="risk_checker",
            data={"value": 123},
        )

        data = alert.to_dict()

        assert data["level"] == "error"
        assert data["message"] == "Something went wrong"
        assert data["source"] == "risk_checker"
        assert data["data"]["value"] == 123
        assert "timestamp" in data
