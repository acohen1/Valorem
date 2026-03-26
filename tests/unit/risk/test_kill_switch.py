"""Unit tests for kill switch module."""

from datetime import date

import pytest

from src.config.schema import KillSwitchConfig
from src.risk.kill_switch import KillSwitch, KillSwitchResult, KillSwitchTrigger
from src.risk.portfolio import Portfolio, Position
from src.risk.stress import StressResult, StressScenario
from src.strategy.types import Greeks, OptionLeg, OptionRight


@pytest.fixture
def kill_switch_config() -> KillSwitchConfig:
    """Standard kill switch config for testing."""
    return KillSwitchConfig(
        halt_on_daily_loss=True,
        halt_on_stress_breach=True,
        halt_on_liquidity_collapse=True,
        max_daily_loss=1000.0,
        max_spread_pct=0.05,
    )


@pytest.fixture
def kill_switch(kill_switch_config: KillSwitchConfig) -> KillSwitch:
    """Kill switch for testing."""
    return KillSwitch(kill_switch_config)


@pytest.fixture
def sample_portfolio() -> Portfolio:
    """Sample portfolio for testing."""
    leg = OptionLeg(
        symbol="SPY240315C00450000",
        qty=1,
        entry_price=5.00,
        strike=450.0,
        expiry=date(2024, 3, 15),
        right=OptionRight.CALL,
        greeks=Greeks(delta=0.5, gamma=0.02, vega=0.15, theta=-0.03),
    )
    return Portfolio(
        positions=[Position(legs=[leg])],
        daily_pnl=0.0,
        max_acceptable_loss=2000.0,
    )


@pytest.fixture
def stress_result_ok() -> StressResult:
    """Stress result within limits."""
    return StressResult(
        scenarios=[StressScenario(description="Test", pnl=-500.0)],
        worst_case_loss=500.0,
        worst_scenario="Test",
    )


@pytest.fixture
def stress_result_breach() -> StressResult:
    """Stress result exceeding limits."""
    return StressResult(
        scenarios=[StressScenario(description="Worst case", pnl=-3000.0)],
        worst_case_loss=3000.0,
        worst_scenario="Worst case",
    )


class TestKillSwitchResult:
    """Tests for KillSwitchResult dataclass."""

    def test_not_triggered(self) -> None:
        """Test result when not triggered."""
        result = KillSwitchResult(triggered=False)
        assert result.triggered is False
        assert result.reason is None
        assert result.trigger_type is None
        assert result.triggered_at is None

    def test_triggered(self) -> None:
        """Test result when triggered."""
        from datetime import UTC, datetime

        now = datetime.now(UTC)
        result = KillSwitchResult(
            triggered=True,
            reason="Test trigger",
            trigger_type=KillSwitchTrigger.DAILY_LOSS,
            triggered_at=now,
        )
        assert result.triggered is True
        assert result.reason == "Test trigger"
        assert result.trigger_type == KillSwitchTrigger.DAILY_LOSS
        assert result.triggered_at == now


class TestKillSwitchTrigger:
    """Tests for KillSwitchTrigger enum."""

    def test_trigger_types(self) -> None:
        """Test all trigger types exist."""
        assert KillSwitchTrigger.DAILY_LOSS.value == "daily_loss"
        assert KillSwitchTrigger.STRESS_BREACH.value == "stress_breach"
        assert KillSwitchTrigger.LIQUIDITY_COLLAPSE.value == "liquidity_collapse"
        assert KillSwitchTrigger.MANUAL.value == "manual"


class TestKillSwitchDailyLoss:
    """Tests for daily loss trigger."""

    def test_no_trigger_within_limit(
        self,
        kill_switch: KillSwitch,
        sample_portfolio: Portfolio,
    ) -> None:
        """Test no trigger when within daily loss limit."""
        sample_portfolio.daily_pnl = -500.0  # Within $1000 limit

        result = kill_switch.check(sample_portfolio)

        assert result.triggered is False
        assert kill_switch.is_triggered is False

    def test_trigger_on_daily_loss_breach(
        self,
        kill_switch: KillSwitch,
        sample_portfolio: Portfolio,
    ) -> None:
        """Test trigger when daily loss exceeds limit."""
        sample_portfolio.daily_pnl = -1500.0  # Exceeds $1000 limit

        result = kill_switch.check(sample_portfolio)

        assert result.triggered is True
        assert result.trigger_type == KillSwitchTrigger.DAILY_LOSS
        assert "Daily loss" in result.reason
        assert kill_switch.is_triggered is True

    def test_check_daily_loss_only(
        self,
        kill_switch: KillSwitch,
        sample_portfolio: Portfolio,
    ) -> None:
        """Test check_daily_loss method."""
        sample_portfolio.daily_pnl = -1500.0

        result = kill_switch.check_daily_loss(sample_portfolio)

        assert result.triggered is True
        assert result.trigger_type == KillSwitchTrigger.DAILY_LOSS

    def test_daily_loss_disabled(
        self,
        kill_switch_config: KillSwitchConfig,
        sample_portfolio: Portfolio,
    ) -> None:
        """Test daily loss check can be disabled."""
        kill_switch_config.halt_on_daily_loss = False
        kill_switch = KillSwitch(kill_switch_config)
        sample_portfolio.daily_pnl = -2000.0  # Would normally trigger

        result = kill_switch.check(sample_portfolio)

        assert result.triggered is False


class TestKillSwitchStressBreach:
    """Tests for stress breach trigger."""

    def test_no_trigger_within_limit(
        self,
        kill_switch: KillSwitch,
        sample_portfolio: Portfolio,
        stress_result_ok: StressResult,
    ) -> None:
        """Test no trigger when stress test within limits."""
        result = kill_switch.check(sample_portfolio, stress_result=stress_result_ok)

        assert result.triggered is False

    def test_trigger_on_stress_breach(
        self,
        kill_switch: KillSwitch,
        sample_portfolio: Portfolio,
        stress_result_breach: StressResult,
    ) -> None:
        """Test trigger when stress test exceeds acceptable loss."""
        # stress_result_breach.worst_case_loss = 3000 > max_acceptable_loss = 2000
        result = kill_switch.check(sample_portfolio, stress_result=stress_result_breach)

        assert result.triggered is True
        assert result.trigger_type == KillSwitchTrigger.STRESS_BREACH
        assert "Stress test" in result.reason

    def test_stress_breach_disabled(
        self,
        kill_switch_config: KillSwitchConfig,
        sample_portfolio: Portfolio,
        stress_result_breach: StressResult,
    ) -> None:
        """Test stress breach check can be disabled."""
        kill_switch_config.halt_on_stress_breach = False
        kill_switch = KillSwitch(kill_switch_config)

        result = kill_switch.check(sample_portfolio, stress_result=stress_result_breach)

        assert result.triggered is False


class TestKillSwitchLiquidityCollapse:
    """Tests for liquidity collapse trigger."""

    def test_no_trigger_within_limit(
        self,
        kill_switch: KillSwitch,
        sample_portfolio: Portfolio,
    ) -> None:
        """Test no trigger when spreads within limits."""
        result = kill_switch.check(sample_portfolio, max_spread_pct=0.03)

        assert result.triggered is False

    def test_trigger_on_spread_blowout(
        self,
        kill_switch: KillSwitch,
        sample_portfolio: Portfolio,
    ) -> None:
        """Test trigger on spread blowout."""
        result = kill_switch.check(sample_portfolio, max_spread_pct=0.10)

        assert result.triggered is True
        assert result.trigger_type == KillSwitchTrigger.LIQUIDITY_COLLAPSE
        assert "Spread blowout" in result.reason

    def test_check_liquidity_only(
        self,
        kill_switch: KillSwitch,
    ) -> None:
        """Test check_liquidity method."""
        result = kill_switch.check_liquidity(0.15)

        assert result.triggered is True
        assert result.trigger_type == KillSwitchTrigger.LIQUIDITY_COLLAPSE

    def test_liquidity_collapse_disabled(
        self,
        kill_switch_config: KillSwitchConfig,
        sample_portfolio: Portfolio,
    ) -> None:
        """Test liquidity collapse check can be disabled."""
        kill_switch_config.halt_on_liquidity_collapse = False
        kill_switch = KillSwitch(kill_switch_config)

        result = kill_switch.check(sample_portfolio, max_spread_pct=0.50)

        assert result.triggered is False


class TestKillSwitchManualTrigger:
    """Tests for manual trigger."""

    def test_manual_trigger(
        self,
        kill_switch: KillSwitch,
    ) -> None:
        """Test manual trigger."""
        result = kill_switch.trigger_manual("Manual stop for review")

        assert result.triggered is True
        assert result.trigger_type == KillSwitchTrigger.MANUAL
        assert "Manual stop" in result.reason
        assert kill_switch.is_triggered is True

    def test_manual_trigger_default_reason(
        self,
        kill_switch: KillSwitch,
    ) -> None:
        """Test manual trigger with default reason."""
        result = kill_switch.trigger_manual()

        assert result.triggered is True
        assert "Manual halt" in result.reason


class TestKillSwitchReset:
    """Tests for kill switch reset."""

    def test_reset(
        self,
        kill_switch: KillSwitch,
        sample_portfolio: Portfolio,
    ) -> None:
        """Test reset clears triggered state."""
        # Trigger first
        sample_portfolio.daily_pnl = -2000.0
        kill_switch.check(sample_portfolio)
        assert kill_switch.is_triggered is True

        # Reset
        kill_switch.reset()

        assert kill_switch.is_triggered is False
        assert kill_switch.trigger_reason is None

    def test_reset_allows_new_checks(
        self,
        kill_switch: KillSwitch,
        sample_portfolio: Portfolio,
    ) -> None:
        """Test that reset allows new checks to proceed."""
        # Trigger and reset
        sample_portfolio.daily_pnl = -2000.0
        kill_switch.check(sample_portfolio)
        kill_switch.reset()

        # New check with safe values should pass
        sample_portfolio.daily_pnl = 0.0
        result = kill_switch.check(sample_portfolio)

        assert result.triggered is False


class TestKillSwitchPersistence:
    """Tests for kill switch state persistence."""

    def test_triggered_state_persists(
        self,
        kill_switch: KillSwitch,
        sample_portfolio: Portfolio,
    ) -> None:
        """Test that triggered state persists across checks."""
        # Trigger
        sample_portfolio.daily_pnl = -2000.0
        kill_switch.check(sample_portfolio)

        # Subsequent check with safe values still shows triggered
        sample_portfolio.daily_pnl = 0.0
        result = kill_switch.check(sample_portfolio)

        assert result.triggered is True

    def test_first_trigger_wins(
        self,
        kill_switch: KillSwitch,
        sample_portfolio: Portfolio,
    ) -> None:
        """Test that first trigger reason is preserved."""
        # Trigger on daily loss
        sample_portfolio.daily_pnl = -2000.0
        kill_switch.check(sample_portfolio)
        first_reason = kill_switch.trigger_reason

        # Even if another condition would trigger, first reason preserved
        result = kill_switch.check(sample_portfolio, max_spread_pct=0.50)

        assert result.reason == first_reason
        assert result.trigger_type == KillSwitchTrigger.DAILY_LOSS


class TestKillSwitchTriggeredAt:
    """Tests for trigger timestamp."""

    def test_triggered_at_set(
        self,
        kill_switch: KillSwitch,
        sample_portfolio: Portfolio,
    ) -> None:
        """Test that triggered_at is set on trigger."""
        from datetime import UTC, datetime

        before = datetime.now(UTC)
        sample_portfolio.daily_pnl = -2000.0
        result = kill_switch.check(sample_portfolio)
        after = datetime.now(UTC)

        assert result.triggered_at is not None
        assert before <= result.triggered_at <= after

    def test_triggered_at_preserved(
        self,
        kill_switch: KillSwitch,
        sample_portfolio: Portfolio,
    ) -> None:
        """Test that triggered_at is preserved across checks."""
        sample_portfolio.daily_pnl = -2000.0
        first_result = kill_switch.check(sample_portfolio)
        first_time = first_result.triggered_at

        # Subsequent check
        second_result = kill_switch.check(sample_portfolio)

        assert second_result.triggered_at == first_time
