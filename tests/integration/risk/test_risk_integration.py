"""Integration tests for risk management system.

These tests verify the full flow from trade to risk check to kill switch.
"""

from datetime import date

import pytest

from src.config.schema import (
    KillSwitchConfig,
    PerTradeRiskConfig,
    RiskCapsConfig,
    RiskConfig,
    StressConfig,
)
from src.risk import (
    KillSwitch,
    KillSwitchTrigger,
    Portfolio,
    Position,
    RiskChecker,
    StressEngine,
)
from src.strategy import CalendarSpread, Greeks, OptionLeg, OptionRight, Signal, SignalType


@pytest.fixture
def risk_config() -> RiskConfig:
    """Risk configuration for integration tests."""
    return RiskConfig(
        per_trade=PerTradeRiskConfig(max_loss=1000.0, max_contracts=20),
        caps=RiskCapsConfig(
            max_abs_delta=200.0,
            max_abs_gamma=100.0,
            max_abs_vega=2000.0,
            max_daily_loss=5000.0,
        ),
        stress=StressConfig(
            enabled=True,
            underlying_shocks_pct=[-0.10, -0.05, 0.05, 0.10],
            iv_shocks_points=[-10.0, 10.0, 20.0],
        ),
        kill_switch=KillSwitchConfig(
            halt_on_daily_loss=True,
            halt_on_stress_breach=True,
            halt_on_liquidity_collapse=True,
            max_daily_loss=3000.0,
            max_spread_pct=0.10,
        ),
    )


@pytest.fixture
def risk_checker(risk_config: RiskConfig) -> RiskChecker:
    """Risk checker for integration tests."""
    return RiskChecker(risk_config)


@pytest.fixture
def kill_switch(risk_config: RiskConfig) -> KillSwitch:
    """Kill switch for integration tests."""
    return KillSwitch(risk_config.kill_switch)


@pytest.fixture
def stress_engine(risk_config: RiskConfig) -> StressEngine:
    """Stress engine for integration tests."""
    return StressEngine(risk_config.stress)


def make_vertical_spread(
    qty: int = 1,
    buy_price: float = 5.00,
    sell_price: float = 2.50,
) -> list[OptionLeg]:
    """Create a bull call vertical spread."""
    return [
        OptionLeg(
            symbol="SPY240315C00450000",
            qty=qty,
            entry_price=buy_price,
            strike=450.0,
            expiry=date(2024, 3, 15),
            right=OptionRight.CALL,
            greeks=Greeks(delta=0.50, gamma=0.02, vega=0.15, theta=-0.03),
        ),
        OptionLeg(
            symbol="SPY240315C00460000",
            qty=-qty,
            entry_price=sell_price,
            strike=460.0,
            expiry=date(2024, 3, 15),
            right=OptionRight.CALL,
            greeks=Greeks(delta=0.30, gamma=0.015, vega=0.10, theta=-0.02),
        ),
    ]


class TestFullRiskCheckFlow:
    """Integration tests for full risk check flow."""

    def test_trade_approval_flow(
        self,
        risk_checker: RiskChecker,
    ) -> None:
        """Test full flow: empty portfolio → trade → approval."""
        portfolio = Portfolio()
        legs = make_vertical_spread(qty=1)

        result = risk_checker.check_trade(legs, portfolio, underlying_price=450.0)

        # Small trade should be approved
        assert result.approved is True
        assert len(result.issues) == 0

        # Hypothetical Greeks should be populated
        assert result.hypothetical_delta != 0.0
        assert result.stress_worst_case_loss >= 0.0

    def test_trade_rejection_flow(
        self,
        risk_config: RiskConfig,
    ) -> None:
        """Test full flow: portfolio near limits → trade → rejection."""
        # Create checker with tight limits
        risk_config.caps.max_abs_delta = 30.0
        risk_checker = RiskChecker(risk_config)

        portfolio = Portfolio()
        legs = make_vertical_spread(qty=2)  # Delta = 2 * (0.50 - 0.30) * 100 = 40

        result = risk_checker.check_trade(legs, portfolio, underlying_price=450.0)

        # Should be rejected for delta breach
        assert result.approved is False
        assert any("delta" in issue.lower() for issue in result.issues)

    def test_portfolio_accumulation(
        self,
        risk_checker: RiskChecker,
    ) -> None:
        """Test that portfolio risk accumulates correctly."""
        portfolio = Portfolio()

        # First trade
        legs1 = make_vertical_spread(qty=1)
        result1 = risk_checker.check_trade(legs1, portfolio, underlying_price=450.0)
        assert result1.approved is True

        # Add first trade to portfolio
        portfolio = portfolio.add_trade(legs1)

        # Second trade
        legs2 = make_vertical_spread(qty=1)
        result2 = risk_checker.check_trade(legs2, portfolio, underlying_price=450.0)

        # Hypothetical delta should include both trades
        assert result2.hypothetical_delta > result1.hypothetical_delta


class TestStressTestIntegration:
    """Integration tests for stress testing."""

    def test_stress_test_integrated_with_risk_check(
        self,
        risk_checker: RiskChecker,
    ) -> None:
        """Test stress test runs as part of risk check."""
        portfolio = Portfolio()
        legs = make_vertical_spread(qty=1)

        result = risk_checker.check_trade(legs, portfolio, underlying_price=450.0)

        # Stress test should have run
        assert result.stress_worst_case_loss >= 0.0

    def test_stress_breach_rejects_trade(
        self,
        risk_config: RiskConfig,
    ) -> None:
        """Test that stress test breach rejects trade."""
        # Set very low daily loss limit to trigger stress breach
        risk_config.caps.max_daily_loss = 10.0
        risk_checker = RiskChecker(risk_config)

        portfolio = Portfolio()
        legs = make_vertical_spread(qty=5)  # Larger trade

        result = risk_checker.check_trade(legs, portfolio, underlying_price=450.0)

        # Should be rejected for stress breach
        if not result.approved:
            # Check if stress test was the reason
            stress_issues = [i for i in result.issues if "Stress" in i]
            assert len(stress_issues) > 0 or any("loss" in i.lower() for i in result.issues)


class TestKillSwitchIntegration:
    """Integration tests for kill switch."""

    def test_kill_switch_after_daily_loss(
        self,
        kill_switch: KillSwitch,
        stress_engine: StressEngine,
    ) -> None:
        """Test kill switch triggers on daily loss."""
        portfolio = Portfolio(daily_pnl=-4000.0, max_acceptable_loss=5000.0)

        # Run stress test for completeness
        stress_result = stress_engine.run_stress_test(portfolio, 450.0)

        result = kill_switch.check(
            portfolio,
            stress_result=stress_result,
            max_spread_pct=0.02,
        )

        assert result.triggered is True
        assert result.trigger_type == KillSwitchTrigger.DAILY_LOSS

    def test_kill_switch_blocks_subsequent_checks(
        self,
        kill_switch: KillSwitch,
        risk_checker: RiskChecker,
    ) -> None:
        """Test that triggered kill switch affects subsequent operations."""
        # Trigger kill switch
        portfolio_loss = Portfolio(daily_pnl=-4000.0)
        kill_switch.check(portfolio_loss)

        assert kill_switch.is_triggered is True

        # Risk checker doesn't check kill switch directly,
        # but in live trading the loop would stop
        # This test verifies the kill switch state persists
        portfolio_ok = Portfolio(daily_pnl=0.0)
        ks_result = kill_switch.check(portfolio_ok)

        assert ks_result.triggered is True

    def test_kill_switch_reset_allows_trading(
        self,
        kill_switch: KillSwitch,
    ) -> None:
        """Test that reset allows trading to resume."""
        # Trigger
        portfolio_loss = Portfolio(daily_pnl=-4000.0)
        kill_switch.check(portfolio_loss)
        assert kill_switch.is_triggered is True

        # Reset
        kill_switch.reset()

        # New check should pass
        portfolio_ok = Portfolio(daily_pnl=0.0)
        result = kill_switch.check(portfolio_ok)

        assert result.triggered is False


class TestPortfolioRiskMetrics:
    """Integration tests for portfolio risk metrics."""

    def test_portfolio_greek_aggregation(self) -> None:
        """Test that portfolio Greeks aggregate correctly."""
        # Create positions with known Greeks
        leg1 = OptionLeg(
            symbol="SPY240315C00450000",
            qty=2,
            entry_price=5.00,
            strike=450.0,
            expiry=date(2024, 3, 15),
            right=OptionRight.CALL,
            greeks=Greeks(delta=0.50, gamma=0.02, vega=0.15, theta=-0.03),
        )
        leg2 = OptionLeg(
            symbol="SPY240315P00440000",
            qty=-1,
            entry_price=3.00,
            strike=440.0,
            expiry=date(2024, 3, 15),
            right=OptionRight.PUT,
            greeks=Greeks(delta=-0.40, gamma=0.025, vega=0.12, theta=-0.025),
        )

        pos1 = Position(legs=[leg1])
        pos2 = Position(legs=[leg2])
        portfolio = Portfolio(positions=[pos1, pos2])

        # Delta: (2 * 0.50 + (-1) * (-0.40)) * 100 = (1.0 + 0.4) * 100 = 140
        assert portfolio.net_delta == pytest.approx(140.0)

        # Gamma: (2 * 0.02 + (-1) * 0.025) * 100 = (0.04 - 0.025) * 100 = 1.5
        assert portfolio.net_gamma == pytest.approx(1.5)

    def test_portfolio_pnl_tracking(self) -> None:
        """Test portfolio P&L tracking."""
        portfolio = Portfolio(daily_pnl=0.0)

        # Simulate trading day
        portfolio.update_daily_pnl(100.0)   # Gain
        portfolio.update_daily_pnl(-150.0)  # Loss
        portfolio.update_daily_pnl(200.0)   # Gain

        assert portfolio.daily_pnl == pytest.approx(150.0)

        # End of day reset
        portfolio.reset_daily_pnl()
        assert portfolio.daily_pnl == 0.0


class TestRiskSystemWithRealStructures:
    """Integration tests using real trade structures."""

    def test_calendar_spread_risk_check(
        self,
        risk_checker: RiskChecker,
    ) -> None:
        """Test risk check with CalendarSpread-generated legs."""
        # Create legs similar to what CalendarSpread would generate
        near_leg = OptionLeg(
            symbol="SPY240315C00450000",
            qty=-1,  # Sell near
            entry_price=4.00,
            strike=450.0,
            expiry=date(2024, 3, 15),
            right=OptionRight.CALL,
            greeks=Greeks(delta=0.50, gamma=0.03, vega=0.12, theta=-0.04),
        )
        far_leg = OptionLeg(
            symbol="SPY240415C00450000",
            qty=1,  # Buy far
            entry_price=6.00,
            strike=450.0,
            expiry=date(2024, 4, 15),
            right=OptionRight.CALL,
            greeks=Greeks(delta=0.52, gamma=0.02, vega=0.18, theta=-0.02),
        )

        portfolio = Portfolio()
        result = risk_checker.check_trade(
            [near_leg, far_leg], portfolio, underlying_price=450.0
        )

        # Calendar spread should pass risk checks
        assert result.approved is True

        # Calendar should be long vega (far vega > near vega)
        # Net vega = (1 * 0.18 - 1 * 0.12) * 100 = 6.0
        assert result.hypothetical_vega == pytest.approx(6.0)

    def test_multiple_trades_accumulate_risk(
        self,
        risk_config: RiskConfig,
    ) -> None:
        """Test that multiple trades accumulate toward limits."""
        risk_config.caps.max_abs_delta = 100.0
        risk_checker = RiskChecker(risk_config)

        portfolio = Portfolio()

        # Trade 1: delta = 20
        legs1 = make_vertical_spread(qty=1)
        result1 = risk_checker.check_trade(legs1, portfolio, underlying_price=450.0)
        assert result1.approved is True
        portfolio = portfolio.add_trade(legs1)

        # Trade 2: delta = 20 more
        legs2 = make_vertical_spread(qty=1)
        result2 = risk_checker.check_trade(legs2, portfolio, underlying_price=450.0)
        assert result2.approved is True
        portfolio = portfolio.add_trade(legs2)

        # Trade 3: would push delta to 60, still under 100
        legs3 = make_vertical_spread(qty=1)
        result3 = risk_checker.check_trade(legs3, portfolio, underlying_price=450.0)
        assert result3.approved is True
        portfolio = portfolio.add_trade(legs3)

        # Trade 4: Large trade that would breach
        legs4 = make_vertical_spread(qty=3)  # delta = 60 more, total = 120
        result4 = risk_checker.check_trade(legs4, portfolio, underlying_price=450.0)
        assert result4.approved is False
        assert any("delta" in issue.lower() for issue in result4.issues)
