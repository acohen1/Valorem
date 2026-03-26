"""Unit tests for stress testing module."""

from datetime import date

import pytest

from src.config.schema import StressConfig
from src.risk.portfolio import Portfolio, Position
from src.risk.stress import StressEngine, StressResult, StressScenario
from src.strategy.types import Greeks, OptionLeg, OptionRight


@pytest.fixture
def stress_config() -> StressConfig:
    """Standard stress config for testing."""
    return StressConfig(
        enabled=True,
        underlying_shocks_pct=[-0.10, -0.05, 0.05, 0.10],
        iv_shocks_points=[-10.0, 10.0, 20.0],
    )


@pytest.fixture
def stress_engine(stress_config: StressConfig) -> StressEngine:
    """Stress engine for testing."""
    return StressEngine(stress_config)


@pytest.fixture
def long_call_position() -> Position:
    """Long call position for testing."""
    leg = OptionLeg(
        symbol="SPY240315C00450000",
        qty=1,
        entry_price=5.00,
        strike=450.0,
        expiry=date(2024, 3, 15),
        right=OptionRight.CALL,
        greeks=Greeks(delta=0.5, gamma=0.02, vega=0.15, theta=-0.03),
    )
    return Position(legs=[leg])


@pytest.fixture
def short_call_position() -> Position:
    """Short call position for testing."""
    leg = OptionLeg(
        symbol="SPY240315C00450000",
        qty=-1,
        entry_price=5.00,
        strike=450.0,
        expiry=date(2024, 3, 15),
        right=OptionRight.CALL,
        greeks=Greeks(delta=0.5, gamma=0.02, vega=0.15, theta=-0.03),
    )
    return Position(legs=[leg])


class TestStressScenario:
    """Tests for StressScenario dataclass."""

    def test_scenario_creation(self) -> None:
        """Test scenario creation."""
        scenario = StressScenario(
            description="Underlying -10%",
            pnl=-500.0,
            underlying_shock_pct=-0.10,
        )
        assert scenario.description == "Underlying -10%"
        assert scenario.pnl == -500.0
        assert scenario.underlying_shock_pct == -0.10
        assert scenario.iv_shock_pts == 0.0

    def test_scenario_with_iv_shock(self) -> None:
        """Test scenario with IV shock."""
        scenario = StressScenario(
            description="IV +10 pts",
            pnl=200.0,
            iv_shock_pts=10.0,
        )
        assert scenario.iv_shock_pts == 10.0
        assert scenario.underlying_shock_pct == 0.0


class TestStressResult:
    """Tests for StressResult dataclass."""

    def test_result_creation(self) -> None:
        """Test result creation."""
        scenarios = [
            StressScenario(description="Test 1", pnl=-100.0),
            StressScenario(description="Test 2", pnl=-500.0),
        ]
        result = StressResult(
            scenarios=scenarios,
            worst_case_loss=500.0,
            worst_scenario="Test 2",
        )
        assert len(result.scenarios) == 2
        assert result.worst_case_loss == 500.0
        assert result.worst_scenario == "Test 2"


class TestStressEngineUnderlyingShocks:
    """Tests for underlying price shocks."""

    def test_long_call_underlying_down(
        self,
        stress_engine: StressEngine,
        long_call_position: Position,
    ) -> None:
        """Test long call loses money on underlying down move."""
        portfolio = Portfolio(positions=[long_call_position])

        # -10% shock on $450 = -$45 price change
        scenario = stress_engine._stress_underlying(portfolio, 450.0, -0.10)

        # Delta P&L: 50 * (-45) = -2250 (portfolio delta is contract-adjusted)
        # Gamma P&L: 0.5 * 2 * 45^2 = 2025
        # Total: -2250 + 2025 = -225
        expected_delta_pnl = portfolio.net_delta * (-45)
        expected_gamma_pnl = 0.5 * portfolio.net_gamma * (45**2)
        expected_total = expected_delta_pnl + expected_gamma_pnl

        assert scenario.pnl == pytest.approx(expected_total)
        assert scenario.underlying_shock_pct == -0.10
        assert "Underlying" in scenario.description

    def test_long_call_underlying_up(
        self,
        stress_engine: StressEngine,
        long_call_position: Position,
    ) -> None:
        """Test long call makes money on underlying up move."""
        portfolio = Portfolio(positions=[long_call_position])

        # +10% shock on $450 = +$45 price change
        scenario = stress_engine._stress_underlying(portfolio, 450.0, 0.10)

        # Should be profitable for long call
        assert scenario.pnl > 0

    def test_short_call_underlying_up(
        self,
        stress_engine: StressEngine,
        short_call_position: Position,
    ) -> None:
        """Test short call loses money on underlying up move."""
        portfolio = Portfolio(positions=[short_call_position])

        # +10% shock
        scenario = stress_engine._stress_underlying(portfolio, 450.0, 0.10)

        # Short call should lose on up move (negative delta effect dominates)
        assert scenario.pnl < 0


class TestStressEngineIVShocks:
    """Tests for IV shocks."""

    def test_long_vega_iv_up(
        self,
        stress_engine: StressEngine,
        long_call_position: Position,
    ) -> None:
        """Test long vega position profits from IV increase."""
        portfolio = Portfolio(positions=[long_call_position])

        # +10 pts IV shock
        scenario = stress_engine._stress_iv(portfolio, 10.0)

        # Vega P&L: 15 * 0.10 = 1.5 (portfolio vega = 15 contract-adjusted)
        expected_pnl = portfolio.net_vega * 0.10  # 10 pts = 0.10

        assert scenario.pnl == pytest.approx(expected_pnl)
        assert scenario.iv_shock_pts == 10.0
        assert "IV" in scenario.description

    def test_long_vega_iv_down(
        self,
        stress_engine: StressEngine,
        long_call_position: Position,
    ) -> None:
        """Test long vega position loses from IV decrease."""
        portfolio = Portfolio(positions=[long_call_position])

        # -10 pts IV shock
        scenario = stress_engine._stress_iv(portfolio, -10.0)

        # Should lose money
        assert scenario.pnl < 0

    def test_short_vega_iv_up(
        self,
        stress_engine: StressEngine,
        short_call_position: Position,
    ) -> None:
        """Test short vega position loses from IV increase."""
        portfolio = Portfolio(positions=[short_call_position])

        # +20 pts IV shock
        scenario = stress_engine._stress_iv(portfolio, 20.0)

        # Short vega should lose on IV up
        assert scenario.pnl < 0


class TestStressEngineCombinedShocks:
    """Tests for combined shocks."""

    def test_combined_shock(
        self,
        stress_engine: StressEngine,
        long_call_position: Position,
    ) -> None:
        """Test combined underlying and IV shock."""
        portfolio = Portfolio(positions=[long_call_position])

        scenario = stress_engine._stress_combined(
            portfolio, 450.0, -0.10, 20.0
        )

        # Should combine both effects
        assert "Combined" in scenario.description
        assert scenario.underlying_shock_pct == -0.10
        assert scenario.iv_shock_pts == 20.0


class TestStressEngineFullTest:
    """Tests for full stress test runs."""

    def test_run_stress_test(
        self,
        stress_engine: StressEngine,
        long_call_position: Position,
    ) -> None:
        """Test full stress test run."""
        portfolio = Portfolio(positions=[long_call_position])

        result = stress_engine.run_stress_test(portfolio, 450.0)

        # Should have scenarios for each shock type
        # 4 underlying + 3 IV + 1 combined = 8 scenarios
        assert len(result.scenarios) == 8

        # Worst case should be identified
        assert result.worst_case_loss >= 0
        assert result.worst_scenario != ""

    def test_run_stress_test_empty_portfolio(
        self,
        stress_engine: StressEngine,
    ) -> None:
        """Test stress test on empty portfolio."""
        portfolio = Portfolio()

        result = stress_engine.run_stress_test(portfolio, 450.0)

        # All scenarios should have zero P&L
        assert all(s.pnl == 0.0 for s in result.scenarios)
        assert result.worst_case_loss == 0.0

    def test_worst_case_is_worst_scenario(
        self,
        stress_engine: StressEngine,
        long_call_position: Position,
    ) -> None:
        """Test that worst_case_loss matches the worst scenario."""
        portfolio = Portfolio(positions=[long_call_position])

        result = stress_engine.run_stress_test(portfolio, 450.0)

        # Find worst scenario manually
        worst = min(result.scenarios, key=lambda s: s.pnl)
        expected_worst_loss = abs(min(worst.pnl, 0.0))

        assert result.worst_case_loss == pytest.approx(expected_worst_loss)


class TestStressEngineSingleScenario:
    """Tests for single custom scenario."""

    def test_single_underlying_shock(
        self,
        stress_engine: StressEngine,
        long_call_position: Position,
    ) -> None:
        """Test single custom underlying shock."""
        portfolio = Portfolio(positions=[long_call_position])

        scenario = stress_engine.stress_single_scenario(
            portfolio, 450.0, underlying_shock_pct=-0.05
        )

        assert scenario.underlying_shock_pct == -0.05
        assert scenario.iv_shock_pts == 0.0

    def test_single_iv_shock(
        self,
        stress_engine: StressEngine,
        long_call_position: Position,
    ) -> None:
        """Test single custom IV shock."""
        portfolio = Portfolio(positions=[long_call_position])

        scenario = stress_engine.stress_single_scenario(
            portfolio, 450.0, iv_shock_pts=15.0
        )

        assert scenario.iv_shock_pts == 15.0
        assert scenario.underlying_shock_pct == 0.0

    def test_single_combined_shock(
        self,
        stress_engine: StressEngine,
        long_call_position: Position,
    ) -> None:
        """Test single custom combined shock."""
        portfolio = Portfolio(positions=[long_call_position])

        scenario = stress_engine.stress_single_scenario(
            portfolio, 450.0, underlying_shock_pct=-0.05, iv_shock_pts=15.0
        )

        assert scenario.underlying_shock_pct == -0.05
        assert scenario.iv_shock_pts == 15.0

    def test_single_no_shock(
        self,
        stress_engine: StressEngine,
        long_call_position: Position,
    ) -> None:
        """Test scenario with no shock."""
        portfolio = Portfolio(positions=[long_call_position])

        scenario = stress_engine.stress_single_scenario(portfolio, 450.0)

        assert scenario.pnl == 0.0
        assert "No shock" in scenario.description
