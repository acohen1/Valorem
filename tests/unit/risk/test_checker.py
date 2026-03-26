"""Unit tests for risk checker module."""

from datetime import date

import pytest

from src.config.schema import (
    KillSwitchConfig,
    PerTradeRiskConfig,
    RiskCapsConfig,
    RiskConfig,
    StressConfig,
)
from src.risk.checker import RiskCheckResult, RiskCheckStatus, RiskChecker
from src.risk.portfolio import Portfolio, Position
from src.strategy.types import Greeks, OptionLeg, OptionRight


@pytest.fixture
def risk_config() -> RiskConfig:
    """Standard risk config for testing."""
    return RiskConfig(
        per_trade=PerTradeRiskConfig(max_loss=500.0, max_contracts=10),
        caps=RiskCapsConfig(
            max_abs_delta=100.0,
            max_abs_gamma=50.0,
            max_abs_vega=1000.0,
            max_daily_loss=2000.0,
        ),
        stress=StressConfig(
            enabled=True,
            underlying_shocks_pct=[-0.10, 0.10],
            iv_shocks_points=[10.0],
        ),
        kill_switch=KillSwitchConfig(),
    )


@pytest.fixture
def risk_checker(risk_config: RiskConfig) -> RiskChecker:
    """Risk checker for testing."""
    return RiskChecker(risk_config)


@pytest.fixture
def small_trade_legs() -> list[OptionLeg]:
    """Small trade that should pass risk checks."""
    return [
        OptionLeg(
            symbol="SPY240315C00450000",
            qty=1,
            entry_price=3.00,
            strike=450.0,
            expiry=date(2024, 3, 15),
            right=OptionRight.CALL,
            greeks=Greeks(delta=0.5, gamma=0.02, vega=0.15, theta=-0.03),
        ),
        OptionLeg(
            symbol="SPY240315C00460000",
            qty=-1,
            entry_price=1.50,
            strike=460.0,
            expiry=date(2024, 3, 15),
            right=OptionRight.CALL,
            greeks=Greeks(delta=0.3, gamma=0.015, vega=0.10, theta=-0.02),
        ),
    ]


@pytest.fixture
def large_loss_trade_legs() -> list[OptionLeg]:
    """Trade with large potential loss."""
    return [
        OptionLeg(
            symbol="SPY240315C00450000",
            qty=5,  # Large qty
            entry_price=10.00,  # Expensive
            strike=450.0,
            expiry=date(2024, 3, 15),
            right=OptionRight.CALL,
            greeks=Greeks(delta=0.5, gamma=0.02, vega=0.15, theta=-0.03),
        ),
    ]


class TestRiskCheckResult:
    """Tests for RiskCheckResult dataclass."""

    def test_approved_result(self) -> None:
        """Test approved result."""
        result = RiskCheckResult(approved=True)
        assert result.approved is True
        assert result.status == RiskCheckStatus.APPROVED
        assert len(result.issues) == 0

    def test_rejected_result(self) -> None:
        """Test rejected result with issues."""
        result = RiskCheckResult(
            approved=False,
            issues=["Max loss exceeded", "Delta cap breached"],
        )
        assert result.approved is False
        assert result.status == RiskCheckStatus.REJECTED
        assert len(result.issues) == 2


class TestRiskCheckerPerTradeChecks:
    """Tests for per-trade risk checks."""

    def test_small_trade_approved(
        self,
        risk_checker: RiskChecker,
        small_trade_legs: list[OptionLeg],
    ) -> None:
        """Test small trade passes all checks."""
        portfolio = Portfolio()

        result = risk_checker.check_trade(
            small_trade_legs, portfolio, underlying_price=450.0
        )

        assert result.approved is True
        assert len(result.issues) == 0

    def test_max_loss_exceeded(
        self,
        risk_checker: RiskChecker,
        large_loss_trade_legs: list[OptionLeg],
    ) -> None:
        """Test trade exceeding max loss is rejected."""
        portfolio = Portfolio()

        result = risk_checker.check_trade(
            large_loss_trade_legs, portfolio, underlying_price=450.0
        )

        # Max loss = 5 * 10 * 100 = $5000, exceeds $500 limit
        assert result.approved is False
        assert any("Max loss" in issue for issue in result.issues)

    def test_max_contracts_exceeded(self, risk_config: RiskConfig) -> None:
        """Test trade exceeding contract limit is rejected."""
        # Set low contract limit
        risk_config.per_trade.max_contracts = 5
        risk_checker = RiskChecker(risk_config)

        legs = [
            OptionLeg(
                symbol="SPY240315C00450000",
                qty=3,
                entry_price=1.00,
                strike=450.0,
                expiry=date(2024, 3, 15),
                right=OptionRight.CALL,
                greeks=Greeks(delta=0.5, gamma=0.02, vega=0.15, theta=-0.03),
            ),
            OptionLeg(
                symbol="SPY240315C00460000",
                qty=-3,
                entry_price=0.50,
                strike=460.0,
                expiry=date(2024, 3, 15),
                right=OptionRight.CALL,
                greeks=Greeks(delta=0.3, gamma=0.015, vega=0.10, theta=-0.02),
            ),
        ]

        portfolio = Portfolio()
        result = risk_checker.check_trade(legs, portfolio, underlying_price=450.0)

        # Total contracts = |3| + |-3| = 6, exceeds 5 limit
        assert result.approved is False
        assert any("contracts" in issue.lower() for issue in result.issues)


class TestRiskCheckerPortfolioCaps:
    """Tests for portfolio-level caps."""

    def test_delta_cap_exceeded(self, risk_config: RiskConfig) -> None:
        """Test trade that would breach delta cap is rejected."""
        # Set low delta cap
        risk_config.caps.max_abs_delta = 50.0
        risk_checker = RiskChecker(risk_config)

        # Trade that adds significant delta
        legs = [
            OptionLeg(
                symbol="SPY240315C00450000",
                qty=2,
                entry_price=1.00,
                strike=450.0,
                expiry=date(2024, 3, 15),
                right=OptionRight.CALL,
                greeks=Greeks(delta=0.5, gamma=0.02, vega=0.15, theta=-0.03),
            ),
        ]

        portfolio = Portfolio()
        result = risk_checker.check_trade(legs, portfolio, underlying_price=450.0)

        # Delta = 2 * 0.5 * 100 = 100, exceeds 50 cap
        assert result.approved is False
        assert any("delta" in issue.lower() for issue in result.issues)

    def test_gamma_cap_exceeded(self, risk_config: RiskConfig) -> None:
        """Test trade that would breach gamma cap is rejected."""
        # Set low gamma cap
        risk_config.caps.max_abs_gamma = 1.0
        risk_checker = RiskChecker(risk_config)

        legs = [
            OptionLeg(
                symbol="SPY240315C00450000",
                qty=1,
                entry_price=1.00,
                strike=450.0,
                expiry=date(2024, 3, 15),
                right=OptionRight.CALL,
                greeks=Greeks(delta=0.5, gamma=0.05, vega=0.15, theta=-0.03),
            ),
        ]

        portfolio = Portfolio()
        result = risk_checker.check_trade(legs, portfolio, underlying_price=450.0)

        # Gamma = 1 * 0.05 * 100 = 5, exceeds 1 cap
        assert result.approved is False
        assert any("gamma" in issue.lower() for issue in result.issues)

    def test_vega_cap_exceeded(self, risk_config: RiskConfig) -> None:
        """Test trade that would breach vega cap is rejected."""
        risk_config.caps.max_abs_vega = 10.0
        risk_checker = RiskChecker(risk_config)

        legs = [
            OptionLeg(
                symbol="SPY240315C00450000",
                qty=1,
                entry_price=1.00,
                strike=450.0,
                expiry=date(2024, 3, 15),
                right=OptionRight.CALL,
                greeks=Greeks(delta=0.5, gamma=0.02, vega=0.20, theta=-0.03),
            ),
        ]

        portfolio = Portfolio()
        result = risk_checker.check_trade(legs, portfolio, underlying_price=450.0)

        # Vega = 1 * 0.20 * 100 = 20, exceeds 10 cap
        assert result.approved is False
        assert any("vega" in issue.lower() for issue in result.issues)

    def test_existing_portfolio_considered(
        self,
        risk_config: RiskConfig,
        small_trade_legs: list[OptionLeg],
    ) -> None:
        """Test that existing portfolio is considered for caps."""
        risk_config.caps.max_abs_delta = 100.0
        risk_checker = RiskChecker(risk_config)

        # Create portfolio with existing delta
        existing_leg = OptionLeg(
            symbol="SPY240315C00440000",
            qty=1,
            entry_price=5.00,
            strike=440.0,
            expiry=date(2024, 3, 15),
            right=OptionRight.CALL,
            greeks=Greeks(delta=0.7, gamma=0.02, vega=0.15, theta=-0.03),
        )
        existing_position = Position(legs=[existing_leg])
        portfolio = Portfolio(positions=[existing_position])

        # Existing delta = 70, new trade adds ~20, total ~90 < 100 -> approved
        result = risk_checker.check_trade(
            small_trade_legs, portfolio, underlying_price=450.0
        )

        assert result.hypothetical_delta == pytest.approx(90.0)


class TestRiskCheckerStressTesting:
    """Tests for stress testing checks."""

    def test_stress_test_passes(
        self,
        risk_checker: RiskChecker,
        small_trade_legs: list[OptionLeg],
    ) -> None:
        """Test that small trade passes stress test."""
        portfolio = Portfolio()

        result = risk_checker.check_trade(
            small_trade_legs, portfolio, underlying_price=450.0
        )

        # Small trade should pass stress test
        assert result.approved is True or not any(
            "Stress test" in issue for issue in result.issues
        )

    def test_stress_test_disabled(
        self,
        risk_config: RiskConfig,
        small_trade_legs: list[OptionLeg],
    ) -> None:
        """Test that stress test can be disabled."""
        risk_config.stress.enabled = False
        risk_checker = RiskChecker(risk_config)

        portfolio = Portfolio()
        result = risk_checker.check_trade(
            small_trade_legs, portfolio, underlying_price=450.0
        )

        # No stress test issues when disabled
        assert not any("Stress test" in issue for issue in result.issues)
        assert result.stress_worst_case_loss == 0.0

    def test_stress_worst_case_returned(
        self,
        risk_checker: RiskChecker,
        small_trade_legs: list[OptionLeg],
    ) -> None:
        """Test that worst case loss is returned in result."""
        portfolio = Portfolio()

        result = risk_checker.check_trade(
            small_trade_legs, portfolio, underlying_price=450.0
        )

        # Stress result should be populated
        assert result.stress_worst_case_loss >= 0.0


class TestRiskCheckerMaxLossCalculation:
    """Tests for max loss calculation logic."""

    def test_debit_spread_max_loss(self, risk_checker: RiskChecker) -> None:
        """Test max loss for debit spread = net debit."""
        # Bull call spread (debit)
        legs = [
            OptionLeg(
                symbol="SPY240315C00450000",
                qty=1,  # Buy
                entry_price=5.00,
                strike=450.0,
                expiry=date(2024, 3, 15),
                right=OptionRight.CALL,
                greeks=Greeks(delta=0.5, gamma=0.02, vega=0.15, theta=-0.03),
            ),
            OptionLeg(
                symbol="SPY240315C00460000",
                qty=-1,  # Sell
                entry_price=2.00,
                strike=460.0,
                expiry=date(2024, 3, 15),
                right=OptionRight.CALL,
                greeks=Greeks(delta=0.3, gamma=0.015, vega=0.10, theta=-0.02),
            ),
        ]

        max_loss = risk_checker._compute_max_loss(legs)

        # Net debit = (1 * 5.00 - 1 * 2.00) * 100 = $300
        assert max_loss == pytest.approx(300.0)

    def test_credit_spread_max_loss(self, risk_checker: RiskChecker) -> None:
        """Test max loss for credit spread = spread width - credit."""
        # Credit spread (sell ATM, buy OTM)
        legs = [
            OptionLeg(
                symbol="SPY240315C00450000",
                qty=-1,  # Sell
                entry_price=5.00,
                strike=450.0,
                expiry=date(2024, 3, 15),
                right=OptionRight.CALL,
                greeks=Greeks(delta=0.5, gamma=0.02, vega=0.15, theta=-0.03),
            ),
            OptionLeg(
                symbol="SPY240315C00460000",
                qty=1,  # Buy
                entry_price=2.00,
                strike=460.0,
                expiry=date(2024, 3, 15),
                right=OptionRight.CALL,
                greeks=Greeks(delta=0.3, gamma=0.015, vega=0.10, theta=-0.02),
            ),
        ]

        max_loss = risk_checker._compute_max_loss(legs)

        # Net credit = (5.00 - 2.00) * 100 = $300
        # Spread width = (460 - 450) * 100 = $1000
        # Max loss = $1000 - $300 = $700
        assert max_loss == pytest.approx(700.0)

    def test_iron_condor_max_loss(self, risk_checker: RiskChecker) -> None:
        """Test max loss for iron condor = wider spread width - net credit."""
        # Iron condor: sell 440P/460C, buy 430P/470C
        legs = [
            OptionLeg(
                symbol="SPY240315P00430000",
                qty=1,  # Buy wing put
                entry_price=1.00,
                strike=430.0,
                expiry=date(2024, 3, 15),
                right=OptionRight.PUT,
                greeks=Greeks(delta=-0.1, gamma=0.01, vega=0.05, theta=-0.01),
            ),
            OptionLeg(
                symbol="SPY240315P00440000",
                qty=-1,  # Sell 25P
                entry_price=3.00,
                strike=440.0,
                expiry=date(2024, 3, 15),
                right=OptionRight.PUT,
                greeks=Greeks(delta=-0.25, gamma=0.02, vega=0.10, theta=-0.02),
            ),
            OptionLeg(
                symbol="SPY240315C00460000",
                qty=-1,  # Sell 25C
                entry_price=3.50,
                strike=460.0,
                expiry=date(2024, 3, 15),
                right=OptionRight.CALL,
                greeks=Greeks(delta=0.25, gamma=0.02, vega=0.10, theta=-0.02),
            ),
            OptionLeg(
                symbol="SPY240315C00470000",
                qty=1,  # Buy wing call
                entry_price=1.50,
                strike=470.0,
                expiry=date(2024, 3, 15),
                right=OptionRight.CALL,
                greeks=Greeks(delta=0.1, gamma=0.01, vega=0.05, theta=-0.01),
            ),
        ]

        max_loss = risk_checker._compute_max_loss(legs)

        # Net credit = (3.00 + 3.50 - 1.00 - 1.50) * 100 = $400
        # Put width = (440 - 430) * 100 = $1000
        # Call width = (470 - 460) * 100 = $1000
        # Max loss = $1000 - $400 = $600
        assert max_loss == pytest.approx(600.0)

    def test_asymmetric_skew_trade_max_loss(self, risk_checker: RiskChecker) -> None:
        """Test max loss for skew trade with unequal spread widths."""
        # Bullish skew: tight put spread ($5 wide), wide call spread ($15 wide)
        legs = [
            OptionLeg(
                symbol="SPY240315P00435000",
                qty=-1,
                entry_price=2.00,
                strike=435.0,
                expiry=date(2024, 3, 15),
                right=OptionRight.PUT,
                greeks=Greeks(delta=-0.15, gamma=0.01, vega=0.05, theta=-0.01),
            ),
            OptionLeg(
                symbol="SPY240315P00440000",
                qty=1,
                entry_price=3.00,
                strike=440.0,
                expiry=date(2024, 3, 15),
                right=OptionRight.PUT,
                greeks=Greeks(delta=-0.25, gamma=0.02, vega=0.10, theta=-0.02),
            ),
            OptionLeg(
                symbol="SPY240315C00455000",
                qty=-1,
                entry_price=4.00,
                strike=455.0,
                expiry=date(2024, 3, 15),
                right=OptionRight.CALL,
                greeks=Greeks(delta=0.30, gamma=0.02, vega=0.10, theta=-0.02),
            ),
            OptionLeg(
                symbol="SPY240315C00470000",
                qty=1,
                entry_price=1.00,
                strike=470.0,
                expiry=date(2024, 3, 15),
                right=OptionRight.CALL,
                greeks=Greeks(delta=0.10, gamma=0.01, vega=0.05, theta=-0.01),
            ),
        ]

        max_loss = risk_checker._compute_max_loss(legs)

        # Net credit = (2.00 + 4.00 - 3.00 - 1.00) * 100 = $200
        # Put width = (440 - 435) * 100 = $500
        # Call width = (470 - 455) * 100 = $1500
        # Max loss = max($500, $1500) - $200 = $1300
        assert max_loss == pytest.approx(1300.0)

    def test_complex_structure_not_inflated_by_strike_price(
        self, risk_checker: RiskChecker
    ) -> None:
        """Regression: max loss must use spread width, not raw strike × 100."""
        # Iron condor at SPY ~$450 — the old bug computed
        # max_loss ≈ (450 + 460) × 100 ≈ $91,000 instead of ~$600.
        legs = [
            OptionLeg(
                symbol="SPY240315P00440000",
                qty=1,
                entry_price=1.50,
                strike=440.0,
                expiry=date(2024, 3, 15),
                right=OptionRight.PUT,
                greeks=Greeks(delta=-0.1, gamma=0.01, vega=0.05, theta=-0.01),
            ),
            OptionLeg(
                symbol="SPY240315P00450000",
                qty=-1,
                entry_price=4.00,
                strike=450.0,
                expiry=date(2024, 3, 15),
                right=OptionRight.PUT,
                greeks=Greeks(delta=-0.25, gamma=0.02, vega=0.10, theta=-0.02),
            ),
            OptionLeg(
                symbol="SPY240315C00460000",
                qty=-1,
                entry_price=3.50,
                strike=460.0,
                expiry=date(2024, 3, 15),
                right=OptionRight.CALL,
                greeks=Greeks(delta=0.25, gamma=0.02, vega=0.10, theta=-0.02),
            ),
            OptionLeg(
                symbol="SPY240315C00470000",
                qty=1,
                entry_price=1.00,
                strike=470.0,
                expiry=date(2024, 3, 15),
                right=OptionRight.CALL,
                greeks=Greeks(delta=0.1, gamma=0.01, vega=0.05, theta=-0.01),
            ),
        ]

        max_loss = risk_checker._compute_max_loss(legs)

        # Put width = (450 - 440) * 100 = $1000
        # Call width = (470 - 460) * 100 = $1000
        # Net credit = (4.00 + 3.50 - 1.50 - 1.00) * 100 = $500
        # Max loss = $1000 - $500 = $500
        assert max_loss == pytest.approx(500.0)
        # The old bug would compute ~$91,000 here
        assert max_loss < 5000.0


class TestRiskCheckerPortfolioLimits:
    """Tests for continuous portfolio monitoring."""

    def test_check_portfolio_limits_passed(
        self,
        risk_checker: RiskChecker,
    ) -> None:
        """Test portfolio within limits."""
        leg = OptionLeg(
            symbol="SPY240315C00450000",
            qty=1,
            entry_price=5.00,
            strike=450.0,
            expiry=date(2024, 3, 15),
            right=OptionRight.CALL,
            greeks=Greeks(delta=0.3, gamma=0.01, vega=0.10, theta=-0.03),
        )
        portfolio = Portfolio(positions=[Position(legs=[leg])])

        result = risk_checker.check_portfolio_limits(portfolio)

        assert result.approved is True

    def test_check_portfolio_limits_breached(
        self,
        risk_config: RiskConfig,
    ) -> None:
        """Test portfolio breaching limits."""
        risk_config.caps.max_abs_delta = 10.0
        risk_checker = RiskChecker(risk_config)

        leg = OptionLeg(
            symbol="SPY240315C00450000",
            qty=1,
            entry_price=5.00,
            strike=450.0,
            expiry=date(2024, 3, 15),
            right=OptionRight.CALL,
            greeks=Greeks(delta=0.5, gamma=0.02, vega=0.15, theta=-0.03),
        )
        portfolio = Portfolio(positions=[Position(legs=[leg])])

        result = risk_checker.check_portfolio_limits(portfolio)

        # Delta = 50, exceeds 10 limit
        assert result.approved is False
        assert any("delta" in issue.lower() for issue in result.issues)


class TestRiskCheckStatus:
    """Tests for RiskCheckStatus enum."""

    def test_approved_status(self) -> None:
        """Test approved status value."""
        assert RiskCheckStatus.APPROVED.value == "approved"

    def test_rejected_status(self) -> None:
        """Test rejected status value."""
        assert RiskCheckStatus.REJECTED.value == "rejected"
