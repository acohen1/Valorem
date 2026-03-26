"""Stress testing engine for portfolio risk analysis.

This module provides the StressEngine class for running stress scenarios
on portfolios to estimate worst-case losses.
"""

from dataclasses import dataclass

from ..config.schema import StressConfig
from .portfolio import CONTRACT_MULTIPLIER, Portfolio


@dataclass
class StressScenario:
    """Result of a single stress scenario.

    Attributes:
        description: Human-readable scenario description
        pnl: Estimated P&L under this scenario (negative = loss)
        underlying_shock_pct: Applied underlying shock (if any)
        iv_shock_pts: Applied IV shock in points (if any)
    """

    description: str
    pnl: float
    underlying_shock_pct: float = 0.0
    iv_shock_pts: float = 0.0


@dataclass
class StressResult:
    """Aggregated stress test results.

    Attributes:
        scenarios: List of all stress scenarios run
        worst_case_loss: Maximum loss across all scenarios (positive number)
        worst_scenario: Description of the worst-case scenario
    """

    scenarios: list[StressScenario]
    worst_case_loss: float
    worst_scenario: str


class StressEngine:
    """Portfolio stress testing engine.

    Runs multiple stress scenarios on a portfolio to estimate potential
    losses under adverse market conditions. Uses first-order Greek
    approximations for speed.

    Stress scenarios include:
    - Underlying price shocks (e.g., -10%, -5%, +5%, +10%)
    - IV shocks (e.g., +10 pts, +20 pts)
    - Combined shocks (worst underlying + worst IV)
    """

    def __init__(self, config: StressConfig) -> None:
        """Initialize stress engine.

        Args:
            config: Stress testing configuration
        """
        self._config = config

    def run_stress_test(
        self,
        portfolio: Portfolio,
        underlying_price: float,
    ) -> StressResult:
        """Run all stress scenarios on portfolio.

        Args:
            portfolio: Portfolio to stress test
            underlying_price: Current underlying price

        Returns:
            StressResult with all scenarios and worst-case loss
        """
        scenarios: list[StressScenario] = []

        # Underlying price shocks
        for shock_pct in self._config.underlying_shocks_pct:
            scenario = self._stress_underlying(portfolio, underlying_price, shock_pct)
            scenarios.append(scenario)

        # IV shocks (parallel shift)
        for shock_pts in self._config.iv_shocks_points:
            scenario = self._stress_iv(portfolio, shock_pts)
            scenarios.append(scenario)

        # Combined shock (worst underlying down + worst IV up)
        worst_underlying_down = min(self._config.underlying_shocks_pct)
        worst_iv_up = max(self._config.iv_shocks_points)
        combined_scenario = self._stress_combined(
            portfolio, underlying_price, worst_underlying_down, worst_iv_up
        )
        scenarios.append(combined_scenario)

        # Find worst-case (minimum P&L = maximum loss)
        worst_scenario = min(scenarios, key=lambda s: s.pnl)
        worst_case_loss = abs(min(worst_scenario.pnl, 0.0))

        return StressResult(
            scenarios=scenarios,
            worst_case_loss=worst_case_loss,
            worst_scenario=worst_scenario.description,
        )

    def _stress_underlying(
        self,
        portfolio: Portfolio,
        underlying_price: float,
        shock_pct: float,
    ) -> StressScenario:
        """Stress test with underlying price shock.

        Uses delta-gamma approximation:
        P&L ≈ delta * dS + 0.5 * gamma * dS²

        Args:
            portfolio: Portfolio to stress
            underlying_price: Current underlying price
            shock_pct: Price shock as decimal (e.g., -0.10 for -10%)

        Returns:
            StressScenario with estimated P&L
        """
        # Calculate price change
        price_change = underlying_price * shock_pct

        # Delta-gamma approximation for P&L
        # Note: portfolio Greeks are already contract-adjusted
        delta_pnl = portfolio.net_delta * price_change
        gamma_pnl = 0.5 * portfolio.net_gamma * (price_change ** 2)

        total_pnl = delta_pnl + gamma_pnl

        return StressScenario(
            description=f"Underlying {shock_pct:+.1%}",
            pnl=total_pnl,
            underlying_shock_pct=shock_pct,
        )

    def _stress_iv(
        self,
        portfolio: Portfolio,
        shock_pts: float,
    ) -> StressScenario:
        """Stress test with IV shock.

        Uses vega approximation:
        P&L ≈ vega * dIV

        Args:
            portfolio: Portfolio to stress
            shock_pts: IV shock in volatility points (e.g., 10 = +10% IV)

        Returns:
            StressScenario with estimated P&L
        """
        # Convert points to decimal (10 pts = 0.10)
        shock_vol = shock_pts / 100.0

        # Vega approximation for P&L
        # Note: portfolio vega is already contract-adjusted
        vega_pnl = portfolio.net_vega * shock_vol

        return StressScenario(
            description=f"IV {shock_pts:+.0f} pts",
            pnl=vega_pnl,
            iv_shock_pts=shock_pts,
        )

    def _stress_combined(
        self,
        portfolio: Portfolio,
        underlying_price: float,
        underlying_shock_pct: float,
        iv_shock_pts: float,
    ) -> StressScenario:
        """Stress test with combined underlying and IV shock.

        Typically models a market crash scenario: underlying down + IV spike.

        Args:
            portfolio: Portfolio to stress
            underlying_price: Current underlying price
            underlying_shock_pct: Price shock as decimal
            iv_shock_pts: IV shock in volatility points

        Returns:
            StressScenario with estimated P&L
        """
        # Calculate underlying impact
        price_change = underlying_price * underlying_shock_pct
        delta_pnl = portfolio.net_delta * price_change
        gamma_pnl = 0.5 * portfolio.net_gamma * (price_change ** 2)

        # Calculate IV impact
        shock_vol = iv_shock_pts / 100.0
        vega_pnl = portfolio.net_vega * shock_vol

        total_pnl = delta_pnl + gamma_pnl + vega_pnl

        return StressScenario(
            description=f"Combined: Underlying {underlying_shock_pct:+.1%}, IV {iv_shock_pts:+.0f} pts",
            pnl=total_pnl,
            underlying_shock_pct=underlying_shock_pct,
            iv_shock_pts=iv_shock_pts,
        )

    def stress_single_scenario(
        self,
        portfolio: Portfolio,
        underlying_price: float,
        underlying_shock_pct: float = 0.0,
        iv_shock_pts: float = 0.0,
    ) -> StressScenario:
        """Run a single custom stress scenario.

        Args:
            portfolio: Portfolio to stress
            underlying_price: Current underlying price
            underlying_shock_pct: Price shock as decimal (default 0)
            iv_shock_pts: IV shock in points (default 0)

        Returns:
            StressScenario with estimated P&L
        """
        if underlying_shock_pct != 0.0 and iv_shock_pts != 0.0:
            return self._stress_combined(
                portfolio, underlying_price, underlying_shock_pct, iv_shock_pts
            )
        elif underlying_shock_pct != 0.0:
            return self._stress_underlying(portfolio, underlying_price, underlying_shock_pct)
        elif iv_shock_pts != 0.0:
            return self._stress_iv(portfolio, iv_shock_pts)
        else:
            return StressScenario(
                description="No shock",
                pnl=0.0,
            )
