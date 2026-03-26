"""Unit tests for analytical Greeks calculator."""

import numpy as np
import pandas as pd
import pytest

from src.surface.greeks.analytical import AnalyticalGreeks
from src.surface.iv.black_scholes import BlackScholesIVSolver


class TestAnalyticalGreeksInitialization:
    """Test AnalyticalGreeks initialization."""

    def test_initialization(self):
        """Test that AnalyticalGreeks can be instantiated."""
        greeks_calc = AnalyticalGreeks()
        assert greeks_calc is not None


class TestDelta:
    """Test delta calculations."""

    @pytest.fixture
    def greeks_calc(self):
        """Create Greeks calculator instance."""
        return AnalyticalGreeks()

    def test_atm_call_delta(self, greeks_calc):
        """Test that ATM call delta is approximately 0.5."""
        S = pd.Series([100.0])
        K = pd.Series([100.0])
        T = pd.Series([1.0])
        r = pd.Series([0.05])
        q = pd.Series([0.0])
        sigma = pd.Series([0.2])
        right = pd.Series(["C"])

        greeks = greeks_calc.compute_greeks_vectorized(S, K, T, r, q, sigma, right)

        # ATM call delta should be around 0.5 (slightly above due to positive drift)
        assert 0.45 < greeks["delta"].iloc[0] < 0.65

    def test_atm_put_delta(self, greeks_calc):
        """Test that ATM put delta is approximately -0.5."""
        S = pd.Series([100.0])
        K = pd.Series([100.0])
        T = pd.Series([1.0])
        r = pd.Series([0.05])
        q = pd.Series([0.0])
        sigma = pd.Series([0.2])
        right = pd.Series(["P"])

        greeks = greeks_calc.compute_greeks_vectorized(S, K, T, r, q, sigma, right)

        # ATM put delta should be around -0.5 (slightly less negative due to positive drift)
        assert -0.65 < greeks["delta"].iloc[0] < -0.35

    def test_itm_call_delta(self, greeks_calc):
        """Test that deep ITM call delta approaches 1."""
        S = pd.Series([150.0])
        K = pd.Series([100.0])
        T = pd.Series([1.0])
        r = pd.Series([0.05])
        q = pd.Series([0.0])
        sigma = pd.Series([0.2])
        right = pd.Series(["C"])

        greeks = greeks_calc.compute_greeks_vectorized(S, K, T, r, q, sigma, right)

        # Deep ITM call should have delta close to 1
        assert greeks["delta"].iloc[0] > 0.95

    def test_otm_call_delta(self, greeks_calc):
        """Test that deep OTM call delta approaches 0."""
        S = pd.Series([100.0])
        K = pd.Series([150.0])
        T = pd.Series([1.0])
        r = pd.Series([0.05])
        q = pd.Series([0.0])
        sigma = pd.Series([0.2])
        right = pd.Series(["C"])

        greeks = greeks_calc.compute_greeks_vectorized(S, K, T, r, q, sigma, right)

        # Deep OTM call should have delta close to 0
        assert greeks["delta"].iloc[0] < 0.1

    def test_put_call_delta_parity(self, greeks_calc):
        """Test that call_delta - put_delta = e^(-qT)."""
        S = pd.Series([100.0])
        K = pd.Series([105.0])
        T = pd.Series([0.5])
        r = pd.Series([0.03])
        q = pd.Series([0.01])
        sigma = pd.Series([0.25])

        call_greeks = greeks_calc.compute_greeks_vectorized(
            S, K, T, r, q, sigma, pd.Series(["C"])
        )
        put_greeks = greeks_calc.compute_greeks_vectorized(
            S, K, T, r, q, sigma, pd.Series(["P"])
        )

        delta_diff = call_greeks["delta"].iloc[0] - put_greeks["delta"].iloc[0]
        expected = np.exp(-q.iloc[0] * T.iloc[0])

        assert np.abs(delta_diff - expected) < 1e-10

    def test_delta_range_calls(self, greeks_calc):
        """Test that call delta is between 0 and 1."""
        S = pd.Series([100.0, 100.0, 100.0])
        K = pd.Series([80.0, 100.0, 120.0])
        T = pd.Series([1.0, 1.0, 1.0])
        r = pd.Series([0.05, 0.05, 0.05])
        q = pd.Series([0.0, 0.0, 0.0])
        sigma = pd.Series([0.2, 0.2, 0.2])
        right = pd.Series(["C", "C", "C"])

        greeks = greeks_calc.compute_greeks_vectorized(S, K, T, r, q, sigma, right)

        assert all(greeks["delta"] >= 0)
        assert all(greeks["delta"] <= 1)

    def test_delta_range_puts(self, greeks_calc):
        """Test that put delta is between -1 and 0."""
        S = pd.Series([100.0, 100.0, 100.0])
        K = pd.Series([80.0, 100.0, 120.0])
        T = pd.Series([1.0, 1.0, 1.0])
        r = pd.Series([0.05, 0.05, 0.05])
        q = pd.Series([0.0, 0.0, 0.0])
        sigma = pd.Series([0.2, 0.2, 0.2])
        right = pd.Series(["P", "P", "P"])

        greeks = greeks_calc.compute_greeks_vectorized(S, K, T, r, q, sigma, right)

        assert all(greeks["delta"] >= -1)
        assert all(greeks["delta"] <= 0)


class TestGamma:
    """Test gamma calculations."""

    @pytest.fixture
    def greeks_calc(self):
        """Create Greeks calculator instance."""
        return AnalyticalGreeks()

    def test_gamma_positive(self, greeks_calc):
        """Test that gamma is always positive."""
        S = pd.Series([100.0, 100.0, 100.0])
        K = pd.Series([95.0, 100.0, 105.0])
        T = pd.Series([1.0, 1.0, 1.0])
        r = pd.Series([0.05, 0.05, 0.05])
        q = pd.Series([0.0, 0.0, 0.0])
        sigma = pd.Series([0.2, 0.2, 0.2])
        right = pd.Series(["C", "C", "C"])

        greeks = greeks_calc.compute_greeks_vectorized(S, K, T, r, q, sigma, right)

        assert all(greeks["gamma"] > 0)

    def test_gamma_same_for_calls_and_puts(self, greeks_calc):
        """Test that gamma is the same for calls and puts."""
        S = pd.Series([100.0])
        K = pd.Series([105.0])
        T = pd.Series([0.5])
        r = pd.Series([0.03])
        q = pd.Series([0.01])
        sigma = pd.Series([0.25])

        call_greeks = greeks_calc.compute_greeks_vectorized(
            S, K, T, r, q, sigma, pd.Series(["C"])
        )
        put_greeks = greeks_calc.compute_greeks_vectorized(
            S, K, T, r, q, sigma, pd.Series(["P"])
        )

        assert np.abs(call_greeks["gamma"].iloc[0] - put_greeks["gamma"].iloc[0]) < 1e-10

    def test_gamma_atm_highest(self, greeks_calc):
        """Test that ATM options have high gamma."""
        S = pd.Series([100.0, 100.0, 100.0])
        K = pd.Series([80.0, 100.0, 120.0])  # ITM, ATM, OTM
        T = pd.Series([1.0, 1.0, 1.0])
        r = pd.Series([0.05, 0.05, 0.05])
        q = pd.Series([0.0, 0.0, 0.0])
        sigma = pd.Series([0.2, 0.2, 0.2])
        right = pd.Series(["C", "C", "C"])

        greeks = greeks_calc.compute_greeks_vectorized(S, K, T, r, q, sigma, right)

        # ATM should have higher gamma than deep ITM
        assert greeks["gamma"].iloc[1] > greeks["gamma"].iloc[0]

    def test_gamma_decreases_with_time(self, greeks_calc):
        """Test that gamma increases as expiration approaches (for ATM)."""
        S = pd.Series([100.0, 100.0, 100.0])
        K = pd.Series([100.0, 100.0, 100.0])
        T = pd.Series([1.0, 0.5, 0.1])
        r = pd.Series([0.05, 0.05, 0.05])
        q = pd.Series([0.0, 0.0, 0.0])
        sigma = pd.Series([0.2, 0.2, 0.2])
        right = pd.Series(["C", "C", "C"])

        greeks = greeks_calc.compute_greeks_vectorized(S, K, T, r, q, sigma, right)

        # For ATM options, gamma increases as expiration approaches
        assert greeks["gamma"].iloc[2] > greeks["gamma"].iloc[1] > greeks["gamma"].iloc[0]


class TestVega:
    """Test vega calculations."""

    @pytest.fixture
    def greeks_calc(self):
        """Create Greeks calculator instance."""
        return AnalyticalGreeks()

    def test_vega_positive(self, greeks_calc):
        """Test that vega is always positive."""
        S = pd.Series([100.0, 100.0, 100.0])
        K = pd.Series([95.0, 100.0, 105.0])
        T = pd.Series([1.0, 1.0, 1.0])
        r = pd.Series([0.05, 0.05, 0.05])
        q = pd.Series([0.0, 0.0, 0.0])
        sigma = pd.Series([0.2, 0.2, 0.2])
        right = pd.Series(["C", "C", "C"])

        greeks = greeks_calc.compute_greeks_vectorized(S, K, T, r, q, sigma, right)

        assert all(greeks["vega"] > 0)

    def test_vega_same_for_calls_and_puts(self, greeks_calc):
        """Test that vega is the same for calls and puts."""
        S = pd.Series([100.0])
        K = pd.Series([105.0])
        T = pd.Series([0.5])
        r = pd.Series([0.03])
        q = pd.Series([0.01])
        sigma = pd.Series([0.25])

        call_greeks = greeks_calc.compute_greeks_vectorized(
            S, K, T, r, q, sigma, pd.Series(["C"])
        )
        put_greeks = greeks_calc.compute_greeks_vectorized(
            S, K, T, r, q, sigma, pd.Series(["P"])
        )

        assert np.abs(call_greeks["vega"].iloc[0] - put_greeks["vega"].iloc[0]) < 1e-10

    def test_vega_atm_highest(self, greeks_calc):
        """Test that ATM options have highest vega."""
        S = pd.Series([100.0, 100.0, 100.0])
        K = pd.Series([90.0, 100.0, 110.0])
        T = pd.Series([1.0, 1.0, 1.0])
        r = pd.Series([0.05, 0.05, 0.05])
        q = pd.Series([0.0, 0.0, 0.0])
        sigma = pd.Series([0.2, 0.2, 0.2])
        right = pd.Series(["C", "C", "C"])

        greeks = greeks_calc.compute_greeks_vectorized(S, K, T, r, q, sigma, right)

        # ATM should have higher vega than OTM
        assert greeks["vega"].iloc[1] > greeks["vega"].iloc[0]

    def test_vega_decreases_with_time(self, greeks_calc):
        """Test that vega decreases as expiration approaches."""
        S = pd.Series([100.0, 100.0, 100.0])
        K = pd.Series([100.0, 100.0, 100.0])
        T = pd.Series([1.0, 0.5, 0.1])
        r = pd.Series([0.05, 0.05, 0.05])
        q = pd.Series([0.0, 0.0, 0.0])
        sigma = pd.Series([0.2, 0.2, 0.2])
        right = pd.Series(["C", "C", "C"])

        greeks = greeks_calc.compute_greeks_vectorized(S, K, T, r, q, sigma, right)

        # Longer dated options have higher vega
        assert greeks["vega"].iloc[0] > greeks["vega"].iloc[1] > greeks["vega"].iloc[2]


class TestTheta:
    """Test theta calculations."""

    @pytest.fixture
    def greeks_calc(self):
        """Create Greeks calculator instance."""
        return AnalyticalGreeks()

    def test_theta_negative_for_long_calls(self, greeks_calc):
        """Test that theta is negative for long call positions."""
        S = pd.Series([100.0, 100.0, 100.0])
        K = pd.Series([95.0, 100.0, 105.0])
        T = pd.Series([1.0, 1.0, 1.0])
        r = pd.Series([0.05, 0.05, 0.05])
        q = pd.Series([0.0, 0.0, 0.0])
        sigma = pd.Series([0.2, 0.2, 0.2])
        right = pd.Series(["C", "C", "C"])

        greeks = greeks_calc.compute_greeks_vectorized(S, K, T, r, q, sigma, right)

        # Theta should be negative for long options (time decay)
        assert all(greeks["theta"] < 0)

    def test_theta_negative_for_long_puts(self, greeks_calc):
        """Test that theta is negative for long put positions."""
        S = pd.Series([100.0, 100.0, 100.0])
        K = pd.Series([95.0, 100.0, 105.0])
        T = pd.Series([1.0, 1.0, 1.0])
        r = pd.Series([0.05, 0.05, 0.05])
        q = pd.Series([0.0, 0.0, 0.0])
        sigma = pd.Series([0.2, 0.2, 0.2])
        right = pd.Series(["P", "P", "P"])

        greeks = greeks_calc.compute_greeks_vectorized(S, K, T, r, q, sigma, right)

        # Theta should be negative for long options (time decay)
        assert all(greeks["theta"] < 0)

    def test_theta_different_for_calls_and_puts(self, greeks_calc):
        """Test that theta is different for calls and puts."""
        S = pd.Series([100.0])
        K = pd.Series([105.0])
        T = pd.Series([0.5])
        r = pd.Series([0.03])
        q = pd.Series([0.0])
        sigma = pd.Series([0.25])

        call_greeks = greeks_calc.compute_greeks_vectorized(
            S, K, T, r, q, sigma, pd.Series(["C"])
        )
        put_greeks = greeks_calc.compute_greeks_vectorized(
            S, K, T, r, q, sigma, pd.Series(["P"])
        )

        # Theta should be different for calls and puts (due to cost of carry)
        assert np.abs(call_greeks["theta"].iloc[0] - put_greeks["theta"].iloc[0]) > 0.01

    def test_theta_atm_highest_magnitude(self, greeks_calc):
        """Test that ATM options have highest theta magnitude."""
        S = pd.Series([100.0, 100.0, 100.0])
        K = pd.Series([90.0, 100.0, 110.0])
        T = pd.Series([1.0, 1.0, 1.0])
        r = pd.Series([0.05, 0.05, 0.05])
        q = pd.Series([0.0, 0.0, 0.0])
        sigma = pd.Series([0.2, 0.2, 0.2])
        right = pd.Series(["C", "C", "C"])

        greeks = greeks_calc.compute_greeks_vectorized(S, K, T, r, q, sigma, right)

        # ATM should have highest theta magnitude (most time decay)
        assert abs(greeks["theta"].iloc[1]) > abs(greeks["theta"].iloc[0])


class TestVectorization:
    """Test vectorized computations."""

    @pytest.fixture
    def greeks_calc(self):
        """Create Greeks calculator instance."""
        return AnalyticalGreeks()

    def test_multiple_options(self, greeks_calc):
        """Test computing Greeks for multiple options."""
        S = pd.Series([100.0, 100.0, 100.0, 100.0])
        K = pd.Series([95.0, 100.0, 105.0, 110.0])
        T = pd.Series([1.0, 1.0, 1.0, 1.0])
        r = pd.Series([0.05, 0.05, 0.05, 0.05])
        q = pd.Series([0.0, 0.0, 0.0, 0.0])
        sigma = pd.Series([0.2, 0.2, 0.2, 0.2])
        right = pd.Series(["C", "C", "C", "C"])

        greeks = greeks_calc.compute_greeks_vectorized(S, K, T, r, q, sigma, right)

        assert len(greeks) == 4
        assert all(col in greeks.columns for col in ["delta", "gamma", "vega", "theta"])

    def test_scalar_rate_and_yield(self, greeks_calc):
        """Test Greeks computation with scalar r and q."""
        S = pd.Series([100.0, 100.0])
        K = pd.Series([95.0, 105.0])
        T = pd.Series([1.0, 1.0])
        r = 0.05  # Scalar
        q = 0.02  # Scalar
        sigma = pd.Series([0.25, 0.30])
        right = pd.Series(["C", "C"])

        greeks = greeks_calc.compute_greeks_vectorized(S, K, T, r, q, sigma, right)

        assert len(greeks) == 2
        assert all(greeks["delta"] > 0)

    def test_mismatched_lengths_raises_error(self, greeks_calc):
        """Test that mismatched input lengths raise ValueError."""
        S = pd.Series([100.0])  # Wrong length
        K = pd.Series([100.0, 105.0])
        T = pd.Series([1.0, 1.0])
        r = 0.05
        q = 0.0
        sigma = pd.Series([0.25, 0.30])
        right = pd.Series(["C", "C"])

        with pytest.raises(ValueError, match="same length"):
            greeks_calc.compute_greeks_vectorized(S, K, T, r, q, sigma, right)


class TestIndividualGreeksMethods:
    """Test individual Greeks computation methods."""

    @pytest.fixture
    def greeks_calc(self):
        """Create Greeks calculator instance."""
        return AnalyticalGreeks()

    def test_compute_delta_only(self, greeks_calc):
        """Test computing delta only."""
        S = pd.Series([100.0])
        K = pd.Series([100.0])
        T = pd.Series([1.0])
        r = pd.Series([0.05])
        q = pd.Series([0.0])
        sigma = pd.Series([0.2])
        right = pd.Series(["C"])

        delta = greeks_calc.compute_delta(S, K, T, r, q, sigma, right)

        assert isinstance(delta, pd.Series)
        assert 0.45 < delta.iloc[0] < 0.65

    def test_compute_gamma_only(self, greeks_calc):
        """Test computing gamma only."""
        S = pd.Series([100.0])
        K = pd.Series([100.0])
        T = pd.Series([1.0])
        r = pd.Series([0.05])
        q = pd.Series([0.0])
        sigma = pd.Series([0.2])
        right = pd.Series(["C"])

        gamma = greeks_calc.compute_gamma(S, K, T, r, q, sigma, right)

        assert isinstance(gamma, pd.Series)
        assert gamma.iloc[0] > 0


class TestIntegrationWithIVSolver:
    """Integration tests with IV solver (IV → Greeks → reprice)."""

    @pytest.fixture
    def greeks_calc(self):
        """Create Greeks calculator instance."""
        return AnalyticalGreeks()

    @pytest.fixture
    def iv_solver(self):
        """Create IV solver instance."""
        return BlackScholesIVSolver()

    def test_iv_greeks_roundtrip(self, iv_solver, greeks_calc):
        """Test full round-trip: price → IV → Greeks."""
        S = pd.Series([100.0])
        K = pd.Series([105.0])
        T = pd.Series([0.5])
        r = pd.Series([0.03])
        q = pd.Series([0.01])
        sigma_true = pd.Series([0.25])
        right = pd.Series(["C"])

        # Generate price with known IV
        price = iv_solver._black_scholes_price(S, K, T, r, q, sigma_true, right)

        # Solve for IV
        iv_solved = iv_solver.solve_iv_vectorized(
            pd.Series(price, index=[0]), S, K, T, r, q, right
        )

        # Compute Greeks using solved IV
        greeks = greeks_calc.compute_greeks_vectorized(
            S, K, T, r, q, iv_solved, right
        )

        # Verify Greeks are reasonable
        assert 0 < greeks["delta"].iloc[0] < 1  # Call delta between 0 and 1
        assert greeks["gamma"].iloc[0] > 0  # Gamma positive
        assert greeks["vega"].iloc[0] > 0  # Vega positive
        assert greeks["theta"].iloc[0] < 0  # Theta negative for long option

    def test_greeks_reprice_consistency(self, iv_solver, greeks_calc):
        """Test that Greeks are consistent with repricing."""
        S = pd.Series([100.0])
        K = pd.Series([100.0])
        T = pd.Series([1.0])
        r = pd.Series([0.05])
        q = pd.Series([0.0])
        sigma = pd.Series([0.25])
        right = pd.Series(["C"])

        # Compute Greeks
        greeks = greeks_calc.compute_greeks_vectorized(S, K, T, r, q, sigma, right)

        # Compute base price
        price_base = iv_solver._black_scholes_price(S, K, T, r, q, sigma, right)[0]

        # Compute price with small spot bump
        dS = 1.0
        S_bump = pd.Series([S.iloc[0] + dS])
        price_bump = iv_solver._black_scholes_price(
            S_bump, K, T, r, q, sigma, right
        )[0]

        # Delta approximation
        delta_approx = (price_bump - price_base) / dS

        # Should match analytical delta (within numerical tolerance)
        assert np.abs(delta_approx - greeks["delta"].iloc[0]) < 0.01
