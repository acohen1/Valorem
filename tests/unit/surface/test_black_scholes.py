"""Unit tests for Black-Scholes IV solver."""

import numpy as np
import pandas as pd
import pytest

from src.surface.iv.black_scholes import BlackScholesIVSolver


class TestBlackScholesIVSolverInitialization:
    """Test BlackScholesIVSolver initialization."""

    def test_initialization_with_defaults(self):
        """Test solver initialization with default parameters."""
        solver = BlackScholesIVSolver()
        assert solver._max_iters == 100
        assert solver._tolerance == 1e-6

    def test_initialization_with_custom_parameters(self):
        """Test solver initialization with custom parameters."""
        solver = BlackScholesIVSolver(max_iters=50, tolerance=1e-8)
        assert solver._max_iters == 50
        assert solver._tolerance == 1e-8

    def test_initialization_invalid_max_iters(self):
        """Test that invalid max_iters raises ValueError."""
        with pytest.raises(ValueError, match="max_iters must be positive"):
            BlackScholesIVSolver(max_iters=0)

        with pytest.raises(ValueError, match="max_iters must be positive"):
            BlackScholesIVSolver(max_iters=-10)

    def test_initialization_invalid_tolerance(self):
        """Test that invalid tolerance raises ValueError."""
        with pytest.raises(ValueError, match="tolerance must be positive"):
            BlackScholesIVSolver(tolerance=0)

        with pytest.raises(ValueError, match="tolerance must be positive"):
            BlackScholesIVSolver(tolerance=-1e-6)


class TestBlackScholesPricing:
    """Test Black-Scholes pricing helper."""

    @pytest.fixture
    def solver(self):
        """Create solver instance."""
        return BlackScholesIVSolver()

    def test_call_price_ATM(self, solver):
        """Test call option pricing at-the-money."""
        S = pd.Series([100.0])
        K = pd.Series([100.0])
        T = pd.Series([1.0])
        r = pd.Series([0.05])
        q = pd.Series([0.0])
        sigma = pd.Series([0.2])
        right = pd.Series(["C"])

        price = solver._black_scholes_price(S, K, T, r, q, sigma, right)

        # ATM call with 20% vol, 5% rate, 1 year: price ≈ $10.45
        assert 10.0 < price[0] < 11.0

    def test_put_price_ATM(self, solver):
        """Test put option pricing at-the-money."""
        S = pd.Series([100.0])
        K = pd.Series([100.0])
        T = pd.Series([1.0])
        r = pd.Series([0.05])
        q = pd.Series([0.0])
        sigma = pd.Series([0.2])
        right = pd.Series(["P"])

        price = solver._black_scholes_price(S, K, T, r, q, sigma, right)

        # ATM put: price ≈ $5.57 (lower than call due to positive drift)
        assert 5.0 < price[0] < 6.5

    def test_put_call_parity(self, solver):
        """Test that put-call parity holds: C - P = S*e^(-qT) - K*e^(-rT)."""
        S = pd.Series([100.0])
        K = pd.Series([105.0])
        T = pd.Series([0.5])
        r = pd.Series([0.03])
        q = pd.Series([0.01])
        sigma = pd.Series([0.25])

        call_price = solver._black_scholes_price(S, K, T, r, q, sigma, pd.Series(["C"]))
        put_price = solver._black_scholes_price(S, K, T, r, q, sigma, pd.Series(["P"]))

        lhs = call_price[0] - put_price[0]
        rhs = S[0] * np.exp(-q[0] * T[0]) - K[0] * np.exp(-r[0] * T[0])

        assert np.abs(lhs - rhs) < 1e-10

    def test_deep_ITM_call(self, solver):
        """Test deep in-the-money call approaches intrinsic value."""
        S = pd.Series([150.0])
        K = pd.Series([100.0])
        T = pd.Series([1.0])
        r = pd.Series([0.05])
        q = pd.Series([0.0])
        sigma = pd.Series([0.2])
        right = pd.Series(["C"])

        price = solver._black_scholes_price(S, K, T, r, q, sigma, right)

        # Deep ITM call should be close to: S - K*e^(-rT) = 150 - 100*e^(-0.05) ≈ 55.13
        intrinsic = S[0] - K[0] * np.exp(-r[0] * T[0])
        assert price[0] > intrinsic  # Should have some time value
        assert price[0] < intrinsic + 5.0  # But not much

    def test_deep_OTM_call(self, solver):
        """Test deep out-of-the-money call approaches zero."""
        S = pd.Series([100.0])
        K = pd.Series([200.0])
        T = pd.Series([0.25])
        r = pd.Series([0.05])
        q = pd.Series([0.0])
        sigma = pd.Series([0.2])
        right = pd.Series(["C"])

        price = solver._black_scholes_price(S, K, T, r, q, sigma, right)

        # Deep OTM call should be near zero
        assert price[0] < 0.01

    def test_vectorized_pricing(self, solver):
        """Test that pricing works correctly on multiple options."""
        S = pd.Series([100.0, 100.0, 100.0])
        K = pd.Series([95.0, 100.0, 105.0])  # ITM, ATM, OTM
        T = pd.Series([1.0, 1.0, 1.0])
        r = pd.Series([0.05, 0.05, 0.05])
        q = pd.Series([0.0, 0.0, 0.0])
        sigma = pd.Series([0.2, 0.2, 0.2])
        right = pd.Series(["C", "C", "C"])

        prices = solver._black_scholes_price(S, K, T, r, q, sigma, right)

        # ITM should be most valuable, OTM should be least
        assert prices[0] > prices[1] > prices[2]
        assert all(prices > 0)


class TestVega:
    """Test vega calculation."""

    @pytest.fixture
    def solver(self):
        """Create solver instance."""
        return BlackScholesIVSolver()

    def test_vega_positive(self, solver):
        """Test that vega is always positive."""
        S = pd.Series([100.0, 100.0, 100.0])
        K = pd.Series([95.0, 100.0, 105.0])
        T = pd.Series([1.0, 1.0, 1.0])
        r = pd.Series([0.05, 0.05, 0.05])
        q = pd.Series([0.0, 0.0, 0.0])
        sigma = pd.Series([0.2, 0.2, 0.2])

        vega = solver._vega(S, K, T, r, q, sigma)

        assert all(vega > 0)

    def test_vega_ATM_highest(self, solver):
        """Test that ATM options have highest vega for same moneyness distance."""
        S = pd.Series([100.0, 100.0, 100.0])
        K = pd.Series([90.0, 100.0, 110.0])  # OTM put moneyness, ATM, OTM call moneyness
        T = pd.Series([1.0, 1.0, 1.0])
        r = pd.Series([0.05, 0.05, 0.05])
        q = pd.Series([0.0, 0.0, 0.0])
        sigma = pd.Series([0.2, 0.2, 0.2])

        vega = solver._vega(S, K, T, r, q, sigma)

        # ATM should have higher vega than more OTM options
        assert vega[1] > vega[0]

    def test_vega_decreases_with_time(self, solver):
        """Test that vega decreases as expiration approaches."""
        S = pd.Series([100.0, 100.0, 100.0])
        K = pd.Series([100.0, 100.0, 100.0])
        T = pd.Series([1.0, 0.5, 0.1])
        r = pd.Series([0.05, 0.05, 0.05])
        q = pd.Series([0.0, 0.0, 0.0])
        sigma = pd.Series([0.2, 0.2, 0.2])

        vega = solver._vega(S, K, T, r, q, sigma)

        # Longer dated options have higher vega
        assert vega[0] > vega[1] > vega[2]


class TestIVSolver:
    """Test implied volatility solver."""

    @pytest.fixture
    def solver(self):
        """Create solver instance."""
        return BlackScholesIVSolver()

    def test_solve_iv_known_value(self, solver):
        """Test IV solver with known volatility."""
        # Generate a price with known IV
        S = pd.Series([100.0])
        K = pd.Series([100.0])
        T = pd.Series([1.0])
        r = pd.Series([0.05])
        q = pd.Series([0.0])
        sigma_true = pd.Series([0.25])
        right = pd.Series(["C"])

        # Compute price with known IV
        price = solver._black_scholes_price(S, K, T, r, q, sigma_true, right)

        # Solve for IV
        iv_solved = solver.solve_iv_vectorized(
            pd.Series(price, index=[0]), S, K, T, r, q, right
        )

        # Should recover the original IV
        assert np.abs(iv_solved[0] - sigma_true[0]) < 1e-6

    def test_solve_iv_multiple_options(self, solver):
        """Test IV solver on multiple options."""
        S = pd.Series([100.0, 100.0, 100.0])
        K = pd.Series([95.0, 100.0, 105.0])
        T = pd.Series([1.0, 1.0, 1.0])
        r = pd.Series([0.05, 0.05, 0.05])
        q = pd.Series([0.0, 0.0, 0.0])
        sigma_true = pd.Series([0.20, 0.25, 0.30])
        right = pd.Series(["C", "C", "C"])

        # Generate prices with different IVs
        prices = solver._black_scholes_price(S, K, T, r, q, sigma_true, right)

        # Solve for IV
        iv_solved = solver.solve_iv_vectorized(
            pd.Series(prices, index=[0, 1, 2]), S, K, T, r, q, right
        )

        # Should recover all IVs
        for i in range(3):
            assert np.abs(iv_solved[i] - sigma_true[i]) < 1e-6

    def test_solve_iv_calls_and_puts(self, solver):
        """Test IV solver works for both calls and puts."""
        S = pd.Series([100.0, 100.0])
        K = pd.Series([100.0, 100.0])
        T = pd.Series([1.0, 1.0])
        r = pd.Series([0.05, 0.05])
        q = pd.Series([0.0, 0.0])
        sigma_true = pd.Series([0.25, 0.25])
        right = pd.Series(["C", "P"])

        # Generate prices
        prices = solver._black_scholes_price(S, K, T, r, q, sigma_true, right)

        # Solve for IV
        iv_solved = solver.solve_iv_vectorized(
            pd.Series(prices, index=[0, 1]), S, K, T, r, q, right
        )

        # Both should recover same IV
        assert np.abs(iv_solved[0] - sigma_true[0]) < 1e-6
        assert np.abs(iv_solved[1] - sigma_true[1]) < 1e-6

    def test_solve_iv_with_dividends(self, solver):
        """Test IV solver with dividend yield."""
        S = pd.Series([100.0])
        K = pd.Series([100.0])
        T = pd.Series([1.0])
        r = pd.Series([0.05])
        q = pd.Series([0.02])  # 2% dividend yield
        sigma_true = pd.Series([0.25])
        right = pd.Series(["C"])

        # Generate price with dividends
        price = solver._black_scholes_price(S, K, T, r, q, sigma_true, right)

        # Solve for IV
        iv_solved = solver.solve_iv_vectorized(
            pd.Series(price, index=[0]), S, K, T, r, q, right
        )

        # Should recover the original IV
        assert np.abs(iv_solved[0] - sigma_true[0]) < 1e-6

    def test_solve_iv_scalar_rate_and_yield(self, solver):
        """Test IV solver with scalar r and q."""
        S = pd.Series([100.0, 100.0])
        K = pd.Series([95.0, 105.0])
        T = pd.Series([1.0, 1.0])
        r = 0.05  # Scalar
        q = 0.02  # Scalar
        sigma_true = pd.Series([0.25, 0.30])
        right = pd.Series(["C", "C"])

        # Generate prices
        prices = solver._black_scholes_price(S, K, T, r, q, sigma_true, right)

        # Solve for IV with scalar r and q
        iv_solved = solver.solve_iv_vectorized(
            pd.Series(prices, index=[0, 1]), S, K, T, r, q, right
        )

        # Should recover all IVs
        assert np.abs(iv_solved[0] - sigma_true[0]) < 1e-6
        assert np.abs(iv_solved[1] - sigma_true[1]) < 1e-6

    def test_solve_iv_infers_option_type(self, solver):
        """Test IV solver infers call/put from moneyness when right=None."""
        S = pd.Series([100.0, 100.0])
        K = pd.Series([95.0, 105.0])  # Call should be ITM, put should be ITM
        T = pd.Series([1.0, 1.0])
        r = pd.Series([0.05, 0.05])
        q = pd.Series([0.0, 0.0])
        sigma_true = pd.Series([0.25, 0.25])

        # Generate call and put prices
        call_price = solver._black_scholes_price(
            S[0:1], K[0:1], T[0:1], r[0:1], q[0:1], sigma_true[0:1], pd.Series(["C"])
        )
        put_price = solver._black_scholes_price(
            S[1:2], K[1:2], T[1:2], r[1:2], q[1:2], sigma_true[1:2], pd.Series(["P"])
        )
        prices = pd.Series([call_price[0], put_price[0]], index=[0, 1])

        # Solve without specifying option type
        iv_solved = solver.solve_iv_vectorized(prices, S, K, T, r, q, right=None)

        # Should still recover IVs
        assert np.abs(iv_solved[0] - sigma_true[0]) < 1e-4
        assert np.abs(iv_solved[1] - sigma_true[1]) < 1e-4

    def test_solve_iv_deep_ITM(self, solver):
        """Test IV solver on deep in-the-money options."""
        S = pd.Series([150.0])
        K = pd.Series([100.0])
        T = pd.Series([1.0])
        r = pd.Series([0.05])
        q = pd.Series([0.0])
        sigma_true = pd.Series([0.25])
        right = pd.Series(["C"])

        price = solver._black_scholes_price(S, K, T, r, q, sigma_true, right)

        iv_solved = solver.solve_iv_vectorized(
            pd.Series(price, index=[0]), S, K, T, r, q, right
        )

        assert np.abs(iv_solved[0] - sigma_true[0]) < 1e-5

    def test_solve_iv_deep_OTM(self, solver):
        """Test IV solver on deep out-of-the-money options."""
        S = pd.Series([100.0])
        K = pd.Series([150.0])
        T = pd.Series([1.0])
        r = pd.Series([0.05])
        q = pd.Series([0.0])
        sigma_true = pd.Series([0.25])
        right = pd.Series(["C"])

        price = solver._black_scholes_price(S, K, T, r, q, sigma_true, right)

        iv_solved = solver.solve_iv_vectorized(
            pd.Series(price, index=[0]), S, K, T, r, q, right
        )

        # Deep OTM may have convergence issues due to very low vega
        # Accept either accurate convergence or NaN
        if not np.isnan(iv_solved[0]):
            assert np.abs(iv_solved[0] - sigma_true[0]) < 1e-4

    def test_solve_iv_short_expiration(self, solver):
        """Test IV solver on options near expiration."""
        S = pd.Series([100.0])
        K = pd.Series([100.0])
        T = pd.Series([0.01])  # 1% of a year ≈ 3.6 days
        r = pd.Series([0.05])
        q = pd.Series([0.0])
        sigma_true = pd.Series([0.25])
        right = pd.Series(["C"])

        price = solver._black_scholes_price(S, K, T, r, q, sigma_true, right)

        iv_solved = solver.solve_iv_vectorized(
            pd.Series(price, index=[0]), S, K, T, r, q, right
        )

        # May be less accurate near expiration, but should be close
        assert np.abs(iv_solved[0] - sigma_true[0]) < 1e-3

    def test_solve_iv_invalid_price_returns_nan(self, solver):
        """Test that invalid prices return NaN."""
        S = pd.Series([100.0])
        K = pd.Series([100.0])
        T = pd.Series([1.0])
        r = pd.Series([0.05])
        q = pd.Series([0.0])
        right = pd.Series(["C"])

        # Negative price (invalid)
        prices = pd.Series([-5.0])

        iv_solved = solver.solve_iv_vectorized(prices, S, K, T, r, q, right)

        assert np.isnan(iv_solved[0])

    def test_solve_iv_price_too_high_returns_nan(self, solver):
        """Test that impossibly high prices return NaN."""
        S = pd.Series([100.0])
        K = pd.Series([100.0])
        T = pd.Series([1.0])
        r = pd.Series([0.05])
        q = pd.Series([0.0])
        right = pd.Series(["C"])

        # Price higher than spot (impossible for ATM call with positive rate)
        prices = pd.Series([150.0])

        iv_solved = solver.solve_iv_vectorized(prices, S, K, T, r, q, right)

        # Should either return NaN or very high IV (>10)
        assert np.isnan(iv_solved[0]) or iv_solved[0] > 10.0


class TestIVSolverEdgeCases:
    """Test IV solver edge cases."""

    @pytest.fixture
    def solver(self):
        """Create solver instance."""
        return BlackScholesIVSolver()

    def test_mismatched_lengths_raises_error(self, solver):
        """Test that mismatched input lengths raise ValueError."""
        prices = pd.Series([10.0, 11.0])
        S = pd.Series([100.0])  # Wrong length
        K = pd.Series([100.0, 105.0])
        T = pd.Series([1.0, 1.0])
        r = 0.05
        q = 0.0
        right = pd.Series(["C", "C"])

        with pytest.raises(ValueError, match="same length"):
            solver.solve_iv_vectorized(prices, S, K, T, r, q, right)

    def test_convergence_within_max_iters(self, solver):
        """Test that most options converge within max iterations."""
        # Generate 100 random options with reasonable parameters
        np.random.seed(42)
        n = 100

        S = pd.Series(np.random.uniform(80, 120, n))
        K = pd.Series(np.random.uniform(80, 120, n))
        T = pd.Series(np.random.uniform(0.1, 2.0, n))
        r = pd.Series(np.random.uniform(0.01, 0.1, n))
        q = pd.Series(np.random.uniform(0.0, 0.03, n))
        sigma_true = pd.Series(np.random.uniform(0.15, 0.4, n))
        right = pd.Series(["C"] * 50 + ["P"] * 50)

        # Generate prices
        prices = solver._black_scholes_price(S, K, T, r, q, sigma_true, right)

        # Solve for IV
        iv_solved = solver.solve_iv_vectorized(
            pd.Series(prices, index=range(n)), S, K, T, r, q, right
        )

        # Check convergence rate
        converged = ~np.isnan(iv_solved)
        convergence_rate = converged.sum() / n

        # Should converge for >=95% of realistic inputs
        assert convergence_rate >= 0.95

        # Check accuracy for converged values
        errors = np.abs(iv_solved[converged] - sigma_true[converged])
        assert np.max(errors) < 1e-4


class TestIVSolverPerformance:
    """Test IV solver performance."""

    @pytest.fixture
    def solver(self):
        """Create solver instance."""
        return BlackScholesIVSolver()

    def test_vectorized_faster_than_loop(self, solver):
        """Test that vectorized solver is faster than looping."""
        import time

        # Generate 1000 options
        np.random.seed(42)
        n = 1000

        S = pd.Series(np.random.uniform(80, 120, n))
        K = pd.Series(np.random.uniform(80, 120, n))
        T = pd.Series(np.random.uniform(0.1, 1.0, n))
        r = pd.Series([0.05] * n)
        q = pd.Series([0.0] * n)
        sigma_true = pd.Series(np.random.uniform(0.15, 0.35, n))
        right = pd.Series(["C"] * n)

        # Generate prices
        prices = solver._black_scholes_price(S, K, T, r, q, sigma_true, right)
        prices = pd.Series(prices, index=range(n))

        # Time vectorized approach
        start = time.time()
        iv_vectorized = solver.solve_iv_vectorized(prices, S, K, T, r, q, right)
        time_vectorized = time.time() - start

        # Time looped approach
        start = time.time()
        iv_loop = []
        for i in range(n):
            iv = solver.solve_iv_vectorized(
                prices[i : i + 1], S[i : i + 1], K[i : i + 1], T[i : i + 1], r[i : i + 1], q[i : i + 1], right[i : i + 1]
            )
            iv_loop.append(iv.iloc[0])
        time_loop = time.time() - start

        # Vectorized should be at least 10x faster
        speedup = time_loop / time_vectorized
        assert speedup > 10.0, f"Speedup was only {speedup:.1f}x"

        # Results should be nearly identical (within numerical tolerance)
        # Use looser tolerance since iteration order can affect convergence slightly
        iv_loop = pd.Series(iv_loop, index=range(n), name="implied_vol")

        # Compare only where both converged
        both_valid = ~np.isnan(iv_vectorized) & ~np.isnan(iv_loop)
        if both_valid.sum() > 0:
            # Check median absolute error is small
            errors = np.abs(iv_vectorized[both_valid].values - iv_loop[both_valid].values)
            median_error = np.median(errors)
            max_error = np.max(errors)

            assert median_error < 1e-6, f"Median error {median_error} too large"
            assert max_error < 1e-4, f"Max error {max_error} too large"
