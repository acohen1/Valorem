"""Black-Scholes implied volatility solver using Newton-Raphson iteration.

This module implements a vectorized IV solver that works on pandas Series,
making it efficient for processing large option chains.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm


class BlackScholesIVSolver:
    """Black-Scholes implied volatility solver using Newton-Raphson.

    This solver uses the Newton-Raphson method to invert the Black-Scholes
    formula and solve for implied volatility given market prices.

    The algorithm:
    1. Start with an initial guess using Brenner-Subrahmanyam approximation
    2. Iterate: IV_new = IV_old - (BS_price - market_price) / vega
    3. Stop when |BS_price - market_price| < tolerance or max iterations reached

    Attributes:
        _max_iters: Maximum number of Newton-Raphson iterations
        _tolerance: Convergence tolerance for price difference
    """

    def __init__(self, max_iters: int = 100, tolerance: float = 1e-6):
        """Initialize IV solver.

        Args:
            max_iters: Maximum number of iterations before giving up
            tolerance: Convergence threshold for price difference

        Raises:
            ValueError: If max_iters or tolerance are invalid
        """
        if max_iters <= 0:
            raise ValueError(f"max_iters must be positive, got {max_iters}")
        if tolerance <= 0:
            raise ValueError(f"tolerance must be positive, got {tolerance}")

        self._max_iters = max_iters
        self._tolerance = tolerance

    def solve_iv_vectorized(
        self,
        prices: pd.Series,
        S: pd.Series,
        K: pd.Series,
        T: pd.Series,
        r: pd.Series | float,
        q: pd.Series | float = 0.0,
        right: pd.Series | None = None,
    ) -> pd.Series:
        """Solve for implied volatility using vectorized Newton-Raphson.

        Args:
            prices: Market prices (mid or last)
            S: Underlying spot prices
            K: Strike prices
            T: Time to expiration in years (annualized)
            r: Risk-free rate (annualized)
            q: Dividend yield (annualized), default 0.0
            right: Option type ('C' for call, 'P' for put). If None, infers from moneyness.

        Returns:
            Series of implied volatilities (annualized). NaN for failed convergence.

        Raises:
            ValueError: If input series have mismatched lengths
        """
        # Validate inputs
        if not (len(prices) == len(S) == len(K) == len(T)):
            raise ValueError("Input series must have same length")

        # Convert scalar r/q to Series
        if isinstance(r, (int, float)):
            r = pd.Series(r, index=prices.index)
        if isinstance(q, (int, float)):
            q = pd.Series(q, index=prices.index)

        # Infer option type if not provided
        if right is None:
            # Calls: S > K, Puts: S < K, ATM: use intrinsic value
            intrinsic_call = np.maximum(S - K, 0)
            intrinsic_put = np.maximum(K - S, 0)
            right = pd.Series(
                np.where(intrinsic_call > intrinsic_put, "C", "P"), index=prices.index
            )

        # Initial guess using improved Brenner-Subrahmanyam approximation
        # For ATM options: IV ≈ sqrt(2π/T) * (price / S)
        # Adjust for moneyness
        moneyness = S / K
        atm_iv = np.sqrt(2 * np.pi / np.maximum(T, 1e-10)) * (prices / np.maximum(S, 1e-10))

        # Better initial guess considering moneyness
        # Start with ATM approximation, adjust for deep ITM/OTM
        iv_guess = np.where(
            (moneyness > 0.8) & (moneyness < 1.2),  # Near ATM
            atm_iv,
            0.25,  # Default for far from ATM
        )
        iv_guess = np.clip(iv_guess, 0.05, 3.0)  # Reasonable bounds

        # Newton-Raphson iteration with per-element convergence tracking.
        # Freeze converged options to prevent precision degradation from
        # continued updates (especially when vega is near zero).
        converged = np.zeros(len(prices), dtype=bool)
        for _ in range(self._max_iters):
            bs_price = self._black_scholes_price(S, K, T, r, q, iv_guess, right)
            vega = self._vega(S, K, T, r, q, iv_guess)

            diff = bs_price - prices
            newly_converged = np.abs(diff) < self._tolerance
            converged = converged | newly_converged

            update = diff / np.maximum(vega, 1e-10)
            iv_guess = np.where(converged, iv_guess, iv_guess - update)

            if converged.all():
                break

        # Mask failed convergence or invalid values
        iv_guess = np.where(
            np.isfinite(iv_guess) & (iv_guess > 0) & (iv_guess < 10.0), iv_guess, np.nan
        )

        return pd.Series(iv_guess, index=prices.index, name="implied_vol")

    def _black_scholes_price(
        self,
        S: pd.Series | np.ndarray,
        K: pd.Series | np.ndarray,
        T: pd.Series | np.ndarray,
        r: pd.Series | np.ndarray,
        q: pd.Series | np.ndarray,
        sigma: pd.Series | np.ndarray,
        right: pd.Series | np.ndarray,
    ) -> np.ndarray:
        """Compute Black-Scholes option price (vectorized).

        Args:
            S: Underlying spot prices
            K: Strike prices
            T: Time to expiration in years
            r: Risk-free rate
            q: Dividend yield
            sigma: Volatility (annualized)
            right: Option type ('C' or 'P')

        Returns:
            Array of option prices
        """
        # Handle near-zero time to expiration
        T = np.maximum(T, 1e-10)
        sigma = np.maximum(sigma, 1e-10)

        # Compute d1 and d2
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        # Determine if call or put
        is_call = (right == "C") if isinstance(right, pd.Series) else (right == "C")

        # Compute call and put prices
        call_price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(
            d2
        )
        put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(
            -d1
        )

        # Select based on option type
        price = np.where(is_call, call_price, put_price)

        return price

    def _vega(
        self,
        S: pd.Series | np.ndarray,
        K: pd.Series | np.ndarray,
        T: pd.Series | np.ndarray,
        r: pd.Series | np.ndarray,
        q: pd.Series | np.ndarray,
        sigma: pd.Series | np.ndarray,
    ) -> np.ndarray:
        """Compute vega (∂price/∂σ) using Black-Scholes formula (vectorized).

        Vega is the same for calls and puts.

        Args:
            S: Underlying spot prices
            K: Strike prices
            T: Time to expiration in years
            r: Risk-free rate
            q: Dividend yield
            sigma: Volatility (annualized)

        Returns:
            Array of vega values (per 1.0 change in volatility)
        """
        # Handle near-zero time to expiration
        T = np.maximum(T, 1e-10)
        sigma = np.maximum(sigma, 1e-10)

        # Compute d1
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

        # Vega = S * exp(-qT) * N'(d1) * sqrt(T)
        vega = S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)

        return vega
