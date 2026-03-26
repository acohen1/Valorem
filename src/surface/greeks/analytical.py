"""Analytical Greeks calculator using Black-Scholes formulas.

This module computes option Greeks (delta, gamma, vega, theta) using
closed-form Black-Scholes formulas for European options.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm


class AnalyticalGreeks:
    """Analytical Greeks calculator using Black-Scholes formulas.

    This class computes option sensitivities (Greeks) using analytical
    Black-Scholes formulas. All computations are vectorized for efficiency.

    Greeks computed:
    - Delta: ∂V/∂S (sensitivity to underlying price)
    - Gamma: ∂²V/∂S² (rate of change of delta)
    - Vega: ∂V/∂σ (sensitivity to volatility)
    - Theta: ∂V/∂t (time decay)

    Note: Gamma and vega are the same for calls and puts with same parameters.
    """

    def compute_greeks_vectorized(
        self,
        S: pd.Series,
        K: pd.Series,
        T: pd.Series,
        r: pd.Series | float,
        q: pd.Series | float,
        sigma: pd.Series,
        right: pd.Series,
    ) -> pd.DataFrame:
        """Compute all Greeks in one vectorized pass.

        Args:
            S: Underlying spot prices
            K: Strike prices
            T: Time to expiration in years (annualized)
            r: Risk-free rate (annualized)
            q: Dividend yield (annualized)
            sigma: Implied volatility (annualized)
            right: Option type ('C' for call, 'P' for put)

        Returns:
            DataFrame with columns [delta, gamma, vega, theta]
            Index matches input series

        Raises:
            ValueError: If input series have mismatched lengths
        """
        # Validate inputs
        if not (len(S) == len(K) == len(T) == len(sigma) == len(right)):
            raise ValueError("Input series must have same length")

        # Convert scalar r/q to Series
        if isinstance(r, (int, float)):
            r = pd.Series(r, index=S.index)
        if isinstance(q, (int, float)):
            q = pd.Series(q, index=S.index)

        # Handle near-zero time to expiration
        T = np.maximum(T, 1e-10)
        sigma = np.maximum(sigma, 1e-10)

        # Compute d1 and d2
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        # Determine if call or put
        is_call = (right == "C")

        # Delta: ∂V/∂S
        # Call: N(d1) * e^(-qT)
        # Put: -N(-d1) * e^(-qT) = [N(d1) - 1] * e^(-qT)
        delta = np.where(
            is_call,
            np.exp(-q * T) * norm.cdf(d1),
            -np.exp(-q * T) * norm.cdf(-d1),
        )

        # Gamma: ∂²V/∂S² (same for calls and puts)
        # Gamma = N'(d1) * e^(-qT) / (S * σ * √T)
        gamma = np.exp(-q * T) * norm.pdf(d1) / np.maximum(S * sigma * np.sqrt(T), 1e-10)

        # Vega: ∂V/∂σ (same for calls and puts, per 1.0 vol change)
        # Vega = S * N'(d1) * √T * e^(-qT)
        vega = S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)

        # Theta: ∂V/∂t (per year, so divide by 365 for per-day)
        # Call: -S*N'(d1)*σ*e^(-qT)/(2√T) - rK*e^(-rT)*N(d2) + qS*e^(-qT)*N(d1)
        # Put: -S*N'(d1)*σ*e^(-qT)/(2√T) + rK*e^(-rT)*N(-d2) - qS*e^(-qT)*N(-d1)
        theta_call = (
            -S * norm.pdf(d1) * sigma * np.exp(-q * T) / (2 * np.sqrt(T))
            - r * K * np.exp(-r * T) * norm.cdf(d2)
            + q * S * np.exp(-q * T) * norm.cdf(d1)
        )
        theta_put = (
            -S * norm.pdf(d1) * sigma * np.exp(-q * T) / (2 * np.sqrt(T))
            + r * K * np.exp(-r * T) * norm.cdf(-d2)
            - q * S * np.exp(-q * T) * norm.cdf(-d1)
        )
        theta = np.where(is_call, theta_call, theta_put)

        return pd.DataFrame(
            {
                "delta": delta,
                "gamma": gamma,
                "vega": vega,
                "theta": theta,
            },
            index=S.index,
        )

    def compute_delta(
        self,
        S: pd.Series,
        K: pd.Series,
        T: pd.Series,
        r: pd.Series | float,
        q: pd.Series | float,
        sigma: pd.Series,
        right: pd.Series,
    ) -> pd.Series:
        """Compute delta only (for performance when other Greeks not needed).

        Args:
            S: Underlying spot prices
            K: Strike prices
            T: Time to expiration in years
            r: Risk-free rate
            q: Dividend yield
            sigma: Implied volatility
            right: Option type ('C' or 'P')

        Returns:
            Series of delta values
        """
        greeks = self.compute_greeks_vectorized(S, K, T, r, q, sigma, right)
        return greeks["delta"]

    def compute_gamma(
        self,
        S: pd.Series,
        K: pd.Series,
        T: pd.Series,
        r: pd.Series | float,
        q: pd.Series | float,
        sigma: pd.Series,
        right: pd.Series,
    ) -> pd.Series:
        """Compute gamma only (for performance when other Greeks not needed).

        Args:
            S: Underlying spot prices
            K: Strike prices
            T: Time to expiration in years
            r: Risk-free rate
            q: Dividend yield
            sigma: Implied volatility
            right: Option type ('C' or 'P')

        Returns:
            Series of gamma values
        """
        greeks = self.compute_greeks_vectorized(S, K, T, r, q, sigma, right)
        return greeks["gamma"]
