"""Marchenko-Pastur covariance cleaning for portfolio risk.

Computes a cleaned covariance matrix across volatility surface nodes
using Random Matrix Theory. Eigenvalues that fall within the
Marchenko-Pastur noise band are shrunk or clipped, leaving only
statistically significant covariance structure.

Usage:
    estimator = CovarianceEstimator(config)
    result = estimator.update(returns_panel)
    # result.covariance is the (p, p) cleaned matrix
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from ..config.schema import CovarianceConfig
from ..exceptions import InsufficientDataError

if TYPE_CHECKING:
    from ..strategy.types import Signal

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CovarianceResult:
    """Output of a covariance cleaning pass."""

    covariance: np.ndarray
    """(p, p) cleaned covariance matrix."""

    correlation: np.ndarray
    """(p, p) cleaned correlation matrix."""

    eigenvalues_raw: np.ndarray
    """(p,) eigenvalues of the sample correlation matrix."""

    eigenvalues_cleaned: np.ndarray
    """(p,) eigenvalues after MP cleaning."""

    mp_lambda_plus: float
    """Upper Marchenko-Pastur bound."""

    mp_lambda_minus: float
    """Lower Marchenko-Pastur bound."""

    noise_eigenvalue_count: int
    signal_eigenvalue_count: int

    timestamp: datetime
    """When this estimate was produced."""

    observations_used: int
    """n (rows) used in gamma = p / n."""

    node_order: list[tuple[int, str]] = field(default_factory=list)
    """Index i maps to (tenor_days, delta_bucket)."""


# ---------------------------------------------------------------------------
# Estimator
# ---------------------------------------------------------------------------

class CovarianceEstimator:
    """Compute and cache a cleaned covariance matrix from node returns."""

    def __init__(self, config: CovarianceConfig) -> None:
        self._config = config
        self._latest: CovarianceResult | None = None

    # -- public API ---------------------------------------------------------

    def update(self, returns_panel: pd.DataFrame) -> CovarianceResult:
        """Recompute the cleaned covariance matrix.

        Args:
            returns_panel: Long-format DataFrame with columns
                [ts_utc, tenor_days, delta_bucket, <returns_source>].

        Returns:
            CovarianceResult with cleaned matrices.

        Raises:
            InsufficientDataError: If fewer than ``min_observations`` rows
                remain after pivoting.
        """
        col = self._config.returns_source
        required = {"ts_utc", "tenor_days", "delta_bucket", col}
        missing = required - set(returns_panel.columns)
        if missing:
            raise ValueError(f"returns_panel missing columns: {missing}")

        X, node_order = self._pivot_returns(returns_panel, col)

        p = X.shape[1]
        n = X.shape[0]

        if n < self._config.min_observations:
            raise InsufficientDataError(
                f"Only {n} observations available, need {self._config.min_observations}"
            )

        # Trim to rolling window
        if n > self._config.window:
            X = X[-self._config.window:]
            n = X.shape[0]

        if self._config.method == "marchenko_pastur":
            result = self._run_marchenko_pastur(X, n, p, node_order)
        else:
            result = self._run_ledoit_wolf(X, n, p, node_order)

        self._latest = result
        return result

    def get_latest(self) -> CovarianceResult | None:
        """Return the most recently computed result, or None."""
        return self._latest

    def correlation_between(
        self,
        node_a: tuple[int, str],
        node_b: tuple[int, str],
    ) -> float:
        """Return pairwise cleaned correlation between two nodes."""
        if self._latest is None:
            raise RuntimeError("No covariance result available; call update() first")
        idx = {n: i for i, n in enumerate(self._latest.node_order)}
        i, j = idx[node_a], idx[node_b]
        return float(self._latest.correlation[i, j])

    def portfolio_variance(self, weights: np.ndarray) -> float:
        """Compute w' Sigma w."""
        if self._latest is None:
            raise RuntimeError("No covariance result available; call update() first")
        return float(weights @ self._latest.covariance @ weights)

    def marginal_contribution(self, weights: np.ndarray) -> np.ndarray:
        """Compute marginal risk contribution per node.

        Returns Sigma @ w / sqrt(w' Sigma w).
        """
        if self._latest is None:
            raise RuntimeError("No covariance result available; call update() first")
        sigma_w = self._latest.covariance @ weights
        port_vol = np.sqrt(float(weights @ sigma_w))
        if port_vol < 1e-12:
            return np.zeros_like(weights)
        return sigma_w / port_vol

    # -- pivot --------------------------------------------------------------

    @staticmethod
    def _pivot_returns(
        df: pd.DataFrame, col: str
    ) -> tuple[np.ndarray, list[tuple[int, str]]]:
        """Pivot long-format returns to (n_obs, p) matrix.

        Returns:
            X: (n, p) array of returns (NaN-free rows only).
            node_order: List mapping column index to (tenor_days, delta_bucket).
        """
        wide = df.pivot_table(
            index="ts_utc",
            columns=["tenor_days", "delta_bucket"],
            values=col,
            aggfunc="last",
        ).sort_index()

        node_order = [(int(t), str(b)) for t, b in wide.columns]

        # Drop rows with any NaN
        pre_drop = len(wide)
        wide = wide.dropna()
        dropped = pre_drop - len(wide)
        if dropped > 0:
            pct = dropped / pre_drop * 100
            logger.warning(
                "Dropped %d / %d rows (%.1f%%) with NaN before covariance estimation",
                dropped, pre_drop, pct,
            )
            if pct > 20:
                logger.warning(
                    "More than 20%% of rows dropped — covariance estimate may be unreliable"
                )

        X = wide.values.astype(np.float64)
        return X, node_order

    # -- Marchenko-Pastur ---------------------------------------------------

    def _run_marchenko_pastur(
        self,
        X: np.ndarray,
        n: int,
        p: int,
        node_order: list[tuple[int, str]],
    ) -> CovarianceResult:
        sample_cov = np.cov(X, rowvar=False, ddof=1)

        # Standardize to correlation
        stds = np.sqrt(np.diag(sample_cov))
        stds = np.where(stds < 1e-12, 1e-12, stds)  # guard zero-variance nodes
        D_inv = np.diag(1.0 / stds)
        corr = D_inv @ sample_cov @ D_inv

        eigenvalues_raw, eigenvectors = np.linalg.eigh(corr)
        eigenvalues_raw = eigenvalues_raw.real

        # MP bounds
        gamma = p / n
        lambda_plus = (1.0 + np.sqrt(gamma)) ** 2
        lambda_minus = (1.0 - np.sqrt(gamma)) ** 2

        # Classify
        noise_mask = (eigenvalues_raw >= lambda_minus) & (eigenvalues_raw <= lambda_plus)
        noise_count = int(noise_mask.sum())
        signal_count = p - noise_count

        eigenvalues_cleaned = eigenvalues_raw.copy()
        method = self._config.mp_eigenvalue_method

        if noise_count > 0:
            noise_mean = eigenvalues_raw[noise_mask].mean()

            if method == "clip_to_mean":
                eigenvalues_cleaned[noise_mask] = noise_mean
            elif method == "clip_to_zero":
                eigenvalues_cleaned[noise_mask] = 0.0
            elif method == "shrink_toward_identity":
                eigenvalues_cleaned[noise_mask] = (
                    0.5 * 1.0 + 0.5 * noise_mean
                )

        # If everything is noise and fallback enabled, use Ledoit-Wolf
        if signal_count == 0 and self._config.fallback_to_ledoit_wolf:
            logger.warning(
                "All %d eigenvalues in noise band — falling back to Ledoit-Wolf", p
            )
            return self._run_ledoit_wolf(X, n, p, node_order)

        # Ensure PSD
        eigenvalues_cleaned = np.maximum(eigenvalues_cleaned, 1e-10)

        # Reconstruct correlation
        corr_cleaned = (
            eigenvectors @ np.diag(eigenvalues_cleaned) @ eigenvectors.T
        )

        # Regularize if condition number too high
        cond = eigenvalues_cleaned.max() / eigenvalues_cleaned.min()
        if cond > self._config.condition_number_cap:
            ridge = eigenvalues_cleaned.max() / self._config.condition_number_cap
            corr_cleaned += ridge * np.eye(p)
            logger.debug(
                "Added ridge %.4e to cap condition number at %.0f",
                ridge, self._config.condition_number_cap,
            )

        # Force unit diagonal (numerical drift from ridge)
        d = np.sqrt(np.diag(corr_cleaned))
        d = np.where(d < 1e-12, 1e-12, d)
        corr_cleaned = corr_cleaned / np.outer(d, d)

        # De-standardize to covariance
        D = np.diag(stds)
        cov_cleaned = D @ corr_cleaned @ D

        return CovarianceResult(
            covariance=cov_cleaned,
            correlation=corr_cleaned,
            eigenvalues_raw=eigenvalues_raw,
            eigenvalues_cleaned=eigenvalues_cleaned,
            mp_lambda_plus=float(lambda_plus),
            mp_lambda_minus=float(lambda_minus),
            noise_eigenvalue_count=noise_count,
            signal_eigenvalue_count=signal_count,
            timestamp=datetime.utcnow(),
            observations_used=n,
            node_order=node_order,
        )

    # -- Ledoit-Wolf --------------------------------------------------------

    def _run_ledoit_wolf(
        self,
        X: np.ndarray,
        n: int,
        p: int,
        node_order: list[tuple[int, str]],
    ) -> CovarianceResult:
        from sklearn.covariance import LedoitWolf

        lw = LedoitWolf().fit(X)
        cov_cleaned = lw.covariance_

        stds = np.sqrt(np.diag(cov_cleaned))
        stds = np.where(stds < 1e-12, 1e-12, stds)
        corr_cleaned = cov_cleaned / np.outer(stds, stds)

        # Eigenvalues for diagnostics (of correlation)
        eigenvalues_raw = np.linalg.eigvalsh(np.cov(X, rowvar=False, ddof=1))
        eigenvalues_cleaned = np.linalg.eigvalsh(corr_cleaned)

        gamma = p / n
        lambda_plus = (1.0 + np.sqrt(gamma)) ** 2
        lambda_minus = (1.0 - np.sqrt(gamma)) ** 2

        return CovarianceResult(
            covariance=cov_cleaned,
            correlation=corr_cleaned,
            eigenvalues_raw=eigenvalues_raw,
            eigenvalues_cleaned=eigenvalues_cleaned,
            mp_lambda_plus=float(lambda_plus),
            mp_lambda_minus=float(lambda_minus),
            noise_eigenvalue_count=0,
            signal_eigenvalue_count=p,
            timestamp=datetime.utcnow(),
            observations_used=n,
            node_order=node_order,
        )


# ---------------------------------------------------------------------------
# Signal diversity filter
# ---------------------------------------------------------------------------

def filter_correlated_signals(
    signals: list[Signal],
    covariance_result: CovarianceResult,
    max_correlation: float = 0.85,
) -> list[Signal]:
    """Greedy selection that drops redundant correlated signals.

    Signals are sorted by ``|edge| * confidence`` descending (best first).
    Each candidate is accepted only if its maximum absolute cleaned
    correlation with all previously-accepted signals is below
    ``max_correlation``.

    Args:
        signals: Candidate signals to filter.
        covariance_result: Cleaned covariance result from CovarianceEstimator.
        max_correlation: Threshold above which a signal is deemed redundant.

    Returns:
        Filtered list of signals preserving input order among accepted ones.
    """
    if not signals or covariance_result.correlation is None:
        return list(signals)

    # Build node -> matrix index lookup
    idx_map: dict[tuple[int, str], int] = {
        n: i for i, n in enumerate(covariance_result.node_order)
    }
    corr = covariance_result.correlation

    # Sort by score descending
    scored = sorted(
        signals,
        key=lambda s: abs(s.edge) * s.confidence,
        reverse=True,
    )

    accepted: list[Signal] = []
    accepted_indices: list[int] = []

    for sig in scored:
        node = (sig.tenor_days, sig.delta_bucket)
        node_idx = idx_map.get(node)
        if node_idx is None:
            # Node not in covariance matrix — accept by default
            accepted.append(sig)
            continue

        # Check correlation with all accepted signals
        redundant = False
        for ai in accepted_indices:
            if abs(float(corr[node_idx, ai])) >= max_correlation:
                redundant = True
                break

        if not redundant:
            accepted.append(sig)
            accepted_indices.append(node_idx)

    filtered_count = len(signals) - len(accepted)
    if filtered_count > 0:
        logger.info(
            "Correlation filter: %d / %d signals removed (max_corr=%.2f)",
            filtered_count, len(signals), max_correlation,
        )

    return accepted
