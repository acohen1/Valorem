"""Unit tests for Marchenko-Pastur covariance cleaning."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, UTC

import numpy as np
import pandas as pd
import pytest

from src.config.schema import CovarianceConfig
from src.exceptions import InsufficientDataError
from src.risk.covariance import CovarianceEstimator, CovarianceResult, filter_correlated_signals


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TENORS = [7, 14, 30, 60, 90, 120]
BUCKETS = ["P10", "P25", "P40", "ATM", "C40", "C25", "C10"]
NUM_NODES = len(TENORS) * len(BUCKETS)  # 42


def _make_returns_panel(
    n_obs: int = 300,
    n_signal_factors: int = 3,
    signal_strength: float = 5.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Build a synthetic long-format returns panel.

    Embeds ``n_signal_factors`` strong factors into otherwise random noise
    so that MP cleaning should find exactly that many signal eigenvalues.
    """
    rng = np.random.default_rng(seed)
    p = NUM_NODES

    # Pure noise
    noise = rng.standard_normal((n_obs, p))

    # Inject signal factors
    factors = rng.standard_normal((n_obs, n_signal_factors))
    loadings = rng.standard_normal((n_signal_factors, p)) * signal_strength
    X = noise + factors @ loadings

    # Build long-format DataFrame
    dates = pd.date_range("2020-01-01", periods=n_obs, freq="B")
    rows = []
    col_idx = 0
    for tenor in TENORS:
        for bucket in BUCKETS:
            for t_idx, dt in enumerate(dates):
                rows.append({
                    "ts_utc": dt,
                    "tenor_days": tenor,
                    "delta_bucket": bucket,
                    "iv_change_1d": X[t_idx, col_idx],
                })
            col_idx += 1

    return pd.DataFrame(rows)


def _default_config(**overrides) -> CovarianceConfig:
    return CovarianceConfig(**overrides)


# ---------------------------------------------------------------------------
# CovarianceEstimator tests
# ---------------------------------------------------------------------------

class TestCovarianceEstimator:
    """Tests for the core estimator."""

    def test_update_returns_result(self):
        est = CovarianceEstimator(_default_config())
        panel = _make_returns_panel()
        result = est.update(panel)

        assert isinstance(result, CovarianceResult)
        assert result.covariance.shape == (NUM_NODES, NUM_NODES)
        assert result.correlation.shape == (NUM_NODES, NUM_NODES)
        assert len(result.node_order) == NUM_NODES

    def test_cleaned_matrix_is_symmetric(self):
        est = CovarianceEstimator(_default_config())
        result = est.update(_make_returns_panel())

        np.testing.assert_allclose(
            result.covariance, result.covariance.T, atol=1e-10
        )
        np.testing.assert_allclose(
            result.correlation, result.correlation.T, atol=1e-10
        )

    def test_cleaned_matrix_is_psd(self):
        est = CovarianceEstimator(_default_config())
        result = est.update(_make_returns_panel())

        eigenvalues = np.linalg.eigvalsh(result.covariance)
        assert np.all(eigenvalues >= -1e-10), f"Negative eigenvalue: {eigenvalues.min()}"

    def test_correlation_diagonal_is_one(self):
        est = CovarianceEstimator(_default_config())
        result = est.update(_make_returns_panel())

        np.testing.assert_allclose(np.diag(result.correlation), 1.0, atol=1e-6)

    def test_noise_eigenvalues_detected(self):
        """With weak signal factors, most eigenvalues should fall in the noise band."""
        # Use low signal strength so noise eigenvalues stay inside [lambda-, lambda+]
        est = CovarianceEstimator(_default_config())
        result = est.update(_make_returns_panel(n_signal_factors=3, signal_strength=0.5))

        # With weak factors on 42 nodes, expect a substantial number in the noise band
        assert result.noise_eigenvalue_count > 20
        # The 3 injected factors should produce at least some signal eigenvalues
        assert result.signal_eigenvalue_count >= 1

    def test_mp_bounds_correct(self):
        """Verify MP bounds match the formula."""
        n_obs = 300
        est = CovarianceEstimator(_default_config(window=n_obs))
        result = est.update(_make_returns_panel(n_obs=n_obs))

        gamma = NUM_NODES / n_obs
        expected_plus = (1.0 + np.sqrt(gamma)) ** 2
        expected_minus = (1.0 - np.sqrt(gamma)) ** 2

        assert abs(result.mp_lambda_plus - expected_plus) < 1e-10
        assert abs(result.mp_lambda_minus - expected_minus) < 1e-10

    def test_insufficient_data_raises(self):
        est = CovarianceEstimator(_default_config(min_observations=100))
        panel = _make_returns_panel(n_obs=50)

        with pytest.raises(InsufficientDataError, match="50 observations"):
            est.update(panel)

    def test_missing_column_raises(self):
        est = CovarianceEstimator(_default_config())
        panel = pd.DataFrame({"ts_utc": [], "tenor_days": [], "delta_bucket": []})

        with pytest.raises(ValueError, match="missing columns"):
            est.update(panel)

    def test_get_latest_none_before_update(self):
        est = CovarianceEstimator(_default_config())
        assert est.get_latest() is None

    def test_get_latest_after_update(self):
        est = CovarianceEstimator(_default_config())
        result = est.update(_make_returns_panel())
        assert est.get_latest() is result

    def test_window_trimming(self):
        """If n_obs > window, only the last `window` rows are used."""
        window = 100
        est = CovarianceEstimator(_default_config(window=window))
        result = est.update(_make_returns_panel(n_obs=300))
        assert result.observations_used == window

    def test_correlation_between(self):
        est = CovarianceEstimator(_default_config())
        est.update(_make_returns_panel())

        node_a = (7, "P10")
        corr_self = est.correlation_between(node_a, node_a)
        assert abs(corr_self - 1.0) < 1e-6

    def test_portfolio_variance_positive(self):
        est = CovarianceEstimator(_default_config())
        est.update(_make_returns_panel())

        w = np.ones(NUM_NODES) / NUM_NODES
        var = est.portfolio_variance(w)
        assert var > 0

    def test_marginal_contribution_shape(self):
        est = CovarianceEstimator(_default_config())
        est.update(_make_returns_panel())

        w = np.ones(NUM_NODES) / NUM_NODES
        mc = est.marginal_contribution(w)
        assert mc.shape == (NUM_NODES,)

    def test_clip_to_mean_preserves_trace(self):
        """clip_to_mean should approximately preserve the trace of the correlation matrix."""
        est = CovarianceEstimator(_default_config(mp_eigenvalue_method="clip_to_mean"))
        result = est.update(_make_returns_panel())

        raw_trace = result.eigenvalues_raw.sum()
        cleaned_trace = result.eigenvalues_cleaned.sum()
        # Should be close (not exact due to regularization)
        assert abs(raw_trace - cleaned_trace) / raw_trace < 0.05

    def test_ledoit_wolf_method(self):
        est = CovarianceEstimator(_default_config(method="ledoit_wolf"))
        result = est.update(_make_returns_panel())

        assert result.covariance.shape == (NUM_NODES, NUM_NODES)
        eigenvalues = np.linalg.eigvalsh(result.covariance)
        assert np.all(eigenvalues >= -1e-10)

    def test_clip_to_zero(self):
        est = CovarianceEstimator(_default_config(mp_eigenvalue_method="clip_to_zero"))
        result = est.update(_make_returns_panel())

        # Noise eigenvalues should be near zero (regularized to 1e-10)
        noise_mask = (result.eigenvalues_raw >= result.mp_lambda_minus) & (
            result.eigenvalues_raw <= result.mp_lambda_plus
        )
        if noise_mask.any():
            assert result.eigenvalues_cleaned[noise_mask].max() < 1e-5


# ---------------------------------------------------------------------------
# filter_correlated_signals tests
# ---------------------------------------------------------------------------

@dataclass
class _FakeSignal:
    """Minimal Signal stand-in for testing."""
    edge: float
    confidence: float
    tenor_days: int
    delta_bucket: str
    signal_type: str = "test"
    timestamp: datetime = datetime(2024, 1, 1, tzinfo=UTC)


class TestFilterCorrelatedSignals:

    @staticmethod
    def _make_corr_result(
        node_order: list[tuple[int, str]],
        corr_matrix: np.ndarray,
    ) -> CovarianceResult:
        p = len(node_order)
        return CovarianceResult(
            covariance=corr_matrix,
            correlation=corr_matrix,
            eigenvalues_raw=np.ones(p),
            eigenvalues_cleaned=np.ones(p),
            mp_lambda_plus=2.0,
            mp_lambda_minus=0.5,
            noise_eigenvalue_count=0,
            signal_eigenvalue_count=p,
            timestamp=datetime.utcnow(),
            observations_used=100,
            node_order=node_order,
        )

    def test_empty_signals(self):
        result = self._make_corr_result([], np.eye(0))
        assert filter_correlated_signals([], result) == []

    def test_uncorrelated_signals_all_pass(self):
        nodes = [(7, "P10"), (30, "ATM"), (90, "C25")]
        corr = np.eye(3)
        result = self._make_corr_result(nodes, corr)

        signals = [
            _FakeSignal(edge=0.05, confidence=0.8, tenor_days=7, delta_bucket="P10"),
            _FakeSignal(edge=0.04, confidence=0.7, tenor_days=30, delta_bucket="ATM"),
            _FakeSignal(edge=0.03, confidence=0.6, tenor_days=90, delta_bucket="C25"),
        ]
        filtered = filter_correlated_signals(signals, result, max_correlation=0.85)
        assert len(filtered) == 3

    def test_highly_correlated_signal_dropped(self):
        nodes = [(7, "P10"), (7, "P25")]
        corr = np.array([[1.0, 0.95], [0.95, 1.0]])
        result = self._make_corr_result(nodes, corr)

        signals = [
            _FakeSignal(edge=0.05, confidence=0.9, tenor_days=7, delta_bucket="P10"),
            _FakeSignal(edge=0.03, confidence=0.8, tenor_days=7, delta_bucket="P25"),
        ]
        filtered = filter_correlated_signals(signals, result, max_correlation=0.85)

        # Best signal kept, correlated one dropped
        assert len(filtered) == 1
        assert filtered[0].delta_bucket == "P10"

    def test_best_signal_wins_tiebreak(self):
        """When two signals are correlated, the one with higher |edge|*confidence survives."""
        nodes = [(7, "P10"), (7, "P25")]
        corr = np.array([[1.0, 0.90], [0.90, 1.0]])
        result = self._make_corr_result(nodes, corr)

        # Second signal has higher score
        signals = [
            _FakeSignal(edge=0.01, confidence=0.5, tenor_days=7, delta_bucket="P10"),
            _FakeSignal(edge=0.10, confidence=0.9, tenor_days=7, delta_bucket="P25"),
        ]
        filtered = filter_correlated_signals(signals, result, max_correlation=0.85)

        assert len(filtered) == 1
        assert filtered[0].delta_bucket == "P25"

    def test_unknown_node_accepted(self):
        """Signals at nodes not in the covariance matrix are accepted by default."""
        nodes = [(7, "P10")]
        corr = np.array([[1.0]])
        result = self._make_corr_result(nodes, corr)

        signals = [
            _FakeSignal(edge=0.05, confidence=0.9, tenor_days=7, delta_bucket="P10"),
            _FakeSignal(edge=0.05, confidence=0.9, tenor_days=999, delta_bucket="UNKNOWN"),
        ]
        filtered = filter_correlated_signals(signals, result, max_correlation=0.85)
        assert len(filtered) == 2

    def test_negative_correlation_also_filtered(self):
        """Strong negative correlation should also trigger filtering."""
        nodes = [(7, "P10"), (7, "C10")]
        corr = np.array([[1.0, -0.92], [-0.92, 1.0]])
        result = self._make_corr_result(nodes, corr)

        signals = [
            _FakeSignal(edge=0.05, confidence=0.9, tenor_days=7, delta_bucket="P10"),
            _FakeSignal(edge=0.04, confidence=0.8, tenor_days=7, delta_bucket="C10"),
        ]
        filtered = filter_correlated_signals(signals, result, max_correlation=0.85)
        assert len(filtered) == 1
