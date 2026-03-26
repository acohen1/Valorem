"""Unit tests for evaluation metrics."""

import numpy as np
import pytest

from src.models.eval import (
    MetricsCalculator,
    compute_ic,
    compute_mae,
    compute_rmse,
)


class TestComputeIC:
    """Tests for Information Coefficient (Pearson correlation)."""

    def test_perfect_correlation(self) -> None:
        """Test IC = 1 for perfectly correlated data."""
        predictions = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        targets = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        ic = compute_ic(predictions, targets)
        assert np.isclose(ic, 1.0, atol=1e-6)

    def test_perfect_negative_correlation(self) -> None:
        """Test IC = -1 for perfectly negatively correlated data."""
        predictions = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        targets = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        ic = compute_ic(predictions, targets)
        assert np.isclose(ic, -1.0, atol=1e-6)

    def test_zero_correlation(self) -> None:
        """Test IC ~ 0 for uncorrelated data."""
        np.random.seed(42)
        predictions = np.random.randn(1000)
        targets = np.random.randn(1000)
        ic = compute_ic(predictions, targets)
        assert abs(ic) < 0.1  # Should be close to 0

    def test_2d_input(self) -> None:
        """Test IC with 2D input (multiple horizons)."""
        predictions = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0], [4.0, 8.0]])
        targets = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0], [4.0, 8.0]])
        ic = compute_ic(predictions, targets)
        assert np.isclose(ic, 1.0, atol=1e-6)

    def test_handles_nan_values(self) -> None:
        """Test that NaN values are handled correctly."""
        predictions = np.array([1.0, np.nan, 3.0, 4.0, 5.0])
        targets = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        ic = compute_ic(predictions, targets)
        assert not np.isnan(ic)

    def test_constant_predictions_returns_zero(self) -> None:
        """Test that constant predictions return 0 IC."""
        predictions = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        targets = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        ic = compute_ic(predictions, targets)
        assert ic == 0.0

    def test_too_few_samples_returns_zero(self) -> None:
        """Test that fewer than 2 valid samples returns 0."""
        predictions = np.array([1.0])
        targets = np.array([1.0])
        ic = compute_ic(predictions, targets)
        assert ic == 0.0


class TestComputeRMSE:
    """Tests for Root Mean Squared Error."""

    def test_perfect_predictions(self) -> None:
        """Test RMSE = 0 for perfect predictions."""
        predictions = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        targets = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        rmse = compute_rmse(predictions, targets)
        assert np.isclose(rmse, 0.0, atol=1e-6)

    def test_known_rmse(self) -> None:
        """Test RMSE with known value."""
        predictions = np.array([1.0, 2.0, 3.0])
        targets = np.array([2.0, 3.0, 4.0])  # All off by 1
        rmse = compute_rmse(predictions, targets)
        assert np.isclose(rmse, 1.0, atol=1e-6)

    def test_larger_errors(self) -> None:
        """Test RMSE with larger errors."""
        predictions = np.array([0.0, 0.0])
        targets = np.array([3.0, 4.0])  # Errors: 3, 4 -> MSE = (9+16)/2 = 12.5
        rmse = compute_rmse(predictions, targets)
        expected = np.sqrt(12.5)
        assert np.isclose(rmse, expected, atol=1e-6)

    def test_handles_nan_values(self) -> None:
        """Test that NaN values are handled correctly."""
        predictions = np.array([1.0, np.nan, 3.0])
        targets = np.array([1.0, 2.0, 3.0])
        rmse = compute_rmse(predictions, targets)
        assert not np.isnan(rmse)
        assert np.isclose(rmse, 0.0, atol=1e-6)  # Only comparing [1, 3] vs [1, 3]

    def test_all_nan_returns_zero(self) -> None:
        """Test that all NaN inputs return 0."""
        predictions = np.array([np.nan, np.nan])
        targets = np.array([1.0, 2.0])
        rmse = compute_rmse(predictions, targets)
        assert rmse == 0.0


class TestComputeMAE:
    """Tests for Mean Absolute Error."""

    def test_perfect_predictions(self) -> None:
        """Test MAE = 0 for perfect predictions."""
        predictions = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        targets = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        mae = compute_mae(predictions, targets)
        assert np.isclose(mae, 0.0, atol=1e-6)

    def test_known_mae(self) -> None:
        """Test MAE with known value."""
        predictions = np.array([1.0, 2.0, 3.0])
        targets = np.array([2.0, 3.0, 4.0])  # All off by 1
        mae = compute_mae(predictions, targets)
        assert np.isclose(mae, 1.0, atol=1e-6)

    def test_mixed_errors(self) -> None:
        """Test MAE with mixed positive and negative errors."""
        predictions = np.array([0.0, 5.0])
        targets = np.array([2.0, 2.0])  # Errors: -2, +3 -> MAE = (2+3)/2 = 2.5
        mae = compute_mae(predictions, targets)
        assert np.isclose(mae, 2.5, atol=1e-6)

    def test_handles_nan_values(self) -> None:
        """Test that NaN values are handled correctly."""
        predictions = np.array([1.0, np.nan, 3.0])
        targets = np.array([1.0, 2.0, 3.0])
        mae = compute_mae(predictions, targets)
        assert not np.isnan(mae)


class TestMetricsCalculator:
    """Tests for MetricsCalculator class."""

    def test_compute_all_returns_all_metrics(self) -> None:
        """Test that compute_all returns all metrics."""
        calculator = MetricsCalculator()
        predictions = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        targets = np.array([1.1, 2.1, 3.1, 4.1, 5.1])

        metrics = calculator.compute_all(predictions, targets)

        assert "rmse" in metrics
        assert "mae" in metrics
        # Pooled IC deliberately excluded — misleading for DHR targets
        assert "ic" not in metrics

    def test_compute_all_values_reasonable(self) -> None:
        """Test that computed metrics have reasonable values."""
        calculator = MetricsCalculator()
        predictions = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        targets = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        metrics = calculator.compute_all(predictions, targets)

        assert np.isclose(metrics["rmse"], 0.0, atol=1e-6)
        assert np.isclose(metrics["mae"], 0.0, atol=1e-6)

    def test_compute_all_2d(self) -> None:
        """Test compute_all with 2D input."""
        calculator = MetricsCalculator()
        predictions = np.random.randn(100, 3)
        targets = predictions + 0.1 * np.random.randn(100, 3)

        metrics = calculator.compute_all(predictions, targets)

        assert metrics["rmse"] < 0.2
        assert metrics["mae"] < 0.2
