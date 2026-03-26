"""Evaluation metrics for model predictions.

This package provides metrics for evaluating volatility surface predictions:
- IC (Information Coefficient): Pearson correlation
- Temporal IC: Per-node correlation across time (honest temporal signal)
- XS-Demeaned IC: Cross-sectionally demeaned correlation (relative ranking)
- RMSE: Root Mean Squared Error
- MAE: Mean Absolute Error
"""

from src.models.eval.metrics import (
    MetricsCalculator,
    compute_ic,
    compute_mae,
    compute_rmse,
    compute_temporal_ic,
    compute_xs_demeaned_ic,
)

__all__ = [
    "compute_ic",
    "compute_temporal_ic",
    "compute_xs_demeaned_ic",
    "compute_rmse",
    "compute_mae",
    "MetricsCalculator",
]
