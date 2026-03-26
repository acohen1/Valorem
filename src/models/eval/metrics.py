"""Evaluation metrics for volatility surface prediction.

This module provides metrics for evaluating model predictions:
- Information Coefficient (IC): Pearson correlation between predictions and targets
- Temporal IC: Per-node IC averaged across nodes (measures temporal prediction quality)
- XS-Demeaned IC: Per-node-mean-demeaned Spearman IC per timestamp (measures time-varying ranking skill)
- RMSE: Root Mean Squared Error
- MAE: Mean Absolute Error

Note on pooled vs decomposed IC:
    The pooled IC (compute_ic) correlates across ALL (sample*node)
    pairs. When labels have a cross-sectional component mechanically determined by
    inputs (e.g., DHR ∝ gamma·ΔS²), pooled IC is dominated by this trivial
    relationship. Use temporal_ic and xs_demeaned_ic for honest evaluation.

References:
    - Bali et al. (RFS 2023) — cross-sectional evaluation via long-short portfolios
    - Goyal & Saretto (JFE 2009) — cross-sectional sorts on HV-IV spread
"""

from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.stats import rankdata


def compute_ic(
    predictions: NDArray[np.floating[Any]],
    targets: NDArray[np.floating[Any]],
) -> float:
    """Compute Information Coefficient (Pearson correlation).

    Computes IC per horizon/feature, then averages. Handles NaN values
    by excluding them from the correlation computation.

    Args:
        predictions: Predicted values, shape (n_samples,) or (n_samples, n_horizons)
        targets: Target values, same shape as predictions

    Returns:
        Average IC across all horizons. Returns 0.0 if no valid correlations.
    """
    predictions = np.asarray(predictions)
    targets = np.asarray(targets)

    # Handle 1D case
    if predictions.ndim == 1:
        predictions = predictions.reshape(-1, 1)
        targets = targets.reshape(-1, 1)

    ics: list[float] = []
    for h in range(predictions.shape[1]):
        pred_h = predictions[:, h]
        target_h = targets[:, h]

        # Remove NaN values
        valid_mask = ~(np.isnan(pred_h) | np.isnan(target_h))
        if valid_mask.sum() < 2:
            continue

        pred_valid = pred_h[valid_mask]
        target_valid = target_h[valid_mask]

        # Check for constant arrays (would give NaN correlation)
        if np.std(pred_valid) < 1e-10 or np.std(target_valid) < 1e-10:
            continue

        ic = float(np.corrcoef(pred_valid, target_valid)[0, 1])
        if not np.isnan(ic):
            ics.append(ic)

    return float(np.mean(ics)) if ics else 0.0


def compute_temporal_ic(
    predictions: NDArray[np.floating[Any]],
    targets: NDArray[np.floating[Any]],
    masks: NDArray[np.bool_],
    num_nodes: int,
) -> float:
    """Compute temporal IC: per-node Spearman correlation across timestamps.

    For each node, correlates predictions with actuals across all timestamps
    where the node is valid. Then averages across nodes. This measures whether
    the model correctly predicts WHEN the gap is large vs small for a given node.

    Args:
        predictions: Shape (num_samples, num_nodes, num_horizons).
        targets: Same shape as predictions.
        masks: Shape (num_samples, num_nodes), boolean.
        num_nodes: Number of nodes in the graph.

    Returns:
        Average temporal IC across nodes and horizons. Returns 0.0
        if insufficient data.
    """
    predictions = np.asarray(predictions)
    targets = np.asarray(targets)
    masks = np.asarray(masks)

    num_horizons = predictions.shape[-1] if predictions.ndim == 3 else 1
    if predictions.ndim == 2:
        predictions = predictions[:, :, np.newaxis]
        targets = targets[:, :, np.newaxis]

    node_ics: list[float] = []

    for n in range(num_nodes):
        node_mask = masks[:, n]
        if node_mask.sum() < 5:
            continue

        for h in range(num_horizons):
            pred_n = predictions[node_mask, n, h]
            target_n = targets[node_mask, n, h]

            valid = np.isfinite(pred_n) & np.isfinite(target_n)
            if valid.sum() < 5:
                continue

            pred_v = pred_n[valid]
            target_v = target_n[valid]

            if np.std(pred_v) < 1e-10 or np.std(target_v) < 1e-10:
                continue

            # Spearman rank correlation — averaged ranks for ties
            pred_ranks = rankdata(pred_v, method="average")
            target_ranks = rankdata(target_v, method="average")

            ic = float(np.corrcoef(pred_ranks, target_ranks)[0, 1])
            if not np.isnan(ic):
                node_ics.append(ic)

    return float(np.mean(node_ics)) if node_ics else 0.0


def compute_xs_demeaned_ic(
    predictions: NDArray[np.floating[Any]],
    targets: NDArray[np.floating[Any]],
    masks: NDArray[np.bool_],
    num_nodes: int,
) -> float:
    """Compute cross-sectionally demeaned IC per timestamp.

    For each node, computes the temporal mean of predictions and targets
    across all valid timestamps. At each timestamp, subtracts these
    per-node means before computing Spearman rank correlation across
    nodes. This removes static cross-sectional structure (e.g., a model
    that always predicts "high-gamma nodes have high DHR") and measures
    whether the model captures time-varying deviations in cross-sectional
    ranking — genuine predictive skill beyond structural knowledge.

    Args:
        predictions: Shape (num_samples, num_nodes, num_horizons).
        targets: Same shape as predictions.
        masks: Shape (num_samples, num_nodes), boolean.
        num_nodes: Number of nodes in the graph.

    Returns:
        Average XS-demeaned IC across timestamps and horizons. Returns 0.0
        if insufficient data.
    """
    predictions = np.asarray(predictions, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    masks = np.asarray(masks)

    num_samples = predictions.shape[0]
    num_horizons = predictions.shape[-1] if predictions.ndim == 3 else 1
    if predictions.ndim == 2:
        predictions = predictions[:, :, np.newaxis]
        targets = targets[:, :, np.newaxis]

    # Compute per-node temporal means for demeaning.
    # This removes static cross-sectional structure so that the metric
    # only rewards time-varying cross-sectional skill.
    pred_node_mean = np.full((num_nodes, num_horizons), np.nan)
    target_node_mean = np.full((num_nodes, num_horizons), np.nan)

    for n in range(num_nodes):
        node_mask = masks[:, n]
        if node_mask.sum() < 2:
            continue
        for h in range(num_horizons):
            p = predictions[node_mask, n, h]
            t = targets[node_mask, n, h]
            valid = np.isfinite(p) & np.isfinite(t)
            if valid.sum() >= 2:
                pred_node_mean[n, h] = np.mean(p[valid])
                target_node_mean[n, h] = np.mean(t[valid])

    timestamp_ics: list[float] = []

    for t in range(num_samples):
        valid_nodes = masks[t]
        if valid_nodes.sum() < 3:
            continue

        for h in range(num_horizons):
            pred_t = predictions[t, valid_nodes, h]
            target_t = targets[t, valid_nodes, h]
            pred_mean_t = pred_node_mean[valid_nodes, h]
            target_mean_t = target_node_mean[valid_nodes, h]

            valid = (
                np.isfinite(pred_t)
                & np.isfinite(target_t)
                & np.isfinite(pred_mean_t)
                & np.isfinite(target_mean_t)
            )
            if valid.sum() < 3:
                continue

            # Subtract per-node temporal means to remove static structure
            pred_v = pred_t[valid] - pred_mean_t[valid]
            target_v = target_t[valid] - target_mean_t[valid]

            if np.std(pred_v) < 1e-10 or np.std(target_v) < 1e-10:
                continue

            pred_ranks = rankdata(pred_v, method="average")
            target_ranks = rankdata(target_v, method="average")

            ic = float(np.corrcoef(pred_ranks, target_ranks)[0, 1])
            if not np.isnan(ic):
                timestamp_ics.append(ic)

    return float(np.mean(timestamp_ics)) if timestamp_ics else 0.0


def compute_rmse(
    predictions: NDArray[np.floating[Any]],
    targets: NDArray[np.floating[Any]],
) -> float:
    """Compute Root Mean Squared Error.

    Args:
        predictions: Predicted values
        targets: Target values, same shape as predictions

    Returns:
        RMSE value. Returns 0.0 if no valid values.
    """
    predictions = np.asarray(predictions)
    targets = np.asarray(targets)

    # Remove NaN values
    valid_mask = ~(np.isnan(predictions) | np.isnan(targets))
    if valid_mask.sum() == 0:
        return 0.0

    diff = predictions[valid_mask] - targets[valid_mask]
    return float(np.sqrt(np.mean(diff**2)))


def compute_mae(
    predictions: NDArray[np.floating[Any]],
    targets: NDArray[np.floating[Any]],
) -> float:
    """Compute Mean Absolute Error.

    Args:
        predictions: Predicted values
        targets: Target values, same shape as predictions

    Returns:
        MAE value. Returns 0.0 if no valid values.
    """
    predictions = np.asarray(predictions)
    targets = np.asarray(targets)

    # Remove NaN values
    valid_mask = ~(np.isnan(predictions) | np.isnan(targets))
    if valid_mask.sum() == 0:
        return 0.0

    diff = predictions[valid_mask] - targets[valid_mask]
    return float(np.mean(np.abs(diff)))


class MetricsCalculator:
    """Compute evaluation metrics for model predictions.

    Requires 3D predictions (n_samples, n_nodes, n_horizons) with masks to
    compute decomposed IC metrics that reflect genuine predictive skill:
    - Temporal IC: per-node Spearman correlation across timestamps
    - XS-Demeaned IC: cross-sectionally demeaned Spearman correlation

    Pooled IC (compute_ic) is available as a standalone function but deliberately
    excluded from compute_all — for DHR targets it is dominated by the trivial
    gamma·ΔS² cross-sectional relationship.

    Example:
        >>> calculator = MetricsCalculator()
        >>> metrics = calculator.compute_all(preds_flat, targets_flat,
        ...     predictions_3d=preds_3d, targets_3d=targets_3d,
        ...     masks_2d=masks, num_nodes=42)
        >>> print(metrics["temporal_ic"], metrics["xs_demeaned_ic"])
    """

    def compute_all(
        self,
        predictions: NDArray[np.floating[Any]],
        targets: NDArray[np.floating[Any]],
        *,
        predictions_3d: NDArray[np.floating[Any]] | None = None,
        targets_3d: NDArray[np.floating[Any]] | None = None,
        masks_2d: NDArray[np.bool_] | None = None,
        num_nodes: int = 0,
    ) -> dict[str, float]:
        """Compute all metrics for given predictions and targets.

        Args:
            predictions: Flat predicted values, shape (n_valid,) or (n_valid, n_horizons).
                Used for RMSE, MAE.
            targets: Flat target values, same shape as predictions.
            predictions_3d: Unflattened predictions (n_samples, n_nodes, n_horizons).
                Required for temporal_ic and xs_demeaned_ic.
            targets_3d: Unflattened targets, same shape as predictions_3d.
            masks_2d: Masks (n_samples, n_nodes). Required for decomposed metrics.
            num_nodes: Number of graph nodes. Required for decomposed metrics.

        Returns:
            Dictionary with keys: 'temporal_ic', 'xs_demeaned_ic', 'rmse', 'mae'.
        """
        result: dict[str, float] = {
            "rmse": compute_rmse(predictions, targets),
            "mae": compute_mae(predictions, targets),
        }

        if predictions_3d is not None and targets_3d is not None and masks_2d is not None and num_nodes > 0:
            result["temporal_ic"] = compute_temporal_ic(
                predictions_3d, targets_3d, masks_2d, num_nodes,
            )
            result["xs_demeaned_ic"] = compute_xs_demeaned_ic(
                predictions_3d, targets_3d, masks_2d, num_nodes,
            )

        return result
