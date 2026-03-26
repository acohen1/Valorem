"""Loss functions for volatility surface prediction.

This module provides loss functions for training:
- HuberLoss: Smooth L1 loss, robust to outliers
- QuantileLoss: For quantile regression / uncertainty estimation
- MaskedLoss: Wrapper that applies mask to any base loss
"""

import torch
import torch.nn as nn
from torch import Tensor


class HuberLoss(nn.Module):
    """Huber loss (smooth L1) for robust regression.

    Combines the best properties of L1 and L2 loss:
    - Quadratic for small errors (like MSE)
    - Linear for large errors (like MAE, more robust to outliers)

    Args:
        delta: Threshold at which to switch from quadratic to linear.
            Default 1.0.
        reduction: Specifies the reduction to apply. One of 'none', 'mean', 'sum'.
            Default 'mean'.
    """

    def __init__(self, delta: float = 1.0, reduction: str = "mean") -> None:
        super().__init__()
        self._delta = delta
        self._reduction = reduction

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """Compute Huber loss.

        Args:
            input: Predicted values
            target: Target values

        Returns:
            Huber loss value
        """
        abs_diff = torch.abs(input - target)
        quadratic = torch.clamp(abs_diff, max=self._delta)
        linear = abs_diff - quadratic

        loss = 0.5 * quadratic**2 + self._delta * linear

        if self._reduction == "mean":
            return loss.mean()
        elif self._reduction == "sum":
            return loss.sum()
        else:
            return loss


class QuantileLoss(nn.Module):
    """Quantile loss for quantile regression.

    Also known as pinball loss. Useful for uncertainty estimation
    by predicting different quantiles (e.g., 0.1, 0.5, 0.9).

    Args:
        quantile: The quantile to predict, between 0 and 1.
            0.5 gives median (equivalent to MAE).
        reduction: Specifies the reduction to apply. One of 'none', 'mean', 'sum'.
            Default 'mean'.
    """

    def __init__(self, quantile: float = 0.5, reduction: str = "mean") -> None:
        super().__init__()
        if not 0 < quantile < 1:
            raise ValueError(f"Quantile must be in (0, 1), got {quantile}")
        self._quantile = quantile
        self._reduction = reduction

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """Compute quantile loss.

        Args:
            input: Predicted values
            target: Target values

        Returns:
            Quantile loss value
        """
        diff = target - input
        loss = torch.where(
            diff >= 0,
            self._quantile * diff,
            (self._quantile - 1) * diff,
        )

        if self._reduction == "mean":
            return loss.mean()
        elif self._reduction == "sum":
            return loss.sum()
        else:
            return loss


class MaskedLoss(nn.Module):
    """Wrapper that applies mask before computing loss.

    Filters predictions and targets to only include valid (masked) elements
    before computing the base loss. This is useful for handling variable-length
    sequences or missing data.

    Args:
        base_loss: The underlying loss function to use.
            Must accept (input, target) and return a scalar.

    Example:
        >>> base_loss = nn.MSELoss()
        >>> masked_loss = MaskedLoss(base_loss)
        >>> preds = torch.randn(4, 10, 3)  # (batch, nodes, horizons)
        >>> targets = torch.randn(4, 10, 3)
        >>> mask = torch.ones(4, 10, dtype=torch.bool)
        >>> mask[:, 5:] = False  # Mask out last 5 nodes
        >>> loss = masked_loss(preds, targets, mask)
    """

    def __init__(self, base_loss: nn.Module) -> None:
        super().__init__()
        self._base_loss = base_loss

    def forward(self, input: Tensor, target: Tensor, mask: Tensor) -> Tensor:
        """Compute masked loss.

        Args:
            input: Predicted values, shape (batch, nodes, ...) or (batch, nodes)
            target: Target values, same shape as input
            mask: Boolean mask, shape (batch, nodes). True = valid, False = invalid.

        Returns:
            Loss computed only on valid (masked) elements.
            Returns 0 if no valid elements.
        """
        # Expand mask to full input shape.
        # Supports:
        # - node-level mask: (batch, nodes)
        # - label-level mask: (batch, nodes, horizons)
        mask_expanded = mask
        while mask_expanded.ndim < input.ndim:
            mask_expanded = mask_expanded.unsqueeze(-1)
        mask_expanded = mask_expanded.expand_as(input)

        # Replace invalid predictions with target values so they contribute
        # zero loss, avoiding expensive boolean gather (GPU prefix-sum + scatter).
        masked_input = torch.where(mask_expanded, input, target)

        # Compute loss over all elements (invalid positions contribute zero)
        raw_loss: Tensor = self._base_loss(masked_input, target)

        # Correct the mean: raw_loss averages over ALL elements, but we want
        # the mean over valid elements only.
        n_valid = mask_expanded.sum().to(input.dtype).clamp(min=1)
        loss: Tensor = raw_loss * (mask_expanded.numel() / n_valid)
        return loss


class VolumeWeightedMaskedLoss(MaskedLoss):
    """MaskedLoss with optional per-node volume weighting.

    When volume weights are provided, higher-volume nodes contribute
    proportionally more to the loss. This biases the model toward
    fitting liquid contracts where prices are reliable.

    When weights=None, falls through to parent MaskedLoss (backward compatible).

    Args:
        base_loss: The underlying loss function (with reduction='mean').
        base_loss_type: Loss type name for creating a reduction='none' variant.
        huber_delta: Delta for Huber loss.
        quantile: Quantile for quantile loss.
    """

    def __init__(
        self,
        base_loss: nn.Module,
        base_loss_type: str = "mse",
        huber_delta: float = 1.0,
        quantile: float = 0.5,
    ) -> None:
        super().__init__(base_loss)
        # Build a reduction='none' variant for element-wise loss computation
        self._base_loss_none = _build_loss_none(base_loss_type, huber_delta, quantile)

    def forward(
        self,
        input: Tensor,
        target: Tensor,
        mask: Tensor,
        weights: Tensor | None = None,
    ) -> Tensor:
        """Compute volume-weighted masked loss.

        Args:
            input: Predicted values (batch, nodes, horizons)
            target: Target values (batch, nodes, horizons)
            mask: Boolean mask (batch, nodes). True = valid.
            weights: Optional per-node weights (batch, nodes).
                If None, falls back to unweighted MaskedLoss behavior.

        Returns:
            Scalar loss value.
        """
        if weights is None:
            return super().forward(input, target, mask)

        # Element-wise loss: (batch, nodes, horizons)
        elem_loss = self._base_loss_none(input, target)

        # Expand mask to element-wise shape.
        mask_expanded = mask
        while mask_expanded.ndim < elem_loss.ndim:
            mask_expanded = mask_expanded.unsqueeze(-1)
        mask_expanded = mask_expanded.expand_as(elem_loss)

        # Expand weights: (batch, nodes) -> (batch, nodes, horizons)
        weights_expanded = weights
        while weights_expanded.ndim < elem_loss.ndim:
            weights_expanded = weights_expanded.unsqueeze(-1)
        weights_expanded = weights_expanded.expand_as(elem_loss)

        # Zero out invalid elements
        masked_loss = elem_loss * mask_expanded.float()
        masked_weights = weights_expanded * mask_expanded.float()

        # Weighted mean
        weighted_sum = (masked_loss * masked_weights).sum()
        weight_total = masked_weights.sum().clamp(min=1e-8)

        return weighted_sum / weight_total


def _build_loss_none(
    loss_type: str,
    huber_delta: float = 1.0,
    quantile: float = 0.5,
) -> nn.Module:
    """Build a loss function with reduction='none' for element-wise computation."""
    if loss_type == "mse":
        return nn.MSELoss(reduction="none")
    elif loss_type == "huber":
        return HuberLoss(delta=huber_delta, reduction="none")
    elif loss_type == "quantile":
        return QuantileLoss(quantile=quantile, reduction="none")
    elif loss_type == "mae":
        return nn.L1Loss(reduction="none")
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def build_loss(
    loss_type: str,
    huber_delta: float = 1.0,
    quantile: float = 0.5,
) -> nn.Module:
    """Build a loss function by name.

    Args:
        loss_type: One of 'mse', 'huber', 'quantile', 'mae'
        huber_delta: Delta parameter for Huber loss
        quantile: Quantile parameter for quantile loss

    Returns:
        Loss function module

    Raises:
        ValueError: If loss_type is not recognized
    """
    if loss_type == "mse":
        return nn.MSELoss()
    elif loss_type == "huber":
        return HuberLoss(delta=huber_delta)
    elif loss_type == "quantile":
        return QuantileLoss(quantile=quantile)
    elif loss_type == "mae":
        return nn.L1Loss()
    else:
        raise ValueError(f"Unknown loss type: {loss_type}. Use 'mse', 'huber', 'quantile', or 'mae'")
