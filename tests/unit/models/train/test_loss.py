"""Unit tests for loss functions."""

import pytest
import torch
import torch.nn as nn

from src.models.train.loss import (
    HuberLoss,
    MaskedLoss,
    QuantileLoss,
    build_loss,
)


class TestHuberLoss:
    """Tests for Huber loss function."""

    def test_small_errors_quadratic(self) -> None:
        """Test that small errors use quadratic loss."""
        loss_fn = HuberLoss(delta=1.0)
        # Error of 0.5 should be quadratic: 0.5 * 0.5^2 = 0.125
        input_tensor = torch.tensor([0.0])
        target = torch.tensor([0.5])
        loss = loss_fn(input_tensor, target)
        expected = 0.5 * 0.5**2
        assert torch.isclose(loss, torch.tensor(expected), atol=1e-6)

    def test_large_errors_linear(self) -> None:
        """Test that large errors use linear loss."""
        loss_fn = HuberLoss(delta=1.0)
        # Error of 2.0 should be: 0.5 * 1^2 + 1 * (2-1) = 0.5 + 1 = 1.5
        input_tensor = torch.tensor([0.0])
        target = torch.tensor([2.0])
        loss = loss_fn(input_tensor, target)
        expected = 0.5 * 1.0**2 + 1.0 * (2.0 - 1.0)
        assert torch.isclose(loss, torch.tensor(expected), atol=1e-6)

    def test_exact_delta_boundary(self) -> None:
        """Test loss at exactly delta boundary."""
        loss_fn = HuberLoss(delta=1.0)
        input_tensor = torch.tensor([0.0])
        target = torch.tensor([1.0])
        loss = loss_fn(input_tensor, target)
        expected = 0.5 * 1.0**2  # Pure quadratic at boundary
        assert torch.isclose(loss, torch.tensor(expected), atol=1e-6)

    def test_custom_delta(self) -> None:
        """Test with custom delta value."""
        loss_fn = HuberLoss(delta=0.5)
        input_tensor = torch.tensor([0.0])
        target = torch.tensor([1.0])  # Error > delta
        loss = loss_fn(input_tensor, target)
        # 0.5 * 0.5^2 + 0.5 * (1.0 - 0.5) = 0.125 + 0.25 = 0.375
        expected = 0.5 * 0.5**2 + 0.5 * (1.0 - 0.5)
        assert torch.isclose(loss, torch.tensor(expected), atol=1e-6)

    def test_reduction_none(self) -> None:
        """Test reduction='none' returns element-wise loss."""
        loss_fn = HuberLoss(delta=1.0, reduction="none")
        input_tensor = torch.tensor([0.0, 0.0])
        target = torch.tensor([0.5, 2.0])
        loss = loss_fn(input_tensor, target)
        assert loss.shape == (2,)

    def test_reduction_sum(self) -> None:
        """Test reduction='sum' returns sum of losses."""
        loss_fn = HuberLoss(delta=1.0, reduction="sum")
        input_tensor = torch.tensor([0.0, 0.0])
        target = torch.tensor([0.5, 0.5])
        loss = loss_fn(input_tensor, target)
        expected = 2 * 0.5 * 0.5**2  # Sum of two identical losses
        assert torch.isclose(loss, torch.tensor(expected), atol=1e-6)

    def test_gradient_flow(self) -> None:
        """Test that gradients flow correctly."""
        loss_fn = HuberLoss(delta=1.0)
        input_tensor = torch.tensor([0.0], requires_grad=True)
        target = torch.tensor([0.5])
        loss = loss_fn(input_tensor, target)
        loss.backward()
        assert input_tensor.grad is not None
        assert not torch.all(input_tensor.grad == 0)


class TestQuantileLoss:
    """Tests for quantile loss function."""

    def test_median_loss_equals_mae(self) -> None:
        """Test that quantile=0.5 gives MAE."""
        quantile_loss = QuantileLoss(quantile=0.5)
        mae_loss = nn.L1Loss()
        input_tensor = torch.tensor([0.0, 1.0, 2.0])
        target = torch.tensor([1.0, 1.0, 1.0])

        q_loss = quantile_loss(input_tensor, target)
        m_loss = mae_loss(input_tensor, target)

        # Quantile loss at 0.5 should be half of MAE
        assert torch.isclose(q_loss, m_loss * 0.5, atol=1e-6)

    def test_low_quantile_penalizes_overestimates(self) -> None:
        """Test that low quantile penalizes overestimates more."""
        loss_fn = QuantileLoss(quantile=0.1)
        # Overestimate: pred > target
        over_input = torch.tensor([2.0])
        over_target = torch.tensor([1.0])
        # Underestimate: pred < target
        under_input = torch.tensor([0.0])
        under_target = torch.tensor([1.0])

        over_loss = loss_fn(over_input, over_target)
        under_loss = loss_fn(under_input, under_target)

        # Overestimate should have higher loss with low quantile
        assert over_loss > under_loss

    def test_high_quantile_penalizes_underestimates(self) -> None:
        """Test that high quantile penalizes underestimates more."""
        loss_fn = QuantileLoss(quantile=0.9)
        over_input = torch.tensor([2.0])
        over_target = torch.tensor([1.0])
        under_input = torch.tensor([0.0])
        under_target = torch.tensor([1.0])

        over_loss = loss_fn(over_input, over_target)
        under_loss = loss_fn(under_input, under_target)

        # Underestimate should have higher loss with high quantile
        assert under_loss > over_loss

    def test_invalid_quantile_raises(self) -> None:
        """Test that invalid quantile values raise ValueError."""
        with pytest.raises(ValueError):
            QuantileLoss(quantile=0.0)
        with pytest.raises(ValueError):
            QuantileLoss(quantile=1.0)
        with pytest.raises(ValueError):
            QuantileLoss(quantile=-0.1)
        with pytest.raises(ValueError):
            QuantileLoss(quantile=1.1)

    def test_reduction_none(self) -> None:
        """Test reduction='none' returns element-wise loss."""
        loss_fn = QuantileLoss(quantile=0.5, reduction="none")
        input_tensor = torch.tensor([0.0, 0.0])
        target = torch.tensor([1.0, 2.0])
        loss = loss_fn(input_tensor, target)
        assert loss.shape == (2,)

    def test_gradient_flow(self) -> None:
        """Test that gradients flow correctly."""
        loss_fn = QuantileLoss(quantile=0.5)
        input_tensor = torch.tensor([0.0], requires_grad=True)
        target = torch.tensor([1.0])
        loss = loss_fn(input_tensor, target)
        loss.backward()
        assert input_tensor.grad is not None


class TestMaskedLoss:
    """Tests for masked loss wrapper."""

    def test_mask_excludes_invalid(self) -> None:
        """Test that mask excludes invalid elements."""
        base_loss = nn.MSELoss()
        masked_loss = MaskedLoss(base_loss)

        # 2 samples, 4 nodes, 1 feature
        input_tensor = torch.tensor([
            [[0.0], [1.0], [2.0], [3.0]],
            [[0.0], [1.0], [2.0], [3.0]],
        ])
        target = torch.tensor([
            [[0.0], [1.0], [2.0], [3.0]],
            [[0.0], [1.0], [2.0], [3.0]],
        ])
        # Mask out half the nodes
        mask = torch.tensor([
            [True, True, False, False],
            [True, True, False, False],
        ])

        loss = masked_loss(input_tensor, target, mask)
        # Only valid elements are compared, and they're equal
        assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)

    def test_masked_elements_dont_affect_loss(self) -> None:
        """Test that masked elements don't contribute to loss."""
        base_loss = nn.MSELoss()
        masked_loss = MaskedLoss(base_loss)

        input_tensor = torch.tensor([
            [[0.0], [0.0], [100.0], [100.0]],  # Large error in masked region
        ])
        target = torch.tensor([
            [[0.0], [0.0], [0.0], [0.0]],
        ])
        mask = torch.tensor([[True, True, False, False]])

        loss = masked_loss(input_tensor, target, mask)
        # Large errors in masked region should not affect loss
        assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)

    def test_full_mask_uses_all_elements(self) -> None:
        """Test that full mask includes all elements."""
        base_loss = nn.MSELoss()
        masked_loss = MaskedLoss(base_loss)

        input_tensor = torch.tensor([[[0.0], [2.0]]])
        target = torch.tensor([[[1.0], [1.0]]])
        mask = torch.ones(1, 2, dtype=torch.bool)

        loss = masked_loss(input_tensor, target, mask)
        expected = base_loss(input_tensor, target)
        assert torch.isclose(loss, expected, atol=1e-6)

    def test_empty_mask_returns_zero(self) -> None:
        """Test that empty mask returns zero loss."""
        base_loss = nn.MSELoss()
        masked_loss = MaskedLoss(base_loss)

        input_tensor = torch.tensor([[[1.0], [2.0]]])
        target = torch.tensor([[[0.0], [0.0]]])
        mask = torch.zeros(1, 2, dtype=torch.bool)

        loss = masked_loss(input_tensor, target, mask)
        assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)

    def test_gradient_flow_through_mask(self) -> None:
        """Test that gradients flow through valid elements."""
        base_loss = nn.MSELoss()
        masked_loss = MaskedLoss(base_loss)

        input_tensor = torch.tensor([[[0.0], [0.0]]], requires_grad=True)
        target = torch.tensor([[[1.0], [1.0]]])
        mask = torch.tensor([[True, False]])

        loss = masked_loss(input_tensor, target, mask)
        loss.backward()

        assert input_tensor.grad is not None
        # Only first element should have gradient (second is masked)
        assert input_tensor.grad[0, 0, 0] != 0
        assert input_tensor.grad[0, 1, 0] == 0

    def test_3d_input(self) -> None:
        """Test mask with 3D input (batch, nodes, horizons)."""
        base_loss = nn.MSELoss()
        masked_loss = MaskedLoss(base_loss)

        input_tensor = torch.randn(4, 10, 3)
        target = torch.randn(4, 10, 3)
        mask = torch.ones(4, 10, dtype=torch.bool)
        mask[:, 5:] = False

        loss = masked_loss(input_tensor, target, mask)
        assert loss.numel() == 1  # Scalar output


class TestBuildLoss:
    """Tests for build_loss factory function."""

    def test_build_mse(self) -> None:
        """Test building MSE loss."""
        loss_fn = build_loss("mse")
        assert isinstance(loss_fn, nn.MSELoss)

    def test_build_mae(self) -> None:
        """Test building MAE (L1) loss."""
        loss_fn = build_loss("mae")
        assert isinstance(loss_fn, nn.L1Loss)

    def test_build_huber(self) -> None:
        """Test building Huber loss."""
        loss_fn = build_loss("huber", huber_delta=0.5)
        assert isinstance(loss_fn, HuberLoss)

    def test_build_quantile(self) -> None:
        """Test building quantile loss."""
        loss_fn = build_loss("quantile", quantile=0.9)
        assert isinstance(loss_fn, QuantileLoss)

    def test_unknown_loss_raises(self) -> None:
        """Test that unknown loss type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown loss type"):
            build_loss("unknown")
