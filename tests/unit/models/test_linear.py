"""Unit tests for LinearBaseline model."""

import pytest
import torch

from src.models.linear import LinearBaseline, LinearBaselineConfig


class TestLinearBaselineConfig:
    """Tests for LinearBaselineConfig."""

    def test_default_config(self):
        config = LinearBaselineConfig()
        assert config.dropout == 0.0

    def test_custom_config(self):
        config = LinearBaselineConfig(dropout=0.5)
        assert config.dropout == 0.5


class TestLinearBaseline:
    """Tests for LinearBaseline model."""

    @pytest.fixture
    def model(self):
        config = LinearBaselineConfig()
        return LinearBaseline(config=config, input_dim=29, output_horizons=3)

    def test_output_shape(self, model):
        """Output shape is (batch, nodes, horizons)."""
        x = torch.randn(4, 22, 42, 29)
        preds = model(x)
        assert preds.shape == (4, 42, 3)

    def test_forward_with_mask(self, model):
        """Masked forward pass produces correct shape."""
        x = torch.randn(4, 22, 42, 29)
        mask = torch.ones(4, 42, dtype=torch.bool)
        preds = model(x, mask)
        assert preds.shape == (4, 42, 3)

    def test_mask_zeros_invalid_nodes(self, model):
        """Invalid nodes (mask=False) have zero predictions."""
        x = torch.randn(4, 22, 42, 29)
        mask = torch.ones(4, 42, dtype=torch.bool)
        mask[:, 5:] = False

        model.eval()
        preds = model(x, mask)

        assert torch.all(preds[:, 5:, :] == 0)
        assert not torch.all(preds[:, :5, :] == 0)

    def test_only_uses_last_timestep(self, model):
        """Model only uses the last timestep of input."""
        model.eval()

        # Create two inputs with different histories but same last timestep
        x1 = torch.randn(4, 22, 42, 29)
        x2 = torch.randn(4, 22, 42, 29)
        x2[:, -1, :, :] = x1[:, -1, :, :]

        with torch.no_grad():
            preds1 = model(x1)
            preds2 = model(x2)

        assert torch.allclose(preds1, preds2)

    def test_gradient_flow(self, model):
        """Gradients flow through all parameters."""
        model.train()
        x = torch.randn(4, 22, 42, 29)
        preds = model(x)
        loss = preds.pow(2).mean()
        loss.backward()

        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"

    @pytest.mark.parametrize("batch_size", [1, 4, 16])
    def test_different_batch_sizes(self, model, batch_size):
        x = torch.randn(batch_size, 22, 42, 29)
        preds = model(x)
        assert preds.shape == (batch_size, 42, 3)

    @pytest.mark.parametrize("num_nodes", [1, 10, 42])
    def test_different_num_nodes(self, num_nodes):
        config = LinearBaselineConfig()
        model = LinearBaseline(config=config, input_dim=29, output_horizons=3)
        x = torch.randn(4, 22, num_nodes, 29)
        preds = model(x)
        assert preds.shape == (4, num_nodes, 3)

    @pytest.mark.parametrize("input_dim", [5, 29, 32])
    def test_different_input_dims(self, input_dim):
        config = LinearBaselineConfig()
        model = LinearBaseline(config=config, input_dim=input_dim, output_horizons=3)
        x = torch.randn(4, 22, 42, input_dim)
        preds = model(x)
        assert preds.shape == (4, 42, 3)

    @pytest.mark.parametrize("output_horizons", [1, 3, 5])
    def test_different_output_horizons(self, output_horizons):
        config = LinearBaselineConfig()
        model = LinearBaseline(config=config, input_dim=29, output_horizons=output_horizons)
        x = torch.randn(4, 22, 42, 29)
        preds = model(x)
        assert preds.shape == (4, 42, output_horizons)

    def test_properties(self, model):
        assert isinstance(model.config, LinearBaselineConfig)
        assert model.input_dim == 29
        assert model.output_horizons == 3

    def test_deterministic_eval_mode(self, model):
        """Two forward passes in eval mode produce identical output."""
        model.eval()
        x = torch.randn(4, 22, 42, 29)
        with torch.no_grad():
            preds1 = model(x)
            preds2 = model(x)
        assert torch.allclose(preds1, preds2)

    def test_train_mode_dropout(self):
        """With dropout, train mode produces stochastic output."""
        config = LinearBaselineConfig(dropout=0.5)
        model = LinearBaseline(config=config, input_dim=29, output_horizons=3)
        model.train()

        x = torch.randn(4, 22, 42, 29)
        # With high dropout, outputs should differ across forward passes
        preds1 = model(x)
        preds2 = model(x)
        # Not guaranteed to differ, but with 50% dropout on 29 features it's near certain
        assert not torch.allclose(preds1, preds2)

    def test_parameter_count(self, model):
        """Linear model has exactly input_dim * horizons + horizons parameters."""
        num_params = sum(p.numel() for p in model.parameters())
        expected = 29 * 3 + 3  # weight + bias
        assert num_params == expected
