"""Unit tests for PatchTST model."""

import pytest
import torch

from src.models.dataset import DEFAULT_FEATURE_COLS
from src.models.patchtst import PatchEmbedding, PatchTSTModel, PatchTSTModelConfig

N_FEATURES = len(DEFAULT_FEATURE_COLS)


class TestPatchEmbedding:
    """Tests for PatchEmbedding layer."""

    def test_output_shape(self):
        """Test that output shape matches expected dimensions."""
        embed = PatchEmbedding(patch_len=12, stride=6, input_dim=N_FEATURES, d_model=128)
        x = torch.randn(32, 22, N_FEATURES)  # (batch, time, features)

        out = embed(x)

        # num_patches = (22 - 12) // 6 + 1 = 2
        assert out.shape == (32, 2, 128)

    def test_num_patches_calculation(self):
        """Test number of patches calculation."""
        embed = PatchEmbedding(patch_len=12, stride=6, input_dim=N_FEATURES, d_model=128)

        # (22 - 12) // 6 + 1 = 2
        assert embed.compute_num_patches(22) == 2

        # (30 - 12) // 6 + 1 = 4
        assert embed.compute_num_patches(30) == 4

        # (12 - 12) // 6 + 1 = 1
        assert embed.compute_num_patches(12) == 1

    def test_different_patch_lengths(self):
        """Test with different patch lengths."""
        # Longer patch
        embed = PatchEmbedding(patch_len=16, stride=8, input_dim=N_FEATURES, d_model=64)
        x = torch.randn(16, 32, N_FEATURES)

        out = embed(x)

        # num_patches = (32 - 16) // 8 + 1 = 3
        assert out.shape == (16, 3, 64)

    def test_different_strides(self):
        """Test with different stride values."""
        # Non-overlapping patches
        embed = PatchEmbedding(patch_len=10, stride=10, input_dim=8, d_model=64)
        x = torch.randn(8, 30, 8)

        out = embed(x)

        # num_patches = (30 - 10) // 10 + 1 = 3
        assert out.shape == (8, 3, 64)

    def test_single_patch(self):
        """Test when input results in single patch."""
        embed = PatchEmbedding(patch_len=12, stride=6, input_dim=N_FEATURES, d_model=128)
        x = torch.randn(4, 12, N_FEATURES)  # Exact patch length

        out = embed(x)

        # num_patches = (12 - 12) // 6 + 1 = 1
        assert out.shape == (4, 1, 128)

    def test_positional_encoding_shape(self):
        """Test that positional encoding has correct shape."""
        embed = PatchEmbedding(patch_len=12, stride=6, input_dim=N_FEATURES, d_model=128)

        assert embed._pos_encoding.shape == (embed._max_num_patches, 128)

    def test_input_too_short_raises(self):
        """Test that input shorter than patch_len raises error."""
        embed = PatchEmbedding(patch_len=12, stride=6, input_dim=N_FEATURES, d_model=128)
        x = torch.randn(4, 10, N_FEATURES)  # 10 < 12

        with pytest.raises(ValueError, match="Time steps.*must be >= patch_len"):
            embed(x)

    def test_properties(self):
        """Test property accessors."""
        embed = PatchEmbedding(patch_len=12, stride=6, input_dim=N_FEATURES, d_model=128)

        assert embed.patch_len == 12
        assert embed.stride == 6
        assert embed.d_model == 128

    def test_gradient_flow(self):
        """Test that gradients flow through the layer."""
        embed = PatchEmbedding(patch_len=12, stride=6, input_dim=N_FEATURES, d_model=128)
        x = torch.randn(4, 22, N_FEATURES, requires_grad=True)

        out = embed(x)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.all(x.grad == 0)


class TestPatchTSTModelConfig:
    """Tests for PatchTSTModelConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = PatchTSTModelConfig()

        assert config.patch_len == 12
        assert config.stride == 6
        assert config.d_model == 128
        assert config.n_heads == 8
        assert config.d_ff == 256
        assert config.n_layers == 3
        assert config.dropout == 0.1

    def test_custom_config(self):
        """Test custom configuration."""
        config = PatchTSTModelConfig(
            patch_len=16,
            stride=8,
            d_model=256,
            n_heads=4,
            d_ff=512,
            n_layers=6,
            dropout=0.2,
        )

        assert config.patch_len == 16
        assert config.stride == 8
        assert config.d_model == 256
        assert config.n_heads == 4
        assert config.d_ff == 512
        assert config.n_layers == 6
        assert config.dropout == 0.2


class TestPatchTSTModel:
    """Tests for PatchTSTModel."""

    def test_output_shape(self):
        """Test that output shape matches expected dimensions."""
        config = PatchTSTModelConfig(d_model=64, n_layers=1)
        model = PatchTSTModel(config, input_dim=N_FEATURES, output_horizons=3)
        model.eval()

        x = torch.randn(32, 22, 42, N_FEATURES)  # (batch, time, nodes, features)

        with torch.no_grad():
            preds = model(x)

        assert preds.shape == (32, 42, 3)

    def test_forward_with_mask(self):
        """Test forward pass with mask."""
        config = PatchTSTModelConfig(d_model=64, n_layers=1)
        model = PatchTSTModel(config, input_dim=N_FEATURES, output_horizons=3)
        model.eval()

        x = torch.randn(4, 22, 10, N_FEATURES)
        mask = torch.ones(4, 10, dtype=torch.bool)

        with torch.no_grad():
            preds = model(x, mask)

        assert preds.shape == (4, 10, 3)

    def test_mask_zeros_invalid_nodes(self):
        """Test that mask correctly zeros out invalid nodes."""
        config = PatchTSTModelConfig(d_model=64, n_layers=1)
        model = PatchTSTModel(config, input_dim=N_FEATURES, output_horizons=3)
        model.eval()

        x = torch.randn(4, 22, 10, N_FEATURES)
        mask = torch.ones(4, 10, dtype=torch.bool)
        mask[:, 5:] = False  # Invalid nodes 5-9

        with torch.no_grad():
            preds = model(x, mask)

        # Invalid nodes should be zeroed
        assert torch.all(preds[:, 5:, :] == 0)
        # Valid nodes should have non-zero values (almost certainly)
        assert not torch.all(preds[:, :5, :] == 0)

    def test_encode_returns_embeddings(self):
        """Test encode method returns correct shape embeddings."""
        config = PatchTSTModelConfig(d_model=64, n_layers=1)
        model = PatchTSTModel(config, input_dim=N_FEATURES, output_horizons=3)
        model.eval()

        x = torch.randn(4, 22, 10, N_FEATURES)

        with torch.no_grad():
            embeddings = model.encode(x)

        assert embeddings.shape == (4, 10, 64)  # (batch, nodes, d_model)

    def test_encode_no_head_model(self):
        """Test that model with output_horizons=0 only supports encode."""
        config = PatchTSTModelConfig(d_model=64, n_layers=1)
        model = PatchTSTModel(config, input_dim=N_FEATURES, output_horizons=0)
        model.eval()

        x = torch.randn(4, 22, 10, N_FEATURES)

        # encode should work
        with torch.no_grad():
            embeddings = model.encode(x)
        assert embeddings.shape == (4, 10, 64)

        # forward should raise
        with pytest.raises(ValueError, match="Cannot call forward"):
            model(x)

    def test_gradient_flow(self):
        """Test that gradients flow through all model layers."""
        config = PatchTSTModelConfig(d_model=64, n_layers=2)
        model = PatchTSTModel(config, input_dim=N_FEATURES, output_horizons=3)
        model.train()

        x = torch.randn(2, 22, 5, N_FEATURES, requires_grad=True)

        preds = model(x)
        loss = preds.sum()
        loss.backward()

        # Check gradients exist for all parameters
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            # Most gradients should be non-zero
            if param.numel() > 1:
                assert not torch.all(param.grad == 0), f"Zero gradient for {name}"

        # Check input gradient
        assert x.grad is not None
        assert not torch.all(x.grad == 0)

    def test_different_batch_sizes(self):
        """Test model handles different batch sizes."""
        config = PatchTSTModelConfig(d_model=64, n_layers=1)
        model = PatchTSTModel(config, input_dim=N_FEATURES, output_horizons=3)
        model.eval()

        for batch_size in [1, 4, 16, 64]:
            x = torch.randn(batch_size, 22, 10, N_FEATURES)
            with torch.no_grad():
                preds = model(x)
            assert preds.shape == (batch_size, 10, 3)

    def test_different_num_nodes(self):
        """Test model handles different numbers of nodes."""
        config = PatchTSTModelConfig(d_model=64, n_layers=1)
        model = PatchTSTModel(config, input_dim=N_FEATURES, output_horizons=3)
        model.eval()

        for num_nodes in [1, 10, 42, 100]:
            x = torch.randn(4, 22, num_nodes, N_FEATURES)
            with torch.no_grad():
                preds = model(x)
            assert preds.shape == (4, num_nodes, 3)

    def test_different_input_dims(self):
        """Test model handles different input feature dimensions."""
        for input_dim in [5, N_FEATURES, 32]:
            config = PatchTSTModelConfig(d_model=64, n_layers=1)
            model = PatchTSTModel(config, input_dim=input_dim, output_horizons=3)
            model.eval()

            x = torch.randn(4, 22, 10, input_dim)
            with torch.no_grad():
                preds = model(x)
            assert preds.shape == (4, 10, 3)

    def test_different_output_horizons(self):
        """Test model handles different numbers of output horizons."""
        for horizons in [1, 3, 5]:
            config = PatchTSTModelConfig(d_model=64, n_layers=1)
            model = PatchTSTModel(config, input_dim=N_FEATURES, output_horizons=horizons)
            model.eval()

            x = torch.randn(4, 22, 10, N_FEATURES)
            with torch.no_grad():
                preds = model(x)
            assert preds.shape == (4, 10, horizons)

    def test_properties(self):
        """Test property accessors."""
        config = PatchTSTModelConfig(d_model=64, n_layers=2)
        model = PatchTSTModel(config, input_dim=N_FEATURES, output_horizons=3)

        assert model.config == config
        assert model.input_dim == N_FEATURES
        assert model.output_horizons == 3
        assert model.d_model == 64

    def test_deterministic_with_eval_mode(self):
        """Test that model is deterministic in eval mode."""
        config = PatchTSTModelConfig(d_model=64, n_layers=1, dropout=0.1)
        model = PatchTSTModel(config, input_dim=N_FEATURES, output_horizons=3)
        model.eval()

        x = torch.randn(4, 22, 10, N_FEATURES)

        with torch.no_grad():
            preds1 = model(x)
            preds2 = model(x)

        assert torch.allclose(preds1, preds2)

    def test_train_mode_dropout(self):
        """Test that dropout creates different outputs in train mode."""
        config = PatchTSTModelConfig(d_model=64, n_layers=2, dropout=0.5)
        model = PatchTSTModel(config, input_dim=N_FEATURES, output_horizons=3)
        model.train()

        x = torch.randn(4, 22, 10, N_FEATURES)

        preds1 = model(x)
        preds2 = model(x)

        # With high dropout, outputs should differ
        assert not torch.allclose(preds1, preds2)
