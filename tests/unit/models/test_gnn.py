"""Unit tests for GNN model."""

import pytest
import torch

from src.models.dataset import DEFAULT_FEATURE_COLS
from src.models.gnn import GNNModelConfig, SurfaceGNN

N_FEATURES = len(DEFAULT_FEATURE_COLS)


class TestGNNModelConfig:
    """Tests for GNNModelConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = GNNModelConfig()

        assert config.model_type == "GAT"
        assert config.hidden_dim == 64
        assert config.n_layers == 2
        assert config.heads == 4
        assert config.dropout == 0.1
        assert config.use_edge_attr is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = GNNModelConfig(
            model_type="GCN",
            hidden_dim=128,
            n_layers=3,
            heads=8,
            dropout=0.2,
            use_edge_attr=False,
        )

        assert config.model_type == "GCN"
        assert config.hidden_dim == 128
        assert config.n_layers == 3
        assert config.heads == 8
        assert config.dropout == 0.2
        assert config.use_edge_attr is False


class TestSurfaceGNN:
    """Tests for SurfaceGNN model."""

    @pytest.fixture
    def simple_graph(self):
        """Create a simple graph for testing."""
        # 10 nodes with edges connecting neighbors
        num_nodes = 10
        edges = []
        for i in range(num_nodes - 1):
            edges.append([i, i + 1])
            edges.append([i + 1, i])  # Bidirectional
        edge_index = torch.tensor(edges, dtype=torch.long).T  # (2, num_edges)
        edge_attr = torch.randn(edge_index.size(1), 2)  # Random edge attributes
        return edge_index, edge_attr, num_nodes

    def test_gat_output_shape(self, simple_graph):
        """Test GAT model output shape."""
        edge_index, edge_attr, num_nodes = simple_graph
        config = GNNModelConfig(model_type="GAT", hidden_dim=64, n_layers=2)
        model = SurfaceGNN(config, input_dim=128)
        model.eval()

        x = torch.randn(4, num_nodes, 128)  # (batch, nodes, features)

        with torch.no_grad():
            out = model(x, edge_index, edge_attr)

        assert out.shape == (4, num_nodes, 64)

    def test_gcn_output_shape(self, simple_graph):
        """Test GCN model output shape."""
        edge_index, edge_attr, num_nodes = simple_graph
        config = GNNModelConfig(model_type="GCN", hidden_dim=64, n_layers=2)
        model = SurfaceGNN(config, input_dim=128)
        model.eval()

        x = torch.randn(4, num_nodes, 128)

        with torch.no_grad():
            out = model(x, edge_index)  # GCN doesn't use edge_attr

        assert out.shape == (4, num_nodes, 64)

    def test_gat_with_edge_attr(self, simple_graph):
        """Test GAT model uses edge attributes."""
        edge_index, edge_attr, num_nodes = simple_graph
        config = GNNModelConfig(model_type="GAT", hidden_dim=32, use_edge_attr=True)
        model = SurfaceGNN(config, input_dim=64)
        model.eval()

        x = torch.randn(2, num_nodes, 64)

        with torch.no_grad():
            # With edge_attr
            out_with_attr = model(x, edge_index, edge_attr)
            # Without edge_attr (should still work but different output)
            config_no_attr = GNNModelConfig(model_type="GAT", hidden_dim=32, use_edge_attr=False)
            model_no_attr = SurfaceGNN(config_no_attr, input_dim=64)
            model_no_attr.eval()
            out_no_attr = model_no_attr(x, edge_index)

        assert out_with_attr.shape == out_no_attr.shape

    def test_gat_without_edge_attr(self, simple_graph):
        """Test GAT model without edge attributes."""
        edge_index, _, num_nodes = simple_graph
        config = GNNModelConfig(model_type="GAT", hidden_dim=32, use_edge_attr=False)
        model = SurfaceGNN(config, input_dim=64)
        model.eval()

        x = torch.randn(2, num_nodes, 64)

        with torch.no_grad():
            out = model(x, edge_index)

        assert out.shape == (2, num_nodes, 32)

    def test_different_batch_sizes(self, simple_graph):
        """Test model handles different batch sizes."""
        edge_index, edge_attr, num_nodes = simple_graph
        config = GNNModelConfig(model_type="GAT", hidden_dim=32, n_layers=1)
        model = SurfaceGNN(config, input_dim=64)
        model.eval()

        for batch_size in [1, 4, 16]:
            x = torch.randn(batch_size, num_nodes, 64)
            with torch.no_grad():
                out = model(x, edge_index, edge_attr)
            assert out.shape == (batch_size, num_nodes, 32)

    def test_different_num_nodes(self):
        """Test model handles different numbers of nodes."""
        config = GNNModelConfig(model_type="GAT", hidden_dim=32, n_layers=1)
        model = SurfaceGNN(config, input_dim=64)
        model.eval()

        for num_nodes in [5, 10, 42]:
            # Create graph for this node count
            edges = []
            for i in range(num_nodes - 1):
                edges.append([i, i + 1])
                edges.append([i + 1, i])
            edge_index = torch.tensor(edges, dtype=torch.long).T
            edge_attr = torch.randn(edge_index.size(1), 2)

            x = torch.randn(2, num_nodes, 64)
            with torch.no_grad():
                out = model(x, edge_index, edge_attr)
            assert out.shape == (2, num_nodes, 32)

    def test_different_n_layers(self, simple_graph):
        """Test model with different numbers of layers."""
        edge_index, edge_attr, num_nodes = simple_graph

        for n_layers in [1, 2, 4]:
            config = GNNModelConfig(model_type="GAT", hidden_dim=32, n_layers=n_layers)
            model = SurfaceGNN(config, input_dim=64)
            model.eval()

            x = torch.randn(2, num_nodes, 64)
            with torch.no_grad():
                out = model(x, edge_index, edge_attr)
            assert out.shape == (2, num_nodes, 32)

    def test_gradient_flow_gat(self, simple_graph):
        """Test that gradients flow through GAT model."""
        edge_index, edge_attr, num_nodes = simple_graph
        config = GNNModelConfig(model_type="GAT", hidden_dim=32, n_layers=2)
        model = SurfaceGNN(config, input_dim=64)
        model.train()

        x = torch.randn(2, num_nodes, 64, requires_grad=True)

        out = model(x, edge_index, edge_attr)
        loss = out.sum()
        loss.backward()

        # Check gradients exist for all parameters
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            if param.numel() > 1:
                assert not torch.all(param.grad == 0), f"Zero gradient for {name}"

        # Check input gradient
        assert x.grad is not None
        assert not torch.all(x.grad == 0)

    def test_gradient_flow_gcn(self, simple_graph):
        """Test that gradients flow through GCN model."""
        edge_index, _, num_nodes = simple_graph
        config = GNNModelConfig(model_type="GCN", hidden_dim=32, n_layers=2)
        model = SurfaceGNN(config, input_dim=64)
        model.train()

        x = torch.randn(2, num_nodes, 64, requires_grad=True)

        out = model(x, edge_index)
        loss = out.sum()
        loss.backward()

        # Check gradients exist for all parameters
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"

        # Check input gradient
        assert x.grad is not None

    def test_properties(self, simple_graph):
        """Test property accessors."""
        config = GNNModelConfig(model_type="GAT", hidden_dim=64, n_layers=2)
        model = SurfaceGNN(config, input_dim=128)

        assert model.config == config
        assert model.input_dim == 128
        assert model.hidden_dim == 64

    def test_gnn_standalone_mode_with_head(self, simple_graph):
        """Test GNN with prediction head (standalone mode for ablation study)."""
        edge_index, edge_attr, num_nodes = simple_graph
        config = GNNModelConfig(model_type="GAT", hidden_dim=32, n_layers=2)
        model = SurfaceGNN(config, input_dim=N_FEATURES, output_horizons=3)
        model.eval()

        x = torch.randn(4, num_nodes, N_FEATURES)  # (batch, nodes, features)
        mask = torch.ones(4, num_nodes, dtype=torch.bool)

        with torch.no_grad():
            preds = model(x, edge_index, edge_attr, mask)

        # Check output shape (predictions, not embeddings)
        assert preds.shape == (4, num_nodes, 3)
        # Check output_horizons property
        assert model.output_horizons == 3
        # Valid nodes should have non-zero predictions
        assert not torch.all(preds == 0)

    def test_gnn_encoder_mode(self, simple_graph):
        """Test GNN in encoder-only mode (output_horizons=0, backward compatible)."""
        edge_index, edge_attr, num_nodes = simple_graph
        config = GNNModelConfig(model_type="GAT", hidden_dim=32, n_layers=2)
        model = SurfaceGNN(config, input_dim=N_FEATURES, output_horizons=0)
        model.eval()

        x = torch.randn(4, num_nodes, N_FEATURES)  # (batch, nodes, features)

        with torch.no_grad():
            embeddings = model(x, edge_index, edge_attr)

        # Check output shape (embeddings, not predictions)
        assert embeddings.shape == (4, num_nodes, 32)
        # Check output_horizons property
        assert model.output_horizons == 0

    def test_gnn_standalone_masking(self, simple_graph):
        """Test that masking works correctly in standalone mode."""
        edge_index, edge_attr, num_nodes = simple_graph
        config = GNNModelConfig(model_type="GAT", hidden_dim=32, n_layers=2)
        model = SurfaceGNN(config, input_dim=N_FEATURES, output_horizons=3)
        model.eval()

        x = torch.randn(4, num_nodes, N_FEATURES)
        # Create mask with some nodes invalid
        mask = torch.ones(4, num_nodes, dtype=torch.bool)
        mask[0, 0] = False  # First node in first sample is invalid
        mask[1, 5] = False  # Sixth node in second sample is invalid

        with torch.no_grad():
            preds = model(x, edge_index, edge_attr, mask)

        # Masked nodes should have zero predictions
        assert torch.all(preds[0, 0, :] == 0)
        assert torch.all(preds[1, 5, :] == 0)
        # Valid nodes should have non-zero predictions
        assert not torch.all(preds[0, 1:, :] == 0)
        assert not torch.all(preds[1, :5, :] == 0)

    def test_deterministic_eval_mode(self, simple_graph):
        """Test that model is deterministic in eval mode."""
        edge_index, edge_attr, num_nodes = simple_graph
        config = GNNModelConfig(model_type="GAT", hidden_dim=32, dropout=0.1)
        model = SurfaceGNN(config, input_dim=64)
        model.eval()

        x = torch.randn(2, num_nodes, 64)

        with torch.no_grad():
            out1 = model(x, edge_index, edge_attr)
            out2 = model(x, edge_index, edge_attr)

        assert torch.allclose(out1, out2)

    def test_train_mode_dropout(self, simple_graph):
        """Test that dropout creates different outputs in train mode."""
        edge_index, edge_attr, num_nodes = simple_graph
        config = GNNModelConfig(model_type="GAT", hidden_dim=32, n_layers=2, dropout=0.5)
        model = SurfaceGNN(config, input_dim=64)
        model.train()

        x = torch.randn(2, num_nodes, 64)

        out1 = model(x, edge_index, edge_attr)
        out2 = model(x, edge_index, edge_attr)

        # With high dropout, outputs should differ
        assert not torch.allclose(out1, out2)

    def test_single_node_graph(self):
        """Test model handles single node graph."""
        config = GNNModelConfig(model_type="GAT", hidden_dim=32, n_layers=1)
        model = SurfaceGNN(config, input_dim=64)
        model.eval()

        # Single node with self-loop
        edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        edge_attr = torch.randn(1, 2)
        x = torch.randn(2, 1, 64)

        with torch.no_grad():
            out = model(x, edge_index, edge_attr)

        assert out.shape == (2, 1, 32)

    def test_disconnected_graph(self):
        """Test model handles disconnected graph (no edges)."""
        config = GNNModelConfig(model_type="GAT", hidden_dim=32, n_layers=1)
        model = SurfaceGNN(config, input_dim=64)
        model.eval()

        # No edges
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 2))
        x = torch.randn(2, 5, 64)

        with torch.no_grad():
            out = model(x, edge_index, edge_attr)

        # Should still produce output (just no message passing)
        assert out.shape == (2, 5, 32)
