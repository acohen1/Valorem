"""Unit tests for PatchTST_GNN_Ensemble model."""

import pytest
import torch
from torch_geometric.data import Data

from src.models.dataset import DEFAULT_FEATURE_COLS
from src.models.ensemble import PatchTST_GNN_Ensemble
from src.models.gnn import GNNModelConfig
from src.models.patchtst import PatchTSTModelConfig

N_FEATURES = len(DEFAULT_FEATURE_COLS)


class TestPatchTSTGNNEnsemble:
    """Tests for PatchTST_GNN_Ensemble model."""

    @pytest.fixture
    def simple_graph(self):
        """Create a simple graph for testing."""
        num_nodes = 10
        edges = []
        for i in range(num_nodes - 1):
            edges.append([i, i + 1])
            edges.append([i + 1, i])
        edge_index = torch.tensor(edges, dtype=torch.long).T
        edge_attr = torch.randn(edge_index.size(1), 2)
        return edge_index, edge_attr, num_nodes

    @pytest.fixture
    def default_model(self):
        """Create a default ensemble model."""
        patchtst_config = PatchTSTModelConfig(d_model=64, n_layers=1)
        gnn_config = GNNModelConfig(hidden_dim=32, n_layers=1)
        return PatchTST_GNN_Ensemble(
            patchtst_config=patchtst_config,
            gnn_config=gnn_config,
            input_dim=N_FEATURES,
            output_horizons=3,
        )

    def test_output_shape(self, default_model, simple_graph):
        """Test that output shape matches expected dimensions."""
        edge_index, edge_attr, num_nodes = simple_graph
        default_model.eval()

        x = torch.randn(4, 22, num_nodes, N_FEATURES)

        with torch.no_grad():
            preds = default_model(x, edge_index, edge_attr)

        assert preds.shape == (4, num_nodes, 3)

    def test_forward_with_mask(self, default_model, simple_graph):
        """Test forward pass with mask."""
        edge_index, edge_attr, num_nodes = simple_graph
        default_model.eval()

        x = torch.randn(4, 22, num_nodes, N_FEATURES)
        mask = torch.ones(4, num_nodes, dtype=torch.bool)

        with torch.no_grad():
            preds = default_model(x, edge_index, edge_attr, mask)

        assert preds.shape == (4, num_nodes, 3)

    def test_mask_zeros_invalid_nodes(self, default_model, simple_graph):
        """Test that mask correctly zeros out invalid nodes."""
        edge_index, edge_attr, num_nodes = simple_graph
        default_model.eval()

        x = torch.randn(4, 22, num_nodes, N_FEATURES)
        mask = torch.ones(4, num_nodes, dtype=torch.bool)
        mask[:, 5:] = False  # Invalid nodes 5-9

        with torch.no_grad():
            preds = default_model(x, edge_index, edge_attr, mask)

        # Invalid nodes should be zeroed
        assert torch.all(preds[:, 5:, :] == 0)
        # Valid nodes should have non-zero values (almost certainly)
        assert not torch.all(preds[:, :5, :] == 0)

    def test_encode_returns_embeddings(self, default_model, simple_graph):
        """Test encode method returns correct shape embeddings."""
        edge_index, edge_attr, num_nodes = simple_graph
        default_model.eval()

        x = torch.randn(4, 22, num_nodes, N_FEATURES)

        with torch.no_grad():
            embeddings = default_model.encode(x, edge_index, edge_attr)

        # Should return GNN hidden_dim (32)
        assert embeddings.shape == (4, num_nodes, 32)

    def test_gradient_flow_through_both_components(self, default_model, simple_graph):
        """Test that gradients flow through both PatchTST and GNN."""
        edge_index, edge_attr, num_nodes = simple_graph
        default_model.train()

        x = torch.randn(2, 22, num_nodes, N_FEATURES, requires_grad=True)

        preds = default_model(x, edge_index, edge_attr)
        loss = preds.sum()
        loss.backward()

        # Check gradients exist for PatchTST parameters
        patchtst_has_grad = False
        for name, param in default_model.patchtst.named_parameters():
            if param.grad is not None and not torch.all(param.grad == 0):
                patchtst_has_grad = True
                break
        assert patchtst_has_grad, "No gradients flowing through PatchTST"

        # Check gradients exist for GNN parameters
        gnn_has_grad = False
        for name, param in default_model.gnn.named_parameters():
            if param.grad is not None and not torch.all(param.grad == 0):
                gnn_has_grad = True
                break
        assert gnn_has_grad, "No gradients flowing through GNN"

        # Check input gradient
        assert x.grad is not None
        assert not torch.all(x.grad == 0)

    def test_different_batch_sizes(self, default_model, simple_graph):
        """Test model handles different batch sizes."""
        edge_index, edge_attr, num_nodes = simple_graph
        default_model.eval()

        for batch_size in [1, 4, 16]:
            x = torch.randn(batch_size, 22, num_nodes, N_FEATURES)
            with torch.no_grad():
                preds = default_model(x, edge_index, edge_attr)
            assert preds.shape == (batch_size, num_nodes, 3)

    def test_different_output_horizons(self, simple_graph):
        """Test model handles different numbers of output horizons."""
        edge_index, edge_attr, num_nodes = simple_graph

        for horizons in [1, 3, 5]:
            patchtst_config = PatchTSTModelConfig(d_model=64, n_layers=1)
            gnn_config = GNNModelConfig(hidden_dim=32, n_layers=1)
            model = PatchTST_GNN_Ensemble(
                patchtst_config=patchtst_config,
                gnn_config=gnn_config,
                input_dim=N_FEATURES,
                output_horizons=horizons,
            )
            model.eval()

            x = torch.randn(4, 22, num_nodes, N_FEATURES)
            with torch.no_grad():
                preds = model(x, edge_index, edge_attr)
            assert preds.shape == (4, num_nodes, horizons)

    def test_different_input_dims(self, simple_graph):
        """Test model handles different input feature dimensions."""
        edge_index, edge_attr, num_nodes = simple_graph

        for input_dim in [5, N_FEATURES, 32]:
            patchtst_config = PatchTSTModelConfig(d_model=64, n_layers=1)
            gnn_config = GNNModelConfig(hidden_dim=32, n_layers=1)
            model = PatchTST_GNN_Ensemble(
                patchtst_config=patchtst_config,
                gnn_config=gnn_config,
                input_dim=input_dim,
                output_horizons=3,
            )
            model.eval()

            x = torch.randn(4, 22, num_nodes, input_dim)
            with torch.no_grad():
                preds = model(x, edge_index, edge_attr)
            assert preds.shape == (4, num_nodes, 3)

    def test_gat_vs_gcn(self, simple_graph):
        """Test model works with both GAT and GCN."""
        edge_index, edge_attr, num_nodes = simple_graph
        patchtst_config = PatchTSTModelConfig(d_model=64, n_layers=1)

        for model_type in ["GAT", "GCN"]:
            gnn_config = GNNModelConfig(
                model_type=model_type,  # type: ignore[arg-type]
                hidden_dim=32,
                n_layers=1,
            )
            model = PatchTST_GNN_Ensemble(
                patchtst_config=patchtst_config,
                gnn_config=gnn_config,
                input_dim=N_FEATURES,
                output_horizons=3,
            )
            model.eval()

            x = torch.randn(4, 22, num_nodes, N_FEATURES)
            with torch.no_grad():
                preds = model(x, edge_index, edge_attr if model_type == "GAT" else None)
            assert preds.shape == (4, num_nodes, 3)

    def test_properties(self, default_model):
        """Test property accessors."""
        assert default_model.input_dim == N_FEATURES
        assert default_model.output_horizons == 3
        assert default_model.patchtst_config.d_model == 64
        assert default_model.gnn_config.hidden_dim == 32
        assert default_model.patchtst is not None
        assert default_model.gnn is not None

    def test_deterministic_eval_mode(self, default_model, simple_graph):
        """Test that model is deterministic in eval mode."""
        edge_index, edge_attr, num_nodes = simple_graph
        default_model.eval()

        x = torch.randn(2, 22, num_nodes, N_FEATURES)

        with torch.no_grad():
            preds1 = default_model(x, edge_index, edge_attr)
            preds2 = default_model(x, edge_index, edge_attr)

        assert torch.allclose(preds1, preds2)

    def test_train_mode_dropout(self, simple_graph):
        """Test that dropout creates different outputs in train mode."""
        edge_index, edge_attr, num_nodes = simple_graph

        patchtst_config = PatchTSTModelConfig(d_model=64, n_layers=2, dropout=0.5)
        gnn_config = GNNModelConfig(hidden_dim=32, n_layers=2, dropout=0.5)
        model = PatchTST_GNN_Ensemble(
            patchtst_config=patchtst_config,
            gnn_config=gnn_config,
            input_dim=N_FEATURES,
            output_horizons=3,
        )
        model.train()

        x = torch.randn(2, 22, num_nodes, N_FEATURES)

        preds1 = model(x, edge_index, edge_attr)
        preds2 = model(x, edge_index, edge_attr)

        # With high dropout, outputs should differ
        assert not torch.allclose(preds1, preds2)

    def test_without_edge_attr(self, simple_graph):
        """Test model works without edge attributes."""
        edge_index, _, num_nodes = simple_graph

        patchtst_config = PatchTSTModelConfig(d_model=64, n_layers=1)
        gnn_config = GNNModelConfig(hidden_dim=32, n_layers=1, use_edge_attr=False)
        model = PatchTST_GNN_Ensemble(
            patchtst_config=patchtst_config,
            gnn_config=gnn_config,
            input_dim=N_FEATURES,
            output_horizons=3,
        )
        model.eval()

        x = torch.randn(4, 22, num_nodes, N_FEATURES)

        with torch.no_grad():
            preds = model(x, edge_index)  # No edge_attr

        assert preds.shape == (4, num_nodes, 3)

    def test_larger_surface_graph(self):
        """Test model with a graph similar to real volatility surface."""
        # 42 nodes = 7 delta buckets x 6 tenors
        num_nodes = 42
        patchtst_config = PatchTSTModelConfig(d_model=128, n_layers=2)
        gnn_config = GNNModelConfig(hidden_dim=64, n_layers=2)
        model = PatchTST_GNN_Ensemble(
            patchtst_config=patchtst_config,
            gnn_config=gnn_config,
            input_dim=N_FEATURES,
            output_horizons=3,
        )
        model.eval()

        # Create grid-like connectivity (similar to real surface)
        edges = []
        num_deltas = 7
        num_tenors = 6
        for d in range(num_deltas):
            for t in range(num_tenors):
                node_idx = d * num_tenors + t
                # Connect to delta neighbor
                if d < num_deltas - 1:
                    neighbor_idx = (d + 1) * num_tenors + t
                    edges.append([node_idx, neighbor_idx])
                    edges.append([neighbor_idx, node_idx])
                # Connect to tenor neighbor
                if t < num_tenors - 1:
                    neighbor_idx = d * num_tenors + (t + 1)
                    edges.append([node_idx, neighbor_idx])
                    edges.append([neighbor_idx, node_idx])

        edge_index = torch.tensor(edges, dtype=torch.long).T
        edge_attr = torch.randn(edge_index.size(1), 2)

        x = torch.randn(8, 22, num_nodes, N_FEATURES)

        with torch.no_grad():
            preds = model(x, edge_index, edge_attr)

        assert preds.shape == (8, num_nodes, 3)

    # M3: Learnable edge attributes tests

    def test_learnable_edge_attr_is_parameter(self):
        """Test that edge_attr appears in model.parameters() when learnable."""
        # Create graph with edge attributes
        num_nodes = 10
        edges = [[i, i + 1] for i in range(num_nodes - 1)]
        edges += [[i + 1, i] for i in range(num_nodes - 1)]
        edge_index = torch.tensor(edges, dtype=torch.long).T
        edge_attr = torch.randn(edge_index.size(1), 2)
        graph = Data(edge_index=edge_index, edge_attr=edge_attr)

        # Create model with learnable edges
        patchtst_config = PatchTSTModelConfig(d_model=64, n_layers=1)
        gnn_config = GNNModelConfig(
            hidden_dim=32,
            n_layers=1,
            use_learnable_edge_attr=True,
        )
        model = PatchTST_GNN_Ensemble(
            patchtst_config=patchtst_config,
            gnn_config=gnn_config,
            input_dim=N_FEATURES,
            output_horizons=3,
            graph=graph,
        )

        # Verify edge_attr is a parameter
        param_names = [name for name, _ in model.named_parameters()]
        assert "_edge_attr" in param_names, "edge_attr should be in model.parameters()"

        # Verify it requires gradients
        assert model._edge_attr.requires_grad

    def test_fixed_edge_attr_is_buffer(self):
        """Test that edge_attr is a buffer when not learnable."""
        # Create graph with edge attributes
        num_nodes = 10
        edges = [[i, i + 1] for i in range(num_nodes - 1)]
        edges += [[i + 1, i] for i in range(num_nodes - 1)]
        edge_index = torch.tensor(edges, dtype=torch.long).T
        edge_attr = torch.randn(edge_index.size(1), 2)
        graph = Data(edge_index=edge_index, edge_attr=edge_attr)

        # Create model with fixed edges (default)
        patchtst_config = PatchTSTModelConfig(d_model=64, n_layers=1)
        gnn_config = GNNModelConfig(
            hidden_dim=32,
            n_layers=1,
            use_learnable_edge_attr=False,
        )
        model = PatchTST_GNN_Ensemble(
            patchtst_config=patchtst_config,
            gnn_config=gnn_config,
            input_dim=N_FEATURES,
            output_horizons=3,
            graph=graph,
        )

        # Verify edge_attr is NOT a parameter
        param_names = [name for name, _ in model.named_parameters()]
        assert "_edge_attr" not in param_names, "edge_attr should not be in parameters when fixed"

        # Verify it's a buffer
        buffer_names = [name for name, _ in model.named_buffers()]
        assert "_edge_attr" in buffer_names, "edge_attr should be a buffer when fixed"

        # Verify it doesn't require gradients
        assert not model._edge_attr.requires_grad

    def test_edge_attr_gradients_flow(self):
        """Test that gradients flow through learnable edge_attr."""
        # Create graph
        num_nodes = 10
        edges = [[i, i + 1] for i in range(num_nodes - 1)]
        edges += [[i + 1, i] for i in range(num_nodes - 1)]
        edge_index = torch.tensor(edges, dtype=torch.long).T
        edge_attr = torch.randn(edge_index.size(1), 2)
        graph = Data(edge_index=edge_index, edge_attr=edge_attr)

        # Create model with learnable edges
        patchtst_config = PatchTSTModelConfig(d_model=64, n_layers=1)
        gnn_config = GNNModelConfig(
            hidden_dim=32,
            n_layers=1,
            use_learnable_edge_attr=True,
        )
        model = PatchTST_GNN_Ensemble(
            patchtst_config=patchtst_config,
            gnn_config=gnn_config,
            input_dim=N_FEATURES,
            output_horizons=3,
            graph=graph,
        )
        model.train()

        # Forward pass with internal edges
        x = torch.randn(2, 22, num_nodes, N_FEATURES)
        preds = model(x)  # No edge args - using internal
        loss = preds.sum()
        loss.backward()

        # Verify gradients exist for edge_attr
        assert model._edge_attr.grad is not None, "edge_attr should have gradients"
        assert not torch.all(model._edge_attr.grad == 0), "edge_attr gradients should be non-zero"

    def test_edge_attr_updates_during_training(self):
        """Test that edge_attr values change after optimizer step."""
        # Create graph
        num_nodes = 10
        edges = [[i, i + 1] for i in range(num_nodes - 1)]
        edges += [[i + 1, i] for i in range(num_nodes - 1)]
        edge_index = torch.tensor(edges, dtype=torch.long).T
        edge_attr = torch.randn(edge_index.size(1), 2)
        graph = Data(edge_index=edge_index, edge_attr=edge_attr)

        # Create model with learnable edges
        patchtst_config = PatchTSTModelConfig(d_model=64, n_layers=1)
        gnn_config = GNNModelConfig(
            hidden_dim=32,
            n_layers=1,
            use_learnable_edge_attr=True,
        )
        model = PatchTST_GNN_Ensemble(
            patchtst_config=patchtst_config,
            gnn_config=gnn_config,
            input_dim=N_FEATURES,
            output_horizons=3,
            graph=graph,
        )
        model.train()

        # Store initial edge_attr values
        initial_edge_attr = model._edge_attr.data.clone()

        # Create optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        # Training step
        x = torch.randn(2, 22, num_nodes, N_FEATURES)
        preds = model(x)
        loss = preds.sum()
        loss.backward()
        optimizer.step()

        # Verify edge_attr changed
        assert not torch.allclose(
            model._edge_attr.data, initial_edge_attr
        ), "edge_attr should update after optimizer step"

    def test_ensemble_without_graph(self):
        """Test backward compatibility: ensemble works without graph parameter."""
        patchtst_config = PatchTSTModelConfig(d_model=64, n_layers=1)
        gnn_config = GNNModelConfig(hidden_dim=32, n_layers=1)

        # Old-style construction (no graph)
        model = PatchTST_GNN_Ensemble(
            patchtst_config=patchtst_config,
            gnn_config=gnn_config,
            input_dim=N_FEATURES,
            output_horizons=3,
        )
        model.eval()

        # Verify internal edges are None
        assert model._edge_index is None
        assert model._edge_attr is None

        # Create external graph
        num_nodes = 10
        edges = [[i, i + 1] for i in range(num_nodes - 1)]
        edges += [[i + 1, i] for i in range(num_nodes - 1)]
        edge_index = torch.tensor(edges, dtype=torch.long).T
        edge_attr = torch.randn(edge_index.size(1), 2)

        # Forward pass with external edges (old behavior)
        x = torch.randn(4, 22, num_nodes, N_FEATURES)
        with torch.no_grad():
            preds = model(x, edge_index, edge_attr)

        assert preds.shape == (4, num_nodes, 3)

    def test_external_edge_attr_override(self):
        """Test that external edge_attr can override internal."""
        # Create graph with internal edge attributes
        num_nodes = 10
        edges = [[i, i + 1] for i in range(num_nodes - 1)]
        edges += [[i + 1, i] for i in range(num_nodes - 1)]
        edge_index = torch.tensor(edges, dtype=torch.long).T
        edge_attr_internal = torch.ones(edge_index.size(1), 2)
        graph = Data(edge_index=edge_index, edge_attr=edge_attr_internal)

        # Create model with learnable edges
        patchtst_config = PatchTSTModelConfig(d_model=64, n_layers=1)
        gnn_config = GNNModelConfig(
            hidden_dim=32,
            n_layers=1,
            use_learnable_edge_attr=True,
        )
        model = PatchTST_GNN_Ensemble(
            patchtst_config=patchtst_config,
            gnn_config=gnn_config,
            input_dim=N_FEATURES,
            output_horizons=3,
            graph=graph,
        )
        model.eval()

        x = torch.randn(2, 22, num_nodes, N_FEATURES)

        # Run with internal edges
        with torch.no_grad():
            preds_internal = model(x)

        # Run with external edges (different values)
        edge_attr_external = torch.zeros(edge_index.size(1), 2)
        with torch.no_grad():
            preds_external = model(x, edge_index, edge_attr_external)

        # Predictions should differ (external overrode internal)
        assert not torch.allclose(preds_internal, preds_external), (
            "External edge_attr should produce different results"
        )
