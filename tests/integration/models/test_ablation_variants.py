"""Integration tests for M1 ablation study variants.

Tests all four model variants (Linear, PatchTST-only, GNN-only, Full ensemble)
to ensure they produce compatible outputs and can be trained end-to-end.
"""

import pytest
import torch

from src.models import (
    GNNModelConfig,
    LinearBaseline,
    LinearBaselineConfig,
    PatchTST_GNN_Ensemble,
    PatchTSTModel,
    PatchTSTModelConfig,
    SurfaceGraphConfig,
    build_surface_graph,
)
from src.models.dataset import DEFAULT_FEATURE_COLS, GNN_ABLATION_FEATURE_COLS
from src.models.gnn import SurfaceGNN

N_FEATURES = len(DEFAULT_FEATURE_COLS)
N_GNN_FEATURES = len(GNN_ABLATION_FEATURE_COLS)


class TestAblationVariants:
    """Test all three ablation variants produce valid outputs."""

    @pytest.fixture
    def graph(self):
        """Create test graph."""
        config = SurfaceGraphConfig()
        return build_surface_graph(config)

    @pytest.fixture
    def test_batch(self, graph):
        """Create test batch matching expected data format (all 31 features)."""
        return {
            "X": torch.randn(4, 22, graph.num_nodes, N_FEATURES),
            "y": torch.randn(4, graph.num_nodes, 3),
            "mask": torch.ones(4, graph.num_nodes, dtype=torch.bool),
        }

    @pytest.fixture
    def gnn_test_batch(self, graph):
        """Create test batch for GNN ablation (22 node-specific features)."""
        return {
            "X": torch.randn(4, 22, graph.num_nodes, N_GNN_FEATURES),
            "y": torch.randn(4, graph.num_nodes, 3),
            "mask": torch.ones(4, graph.num_nodes, dtype=torch.bool),
        }

    def test_patchtst_only_variant(self, test_batch):
        """Test PatchTST-only ablation variant (temporal baseline)."""
        config = PatchTSTModelConfig(d_model=64, n_layers=2)
        model = PatchTSTModel(
            config=config,
            input_dim=N_FEATURES,
            output_horizons=3,
        )
        model.eval()

        with torch.no_grad():
            preds = model(test_batch["X"], test_batch["mask"])

        # Check output shape
        assert preds.shape == test_batch["y"].shape
        # Valid nodes should have non-zero predictions
        assert not torch.all(preds == 0)
        # Masked nodes should be zeroed
        assert torch.all(preds[~test_batch["mask"]] == 0)

    def test_gnn_only_variant(self, graph, gnn_test_batch):
        """Test GNN-only ablation variant (cross-sectional baseline)."""
        config = GNNModelConfig(hidden_dim=32, n_layers=2)
        model = SurfaceGNN(
            config=config,
            input_dim=N_GNN_FEATURES,
            output_horizons=3,
        )
        model.eval()

        # Last timestep only (true cross-sectional baseline)
        X_agg = gnn_test_batch["X"][:, -1, :, :]

        with torch.no_grad():
            preds = model(
                X_agg,
                graph.edge_index,
                graph.edge_attr,
                gnn_test_batch["mask"],
            )

        # Check output shape
        assert preds.shape == gnn_test_batch["y"].shape
        # Valid nodes should have non-zero predictions
        assert not torch.all(preds == 0)
        # Masked nodes should be zeroed
        assert torch.all(preds[~gnn_test_batch["mask"]] == 0)

    def test_ensemble_variant(self, graph, test_batch):
        """Test full ensemble ablation variant (combined model)."""
        patchtst_config = PatchTSTModelConfig(d_model=64, n_layers=2)
        gnn_config = GNNModelConfig(hidden_dim=32, n_layers=2)
        model = PatchTST_GNN_Ensemble(
            patchtst_config=patchtst_config,
            gnn_config=gnn_config,
            input_dim=N_FEATURES,
            output_horizons=3,
        )
        model.eval()

        with torch.no_grad():
            preds = model(
                test_batch["X"],
                graph.edge_index,
                graph.edge_attr,
                test_batch["mask"],
            )

        # Check output shape
        assert preds.shape == test_batch["y"].shape
        # Valid nodes should have non-zero predictions
        assert not torch.all(preds == 0)
        # Masked nodes should be zeroed
        assert torch.all(preds[~test_batch["mask"]] == 0)

    def test_all_variants_have_same_output_shape(self, graph, test_batch, gnn_test_batch):
        """Ensure all four variants produce compatible output shapes.

        Linear, PatchTST, and ensemble use all features; GNN uses node-specific.
        Despite different input dimensions, output shapes must match.
        """
        # Linear baseline
        linear_model = LinearBaseline(
            LinearBaselineConfig(),
            input_dim=N_FEATURES,
            output_horizons=3,
        )

        # PatchTST-only
        patchtst_model = PatchTSTModel(
            PatchTSTModelConfig(d_model=64),
            input_dim=N_FEATURES,
            output_horizons=3,
        )

        # GNN-only (node-specific features)
        gnn_model = SurfaceGNN(
            GNNModelConfig(hidden_dim=32),
            input_dim=N_GNN_FEATURES,
            output_horizons=3,
        )

        # Ensemble
        ensemble_model = PatchTST_GNN_Ensemble(
            PatchTSTModelConfig(d_model=64),
            GNNModelConfig(hidden_dim=32),
            input_dim=N_FEATURES,
            output_horizons=3,
        )

        with torch.no_grad():
            preds0 = linear_model(test_batch["X"], test_batch["mask"])
            preds1 = patchtst_model(test_batch["X"], test_batch["mask"])
            preds2 = gnn_model(
                gnn_test_batch["X"][:, -1, :, :],
                graph.edge_index,
                graph.edge_attr,
                gnn_test_batch["mask"],
            )
            preds3 = ensemble_model(
                test_batch["X"],
                graph.edge_index,
                graph.edge_attr,
                test_batch["mask"],
            )

        # All should have same output shape
        assert (
            preds0.shape == preds1.shape == preds2.shape == preds3.shape == test_batch["y"].shape
        )

    def test_gnn_only_backward_pass(self, graph, gnn_test_batch):
        """Test GNN-only variant can compute gradients (trainable)."""
        config = GNNModelConfig(hidden_dim=32, n_layers=2)
        model = SurfaceGNN(
            config=config,
            input_dim=N_GNN_FEATURES,
            output_horizons=3,
        )
        model.train()

        # Last timestep only (true cross-sectional baseline)
        X_agg = gnn_test_batch["X"][:, -1, :, :]

        # Forward pass
        preds = model(
            X_agg,
            graph.edge_index,
            graph.edge_attr,
            gnn_test_batch["mask"],
        )

        # Compute loss and backward
        loss = (preds - gnn_test_batch["y"]).pow(2).mean()
        loss.backward()

        # Check gradients exist
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"

    def test_patchtst_only_backward_pass(self, test_batch):
        """Test PatchTST-only variant can compute gradients (trainable)."""
        config = PatchTSTModelConfig(d_model=64, n_layers=2)
        model = PatchTSTModel(
            config=config,
            input_dim=N_FEATURES,
            output_horizons=3,
        )
        model.train()

        # Forward pass
        preds = model(test_batch["X"], test_batch["mask"])

        # Compute loss and backward
        loss = (preds - test_batch["y"]).pow(2).mean()
        loss.backward()

        # Check gradients exist
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"

    def test_ensemble_backward_pass(self, graph, test_batch):
        """Test full ensemble can compute gradients (trainable)."""
        patchtst_config = PatchTSTModelConfig(d_model=64, n_layers=2)
        gnn_config = GNNModelConfig(hidden_dim=32, n_layers=2)
        model = PatchTST_GNN_Ensemble(
            patchtst_config=patchtst_config,
            gnn_config=gnn_config,
            input_dim=N_FEATURES,
            output_horizons=3,
        )
        model.train()

        # Forward pass
        preds = model(
            test_batch["X"],
            graph.edge_index,
            graph.edge_attr,
            test_batch["mask"],
        )

        # Compute loss and backward
        loss = (preds - test_batch["y"]).pow(2).mean()
        loss.backward()

        # Check gradients exist
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"

    def test_linear_only_variant(self, test_batch):
        """Test linear-only ablation variant (feature baseline)."""
        config = LinearBaselineConfig()
        model = LinearBaseline(
            config=config,
            input_dim=N_FEATURES,
            output_horizons=3,
        )
        model.eval()

        with torch.no_grad():
            preds = model(test_batch["X"], test_batch["mask"])

        # Check output shape
        assert preds.shape == test_batch["y"].shape
        # Valid nodes should have non-zero predictions
        assert not torch.all(preds == 0)
        # Masked nodes should be zeroed
        assert torch.all(preds[~test_batch["mask"]] == 0)

    def test_linear_only_backward_pass(self, test_batch):
        """Test linear-only variant can compute gradients (trainable)."""
        config = LinearBaselineConfig()
        model = LinearBaseline(
            config=config,
            input_dim=N_FEATURES,
            output_horizons=3,
        )
        model.train()

        # Forward pass
        preds = model(test_batch["X"], test_batch["mask"])

        # Compute loss and backward
        loss = (preds - test_batch["y"]).pow(2).mean()
        loss.backward()

        # Check gradients exist
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"

    def test_gnn_ablation_feature_cols_subset(self):
        """Verify GNN ablation features are a valid node-specific subset."""
        assert set(GNN_ABLATION_FEATURE_COLS).issubset(set(DEFAULT_FEATURE_COLS))
        assert len(GNN_ABLATION_FEATURE_COLS) == 20

        # Verify no global features present
        global_features = {
            "underlying_rv_5d", "underlying_rv_10d", "underlying_rv_21d",
            "VIXCLS_level", "VIXCLS_change_1w",
            "DGS10_level", "DGS10_change_1w",
            "DGS2_level", "DGS2_change_1w",
        }
        assert set(GNN_ABLATION_FEATURE_COLS).isdisjoint(global_features)
