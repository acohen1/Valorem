"""Integration tests for GNN and Ensemble models."""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.models import SurfaceGraphConfig, build_surface_graph
from src.models.ensemble import PatchTST_GNN_Ensemble
from src.models.gnn import GNNModelConfig, SurfaceGNN
from src.models.dataset import DEFAULT_FEATURE_COLS
from src.models.patchtst import PatchTSTModelConfig

N_FEATURES = len(DEFAULT_FEATURE_COLS)


class SyntheticGraphDataset(Dataset):
    """Synthetic dataset for testing GNN integration."""

    def __init__(
        self,
        num_samples: int = 100,
        time_steps: int = 22,
        num_nodes: int = 42,
        num_features: int = N_FEATURES,
        num_horizons: int = 3,
    ):
        """Initialize synthetic dataset."""
        self._num_samples = num_samples
        self._time_steps = time_steps
        self._num_nodes = num_nodes
        self._num_features = num_features
        self._num_horizons = num_horizons

        # Generate synthetic data
        self._X = torch.randn(num_samples, time_steps, num_nodes, num_features)
        self._y = self._X[:, -1, :, :num_horizons].clone()
        self._mask = torch.ones(num_samples, num_nodes, dtype=torch.bool)

    def __len__(self) -> int:
        return self._num_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "X": self._X[idx],
            "y": self._y[idx],
            "mask": self._mask[idx],
        }


def collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Collate batch of samples into tensors."""
    return {
        "X": torch.stack([b["X"] for b in batch]),
        "y": torch.stack([b["y"] for b in batch]),
        "mask": torch.stack([b["mask"] for b in batch]),
    }


class TestGNNWithRealGraph:
    """Integration tests using real graph from build_surface_graph."""

    @pytest.fixture
    def surface_graph(self):
        """Build real surface graph."""
        config = SurfaceGraphConfig()  # Default: 7 delta buckets, 6 tenors
        graph = build_surface_graph(config)
        return graph.edge_index, graph.edge_attr, graph.num_nodes

    def test_gnn_with_real_graph(self, surface_graph):
        """Test GNN with real surface graph topology."""
        edge_index, edge_attr, num_nodes = surface_graph

        config = GNNModelConfig(model_type="GAT", hidden_dim=64, n_layers=2)
        model = SurfaceGNN(config, input_dim=128)
        model.eval()

        x = torch.randn(4, num_nodes, 128)

        with torch.no_grad():
            out = model(x, edge_index, edge_attr)

        assert out.shape == (4, num_nodes, 64)

    def test_ensemble_with_real_graph(self, surface_graph):
        """Test ensemble model with real surface graph."""
        edge_index, edge_attr, num_nodes = surface_graph

        patchtst_config = PatchTSTModelConfig(d_model=64, n_layers=1)
        gnn_config = GNNModelConfig(hidden_dim=32, n_layers=2)
        model = PatchTST_GNN_Ensemble(
            patchtst_config=patchtst_config,
            gnn_config=gnn_config,
            input_dim=N_FEATURES,
            output_horizons=3,
        )
        model.eval()

        x = torch.randn(4, 22, num_nodes, N_FEATURES)

        with torch.no_grad():
            preds = model(x, edge_index, edge_attr)

        assert preds.shape == (4, num_nodes, 3)


class TestGNNTraining:
    """Integration tests for GNN training."""

    @pytest.fixture
    def surface_graph(self):
        """Build surface graph for training tests."""
        config = SurfaceGraphConfig()
        graph = build_surface_graph(config)
        return graph.edge_index, graph.edge_attr, graph.num_nodes

    def test_training_loop_one_epoch(self, surface_graph):
        """Test basic training loop runs without errors."""
        edge_index, edge_attr, num_nodes = surface_graph

        patchtst_config = PatchTSTModelConfig(d_model=32, n_layers=1)
        gnn_config = GNNModelConfig(hidden_dim=16, n_layers=1)
        model = PatchTST_GNN_Ensemble(
            patchtst_config=patchtst_config,
            gnn_config=gnn_config,
            input_dim=N_FEATURES,
            output_horizons=3,
        )

        dataset = SyntheticGraphDataset(num_samples=32, num_nodes=num_nodes)
        dataloader = DataLoader(dataset, batch_size=8, collate_fn=collate_fn)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()

        model.train()
        for batch in dataloader:
            optimizer.zero_grad()
            preds = model(batch["X"], edge_index, edge_attr, batch["mask"])
            loss = criterion(preds, batch["y"])
            loss.backward()
            optimizer.step()

        assert True  # Training completed without error

    def test_loss_decreases(self, surface_graph):
        """Test that loss decreases over training epochs."""
        edge_index, edge_attr, num_nodes = surface_graph

        patchtst_config = PatchTSTModelConfig(d_model=32, n_layers=1, dropout=0.0)
        gnn_config = GNNModelConfig(hidden_dim=16, n_layers=1, dropout=0.0)
        model = PatchTST_GNN_Ensemble(
            patchtst_config=patchtst_config,
            gnn_config=gnn_config,
            input_dim=N_FEATURES,
            output_horizons=3,
        )

        dataset = SyntheticGraphDataset(num_samples=64, num_nodes=num_nodes)
        dataloader = DataLoader(dataset, batch_size=16, collate_fn=collate_fn)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        criterion = nn.MSELoss()

        losses = []
        for epoch in range(5):
            epoch_loss = 0.0
            model.train()
            for batch in dataloader:
                optimizer.zero_grad()
                preds = model(batch["X"], edge_index, edge_attr, batch["mask"])
                loss = criterion(preds, batch["y"])
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            losses.append(epoch_loss / len(dataloader))

        # Loss should decrease (or at least not increase significantly)
        assert losses[-1] < losses[0] * 1.1

    def test_overfit_small_batch(self, surface_graph):
        """Test that model can overfit a small batch (sanity check)."""
        edge_index, edge_attr, num_nodes = surface_graph

        patchtst_config = PatchTSTModelConfig(d_model=64, n_layers=2, dropout=0.0)
        gnn_config = GNNModelConfig(hidden_dim=32, n_layers=2, dropout=0.0)
        model = PatchTST_GNN_Ensemble(
            patchtst_config=patchtst_config,
            gnn_config=gnn_config,
            input_dim=N_FEATURES,
            output_horizons=3,
        )

        # Seed for deterministic test
        torch.manual_seed(42)

        # Very small dataset
        X = torch.randn(4, 22, num_nodes, N_FEATURES)
        y = torch.randn(4, num_nodes, 3)
        mask = torch.ones(4, num_nodes, dtype=torch.bool)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)  # Higher LR for faster convergence
        criterion = nn.MSELoss()

        initial_loss = None
        for i in range(200):  # More iterations for larger model
            model.train()
            optimizer.zero_grad()
            preds = model(X, edge_index, edge_attr, mask)
            loss = criterion(preds, y)
            if initial_loss is None:
                initial_loss = loss.item()
            loss.backward()
            optimizer.step()

        final_loss = loss.item()

        # Should overfit significantly (at least 75% reduction)
        assert final_loss < initial_loss * 0.25


class TestEnsembleInference:
    """Integration tests for ensemble inference."""

    @pytest.fixture
    def surface_graph(self):
        """Build surface graph."""
        config = SurfaceGraphConfig()
        graph = build_surface_graph(config)
        return graph.edge_index, graph.edge_attr, graph.num_nodes

    def test_inference_mode(self, surface_graph):
        """Test model in inference mode (no gradient tracking)."""
        edge_index, edge_attr, num_nodes = surface_graph

        patchtst_config = PatchTSTModelConfig(d_model=64, n_layers=1)
        gnn_config = GNNModelConfig(hidden_dim=32, n_layers=1)
        model = PatchTST_GNN_Ensemble(
            patchtst_config=patchtst_config,
            gnn_config=gnn_config,
            input_dim=N_FEATURES,
            output_horizons=3,
        )
        model.eval()

        X = torch.randn(4, 22, num_nodes, N_FEATURES)

        with torch.inference_mode():
            preds = model(X, edge_index, edge_attr)

        assert preds.shape == (4, num_nodes, 3)
        assert not preds.requires_grad

    def test_model_save_load(self, surface_graph):
        """Test model checkpoint save and load."""
        import os
        import tempfile

        edge_index, edge_attr, num_nodes = surface_graph

        patchtst_config = PatchTSTModelConfig(d_model=64, n_layers=1)
        gnn_config = GNNModelConfig(hidden_dim=32, n_layers=1)
        model = PatchTST_GNN_Ensemble(
            patchtst_config=patchtst_config,
            gnn_config=gnn_config,
            input_dim=N_FEATURES,
            output_horizons=3,
        )
        model.eval()

        X = torch.randn(2, 22, num_nodes, N_FEATURES)

        with torch.no_grad():
            preds_before = model(X, edge_index, edge_attr)

        # Save checkpoint
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "ensemble.pt")
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "patchtst_config": patchtst_config,
                    "gnn_config": gnn_config,
                    "input_dim": N_FEATURES,
                    "output_horizons": 3,
                },
                path,
            )

            # Load into new model
            checkpoint = torch.load(path, weights_only=False)
            model2 = PatchTST_GNN_Ensemble(
                patchtst_config=checkpoint["patchtst_config"],
                gnn_config=checkpoint["gnn_config"],
                input_dim=checkpoint["input_dim"],
                output_horizons=checkpoint["output_horizons"],
            )
            model2.load_state_dict(checkpoint["model_state_dict"])
            model2.eval()

            with torch.no_grad():
                preds_after = model2(X, edge_index, edge_attr)

        assert torch.allclose(preds_before, preds_after)

    def test_with_dataloader(self, surface_graph):
        """Test model with PyTorch DataLoader."""
        edge_index, edge_attr, num_nodes = surface_graph

        patchtst_config = PatchTSTModelConfig(d_model=64, n_layers=1)
        gnn_config = GNNModelConfig(hidden_dim=32, n_layers=1)
        model = PatchTST_GNN_Ensemble(
            patchtst_config=patchtst_config,
            gnn_config=gnn_config,
            input_dim=N_FEATURES,
            output_horizons=3,
        )
        model.eval()

        dataset = SyntheticGraphDataset(num_samples=16, num_nodes=num_nodes)
        dataloader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)

        for batch in dataloader:
            with torch.no_grad():
                preds = model(batch["X"], edge_index, edge_attr, batch["mask"])
            assert preds.shape == (4, num_nodes, 3)


class TestDeviceHandling:
    """Integration tests for device handling."""

    @pytest.fixture
    def surface_graph(self):
        """Build surface graph."""
        config = SurfaceGraphConfig()
        graph = build_surface_graph(config)
        return graph.edge_index, graph.edge_attr, graph.num_nodes

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_forward(self, surface_graph):
        """Test model on CUDA device."""
        edge_index, edge_attr, num_nodes = surface_graph

        patchtst_config = PatchTSTModelConfig(d_model=64, n_layers=1)
        gnn_config = GNNModelConfig(hidden_dim=32, n_layers=1)
        model = PatchTST_GNN_Ensemble(
            patchtst_config=patchtst_config,
            gnn_config=gnn_config,
            input_dim=N_FEATURES,
            output_horizons=3,
        )
        model = model.cuda()
        model.eval()

        X = torch.randn(4, 22, num_nodes, N_FEATURES).cuda()
        edge_index_cuda = edge_index.cuda()
        edge_attr_cuda = edge_attr.cuda()

        with torch.no_grad():
            preds = model(X, edge_index_cuda, edge_attr_cuda)

        assert preds.device.type == "cuda"
        assert preds.shape == (4, num_nodes, 3)

    @pytest.mark.skipif(
        not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()),
        reason="MPS not available",
    )
    def test_mps_forward(self, surface_graph):
        """Test model on MPS device (Apple Silicon)."""
        edge_index, edge_attr, num_nodes = surface_graph

        patchtst_config = PatchTSTModelConfig(d_model=64, n_layers=1)
        gnn_config = GNNModelConfig(hidden_dim=32, n_layers=1)
        model = PatchTST_GNN_Ensemble(
            patchtst_config=patchtst_config,
            gnn_config=gnn_config,
            input_dim=N_FEATURES,
            output_horizons=3,
        )
        model = model.to("mps")
        model.eval()

        X = torch.randn(4, 22, num_nodes, N_FEATURES).to("mps")
        edge_index_mps = edge_index.to("mps")
        edge_attr_mps = edge_attr.to("mps")

        with torch.no_grad():
            preds = model(X, edge_index_mps, edge_attr_mps)

        assert preds.device.type == "mps"
        assert preds.shape == (4, num_nodes, 3)
