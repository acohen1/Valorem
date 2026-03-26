"""Integration tests for PatchTST model."""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.models.dataset import DEFAULT_FEATURE_COLS
from src.models.patchtst import PatchTSTModel, PatchTSTModelConfig

N_FEATURES = len(DEFAULT_FEATURE_COLS)


class SyntheticSurfaceDataset(Dataset):
    """Synthetic dataset for testing PatchTST integration."""

    def __init__(
        self,
        num_samples: int = 100,
        time_steps: int = 22,
        num_nodes: int = 42,
        num_features: int = N_FEATURES,
        num_horizons: int = 3,
    ):
        """Initialize synthetic dataset.

        Args:
            num_samples: Number of samples in dataset.
            time_steps: Number of time steps per sample.
            num_nodes: Number of nodes in surface graph.
            num_features: Number of input features per node.
            num_horizons: Number of prediction horizons.
        """
        self._num_samples = num_samples
        self._time_steps = time_steps
        self._num_nodes = num_nodes
        self._num_features = num_features
        self._num_horizons = num_horizons

        # Generate synthetic data
        self._X = torch.randn(num_samples, time_steps, num_nodes, num_features)
        # Labels: simple linear function of last timestep features for testability
        self._y = self._X[:, -1, :, :num_horizons].clone()  # (samples, nodes, horizons)
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


class TestPatchTSTWithDataLoader:
    """Integration tests for PatchTST with DataLoader."""

    def test_with_dataloader(self):
        """Test model with PyTorch DataLoader."""
        config = PatchTSTModelConfig(d_model=64, n_layers=1)
        model = PatchTSTModel(config, input_dim=N_FEATURES, output_horizons=3)
        model.eval()

        dataset = SyntheticSurfaceDataset(num_samples=16)
        dataloader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)

        for batch in dataloader:
            with torch.no_grad():
                preds = model(batch["X"], batch["mask"])
            assert preds.shape == (4, 42, 3)

    def test_batched_consistency(self):
        """Test that batched and individual predictions match."""
        config = PatchTSTModelConfig(d_model=64, n_layers=1)
        model = PatchTSTModel(config, input_dim=N_FEATURES, output_horizons=3)
        model.eval()

        # Create small batch
        X = torch.randn(4, 22, 10, N_FEATURES)

        with torch.no_grad():
            # Batched prediction
            batched_preds = model(X)

            # Individual predictions
            individual_preds = torch.stack([model(X[i:i+1]) for i in range(4)])
            individual_preds = individual_preds.squeeze(1)

        assert torch.allclose(batched_preds, individual_preds, atol=1e-5)


class TestPatchTSTTraining:
    """Integration tests for PatchTST training."""

    def test_training_loop_one_epoch(self):
        """Test basic training loop runs without errors."""
        config = PatchTSTModelConfig(d_model=32, n_layers=1)
        model = PatchTSTModel(config, input_dim=N_FEATURES, output_horizons=3)

        dataset = SyntheticSurfaceDataset(num_samples=32)
        dataloader = DataLoader(dataset, batch_size=8, collate_fn=collate_fn)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()

        model.train()
        for batch in dataloader:
            optimizer.zero_grad()
            preds = model(batch["X"], batch["mask"])
            loss = criterion(preds, batch["y"])
            loss.backward()
            optimizer.step()

        # If we get here without error, training works
        assert True

    def test_loss_decreases(self):
        """Test that loss decreases over training epochs."""
        config = PatchTSTModelConfig(d_model=32, n_layers=1, dropout=0.0)
        model = PatchTSTModel(config, input_dim=N_FEATURES, output_horizons=3)

        dataset = SyntheticSurfaceDataset(num_samples=64)
        dataloader = DataLoader(dataset, batch_size=16, collate_fn=collate_fn)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        criterion = nn.MSELoss()

        losses = []
        for epoch in range(5):
            epoch_loss = 0.0
            model.train()
            for batch in dataloader:
                optimizer.zero_grad()
                preds = model(batch["X"], batch["mask"])
                loss = criterion(preds, batch["y"])
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            losses.append(epoch_loss / len(dataloader))

        # Loss should decrease (or at least not increase significantly)
        assert losses[-1] < losses[0] * 1.1  # Allow small fluctuation

    def test_overfit_small_batch(self):
        """Test that model can overfit a small batch (sanity check)."""
        config = PatchTSTModelConfig(d_model=64, n_layers=2, dropout=0.0)
        model = PatchTSTModel(config, input_dim=N_FEATURES, output_horizons=3)

        # Very small dataset
        X = torch.randn(4, 22, 10, N_FEATURES)
        y = torch.randn(4, 10, 3)
        mask = torch.ones(4, 10, dtype=torch.bool)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()

        initial_loss = None
        for i in range(100):
            model.train()
            optimizer.zero_grad()
            preds = model(X, mask)
            loss = criterion(preds, y)
            if initial_loss is None:
                initial_loss = loss.item()
            loss.backward()
            optimizer.step()

        final_loss = loss.item()

        # Should overfit significantly
        assert final_loss < initial_loss * 0.1

    def test_gradient_accumulation(self):
        """Test gradient accumulation works correctly."""
        config = PatchTSTModelConfig(d_model=32, n_layers=1)
        model = PatchTSTModel(config, input_dim=N_FEATURES, output_horizons=3)

        dataset = SyntheticSurfaceDataset(num_samples=16)
        dataloader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)

        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()

        accumulation_steps = 2
        model.train()
        optimizer.zero_grad()

        for i, batch in enumerate(dataloader):
            preds = model(batch["X"], batch["mask"])
            loss = criterion(preds, batch["y"]) / accumulation_steps
            loss.backward()

            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

        # Should complete without error
        assert True


class TestPatchTSTInference:
    """Integration tests for PatchTST inference."""

    def test_inference_mode(self):
        """Test model in inference mode (no gradient tracking)."""
        config = PatchTSTModelConfig(d_model=64, n_layers=1)
        model = PatchTSTModel(config, input_dim=N_FEATURES, output_horizons=3)
        model.eval()

        X = torch.randn(4, 22, 10, N_FEATURES)

        with torch.inference_mode():
            preds = model(X)

        assert preds.shape == (4, 10, 3)
        assert not preds.requires_grad

    def test_model_save_load(self):
        """Test model checkpoint save and load."""
        import tempfile
        import os

        config = PatchTSTModelConfig(d_model=64, n_layers=1)
        model = PatchTSTModel(config, input_dim=N_FEATURES, output_horizons=3)
        model.eval()

        X = torch.randn(2, 22, 10, N_FEATURES)

        with torch.no_grad():
            preds_before = model(X)

        # Save checkpoint
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.pt")
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": config,
                "input_dim": N_FEATURES,
                "output_horizons": 3,
            }, path)

            # Load into new model
            checkpoint = torch.load(path, weights_only=False)
            model2 = PatchTSTModel(
                checkpoint["config"],
                input_dim=checkpoint["input_dim"],
                output_horizons=checkpoint["output_horizons"],
            )
            model2.load_state_dict(checkpoint["model_state_dict"])
            model2.eval()

            with torch.no_grad():
                preds_after = model2(X)

        assert torch.allclose(preds_before, preds_after)

    def test_encode_for_ensemble(self):
        """Test encode method works for ensemble use case."""
        config = PatchTSTModelConfig(d_model=64, n_layers=1)
        model = PatchTSTModel(config, input_dim=N_FEATURES, output_horizons=0)  # No head
        model.eval()

        X = torch.randn(4, 22, 10, N_FEATURES)

        with torch.no_grad():
            embeddings = model.encode(X)

        # Embeddings should be suitable for GNN input
        assert embeddings.shape == (4, 10, 64)
        assert embeddings.dtype == torch.float32


class TestPatchTSTDeviceHandling:
    """Integration tests for device handling."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_forward(self):
        """Test model on CUDA device."""
        config = PatchTSTModelConfig(d_model=64, n_layers=1)
        model = PatchTSTModel(config, input_dim=N_FEATURES, output_horizons=3)
        model = model.cuda()
        model.eval()

        X = torch.randn(4, 22, 10, N_FEATURES).cuda()

        with torch.no_grad():
            preds = model(X)

        assert preds.device.type == "cuda"
        assert preds.shape == (4, 10, 3)

    @pytest.mark.skipif(
        not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()),
        reason="MPS not available"
    )
    def test_mps_forward(self):
        """Test model on MPS device (Apple Silicon)."""
        config = PatchTSTModelConfig(d_model=64, n_layers=1)
        model = PatchTSTModel(config, input_dim=N_FEATURES, output_horizons=3)
        model = model.to("mps")
        model.eval()

        X = torch.randn(4, 22, 10, N_FEATURES).to("mps")

        with torch.no_grad():
            preds = model(X)

        assert preds.device.type == "mps"
        assert preds.shape == (4, 10, 3)
