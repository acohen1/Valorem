"""Unit tests for Trainer class."""

import os
import tempfile

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.models import SurfaceGraphConfig, build_surface_graph
from src.models.dataset import DEFAULT_FEATURE_COLS
from src.models.train import Trainer, TrainerConfig, surface_collate_fn

N_FEATURES = len(DEFAULT_FEATURE_COLS)


class SimpleModel(nn.Module):
    """Simple model for testing trainer."""

    def __init__(self, input_dim: int, num_nodes: int, output_horizons: int) -> None:
        super().__init__()
        self._num_nodes = num_nodes
        self._output_horizons = output_horizons
        self._linear = nn.Linear(input_dim, output_horizons)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # x: (batch, time, nodes, features)
        # Take last time step and project
        batch_size = x.shape[0]
        x_last = x[:, -1, :, :]  # (batch, nodes, features)
        out = self._linear(x_last)  # (batch, nodes, horizons)

        if mask is not None:
            out = out * mask.unsqueeze(-1).float()

        return out


class SyntheticDataset(Dataset[dict[str, torch.Tensor]]):
    """Synthetic dataset for testing."""

    def __init__(
        self,
        num_samples: int,
        num_nodes: int,
        time_steps: int = 22,
        input_dim: int = N_FEATURES,
        output_horizons: int = 3,
    ) -> None:
        self._num_samples = num_samples
        self._X = torch.randn(num_samples, time_steps, num_nodes, input_dim)
        self._y = torch.randn(num_samples, num_nodes, output_horizons)
        self._mask = torch.ones(num_samples, num_nodes, dtype=torch.bool)

    def __len__(self) -> int:
        return self._num_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {"X": self._X[idx], "y": self._y[idx], "mask": self._mask[idx]}


class TestTrainerConfig:
    """Tests for TrainerConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = TrainerConfig()
        assert config.device == "auto"
        assert config.batch_size == 32
        assert config.max_epochs == 100
        assert config.learning_rate == 1e-3
        assert config.early_stopping_patience == 10
        assert config.loss_type == "huber"
        assert config.scheduler_type == "cosine"

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = TrainerConfig(
            device="cpu",
            batch_size=64,
            max_epochs=50,
            learning_rate=1e-4,
        )
        assert config.device == "cpu"
        assert config.batch_size == 64
        assert config.max_epochs == 50
        assert config.learning_rate == 1e-4


class TestTrainer:
    """Tests for Trainer class."""

    @pytest.fixture
    def graph(self) -> torch.Tensor:
        """Create a surface graph."""
        config = SurfaceGraphConfig()
        return build_surface_graph(config)

    @pytest.fixture
    def model(self, graph: torch.Tensor) -> nn.Module:
        """Create a simple test model."""
        return SimpleModel(input_dim=N_FEATURES, num_nodes=graph.num_nodes, output_horizons=3)

    @pytest.fixture
    def train_loader(self, graph: torch.Tensor) -> DataLoader[dict[str, torch.Tensor]]:
        """Create training data loader."""
        dataset = SyntheticDataset(num_samples=32, num_nodes=graph.num_nodes)
        return DataLoader(dataset, batch_size=8, collate_fn=surface_collate_fn)

    @pytest.fixture
    def val_loader(self, graph: torch.Tensor) -> DataLoader[dict[str, torch.Tensor]]:
        """Create validation data loader."""
        dataset = SyntheticDataset(num_samples=16, num_nodes=graph.num_nodes)
        return DataLoader(dataset, batch_size=8, collate_fn=surface_collate_fn)

    def test_trainer_init(self, model: nn.Module, graph: torch.Tensor) -> None:
        """Test trainer initialization."""
        config = TrainerConfig(device="cpu", max_epochs=1)
        trainer = Trainer(model, config, graph)
        assert trainer is not None

    def test_train_one_epoch(
        self,
        model: nn.Module,
        graph: torch.Tensor,
        train_loader: DataLoader[dict[str, torch.Tensor]],
        val_loader: DataLoader[dict[str, torch.Tensor]],
    ) -> None:
        """Test training for one epoch."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainerConfig(
                device="cpu",
                max_epochs=1,
                checkpoint_dir=tmpdir,
            )
            trainer = Trainer(model, config, graph)
            result = trainer.train(train_loader, val_loader)

            assert result.epochs_trained == 1
            assert len(result.training_history) == 1
            assert "train_loss" in result.training_history.columns
            assert "val_loss" in result.training_history.columns

    def test_train_returns_metrics(
        self,
        model: nn.Module,
        graph: torch.Tensor,
        train_loader: DataLoader[dict[str, torch.Tensor]],
        val_loader: DataLoader[dict[str, torch.Tensor]],
    ) -> None:
        """Test that training returns all expected metrics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainerConfig(
                device="cpu",
                max_epochs=2,
                checkpoint_dir=tmpdir,
            )
            trainer = Trainer(model, config, graph)
            result = trainer.train(train_loader, val_loader)

            # Check metrics are computed in final_metrics
            assert "temporal_ic" in result.final_metrics
            assert "xs_demeaned_ic" in result.final_metrics
            assert "rmse" in result.final_metrics
            assert "mae" in result.final_metrics

    def test_early_stopping(
        self,
        graph: torch.Tensor,
        train_loader: DataLoader[dict[str, torch.Tensor]],
        val_loader: DataLoader[dict[str, torch.Tensor]],
    ) -> None:
        """Test that early stopping works."""
        # Create a model that won't improve
        model = SimpleModel(input_dim=N_FEATURES, num_nodes=graph.num_nodes, output_horizons=3)

        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainerConfig(
                device="cpu",
                max_epochs=100,  # High max epochs
                early_stopping_patience=2,  # Low patience
                learning_rate=0.0,  # No learning -> no improvement
                checkpoint_dir=tmpdir,
            )
            trainer = Trainer(model, config, graph)
            result = trainer.train(train_loader, val_loader)

            # Should stop early due to no improvement
            assert result.epochs_trained < 100
            assert result.epochs_trained <= 3  # patience + 1

    def test_checkpoint_save_load(
        self,
        model: nn.Module,
        graph: torch.Tensor,
        train_loader: DataLoader[dict[str, torch.Tensor]],
        val_loader: DataLoader[dict[str, torch.Tensor]],
    ) -> None:
        """Test checkpoint save and load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainerConfig(
                device="cpu",
                max_epochs=1,
                checkpoint_dir=tmpdir,
            )
            trainer = Trainer(model, config, graph)

            # Train and save checkpoint
            trainer.train(train_loader, val_loader)

            # Checkpoint uses run_id in filename
            import glob
            checkpoint_files = glob.glob(os.path.join(tmpdir, "best_model_*.pt"))
            assert len(checkpoint_files) == 1
            checkpoint_path = checkpoint_files[0]

            # Load checkpoint
            metrics = trainer.load_checkpoint(checkpoint_path)
            assert "train_loss" in metrics

    def test_gradient_clipping(
        self,
        graph: torch.Tensor,
        train_loader: DataLoader[dict[str, torch.Tensor]],
        val_loader: DataLoader[dict[str, torch.Tensor]],
    ) -> None:
        """Test that gradient clipping is applied."""
        model = SimpleModel(input_dim=N_FEATURES, num_nodes=graph.num_nodes, output_horizons=3)

        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainerConfig(
                device="cpu",
                max_epochs=1,
                gradient_clip_val=0.1,  # Very small clip value
                checkpoint_dir=tmpdir,
            )
            trainer = Trainer(model, config, graph)

            # Should not raise any errors with gradient clipping
            result = trainer.train(train_loader, val_loader)
            assert result.epochs_trained == 1

    def test_different_loss_types(
        self,
        model: nn.Module,
        graph: torch.Tensor,
        train_loader: DataLoader[dict[str, torch.Tensor]],
        val_loader: DataLoader[dict[str, torch.Tensor]],
    ) -> None:
        """Test training with different loss types."""
        for loss_type in ["mse", "huber", "mae"]:
            with tempfile.TemporaryDirectory() as tmpdir:
                config = TrainerConfig(
                    device="cpu",
                    max_epochs=1,
                    loss_type=loss_type,
                    checkpoint_dir=tmpdir,
                )
                # Create fresh model for each test
                test_model = SimpleModel(
                    input_dim=N_FEATURES, num_nodes=graph.num_nodes, output_horizons=3
                )
                trainer = Trainer(test_model, config, graph)
                result = trainer.train(train_loader, val_loader)
                assert result.epochs_trained == 1

    def test_scheduler_types(
        self,
        model: nn.Module,
        graph: torch.Tensor,
        train_loader: DataLoader[dict[str, torch.Tensor]],
        val_loader: DataLoader[dict[str, torch.Tensor]],
    ) -> None:
        """Test training with different scheduler types."""
        for scheduler_type in ["cosine", "step", "none"]:
            with tempfile.TemporaryDirectory() as tmpdir:
                config = TrainerConfig(
                    device="cpu",
                    max_epochs=2,
                    scheduler_type=scheduler_type,
                    checkpoint_dir=tmpdir,
                )
                test_model = SimpleModel(
                    input_dim=N_FEATURES, num_nodes=graph.num_nodes, output_horizons=3
                )
                trainer = Trainer(test_model, config, graph)
                result = trainer.train(train_loader, val_loader)
                assert result.epochs_trained == 2

    def test_val_loss_early_stopping(
        self,
        graph: torch.Tensor,
        train_loader: DataLoader[dict[str, torch.Tensor]],
        val_loader: DataLoader[dict[str, torch.Tensor]],
    ) -> None:
        """Test early stopping based on validation loss."""
        model = SimpleModel(input_dim=N_FEATURES, num_nodes=graph.num_nodes, output_horizons=3)

        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainerConfig(
                device="cpu",
                max_epochs=100,
                early_stopping_patience=2,
                early_stopping_metric="val_loss",
                early_stopping_mode="min",
                learning_rate=0.0,
                checkpoint_dir=tmpdir,
            )
            trainer = Trainer(model, config, graph)
            result = trainer.train(train_loader, val_loader)

            # Should stop early
            assert result.epochs_trained < 100


class TestCollateFunction:
    """Tests for surface_collate_fn."""

    def test_collate_batches_correctly(self) -> None:
        """Test that collate function batches samples correctly."""
        samples = [
            {
                "X": torch.randn(22, 10, N_FEATURES),
                "y": torch.randn(10, 3),
                "mask": torch.ones(10, dtype=torch.bool),
            },
            {
                "X": torch.randn(22, 10, N_FEATURES),
                "y": torch.randn(10, 3),
                "mask": torch.ones(10, dtype=torch.bool),
            },
        ]

        batch = surface_collate_fn(samples)

        assert batch["X"].shape == (2, 22, 10, N_FEATURES)
        assert batch["y"].shape == (2, 10, 3)
        assert batch["mask"].shape == (2, 10)
