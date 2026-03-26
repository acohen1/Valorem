"""Integration tests for training infrastructure."""

import os
import tempfile

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.models import (
    GNNModelConfig,
    PatchTST_GNN_Ensemble,
    PatchTSTModelConfig,
    SurfaceGraphConfig,
    build_surface_graph,
)
from src.models.dataset import DEFAULT_FEATURE_COLS
from src.models.train import Trainer, TrainerConfig, surface_collate_fn

N_FEATURES = len(DEFAULT_FEATURE_COLS)


class SyntheticSurfaceDataset(Dataset[dict[str, torch.Tensor]]):
    """Synthetic dataset for integration testing."""

    def __init__(
        self,
        num_samples: int,
        num_nodes: int,
        time_steps: int = 22,
        num_features: int = N_FEATURES,
        num_horizons: int = 3,
    ) -> None:
        self._num_samples = num_samples
        self._X = torch.randn(num_samples, time_steps, num_nodes, num_features)
        self._y = self._X[:, -1, :, :num_horizons].clone()  # Predictable target
        self._mask = torch.ones(num_samples, num_nodes, dtype=torch.bool)

    def __len__(self) -> int:
        return self._num_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {"X": self._X[idx], "y": self._y[idx], "mask": self._mask[idx]}


class TestFullTrainingLoop:
    """Integration tests for full training loop with real ensemble model."""

    @pytest.fixture
    def graph(self) -> torch.Tensor:
        """Build surface graph."""
        config = SurfaceGraphConfig()
        return build_surface_graph(config)

    @pytest.fixture
    def ensemble_model(self, graph: torch.Tensor) -> nn.Module:
        """Create ensemble model."""
        patchtst_config = PatchTSTModelConfig(d_model=32, n_layers=1)
        gnn_config = GNNModelConfig(hidden_dim=16, n_layers=1)
        return PatchTST_GNN_Ensemble(
            patchtst_config=patchtst_config,
            gnn_config=gnn_config,
            input_dim=N_FEATURES,
            output_horizons=3,
        )

    @pytest.fixture
    def train_loader(self, graph: torch.Tensor) -> DataLoader[dict[str, torch.Tensor]]:
        """Create training data loader."""
        dataset = SyntheticSurfaceDataset(num_samples=64, num_nodes=graph.num_nodes)
        return DataLoader(
            dataset, batch_size=16, shuffle=True, collate_fn=surface_collate_fn
        )

    @pytest.fixture
    def val_loader(self, graph: torch.Tensor) -> DataLoader[dict[str, torch.Tensor]]:
        """Create validation data loader."""
        dataset = SyntheticSurfaceDataset(num_samples=32, num_nodes=graph.num_nodes)
        return DataLoader(
            dataset, batch_size=16, shuffle=False, collate_fn=surface_collate_fn
        )

    def test_full_training_loop(
        self,
        ensemble_model: nn.Module,
        graph: torch.Tensor,
        train_loader: DataLoader[dict[str, torch.Tensor]],
        val_loader: DataLoader[dict[str, torch.Tensor]],
    ) -> None:
        """Test full training loop runs without errors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainerConfig(
                device="cpu",
                max_epochs=3,
                learning_rate=1e-3,
                early_stopping_patience=10,  # Don't trigger early stopping
                checkpoint_dir=tmpdir,
            )
            trainer = Trainer(ensemble_model, config, graph)
            result = trainer.train(train_loader, val_loader)

            # Verify training completed
            assert result.epochs_trained == 3
            assert len(result.training_history) == 3
            assert "train_loss" in result.training_history.columns
            assert "val_loss" in result.training_history.columns

            # Verify checkpoint was saved
            assert os.path.exists(result.metadata["checkpoint_path"])

    def test_loss_decreases(
        self,
        ensemble_model: nn.Module,
        graph: torch.Tensor,
        train_loader: DataLoader[dict[str, torch.Tensor]],
        val_loader: DataLoader[dict[str, torch.Tensor]],
    ) -> None:
        """Test that training loss decreases over epochs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainerConfig(
                device="cpu",
                max_epochs=5,
                learning_rate=1e-2,  # Higher LR for faster convergence
                early_stopping_patience=10,
                checkpoint_dir=tmpdir,
            )
            trainer = Trainer(ensemble_model, config, graph)
            result = trainer.train(train_loader, val_loader)

            # Loss should decrease (with some tolerance)
            train_losses = result.training_history["train_loss"].tolist()
            assert train_losses[-1] < train_losses[0] * 1.1

    def test_checkpoint_resume(
        self,
        graph: torch.Tensor,
        train_loader: DataLoader[dict[str, torch.Tensor]],
        val_loader: DataLoader[dict[str, torch.Tensor]],
    ) -> None:
        """Test checkpoint save and resume training."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create model and train for a bit
            patchtst_config = PatchTSTModelConfig(d_model=32, n_layers=1)
            gnn_config = GNNModelConfig(hidden_dim=16, n_layers=1)
            model1 = PatchTST_GNN_Ensemble(
                patchtst_config=patchtst_config,
                gnn_config=gnn_config,
                input_dim=N_FEATURES,
                output_horizons=3,
            )

            config = TrainerConfig(
                device="cpu",
                max_epochs=2,
                checkpoint_dir=tmpdir,
            )
            trainer1 = Trainer(model1, config, graph)
            result1 = trainer1.train(train_loader, val_loader)

            # Create two new models and load checkpoint into each
            model2 = PatchTST_GNN_Ensemble(
                patchtst_config=patchtst_config,
                gnn_config=gnn_config,
                input_dim=N_FEATURES,
                output_horizons=3,
            )
            model3 = PatchTST_GNN_Ensemble(
                patchtst_config=patchtst_config,
                gnn_config=gnn_config,
                input_dim=N_FEATURES,
                output_horizons=3,
            )
            trainer2 = Trainer(model2, config, graph)
            trainer3 = Trainer(model3, config, graph)

            metrics2 = trainer2.load_checkpoint(result1.metadata["checkpoint_path"])
            metrics3 = trainer3.load_checkpoint(result1.metadata["checkpoint_path"])

            assert "train_loss" in metrics2
            assert "train_loss" in metrics3

            # Both models loaded from same checkpoint should produce identical outputs
            model2.eval()
            model3.eval()
            test_input = torch.randn(2, 22, graph.num_nodes, N_FEATURES)
            edge_index = graph.edge_index
            edge_attr = graph.edge_attr

            with torch.no_grad():
                out2 = model2(test_input, edge_index, edge_attr)
                out3 = model3(test_input, edge_index, edge_attr)

            assert torch.allclose(out2, out3, atol=1e-5)

    def test_early_stopping_triggers(
        self,
        graph: torch.Tensor,
        train_loader: DataLoader[dict[str, torch.Tensor]],
        val_loader: DataLoader[dict[str, torch.Tensor]],
    ) -> None:
        """Test that early stopping triggers when no improvement."""
        # Create a model with zero learning rate (won't improve)
        patchtst_config = PatchTSTModelConfig(d_model=32, n_layers=1)
        gnn_config = GNNModelConfig(hidden_dim=16, n_layers=1)
        model = PatchTST_GNN_Ensemble(
            patchtst_config=patchtst_config,
            gnn_config=gnn_config,
            input_dim=N_FEATURES,
            output_horizons=3,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainerConfig(
                device="cpu",
                max_epochs=50,
                learning_rate=0.0,  # No learning
                early_stopping_patience=3,
                checkpoint_dir=tmpdir,
            )
            trainer = Trainer(model, config, graph)
            result = trainer.train(train_loader, val_loader)

            # Should stop early
            assert result.epochs_trained < 50
            assert result.epochs_trained <= 4  # patience + 1


class TestOverfitting:
    """Tests for model overfitting capability."""

    @pytest.fixture
    def graph(self) -> torch.Tensor:
        """Build surface graph."""
        config = SurfaceGraphConfig()
        return build_surface_graph(config)

    def test_overfit_small_batch(self, graph: torch.Tensor) -> None:
        """Test that model can overfit a small batch."""
        # Set seed for deterministic behavior across test runs
        torch.manual_seed(42)

        # Small dataset that should be easy to overfit
        num_nodes = graph.num_nodes
        X = torch.randn(4, 22, num_nodes, N_FEATURES)
        y = torch.randn(4, num_nodes, 3)
        mask = torch.ones(4, num_nodes, dtype=torch.bool)

        # Create a small dataset
        class TinyDataset(Dataset[dict[str, torch.Tensor]]):
            def __len__(self) -> int:
                return 4

            def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
                return {"X": X[idx], "y": y[idx], "mask": mask[idx]}

        train_loader = DataLoader(
            TinyDataset(), batch_size=4, collate_fn=surface_collate_fn
        )

        # Create model
        patchtst_config = PatchTSTModelConfig(d_model=64, n_layers=2, dropout=0.0)
        gnn_config = GNNModelConfig(hidden_dim=32, n_layers=2, dropout=0.0)
        model = PatchTST_GNN_Ensemble(
            patchtst_config=patchtst_config,
            gnn_config=gnn_config,
            input_dim=N_FEATURES,
            output_horizons=3,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainerConfig(
                device="cpu",
                max_epochs=100,  # More epochs for reliable overfitting
                learning_rate=1e-2,
                early_stopping_patience=200,  # Don't stop early
                scheduler_type="none",
                checkpoint_dir=tmpdir,
            )
            trainer = Trainer(model, config, graph)
            result = trainer.train(train_loader, train_loader)  # Same data for train/val

            # Loss should decrease significantly (at least 50% reduction)
            train_losses = result.training_history["train_loss"].tolist()
            assert train_losses[-1] < train_losses[0] * 0.5


class TestMetricsComputation:
    """Tests for metrics computation during training."""

    @pytest.fixture
    def graph(self) -> torch.Tensor:
        """Build surface graph."""
        config = SurfaceGraphConfig()
        return build_surface_graph(config)

    def test_metrics_computed_correctly(self, graph: torch.Tensor) -> None:
        """Test that all metrics are computed during validation."""
        # Create predictable dataset
        num_nodes = graph.num_nodes
        dataset = SyntheticSurfaceDataset(num_samples=32, num_nodes=num_nodes)
        train_loader = DataLoader(
            dataset, batch_size=16, collate_fn=surface_collate_fn
        )
        val_loader = DataLoader(
            dataset, batch_size=16, collate_fn=surface_collate_fn
        )

        patchtst_config = PatchTSTModelConfig(d_model=32, n_layers=1)
        gnn_config = GNNModelConfig(hidden_dim=16, n_layers=1)
        model = PatchTST_GNN_Ensemble(
            patchtst_config=patchtst_config,
            gnn_config=gnn_config,
            input_dim=N_FEATURES,
            output_horizons=3,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainerConfig(
                device="cpu",
                max_epochs=1,
                checkpoint_dir=tmpdir,
            )
            trainer = Trainer(model, config, graph)
            result = trainer.train(train_loader, val_loader)

            # Check all metrics are present in final_metrics
            assert "temporal_ic" in result.final_metrics
            assert "xs_demeaned_ic" in result.final_metrics
            assert "rmse" in result.final_metrics
            assert "mae" in result.final_metrics

            # Check metrics are reasonable (not NaN, not extreme)
            for name, value in result.final_metrics.items():
                assert not torch.isnan(torch.tensor(value)), f"{name} is NaN"
                if name in ["temporal_ic", "xs_demeaned_ic"]:
                    assert -1.0 <= value <= 1.0, f"{name} out of range"


class TestDifferentConfigurations:
    """Tests for various training configurations."""

    @pytest.fixture
    def graph(self) -> torch.Tensor:
        """Build surface graph."""
        config = SurfaceGraphConfig()
        return build_surface_graph(config)

    @pytest.fixture
    def loaders(
        self, graph: torch.Tensor
    ) -> tuple[DataLoader[dict[str, torch.Tensor]], DataLoader[dict[str, torch.Tensor]]]:
        """Create data loaders."""
        train = SyntheticSurfaceDataset(num_samples=32, num_nodes=graph.num_nodes)
        val = SyntheticSurfaceDataset(num_samples=16, num_nodes=graph.num_nodes)
        return (
            DataLoader(train, batch_size=8, collate_fn=surface_collate_fn),
            DataLoader(val, batch_size=8, collate_fn=surface_collate_fn),
        )

    def test_huber_loss(
        self,
        graph: torch.Tensor,
        loaders: tuple[DataLoader[dict[str, torch.Tensor]], DataLoader[dict[str, torch.Tensor]]],
    ) -> None:
        """Test training with Huber loss."""
        train_loader, val_loader = loaders
        patchtst_config = PatchTSTModelConfig(d_model=32, n_layers=1)
        gnn_config = GNNModelConfig(hidden_dim=16, n_layers=1)
        model = PatchTST_GNN_Ensemble(
            patchtst_config=patchtst_config,
            gnn_config=gnn_config,
            input_dim=N_FEATURES,
            output_horizons=3,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainerConfig(
                device="cpu",
                max_epochs=2,
                loss_type="huber",
                huber_delta=0.5,
                checkpoint_dir=tmpdir,
            )
            trainer = Trainer(model, config, graph)
            result = trainer.train(train_loader, val_loader)
            assert result.epochs_trained == 2

    def test_step_scheduler(
        self,
        graph: torch.Tensor,
        loaders: tuple[DataLoader[dict[str, torch.Tensor]], DataLoader[dict[str, torch.Tensor]]],
    ) -> None:
        """Test training with step scheduler."""
        train_loader, val_loader = loaders
        patchtst_config = PatchTSTModelConfig(d_model=32, n_layers=1)
        gnn_config = GNNModelConfig(hidden_dim=16, n_layers=1)
        model = PatchTST_GNN_Ensemble(
            patchtst_config=patchtst_config,
            gnn_config=gnn_config,
            input_dim=N_FEATURES,
            output_horizons=3,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainerConfig(
                device="cpu",
                max_epochs=3,
                scheduler_type="step",
                scheduler_step_size=1,
                scheduler_gamma=0.5,
                checkpoint_dir=tmpdir,
            )
            trainer = Trainer(model, config, graph)
            result = trainer.train(train_loader, val_loader)
            assert result.epochs_trained == 3

    def test_no_scheduler(
        self,
        graph: torch.Tensor,
        loaders: tuple[DataLoader[dict[str, torch.Tensor]], DataLoader[dict[str, torch.Tensor]]],
    ) -> None:
        """Test training without learning rate scheduler."""
        train_loader, val_loader = loaders
        patchtst_config = PatchTSTModelConfig(d_model=32, n_layers=1)
        gnn_config = GNNModelConfig(hidden_dim=16, n_layers=1)
        model = PatchTST_GNN_Ensemble(
            patchtst_config=patchtst_config,
            gnn_config=gnn_config,
            input_dim=N_FEATURES,
            output_horizons=3,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainerConfig(
                device="cpu",
                max_epochs=2,
                scheduler_type="none",
                checkpoint_dir=tmpdir,
            )
            trainer = Trainer(model, config, graph)
            result = trainer.train(train_loader, val_loader)
            assert result.epochs_trained == 2

    def test_learnable_edge_weights_training(
        self,
        graph: torch.Tensor,
        loaders: tuple[DataLoader[dict[str, torch.Tensor]], DataLoader[dict[str, torch.Tensor]]],
    ) -> None:
        """Test end-to-end training with learnable edge weights (M3)."""
        train_loader, val_loader = loaders

        # Create model with learnable edge weights
        patchtst_config = PatchTSTModelConfig(d_model=32, n_layers=1)
        gnn_config = GNNModelConfig(
            hidden_dim=16,
            n_layers=1,
            use_learnable_edge_attr=True,  # Enable M3 feature
        )
        model = PatchTST_GNN_Ensemble(
            patchtst_config=patchtst_config,
            gnn_config=gnn_config,
            input_dim=N_FEATURES,
            output_horizons=3,
            graph=graph,  # Pass graph to store internal edges
        )

        # Store initial edge_attr values
        initial_edge_attr = model._edge_attr.data.clone()

        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainerConfig(
                device="cpu",
                max_epochs=3,
                learning_rate=1e-2,  # Higher LR for more visible changes
                checkpoint_dir=tmpdir,
            )
            trainer = Trainer(model, config, graph)
            result = trainer.train(train_loader, val_loader)

            # Verify training completed
            assert result.epochs_trained == 3

            # Verify edge_attr values changed during training
            assert not torch.allclose(
                model._edge_attr.data, initial_edge_attr, atol=1e-6
            ), "Edge weights should change during training"

            # Verify edge_attr is still a parameter after training
            param_names = [name for name, _ in model.named_parameters()]
            assert "_edge_attr" in param_names

            # Verify checkpoint contains edge_attr
            checkpoint = torch.load(result.metadata["checkpoint_path"], weights_only=False)
            assert "_edge_attr" in checkpoint["model_state_dict"]
