"""Unit tests for TrainingResults and ResultsRepository."""

import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest
import torch

from src.models.repository import ResultsRepository
from src.models.results import TrainingResults


class TestTrainingResults:
    """Tests for TrainingResults dataclass."""

    @pytest.fixture
    def sample_results(self):
        """Create sample training results for testing."""
        training_history = pd.DataFrame({
            "epoch": [1, 2, 3],
            "train_loss": [1.0, 0.8, 0.6],
            "val_loss": [0.9, 0.7, 0.5],
            "temporal_ic": [0.35, 0.45, 0.55],
            "xs_demeaned_ic": [0.2, 0.3, 0.4],
            "rmse": [0.8, 0.7, 0.6],
            "mae": [0.6, 0.5, 0.4],
            "lr": [1e-3, 1e-3, 1e-3],
        })

        final_metrics = {
            "temporal_ic": 0.55,
            "xs_demeaned_ic": 0.4,
            "rmse": 0.6,
            "mae": 0.4,
        }

        config = {
            "ablation": "ensemble",
            "batch_size": 64,
            "learning_rate": 1e-3,
        }

        metadata = {
            "timestamp": datetime.now().isoformat(),
            "device": "cuda",
            "duration_seconds": 120.5,
        }

        model_state = {"layer.weight": torch.randn(10, 10)}

        return TrainingResults(
            run_id="test_run_001",
            config=config,
            training_history=training_history,
            final_metrics=final_metrics,
            best_epoch=3,
            best_metric=0.55,
            epochs_trained=3,
            model_state_dict=model_state,
            metadata=metadata,
        )

    def test_initialization(self, sample_results):
        """Test TrainingResults initialization."""
        assert sample_results.run_id == "test_run_001"
        assert sample_results.best_epoch == 3
        assert sample_results.best_metric == 0.55
        assert sample_results.epochs_trained == 3
        assert len(sample_results.training_history) == 3
        assert "temporal_ic" in sample_results.final_metrics

    def test_to_summary_dict(self, sample_results):
        """Test conversion to summary dictionary."""
        summary = sample_results.to_summary_dict()

        assert summary["run_id"] == "test_run_001"
        assert summary["epochs_trained"] == 3
        assert summary["best_epoch"] == 3
        assert summary["best_metric"] == 0.55
        assert summary["final_temporal_ic"] == 0.55
        assert summary["final_xs_demeaned_ic"] == 0.4
        assert "timestamp" in summary

    def test_repr(self, sample_results):
        """Test string representation."""
        repr_str = repr(sample_results)

        assert "test_run_001" in repr_str
        assert "epochs=3" in repr_str
        assert "0.5500" in repr_str

    def test_save_and_load(self, sample_results):
        """Test saving and loading artifacts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_path = Path(tmpdir) / "test_results.pkl"

            # Save
            sample_results.to_artifact(artifact_path, include_weights=True)
            assert artifact_path.exists()

            # Load
            loaded_results = TrainingResults.from_artifact(artifact_path)

            # Verify
            assert loaded_results.run_id == sample_results.run_id
            assert loaded_results.best_epoch == sample_results.best_epoch
            assert loaded_results.best_metric == sample_results.best_metric
            pd.testing.assert_frame_equal(
                loaded_results.training_history,
                sample_results.training_history,
            )
            assert loaded_results.model_state_dict is not None

    def test_save_without_weights(self, sample_results):
        """Test saving without model weights."""
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_path = Path(tmpdir) / "test_results_no_weights.pkl"

            # Save without weights
            sample_results.to_artifact(artifact_path, include_weights=False)

            # Load
            loaded_results = TrainingResults.from_artifact(artifact_path)

            # Verify weights are None
            assert loaded_results.model_state_dict is None
            assert loaded_results.run_id == sample_results.run_id


class TestResultsRepository:
    """Tests for ResultsRepository."""

    @pytest.fixture
    def temp_repo(self):
        """Create temporary repository for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = ResultsRepository(base_path=tmpdir)
            yield repo

    @pytest.fixture
    def sample_results(self):
        """Create sample results for testing."""
        training_history = pd.DataFrame({
            "epoch": [1, 2],
            "train_loss": [1.0, 0.8],
            "val_loss": [0.9, 0.7],
            "temporal_ic": [0.35, 0.45],
            "xs_demeaned_ic": [0.2, 0.3],
            "rmse": [0.8, 0.7],
            "mae": [0.6, 0.5],
            "lr": [1e-3, 1e-3],
        })

        return TrainingResults(
            run_id="ensemble_fixed_20260203_001",
            config={"ablation": "ensemble", "learnable_edges": False},
            training_history=training_history,
            final_metrics={"temporal_ic": 0.45, "xs_demeaned_ic": 0.3},
            best_epoch=2,
            best_metric=0.45,
            epochs_trained=2,
            metadata={"timestamp": datetime.now().isoformat()},
        )

    def test_save_and_load(self, temp_repo, sample_results):
        """Test saving and loading results."""
        # Save
        name = temp_repo.save(sample_results, tags=["ensemble", "test"])

        # Load
        loaded = temp_repo.load(name)

        assert loaded.run_id == sample_results.run_id
        assert loaded.best_metric == sample_results.best_metric

    def test_list_all(self, temp_repo, sample_results):
        """Test listing all results."""
        # Save multiple results
        temp_repo.save(sample_results, name="run1", tags=["ensemble"])
        temp_repo.save(sample_results, name="run2", tags=["patchtst"])

        # List all
        df = temp_repo.list()

        assert len(df) == 2
        assert "run1" in df["name"].values
        assert "run2" in df["name"].values

    def test_list_with_tag_filter(self, temp_repo, sample_results):
        """Test filtering by tags."""
        # Save with different tags
        temp_repo.save(sample_results, name="run1", tags=["ensemble"])
        temp_repo.save(sample_results, name="run2", tags=["patchtst"])
        temp_repo.save(sample_results, name="run3", tags=["ensemble", "learnable"])

        # Filter by tag
        df = temp_repo.list(tags=["ensemble"])

        assert len(df) == 2
        assert "run1" in df["name"].values
        assert "run3" in df["name"].values

    def test_list_with_limit(self, temp_repo, sample_results):
        """Test limiting results."""
        # Save multiple
        for i in range(5):
            temp_repo.save(sample_results, name=f"run{i}")

        # List with limit
        df = temp_repo.list(limit=2)

        assert len(df) == 2

    def test_delete(self, temp_repo, sample_results):
        """Test deleting results."""
        # Save
        name = temp_repo.save(sample_results)

        # Verify exists
        assert len(temp_repo.list()) == 1

        # Delete
        deleted = temp_repo.delete(name)
        assert deleted is True

        # Verify gone
        assert len(temp_repo.list()) == 0

    def test_get_latest(self, temp_repo, sample_results):
        """Test getting most recent run."""
        # Save multiple
        temp_repo.save(sample_results, name="old_run")
        temp_repo.save(sample_results, name="new_run", tags=["latest"])

        # Get latest
        latest = temp_repo.get_latest()

        assert latest.run_id == sample_results.run_id

    def test_get_latest_with_tags(self, temp_repo, sample_results):
        """Test getting latest with tag filter."""
        # Save with different tags
        temp_repo.save(sample_results, name="run1", tags=["ensemble"])
        temp_repo.save(sample_results, name="run2", tags=["patchtst"])

        # Get latest ensemble run
        latest = temp_repo.get_latest(tags=["ensemble"])

        df = temp_repo.list()
        assert "ensemble" in df[df["name"] == "run1"]["tags"].values[0]

    def test_update_existing_run(self, temp_repo, sample_results):
        """Test updating an existing run name."""
        # Save
        temp_repo.save(sample_results, name="test_run")

        # Update with same name
        temp_repo.save(sample_results, name="test_run", tags=["updated"])

        # Should only have one entry
        df = temp_repo.list()
        assert len(df) == 1
        assert "updated" in df.iloc[0]["tags"]

    def test_load_nonexistent(self, temp_repo):
        """Test loading nonexistent run raises error."""
        with pytest.raises(FileNotFoundError):
            temp_repo.load("nonexistent_run")

    def test_get_latest_empty_repo(self, temp_repo):
        """Test get_latest on empty repo raises error."""
        with pytest.raises(FileNotFoundError):
            temp_repo.get_latest()
