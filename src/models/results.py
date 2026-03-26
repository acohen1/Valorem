"""Training results data structures.

This module provides structured, immutable representations of training outcomes
for use in notebooks and analysis scripts without requiring file path dependencies.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import torch


@dataclass(frozen=True)
class TrainingResults:
    """Immutable training results with all metrics and history.

    This class provides a structured interface to training outcomes,
    eliminating the need for notebooks to parse log files or navigate
    artifact directories.

    Attributes:
        run_id: Unique identifier for this training run (e.g., "M3-learnable-edges-20260203").
        config: Full configuration dict used for training.
        training_history: DataFrame with columns [epoch, train_loss, val_loss, temporal_ic, xs_demeaned_ic, rmse, mae, lr].
        final_metrics: Dict with best-epoch performance metrics {temporal_ic, xs_demeaned_ic, rmse, mae}.
        best_epoch: Epoch number with best validation metric.
        best_metric: Best validation metric value achieved.
        epochs_trained: Total number of epochs completed.
        model_state_dict: Optional model weights (can be None to save space).
        metadata: Additional metadata {timestamp, git_hash, hostname, duration_seconds, etc.}.

    Example:
        >>> results = TrainingResults.from_artifact("artifacts/checkpoints/best_model.pt")
        >>> results.training_history.plot(x="epoch", y="temporal_ic")
        >>> print(f"Best metric: {results.best_metric:.4f} at epoch {results.best_epoch}")
    """

    run_id: str
    config: dict
    training_history: pd.DataFrame
    final_metrics: dict
    best_epoch: int
    best_metric: float
    epochs_trained: int
    model_state_dict: dict | None = None
    metadata: dict = field(default_factory=dict)

    def to_artifact(self, path: str | Path, include_weights: bool = True) -> None:
        """Save results as versioned artifact.

        Saves in a structured format that can be loaded back into TrainingResults
        or used with existing checkpoint loading code.

        Args:
            path: Output path for artifact (e.g., "artifacts/results/M3-learnable-edges.pkl").
            include_weights: Whether to include model weights (can be large).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "run_id": self.run_id,
            "config": self.config,
            "training_history": self.training_history.to_dict(orient="records"),
            "final_metrics": self.final_metrics,
            "best_epoch": self.best_epoch,
            "best_metric": self.best_metric,
            "epochs_trained": self.epochs_trained,
            "model_state_dict": self.model_state_dict if include_weights else None,
            "metadata": self.metadata,
            "version": "1.0",  # For future compatibility
        }

        with open(path, "wb") as f:
            pickle.dump(data, f)

    @classmethod
    def from_artifact(cls, path: str | Path) -> "TrainingResults":
        """Load results from artifact.

        Args:
            path: Path to saved artifact.

        Returns:
            TrainingResults instance.

        Raises:
            FileNotFoundError: If artifact doesn't exist.
            ValueError: If artifact format is invalid.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Artifact not found: {path}")

        with open(path, "rb") as f:
            data = pickle.load(f)

        # Handle legacy checkpoints (backward compatibility)
        if "run_id" not in data:
            return cls._from_legacy_checkpoint(data, path)

        return cls(
            run_id=data["run_id"],
            config=data["config"],
            training_history=pd.DataFrame(data["training_history"]),
            final_metrics=data["final_metrics"],
            best_epoch=data["best_epoch"],
            best_metric=data["best_metric"],
            epochs_trained=data["epochs_trained"],
            model_state_dict=data.get("model_state_dict"),
            metadata=data.get("metadata", {}),
        )

    @classmethod
    def _from_legacy_checkpoint(
        cls, checkpoint: dict, path: Path
    ) -> "TrainingResults":
        """Convert legacy checkpoint format to TrainingResults.

        Supports loading old-style checkpoints that don't have the structured
        results format, for backward compatibility.

        Args:
            checkpoint: Raw checkpoint dict from torch.load().
            path: Path to checkpoint (used for run_id).

        Returns:
            TrainingResults instance reconstructed from checkpoint.
        """
        # Extract what we can from legacy format
        run_id = path.stem  # Use filename as run_id

        # Reconstruct config (may be incomplete)
        config = checkpoint.get("config", {})

        # Try to reconstruct training history if available
        history_records = []
        if "train_losses" in checkpoint:
            for epoch, (train_loss, val_loss) in enumerate(
                zip(checkpoint["train_losses"], checkpoint["val_losses"]), start=1
            ):
                history_records.append({
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                })

        training_history = pd.DataFrame(history_records)

        # Extract final metrics
        final_metrics = {
            # Legacy checkpoints may contain only pooled IC.
            # Map to temporal_ic as a best-effort compatibility fallback.
            "temporal_ic": checkpoint.get(
                "temporal_ic",
                checkpoint.get("ic", 0.0),
            ),
            "xs_demeaned_ic": checkpoint.get("xs_demeaned_ic", 0.0),
            "rmse": checkpoint.get("rmse", 0.0),
            "mae": checkpoint.get("mae", 0.0),
        }

        return cls(
            run_id=run_id,
            config=config,
            training_history=training_history,
            final_metrics=final_metrics,
            best_epoch=checkpoint.get("best_epoch", 0),
            best_metric=checkpoint.get("best_metric", 0.0),
            epochs_trained=checkpoint.get("epoch", len(history_records)),
            model_state_dict=checkpoint.get("model_state_dict"),
            metadata={
                "source": "legacy_checkpoint",
                "original_path": str(path),
            },
        )

    def to_summary_dict(self) -> dict[str, Any]:
        """Get summary dict for display or logging.

        Returns:
            Dict with key metrics and metadata.
        """
        return {
            "run_id": self.run_id,
            "epochs_trained": self.epochs_trained,
            "best_epoch": self.best_epoch,
            "best_metric": self.best_metric,
            "final_temporal_ic": self.final_metrics.get(
                "temporal_ic",
                self.final_metrics.get("ic", 0.0),
            ),
            "final_xs_demeaned_ic": self.final_metrics.get("xs_demeaned_ic", 0.0),
            "final_rmse": self.final_metrics.get("rmse", 0.0),
            "final_mae": self.final_metrics.get("mae", 0.0),
            "timestamp": self.metadata.get("timestamp"),
        }

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"TrainingResults(run_id='{self.run_id}', "
            f"epochs={self.epochs_trained}, "
            f"best_metric={self.best_metric:.4f}@epoch{self.best_epoch})"
        )
