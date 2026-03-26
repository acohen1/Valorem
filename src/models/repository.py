"""Training results repository for versioned storage and retrieval.

This module provides a repository pattern for managing training results,
eliminating the need for notebooks to navigate artifact directories.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.models.results import TrainingResults


class ResultsRepository:
    """Repository for managing training results with versioning and metadata.

    Provides a clean interface for saving and loading training results by name,
    with automatic indexing and metadata tracking.

    Example:
        >>> repo = ResultsRepository()
        >>> repo.save(results, name="M3-learnable-edges", tags=["M3", "ensemble"])
        >>> runs = repo.list(tags=["M3"])
        >>> results = repo.load("M3-learnable-edges")
    """

    def __init__(self, base_path: str | Path = "artifacts/results"):
        """Initialize repository.

        Args:
            base_path: Base directory for storing results artifacts.
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.index_path = self.base_path / "index.json"
        self._ensure_index()

    def _ensure_index(self) -> None:
        """Ensure index file exists."""
        if not self.index_path.exists():
            self._save_index({"runs": []})

    def _load_index(self) -> dict:
        """Load index from disk."""
        with open(self.index_path) as f:
            return json.load(f)

    def _save_index(self, index: dict) -> None:
        """Save index to disk."""
        with open(self.index_path, "w") as f:
            json.dump(index, f, indent=2)

    def save(
        self,
        results: TrainingResults,
        name: str | None = None,
        tags: list[str] | None = None,
        include_weights: bool = True,
    ) -> str:
        """Save training results with metadata.

        Args:
            results: TrainingResults to save.
            name: Optional custom name (defaults to run_id).
            tags: Optional tags for filtering (e.g., ["M3", "learnable-edges"]).
            include_weights: Whether to include model weights.

        Returns:
            Name used for saving (for reference).
        """
        name = name or results.run_id
        tags = tags or []

        # Save artifact
        artifact_path = self.base_path / f"{name}.pkl"
        results.to_artifact(artifact_path, include_weights=include_weights)

        # Update index
        index = self._load_index()

        # Remove existing entry with same name
        index["runs"] = [r for r in index["runs"] if r["name"] != name]

        # Add new entry
        index["runs"].append({
            "name": name,
            "run_id": results.run_id,
            "tags": tags,
            "timestamp": results.metadata.get("timestamp", datetime.now().isoformat()),
            "epochs_trained": results.epochs_trained,
            "best_epoch": results.best_epoch,
            "best_metric": results.best_metric,
            "final_metrics": results.final_metrics,
            "config_summary": {
                "ablation": results.config.get("ablation", "unknown"),
                "learnable_edges": results.config.get("learnable_edges", False),
                "batch_size": results.config.get("batch_size", 0),
                "epochs": results.config.get("epochs", 0),
                "feature_version": results.config.get("feature_version", "unknown"),
                "train_start": results.config.get("train_start"),
                "val_start": results.config.get("val_start"),
                "test_start": results.config.get("test_start"),
                "test_end": results.config.get("test_end"),
            },
            "artifact_path": str(artifact_path.relative_to(self.base_path)),
            "has_weights": include_weights,
        })

        # Sort by timestamp (newest first)
        index["runs"].sort(key=lambda x: x["timestamp"], reverse=True)

        self._save_index(index)

        return name

    def load(
        self,
        name: str,
        *,
        include_legacy: bool = True,
    ) -> TrainingResults:
        """Load training results by name.

        Args:
            name: Name of the run to load.
            include_legacy: If True, will also search artifacts/checkpoints/
                for legacy checkpoint format.

        Returns:
            TrainingResults instance.

        Raises:
            FileNotFoundError: If run not found.
        """
        # Check index first
        index = self._load_index()
        for run in index["runs"]:
            if run["name"] == name:
                artifact_path = self.base_path / run["artifact_path"]
                return TrainingResults.from_artifact(artifact_path)

        # Try direct path if not in index
        artifact_path = self.base_path / f"{name}.pkl"
        if artifact_path.exists():
            return TrainingResults.from_artifact(artifact_path)

        # Try legacy checkpoint if enabled
        if include_legacy:
            legacy_path = Path("artifacts/checkpoints") / f"{name}.pt"
            if legacy_path.exists():
                return TrainingResults.from_artifact(legacy_path)

        raise FileNotFoundError(
            f"Run '{name}' not found. Use list() to see available runs."
        )

    def list(
        self,
        tags: list[str] | None = None,
        limit: int | None = None,
    ) -> pd.DataFrame:
        """List available training runs with metadata.

        Args:
            tags: Optional filter by tags (returns runs matching ANY tag).
            limit: Optional limit on number of results.

        Returns:
            DataFrame with columns [name, run_id, tags, timestamp, best_metric, etc.].
        """
        index = self._load_index()
        runs = index["runs"]

        # Filter by tags if specified
        if tags:
            runs = [
                r
                for r in runs
                if any(tag in r.get("tags", []) for tag in tags)
            ]

        # Apply limit
        if limit:
            runs = runs[:limit]

        if not runs:
            return pd.DataFrame()

        # Convert to DataFrame
        records = []
        for run in runs:
            record = {
                "name": run["name"],
                "run_id": run["run_id"],
                "tags": ", ".join(run.get("tags", [])),
                "timestamp": run["timestamp"],
                "epochs": run["epochs_trained"],
                "best_epoch": run["best_epoch"],
                "best_metric": run["best_metric"],
                "final_temporal_ic": run["final_metrics"].get(
                    "temporal_ic",
                    run["final_metrics"].get("ic", 0.0),
                ),
                "final_xs_demeaned_ic": run["final_metrics"].get("xs_demeaned_ic", 0.0),
                "ablation": run["config_summary"].get("ablation", "unknown"),
                "learnable_edges": run["config_summary"].get("learnable_edges", False),
                "feature_version": run["config_summary"].get("feature_version", "unknown"),
                "train_start": run["config_summary"].get("train_start"),
                "val_start": run["config_summary"].get("val_start"),
                "test_start": run["config_summary"].get("test_start"),
                "test_end": run["config_summary"].get("test_end"),
                "has_weights": run.get("has_weights", True),
            }
            records.append(record)

        return pd.DataFrame(records)

    def delete(self, name: str) -> bool:
        """Delete a training run.

        Args:
            name: Name of the run to delete.

        Returns:
            True if deleted, False if not found.
        """
        index = self._load_index()

        # Find run in index
        run = None
        for r in index["runs"]:
            if r["name"] == name:
                run = r
                break

        if not run:
            return False

        # Delete artifact
        artifact_path = self.base_path / run["artifact_path"]
        if artifact_path.exists():
            artifact_path.unlink()

        # Remove from index
        index["runs"] = [r for r in index["runs"] if r["name"] != name]
        self._save_index(index)

        return True

    def get_latest(self, tags: list[str] | None = None) -> TrainingResults:
        """Get most recent training run.

        Args:
            tags: Optional filter by tags.

        Returns:
            TrainingResults for most recent run.

        Raises:
            FileNotFoundError: If no runs found.
        """
        df = self.list(tags=tags, limit=1)
        if df.empty:
            raise FileNotFoundError("No training runs found")

        name = df.iloc[0]["name"]
        return self.load(name)
