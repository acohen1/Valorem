#!/usr/bin/env python3
"""Evaluate naive gamma baseline on volatility surface data.

Uses raw gamma values as "predictions" (no training, no parameters) and
computes the same metrics reported during training. This establishes the
statistical floor: any trained model should beat this to be useful.

Since DHR = 0.5 * gamma * dS^2, raw gamma has strong cross-sectional
correlation with DHR but limited temporal predictive power. IC metrics
are rank-based, so normalized gamma gives the same IC as raw gamma.

Example:
    # Evaluate on real data
    python scripts/eval_naive_baseline.py --env dev

    # Synthetic data for testing the script
    python scripts/eval_naive_baseline.py --synthetic
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.logging import setup_logging
from src.models.dataset import DEFAULT_FEATURE_COLS, LabelsConfig, SplitsConfig
from src.models.eval import MetricsCalculator
from src.models.train import surface_collate_fn
from src.models.train.data_pipeline import (
    TrainingDataConfig,
    TrainingDataPipeline,
    build_splits_from_yaml,
)


class SyntheticSurfaceDataset(Dataset[dict[str, torch.Tensor]]):
    """Synthetic dataset for testing the evaluation pipeline."""

    def __init__(
        self,
        num_samples: int = 64,
        time_steps: int = 22,
        num_nodes: int = 42,
        num_features: int = len(DEFAULT_FEATURE_COLS),
        num_horizons: int = 3,
    ) -> None:
        self._num_samples = num_samples
        self._X = torch.randn(num_samples, time_steps, num_nodes, num_features)
        self._y = torch.randn(num_samples, num_nodes, num_horizons)
        self._mask = torch.ones(num_samples, num_nodes, dtype=torch.bool)

    def __len__(self) -> int:
        return self._num_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {"X": self._X[idx], "y": self._y[idx], "mask": self._mask[idx]}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate naive gamma baseline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--env", type=str, default="dev")
    parser.add_argument("--feature-version", type=str, default="v1.0")
    parser.add_argument("--train-start", type=str, default=None)
    parser.add_argument("--val-start", type=str, default=None)
    parser.add_argument("--test-start", type=str, default=None)
    parser.add_argument("--test-end", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args()


def _build_splits_with_overrides(args: argparse.Namespace, config) -> SplitsConfig:
    """Build SplitsConfig from YAML config with optional CLI overrides."""
    splits = build_splits_from_yaml(config.dataset.splits)
    has_overrides = any([args.train_start, args.val_start, args.test_start, args.test_end])
    if has_overrides:
        splits = SplitsConfig(
            train_start=(
                datetime.strptime(args.train_start, "%Y-%m-%d")
                if args.train_start else splits.train_start
            ),
            val_start=(
                datetime.strptime(args.val_start, "%Y-%m-%d")
                if args.val_start else splits.val_start
            ),
            test_start=(
                datetime.strptime(args.test_start, "%Y-%m-%d")
                if args.test_start else splits.test_start
            ),
            test_end=(
                datetime.strptime(args.test_end, "%Y-%m-%d")
                if args.test_end else splits.test_end
            ),
        )
    return splits


def evaluate_loader(
    loader: DataLoader,
    gamma_idx: int,
    num_horizons: int = 3,
) -> dict[str, float]:
    """Evaluate naive gamma baseline on a DataLoader.

    Uses gamma at the last timestep as prediction, broadcast to all horizons.

    Args:
        loader: DataLoader yielding batches with X, y, mask.
        gamma_idx: Index of gamma feature in the feature dimension.
        num_horizons: Number of prediction horizons.

    Returns:
        Dictionary of metrics (temporal_ic, xs_demeaned_ic, rmse, mae).
    """
    all_preds = []
    all_targets = []
    all_masks = []
    all_label_masks = []

    for batch in loader:
        X = batch["X"]  # (batch, time, nodes, features)
        y = batch["y"]  # (batch, nodes, horizons)
        mask = batch["mask"]  # (batch, nodes)
        label_mask = batch.get("label_mask")
        if label_mask is None:
            label_mask = mask.unsqueeze(-1).expand_as(y)

        # Naive prediction: gamma at last timestep, broadcast to all horizons
        gamma = X[:, -1, :, gamma_idx]  # (batch, nodes)
        preds = gamma.unsqueeze(-1).expand(-1, -1, num_horizons)  # (batch, nodes, horizons)

        all_preds.append(preds)
        all_targets.append(y)
        all_masks.append(mask)
        all_label_masks.append(label_mask)

    preds_arr = torch.cat(all_preds, dim=0).numpy()
    targets_arr = torch.cat(all_targets, dim=0).numpy()
    masks_arr = torch.cat(all_masks, dim=0).numpy()
    label_masks_arr = torch.cat(all_label_masks, dim=0).numpy()

    # Exclude invalid labels at horizon level.
    preds_arr = np.where(label_masks_arr, preds_arr, np.nan)
    targets_arr = np.where(label_masks_arr, targets_arr, np.nan)

    num_nodes = preds_arr.shape[1]

    # Flatten for pooled metrics. Invalid label positions are NaN.
    preds_flat = preds_arr.reshape(-1, preds_arr.shape[-1])
    targets_flat = targets_arr.reshape(-1, targets_arr.shape[-1])

    calculator = MetricsCalculator()
    return calculator.compute_all(
        preds_flat,
        targets_flat,
        predictions_3d=preds_arr,
        targets_3d=targets_arr,
        masks_2d=masks_arr,
        num_nodes=num_nodes,
    )


def main() -> int:
    args = parse_args()
    setup_logging(verbose=args.verbose)

    gamma_idx = DEFAULT_FEATURE_COLS.index("gamma")
    logger.info(f"Naive gamma baseline evaluation")
    logger.info(f"  Gamma feature index: {gamma_idx}")

    if args.synthetic:
        logger.info("Using synthetic data")
        dataset = SyntheticSurfaceDataset()
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=surface_collate_fn,
        )
        metrics = evaluate_loader(loader, gamma_idx)
        logger.info("Synthetic results (should be near zero):")
        for name, value in metrics.items():
            logger.info(f"  {name}: {value:.4f}")
        return 0

    # Real data path
    from src.config.loader import ConfigLoader
    from src.data.storage.engine import create_engine
    from src.data.storage.repository import DerivedRepository, RawRepository

    config = ConfigLoader.load(Path(args.config), env=args.env)
    splits = _build_splits_with_overrides(args, config)

    logger.info(f"  Config: {args.config} (env={args.env})")
    logger.info(f"  Feature version: {args.feature_version}")
    logger.info(f"  Train: {splits.train_start:%Y-%m-%d} to {splits.val_start:%Y-%m-%d}")
    logger.info(f"  Val:   {splits.val_start:%Y-%m-%d} to {splits.test_start:%Y-%m-%d}")
    logger.info(f"  Test:  {splits.test_start:%Y-%m-%d} to {splits.test_end:%Y-%m-%d}")

    db_engine = create_engine(config.paths.db_path)
    raw_repo = RawRepository(db_engine.engine)
    derived_repo = DerivedRepository(db_engine.engine)

    pipeline_config = TrainingDataConfig(
        feature_version=args.feature_version,
        underlying_symbol=config.universe.underlying,
        splits=splits,
        labels=LabelsConfig(horizons_days=config.labels.horizons),
        batch_size=args.batch_size,
    )
    pipeline = TrainingDataPipeline(raw_repo, derived_repo, pipeline_config)
    training_data = pipeline.load()

    # Evaluate on validation set
    logger.info("Evaluating on validation set...")
    val_metrics = evaluate_loader(training_data.val_loader, gamma_idx)

    logger.info("============================================================")
    logger.info("Naive Gamma Baseline Results")
    logger.info("============================================================")
    logger.info("  Validation set:")
    logger.info(f"    Temporal IC:        {val_metrics.get('temporal_ic', 0):.4f}")
    logger.info(f"    XS-Demeaned IC:     {val_metrics.get('xs_demeaned_ic', 0):.4f}")
    logger.info(f"    RMSE:               {val_metrics.get('rmse', 0):.4f}")
    logger.info(f"    MAE:                {val_metrics.get('mae', 0):.4f}")
    logger.info("============================================================")

    return 0


if __name__ == "__main__":
    sys.exit(main())
