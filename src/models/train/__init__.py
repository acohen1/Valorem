"""Training infrastructure for volatility surface models.

This package provides:
- Trainer: Main training loop with early stopping, checkpointing
- TrainerConfig: Configuration for training
- TrainResult: Results from training run
- Loss functions: HuberLoss, QuantileLoss, MaskedLoss
- Collate function for DataLoader
- TrainingDataPipeline: Load real data from DB into DataLoaders
"""

from src.models.train.collate import surface_collate_fn
from src.models.train.data_pipeline import (
    TrainingData,
    TrainingDataConfig,
    TrainingDataPipeline,
    build_splits_from_yaml,
)
from src.models.train.loss import (
    HuberLoss,
    MaskedLoss,
    QuantileLoss,
    build_loss,
)
from src.models.train.trainer import Trainer, TrainerConfig, TrainResult

__all__ = [
    # Trainer
    "Trainer",
    "TrainerConfig",
    "TrainResult",
    # Data pipeline
    "TrainingDataConfig",
    "TrainingData",
    "TrainingDataPipeline",
    "build_splits_from_yaml",
    # Loss functions
    "HuberLoss",
    "QuantileLoss",
    "MaskedLoss",
    "build_loss",
    # Collate
    "surface_collate_fn",
]
