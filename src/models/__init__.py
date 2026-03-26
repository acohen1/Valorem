"""Machine learning models and datasets for Valorem.

This package provides:
- Graph construction for volatility surface topology
- PyTorch datasets for time series + graph learning
- PatchTST model for temporal encoding (M17)
- GNN model for cross-sectional encoding (M18)
- PatchTST+GNN ensemble model (M18)
- Training infrastructure (M19)
- Evaluation metrics (M19)
"""

from src.models.graph import SurfaceGraphConfig, build_surface_graph
from src.models.dataset import (
    DatasetConfig,
    GNN_ABLATION_FEATURE_COLS,
    LabelsConfig,
    SplitsConfig,
    SurfaceDataset,
    DatasetBuilder,
)
from src.models.patchtst import (
    PatchTSTModel,
    PatchTSTModelConfig,
    PatchEmbedding,
)
from src.models.gnn import (
    GNNModelConfig,
    SurfaceGNN,
)
from src.models.linear import LinearBaseline, LinearBaselineConfig
from src.models.ensemble import PatchTST_GNN_Ensemble
from src.models.train import (
    Trainer,
    TrainerConfig,
    TrainResult,
    TrainingData,
    TrainingDataConfig,
    TrainingDataPipeline,
    build_splits_from_yaml,
    HuberLoss,
    QuantileLoss,
    MaskedLoss,
    build_loss,
    surface_collate_fn,
)
from src.models.eval import (
    compute_ic,
    compute_rmse,
    compute_mae,
    MetricsCalculator,
)

__all__ = [
    # Graph
    "SurfaceGraphConfig",
    "build_surface_graph",
    # Dataset
    "DatasetConfig",
    "GNN_ABLATION_FEATURE_COLS",
    "LabelsConfig",
    "SplitsConfig",
    "SurfaceDataset",
    "DatasetBuilder",
    # PatchTST
    "PatchTSTModel",
    "PatchTSTModelConfig",
    "PatchEmbedding",
    # GNN
    "GNNModelConfig",
    "SurfaceGNN",
    # Linear baseline
    "LinearBaseline",
    "LinearBaselineConfig",
    # Ensemble
    "PatchTST_GNN_Ensemble",
    # Training (M19)
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
    "surface_collate_fn",
    # Evaluation (M19)
    "compute_ic",
    "compute_rmse",
    "compute_mae",
    "MetricsCalculator",
]
