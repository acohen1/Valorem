"""PatchTST model for temporal encoding.

This package implements the PatchTST architecture for per-node time series
forecasting on volatility surfaces.

Classes:
    PatchTSTModel: Main model class for temporal encoding and prediction.
    PatchTSTModelConfig: Configuration dataclass for model hyperparameters.
    PatchEmbedding: Patch embedding layer for converting time series to patches.
"""

from src.models.patchtst.encoder import PatchEmbedding
from src.models.patchtst.model import PatchTSTModel, PatchTSTModelConfig

__all__ = [
    "PatchTSTModel",
    "PatchTSTModelConfig",
    "PatchEmbedding",
]
