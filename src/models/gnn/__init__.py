"""Graph Neural Network for cross-sectional surface learning.

This module provides the GNN component of the PatchTST+GNN ensemble,
capturing structural dependencies between adjacent nodes on the
volatility surface.
"""

from src.models.gnn.model import GNNModelConfig, SurfaceGNN

__all__ = [
    "GNNModelConfig",
    "SurfaceGNN",
]
