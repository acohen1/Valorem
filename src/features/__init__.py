"""Feature engineering module for Valorem.

This module provides feature generation from surface snapshots:
- Node features: IV changes, microstructure, cross-sectional
- Global features: Underlying returns, realized volatility
- Macro features: FRED series transforms

Architecture:
    FeatureEngine uses dependency injection for all components.
    Use create_feature_engine() for production, or inject mocks
    directly into FeatureEngine for testing.
"""

from src.features.engine import (
    FeatureEngine,
    FeatureEngineConfig,
    FeatureEngineResult,
    FeatureResult,
    GlobalFeatureConfig,
    NodeFeatureConfig,
    NodeFeatureGenerator,
    create_feature_engine,
)
from src.features.global_.realized_vol import RealizedVolConfig, RealizedVolGenerator
from src.features.global_.returns import ReturnsConfig, ReturnsGenerator
from src.features.macro.alignment import AlignmentConfig, ReleaseTimeAligner
from src.features.macro.transforms import MacroTransformConfig, MacroTransformGenerator
from src.features.node.iv_features import IVFeatureConfig, IVFeatureGenerator
from src.features.node.microstructure import (
    MicrostructureConfig,
    MicrostructureFeatureGenerator,
)
from src.features.node.surface import SurfaceFeatureConfig, SurfaceFeatureGenerator
from src.features.validators import (
    FeatureValidator,
    IssueSeverity,
    ValidationIssue,
    ValidationResult,
)

__all__ = [
    # Feature Engine (full orchestrator)
    "FeatureEngine",
    "create_feature_engine",
    "FeatureEngineConfig",
    "FeatureEngineResult",
    "GlobalFeatureConfig",
    # Node Feature Generator
    "NodeFeatureGenerator",
    "NodeFeatureConfig",
    "FeatureResult",
    # IV features
    "IVFeatureGenerator",
    "IVFeatureConfig",
    # Microstructure features
    "MicrostructureFeatureGenerator",
    "MicrostructureConfig",
    # Surface features
    "SurfaceFeatureGenerator",
    "SurfaceFeatureConfig",
    # Global features
    "ReturnsGenerator",
    "ReturnsConfig",
    "RealizedVolGenerator",
    "RealizedVolConfig",
    # Macro features
    "MacroTransformGenerator",
    "MacroTransformConfig",
    "ReleaseTimeAligner",
    "AlignmentConfig",
    # Validation
    "FeatureValidator",
    "ValidationResult",
    "ValidationIssue",
    "IssueSeverity",
]
