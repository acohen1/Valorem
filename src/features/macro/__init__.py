"""Macro feature generators.

This module provides feature generators for macroeconomic data (FRED series):
- Transforms: level, change, z-score
- Alignment: release-time aware feature generation
"""

from src.features.macro.alignment import (
    AlignmentConfig,
    ReleaseTimeAligner,
)
from src.features.macro.transforms import (
    MacroTransformConfig,
    MacroTransformGenerator,
)

__all__ = [
    "MacroTransformGenerator",
    "MacroTransformConfig",
    "ReleaseTimeAligner",
    "AlignmentConfig",
]
