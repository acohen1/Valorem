"""Configuration management package.

This package provides type-safe configuration loading, validation, and path resolution.

Constants:
    SurfaceConstants: Tenor and delta bucket definitions for surface grids
    TradingConstants: Contract multiplier and execution constants
    MarketConstants: Market hours and default data sources
"""

from src.config.constants import MarketConstants, SurfaceConstants, TradingConstants
from src.config.env import (
    EnvironmentConfig,
    TradingMode,
    print_validation_results,
    validate_cli_config,
)
from src.config.loader import ConfigLoader
from src.config.logging import setup_logging
from src.config.paths import PathResolver
from src.config.schema import ConfigSchema

__all__ = [
    "ConfigSchema",
    "ConfigLoader",
    "PathResolver",
    "setup_logging",
    # Constants
    "SurfaceConstants",
    "TradingConstants",
    "MarketConstants",
    # Trading environment config
    "EnvironmentConfig",
    "TradingMode",
    "validate_cli_config",
    "print_validation_results",
]
