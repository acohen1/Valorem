"""Configuration loader with environment overlay support.

This module provides utilities to load and validate configuration from YAML files
with environment-specific overrides.
"""

import logging
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from src.config.schema import ConfigSchema

logger = logging.getLogger(__name__)


class ConfigLoadError(Exception):
    """Raised when configuration fails to load."""

    pass


class ConfigLoader:
    """Loads and validates configuration with environment overlays."""

    @staticmethod
    def load(config_path: Path, env: str = "dev") -> ConfigSchema:
        """Load configuration with environment overlay.

        Args:
            config_path: Path to base config.yaml file
            env: Environment name (dev, backtest, paper, live)

        Returns:
            Validated ConfigSchema instance

        Raises:
            ConfigLoadError: If config fails to load or validate
            FileNotFoundError: If config file doesn't exist
            ValidationError: If config validation fails
        """
        # Load base config
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        logger.info(f"Loading base config from {config_path}")
        base_data = ConfigLoader._load_yaml(config_path)

        # Load environment overlay if exists
        env_path = config_path.parent / "environments" / f"{env}.yaml"
        if env_path.exists():
            logger.info(f"Loading environment overlay from {env_path}")
            env_data = ConfigLoader._load_yaml(env_path)
            merged_data = ConfigLoader._deep_merge(base_data, env_data)
        else:
            logger.warning(
                f"Environment overlay {env_path} not found, using base config only"
            )
            merged_data = base_data

        # Validate with Pydantic
        try:
            config = ConfigSchema(**merged_data)
            ConfigLoader.validate(config)
            logger.info(f"Configuration loaded successfully (env={env})")
            return config
        except ValidationError as e:
            logger.error(f"Configuration validation failed: {e}")
            raise

    @staticmethod
    def _load_yaml(path: Path) -> dict[str, Any]:
        """Load YAML file into dictionary.

        Args:
            path: Path to YAML file

        Returns:
            Parsed YAML data as dictionary

        Raises:
            ConfigLoadError: If YAML parsing fails
        """
        try:
            with open(path, "r") as f:
                data = yaml.safe_load(f)
                if data is None:
                    return {}
                return data
        except yaml.YAMLError as e:
            raise ConfigLoadError(f"Failed to parse YAML file {path}: {e}")
        except Exception as e:
            raise ConfigLoadError(f"Failed to load config file {path}: {e}")

    @staticmethod
    def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
        """Deep merge two dictionaries (override takes precedence).

        Args:
            base: Base dictionary
            override: Override dictionary

        Returns:
            Merged dictionary
        """
        result = base.copy()

        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                # Recursively merge nested dicts
                result[key] = ConfigLoader._deep_merge(result[key], value)
            else:
                # Override value
                result[key] = value

        return result

    @staticmethod
    def validate(config: ConfigSchema) -> None:
        """Run additional cross-field validation.

        Args:
            config: Configuration to validate

        Raises:
            ValueError: If validation fails
        """
        # Validate execution pricing (buy_at must be "ask" in v1)
        if config.execution.pricing.buy_at != "ask":
            raise ValueError(
                "execution.pricing.buy_at must be 'ask' (mid pricing not supported in v1)"
            )

        # Validate risk caps are positive (already enforced by Pydantic gt=0, but good practice)
        if config.risk.caps.max_portfolio_delta <= 0:
            raise ValueError("risk.caps.max_portfolio_delta must be positive")
        if config.risk.caps.max_portfolio_vega <= 0:
            raise ValueError("risk.caps.max_portfolio_vega must be positive")
        if config.risk.caps.max_position_size_usd <= 0:
            raise ValueError("risk.caps.max_position_size_usd must be positive")
        if config.risk.caps.max_total_notional_usd <= 0:
            raise ValueError("risk.caps.max_total_notional_usd must be positive")

        # Validate dataset min/max DTE
        if config.dataset.min_dte >= config.dataset.max_dte:
            raise ValueError("dataset.min_dte must be less than dataset.max_dte")

        # Validate backtest dates
        if config.backtest.start_date >= config.backtest.end_date:
            raise ValueError("backtest.start_date must be before backtest.end_date")

        logger.debug("Cross-field validation passed")
