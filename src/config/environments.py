"""Environment-specific configuration utilities.

This module provides utilities for managing environment-specific configuration overlays.
"""

import logging
from enum import Enum
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class Environment(str, Enum):
    """Supported environment types."""

    DEV = "dev"
    BACKTEST = "backtest"
    PAPER = "paper"
    LIVE = "live"


def get_environment() -> Environment:
    """Get current environment from RHUBARB_ENV environment variable.

    Returns:
        Environment enum value (defaults to DEV)
    """
    import os

    env_str = os.getenv("RHUBARB_ENV", "dev").lower()

    try:
        env = Environment(env_str)
        logger.info(f"Running in {env.value} environment")
        return env
    except ValueError:
        logger.warning(
            f"Invalid environment '{env_str}', defaulting to dev. "
            f"Valid environments: {[e.value for e in Environment]}"
        )
        return Environment.DEV


def get_env_config_path(base_config_dir: Path, env: Optional[Environment] = None) -> Optional[Path]:
    """Get path to environment-specific config file.

    Args:
        base_config_dir: Base configuration directory (e.g., config/)
        env: Environment (uses current environment if None)

    Returns:
        Path to environment config file, or None if it doesn't exist
    """
    if env is None:
        env = get_environment()

    env_path = base_config_dir / "environments" / f"{env.value}.yaml"

    if env_path.exists():
        logger.debug(f"Found environment config: {env_path}")
        return env_path
    else:
        logger.debug(f"No environment config found at {env_path}")
        return None


def validate_environment_transition(from_env: Environment, to_env: Environment) -> None:
    """Validate that environment transition is allowed.

    Args:
        from_env: Current environment
        to_env: Target environment

    Raises:
        ValueError: If transition is not allowed
    """
    # Define allowed transitions
    allowed_transitions = {
        Environment.DEV: {Environment.DEV, Environment.BACKTEST},
        Environment.BACKTEST: {Environment.BACKTEST, Environment.PAPER},
        Environment.PAPER: {Environment.PAPER, Environment.LIVE},
        Environment.LIVE: {Environment.LIVE},  # Live can only stay in live
    }

    if to_env not in allowed_transitions.get(from_env, set()):
        raise ValueError(
            f"Environment transition from {from_env.value} to {to_env.value} is not allowed. "
            f"Allowed transitions from {from_env.value}: "
            f"{[e.value for e in allowed_transitions.get(from_env, set())]}"
        )

    logger.info(f"Environment transition validated: {from_env.value} -> {to_env.value}")
