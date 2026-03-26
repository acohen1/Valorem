"""Trading environment configuration.

This module provides trading-specific environment configuration,
including mode switching (mock, paper_live, paper_db) and API key management.
"""

import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class TradingMode(str, Enum):
    """Trading environment mode.

    Determines which providers and data sources are used.

    Attributes:
        MOCK: Synthetic data, no external connections. For testing and development.
        PAPER_LIVE: Live data from Databento, simulated execution via PaperOrderRouter.
        PAPER_DB: Database replay of historical data, simulated execution.
        LIVE: Live data from Databento, real execution via broker (IBKR).
    """

    MOCK = "mock"
    PAPER_LIVE = "paper_live"
    PAPER_DB = "paper_db"
    LIVE = "live"  # Live data, real execution via broker


@dataclass
class EnvironmentConfig:
    """Environment-specific configuration for trading.

    Centralizes all external service configuration and API keys.
    Validates that required keys are present for the selected mode.

    Example:
        # Load from environment
        config = EnvironmentConfig.from_env(mode="paper_live")
        config.validate()

        # Use in trading setup
        if config.mode == TradingMode.PAPER_LIVE:
            ingestion = DatabentoIngestionService(api_key=config.databento_api_key, ...)
            surface_provider = DatabaseSurfaceProvider(derived_repo, version="live")
        elif config.mode == TradingMode.MOCK:
            surface_provider = MockSurfaceProvider()
    """

    mode: TradingMode

    # Data provider API keys
    databento_api_key: str | None = None

    # Database configuration (for PAPER_DB mode)
    database_url: str | None = None

    # Broker configuration (for LIVE mode)
    ibkr_host: str = "127.0.0.1"
    ibkr_port: int = 7497
    ibkr_client_id: int = 1

    # Optional logging level override
    log_level: str = "INFO"

    @classmethod
    def from_env(cls, mode: TradingMode | str = "mock") -> "EnvironmentConfig":
        """Load configuration from environment variables.

        Environment variables:
            VALOREM_MODE: Override for mode parameter
            DATABENTO_API_KEY: Databento API key for live data
            VALOREM_DATABASE_URL: Database URL for paper_db mode
            VALOREM_LOG_LEVEL: Logging level override

        Args:
            mode: Default trading mode (can be overridden by VALOREM_MODE env var)

        Returns:
            EnvironmentConfig instance with values from environment
        """
        # Parse mode parameter
        if isinstance(mode, str):
            mode = TradingMode(mode)

        # Override mode from env if set
        env_mode = os.getenv("VALOREM_MODE")
        if env_mode:
            try:
                mode = TradingMode(env_mode.lower())
                logger.info(f"Using trading mode from environment: {mode.value}")
            except ValueError:
                logger.warning(
                    f"Invalid VALOREM_MODE '{env_mode}', using default: {mode.value}. "
                    f"Valid modes: {[m.value for m in TradingMode]}"
                )

        return cls(
            mode=mode,
            databento_api_key=os.getenv("DATABENTO_API_KEY"),
            database_url=os.getenv("VALOREM_DATABASE_URL"),
            ibkr_host=os.getenv("IBKR_HOST") or "127.0.0.1",
            ibkr_port=int(os.getenv("IBKR_PORT") or "7497"),
            ibkr_client_id=int(os.getenv("IBKR_CLIENT_ID") or "1"),
            log_level=os.getenv("VALOREM_LOG_LEVEL") or "INFO",
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EnvironmentConfig":
        """Create configuration from dictionary (e.g., from YAML config).

        Args:
            data: Dictionary with configuration values

        Returns:
            EnvironmentConfig instance
        """
        mode_str = data.get("mode", "mock")
        if isinstance(mode_str, str):
            mode = TradingMode(mode_str.lower())
        else:
            mode = mode_str

        return cls(
            mode=mode,
            databento_api_key=data.get("databento_api_key") or os.getenv("DATABENTO_API_KEY"),
            database_url=data.get("database_url") or os.getenv("VALOREM_DATABASE_URL"),
            ibkr_host=data.get("ibkr_host") or os.getenv("IBKR_HOST", "127.0.0.1"),
            ibkr_port=int(data.get("ibkr_port") or os.getenv("IBKR_PORT", "7497")),
            ibkr_client_id=int(data.get("ibkr_client_id") or os.getenv("IBKR_CLIENT_ID", "1")),
            log_level=data.get("log_level", "INFO"),
        )

    def validate(self) -> None:
        """Validate configuration for selected mode.

        Ensures all required API keys and settings are present for the
        selected trading mode.

        Raises:
            ValueError: If required configuration is missing
        """
        errors = []

        if self.mode == TradingMode.PAPER_LIVE:
            if not self.databento_api_key:
                errors.append(
                    "DATABENTO_API_KEY required for paper_live mode. "
                    "Set the DATABENTO_API_KEY environment variable."
                )

        elif self.mode == TradingMode.PAPER_DB:
            if not self.database_url:
                errors.append(
                    "VALOREM_DATABASE_URL required for paper_db mode. "
                    "Set the VALOREM_DATABASE_URL environment variable."
                )

        elif self.mode == TradingMode.LIVE:
            if not self.databento_api_key:
                errors.append(
                    "DATABENTO_API_KEY required for live mode. "
                    "Set the DATABENTO_API_KEY environment variable."
                )
            # Note: Broker connection is validated at runtime when connecting
            # We just check that the config values are reasonable
            if self.ibkr_port < 1 or self.ibkr_port > 65535:
                errors.append(
                    f"Invalid IBKR_PORT: {self.ibkr_port}. Must be between 1 and 65535."
                )

        # MOCK mode has no external dependencies

        if errors:
            error_msg = "\n".join(f"  - {e}" for e in errors)
            raise ValueError(f"Configuration validation failed:\n{error_msg}")

        logger.info(f"Configuration validated for mode: {self.mode.value}")

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary.

        Note: Sensitive values (API keys) are masked.

        Returns:
            Dictionary representation of configuration
        """
        return {
            "mode": self.mode.value,
            "databento_api_key": "***" if self.databento_api_key else None,
            "database_url": self._mask_url(self.database_url) if self.database_url else None,
            "ibkr_host": self.ibkr_host,
            "ibkr_port": self.ibkr_port,
            "ibkr_client_id": self.ibkr_client_id,
            "log_level": self.log_level,
        }

    def _mask_url(self, url: str) -> str:
        """Mask sensitive parts of database URL.

        Args:
            url: Database connection URL

        Returns:
            URL with password masked
        """
        # Simple masking - replace password in URL pattern user:pass@host
        import re
        return re.sub(r":([^:@]+)@", ":***@", url)

    def get_summary(self) -> str:
        """Get human-readable configuration summary.

        Returns:
            Multi-line summary string
        """
        lines = [
            f"Trading Mode: {self.mode.value}",
            f"Databento API Key: {'configured' if self.databento_api_key else 'not set'}",
            f"Database URL: {'configured' if self.database_url else 'not set'}",
        ]

        if self.mode == TradingMode.LIVE:
            lines.append(f"IBKR: {self.ibkr_host}:{self.ibkr_port} (client {self.ibkr_client_id})")

        lines.append(f"Log Level: {self.log_level}")
        return "\n".join(lines)

    @property
    def is_live_data(self) -> bool:
        """Check if mode uses live market data."""
        return self.mode in (TradingMode.PAPER_LIVE, TradingMode.LIVE)

    @property
    def is_simulated_execution(self) -> bool:
        """Check if mode uses simulated order execution."""
        return self.mode != TradingMode.LIVE

    @property
    def is_real_execution(self) -> bool:
        """Check if mode uses real broker execution (real money!)."""
        return self.mode == TradingMode.LIVE

    @property
    def requires_api_key(self) -> bool:
        """Check if mode requires Databento API key."""
        return self.mode in (TradingMode.PAPER_LIVE, TradingMode.LIVE)

    @property
    def requires_broker(self) -> bool:
        """Check if mode requires broker connection."""
        return self.mode == TradingMode.LIVE


def validate_cli_config(
    mode: str,
    symbols_file: str | None = None,
    checkpoint: str | None = None,
) -> dict[str, Any]:
    """Validate CLI configuration and return validation results.

    Args:
        mode: Trading mode string
        symbols_file: Optional path to symbols manifest
        checkpoint: Optional path to model checkpoint

    Returns:
        Dictionary with validation results:
        {
            "valid": bool,
            "config": EnvironmentConfig,
            "warnings": list[str],
            "errors": list[str],
        }
    """
    warnings = []
    errors = []

    # Parse and validate mode
    try:
        trading_mode = TradingMode(mode.lower())
    except ValueError:
        errors.append(
            f"Invalid mode '{mode}'. Valid modes: {[m.value for m in TradingMode]}"
        )
        return {"valid": False, "config": None, "warnings": warnings, "errors": errors}

    # Create config
    config = EnvironmentConfig.from_env(mode=trading_mode)

    # Validate config
    try:
        config.validate()
    except ValueError as e:
        errors.append(str(e))

    # Validate symbols file
    if symbols_file:
        from pathlib import Path
        if not Path(symbols_file).exists():
            errors.append(f"Symbols file not found: {symbols_file}")
    elif trading_mode in (TradingMode.PAPER_LIVE, TradingMode.LIVE):
        warnings.append(
            "No symbols file specified. Use --symbols-file or --discover-symbols."
        )

    # Warn about LIVE mode (real money!)
    if trading_mode == TradingMode.LIVE:
        warnings.append(
            "⚠️  LIVE MODE: This will execute real trades with real money! "
            "Ensure broker connection and risk limits are configured correctly."
        )

    # Validate checkpoint
    if checkpoint:
        from pathlib import Path
        if not Path(checkpoint).exists():
            errors.append(f"Model checkpoint not found: {checkpoint}")

    return {
        "valid": len(errors) == 0,
        "config": config if len(errors) == 0 else None,
        "warnings": warnings,
        "errors": errors,
    }


def print_validation_results(results: dict[str, Any]) -> None:
    """Print validation results to console.

    Args:
        results: Dictionary from validate_cli_config()
    """
    if results["valid"]:
        print("\u2713 Configuration valid")
        if results["config"]:
            print(f"\u2713 Mode: {results['config'].mode.value}")
            if results["config"].databento_api_key:
                print("\u2713 DATABENTO_API_KEY found")
            if results["config"].database_url:
                print("\u2713 Database URL configured")
            if results["config"].mode == TradingMode.LIVE:
                print(f"\u2713 Broker: {results['config'].ibkr_host}:{results['config'].ibkr_port}")
    else:
        print("\u2717 Configuration invalid")

    for warning in results["warnings"]:
        print(f"\u26A0 Warning: {warning}")

    for error in results["errors"]:
        print(f"\u2717 Error: {error}")

    if results["valid"]:
        print("\nReady to start paper trading")
