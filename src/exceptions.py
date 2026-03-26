"""Domain exception hierarchy for Valorem.

This module defines a structured exception hierarchy for the trading system,
enabling specific error handling and recovery strategies per error type.

Exception Hierarchy:
    ValoremError (base)
    ├── DataError (data layer)
    │   ├── DataWriteError (failed writes)
    │   ├── DataReadError (failed reads)
    │   └── DataValidationError (validation failures)
    ├── ConfigError (configuration)
    ├── ProviderError (external data providers)
    ├── SignalError (signal generation)
    ├── StructureError (trade structures)
    └── ExecutionError (order execution)

Usage:
    from src.exceptions import DataWriteError, DataReadError

    try:
        repo.write_underlying_bars(df, run_id)
    except DataWriteError as e:
        logger.error(f"Write failed: {e}")
        # Implement retry logic for write errors
"""


class ValoremError(Exception):
    """Base exception for all Valorem errors.

    All domain-specific exceptions inherit from this class,
    allowing callers to catch all Valorem errors with a single handler.
    """

    pass


# =============================================================================
# Data Layer Exceptions
# =============================================================================


class DataError(ValoremError):
    """Base exception for data layer errors.

    Parent class for all data-related exceptions including
    database operations, data validation, and data quality issues.
    """

    pass


class DataWriteError(DataError):
    """Failed to write data to storage.

    Raised when database writes fail due to:
    - Connection errors
    - Constraint violations
    - Transaction rollbacks
    """

    pass


class DataReadError(DataError):
    """Failed to read data from storage.

    Raised when database reads fail due to:
    - Connection errors
    - Query errors
    - Missing data
    """

    pass


class DataValidationError(DataError):
    """Data validation failed.

    Raised when data does not meet expected format or constraints:
    - Missing required columns
    - Invalid data types
    - Out of range values
    """

    pass


class InsufficientDataError(DataError):
    """Not enough data to perform estimation.

    Raised when a statistical estimator (e.g., covariance) does not
    have enough observations to produce a reliable result.
    """

    pass


# =============================================================================
# Configuration Exceptions
# =============================================================================


class ConfigError(ValoremError):
    """Configuration error.

    Raised for configuration-related issues:
    - Invalid configuration values
    - Missing required configuration
    - Conflicting configuration options
    """

    pass


# =============================================================================
# External Provider Exceptions
# =============================================================================


class ProviderError(ValoremError):
    """External data provider error.

    Raised when external API calls fail:
    - Databento API errors
    - FRED API errors
    - Network/connection failures
    """

    pass


# =============================================================================
# Signal Generation Exceptions
# =============================================================================


class SignalError(ValoremError):
    """Signal generation error.

    Raised during signal generation:
    - Model inference failures
    - Invalid model configuration
    - Insufficient data for signal generation
    """

    pass


# =============================================================================
# Trade Structure Exceptions
# =============================================================================


class StructureError(ValoremError):
    """Trade structure creation error.

    Raised when trade structures cannot be created:
    - Missing option contracts
    - Invalid strike/expiry combinations
    - Insufficient market data
    """

    pass


# =============================================================================
# Execution Exceptions
# =============================================================================


class ExecutionError(ValoremError):
    """Order execution error.

    Raised during order execution:
    - Order routing failures
    - Fill errors
    - Position management errors
    """

    pass
