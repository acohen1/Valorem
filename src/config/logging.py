"""Centralized logging configuration.

Installs loguru as the single logging backend for the entire application.
All stdlib logging.getLogger() calls in library code are intercepted and
routed through loguru automatically via InterceptHandler.

Usage in scripts::

    from src.config.logging import setup_logging, add_file_handler
    setup_logging(verbose=True)

    # After config loads:
    config = ConfigLoader.load(...)
    add_file_handler(workflow="backtest", logs_dir=config.paths.logs_dir)
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from loguru import logger

if TYPE_CHECKING:
    from src.config.schema import LoggingConfig


class InterceptHandler(logging.Handler):
    """Route stdlib logging records to loguru.

    This is loguru's officially recommended pattern for unifying
    stdlib logging with loguru's formatting and sinks.
    """

    def emit(self, record: logging.LogRecord) -> None:
        # Get corresponding Loguru level
        try:
            level: str | int = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where the logged message originated.
        # logging.currentframe() returns emit()'s frame (not in logging module),
        # so we step into the stdlib call chain first, then walk past it.
        frame, depth = logging.currentframe(), 1
        frame = frame.f_back
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def setup_logging(
    *,
    verbose: bool = False,
    config: LoggingConfig | None = None,
) -> None:
    """Configure loguru as the application-wide logging backend.

    Call this once at the top of each script's ``main()`` function.
    After this call, all stdlib ``logging.getLogger()`` calls throughout
    ``src/`` are automatically routed through loguru.

    Args:
        verbose: If True, set level to DEBUG. Overrides config.level.
        config: Optional LoggingConfig from the loaded YAML config.
                If provided, respects ``config.level`` (unless verbose=True)
                and ``config.console_enabled``.
    """
    # Determine level
    if verbose:
        level = "DEBUG"
    elif config is not None:
        level = config.level
    else:
        level = "INFO"

    # Remove default loguru handler
    logger.remove()

    # Add console handler (stderr) unless config explicitly disables it
    console_enabled = config.console_enabled if config is not None else True
    if console_enabled:
        logger.add(sys.stderr, level=level)

    # Install InterceptHandler on Python's root logger so that all
    # stdlib logging.getLogger(__name__) calls route through loguru.
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)

    # Route Python warnings (e.g. BentoWarning) through logging → loguru
    logging.captureWarnings(True)


def add_file_handler(
    *,
    workflow: str,
    logs_dir: str = "artifacts/logs",
    level: str = "DEBUG",
    fmt: Literal["text", "json"] = "text",
    enabled: bool = True,
    rotation: str = "100 MB",
    retention: str = "30 days",
) -> Path | None:
    """Add a file sink to loguru for the given workflow.

    Call this after config loads to persist logs to disk. Each invocation
    creates a new session log file named by UTC timestamp.

    Args:
        workflow: Workflow subdirectory name (e.g. "backtest", "live").
        logs_dir: Root logs directory (default: ``artifacts/logs``).
        level: Minimum log level for the file sink.
        fmt: ``"text"`` for human-readable or ``"json"`` for structured output.
        enabled: If False, skip adding the file handler entirely.
        rotation: loguru rotation spec (e.g. ``"100 MB"``).
        retention: loguru retention spec (e.g. ``"30 days"``).

    Returns:
        Path to the created log file, or None if disabled.
    """
    if not enabled:
        return None

    log_dir = Path(logs_dir) / workflow
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"{timestamp}.log"

    logger.add(
        log_path,
        level=level,
        rotation=rotation,
        retention=retention,
        serialize=(fmt == "json"),
    )

    logger.info(f"File logging enabled: {log_path}")
    return log_path
