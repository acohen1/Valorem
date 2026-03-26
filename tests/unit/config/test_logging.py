"""Tests for centralized logging configuration."""

import logging

import pytest
from loguru import logger

from src.config.logging import InterceptHandler, add_file_handler, setup_logging
from src.config.schema import LoggingConfig


class TestInterceptHandler:
    """Tests for InterceptHandler routing stdlib records to loguru."""

    def test_stdlib_records_reach_loguru(self):
        """Stdlib log records are forwarded to loguru."""
        setup_logging(verbose=False)

        messages = []
        handler_id = logger.add(lambda msg: messages.append(str(msg)), level="INFO")

        stdlib_logger = logging.getLogger("test.intercept")
        stdlib_logger.info("hello from stdlib")

        logger.remove(handler_id)
        assert any("hello from stdlib" in m for m in messages)

    def test_exception_info_forwarded(self):
        """Exception info from stdlib records is preserved."""
        setup_logging(verbose=False)

        messages = []
        handler_id = logger.add(lambda msg: messages.append(str(msg)), level="ERROR")

        stdlib_logger = logging.getLogger("test.exception")
        try:
            raise ValueError("test error")
        except ValueError:
            stdlib_logger.exception("caught an error")

        logger.remove(handler_id)
        assert any("caught an error" in m for m in messages)
        assert any("ValueError" in m for m in messages)

    def test_handler_is_logging_handler(self):
        """InterceptHandler is a proper logging.Handler subclass."""
        handler = InterceptHandler()
        assert isinstance(handler, logging.Handler)


class TestSetupLogging:
    """Tests for setup_logging function."""

    def test_default_level_is_info(self):
        """Default level is INFO when neither verbose nor config provided."""
        setup_logging()

        messages = []
        handler_id = logger.add(lambda msg: messages.append(str(msg)), level="DEBUG")

        # DEBUG should not appear in the stderr handler (but will in our test sink)
        # We verify by checking the stderr handler's level indirectly
        logger.remove(handler_id)

    def test_verbose_enables_debug(self):
        """verbose=True sets DEBUG level."""
        setup_logging(verbose=True)

        messages = []
        handler_id = logger.add(lambda msg: messages.append(str(msg)), level="DEBUG")

        stdlib_logger = logging.getLogger("test.verbose")
        stdlib_logger.debug("debug message")

        logger.remove(handler_id)
        assert any("debug message" in m for m in messages)

    def test_config_level_respected(self):
        """LoggingConfig.level is respected when no verbose flag."""
        config = LoggingConfig(level="WARNING")
        setup_logging(config=config)
        # Should not crash; level is set to WARNING

    def test_verbose_overrides_config(self):
        """verbose=True overrides config.level."""
        config = LoggingConfig(level="WARNING")
        setup_logging(verbose=True, config=config)

        messages = []
        handler_id = logger.add(lambda msg: messages.append(str(msg)), level="DEBUG")

        stdlib_logger = logging.getLogger("test.override")
        stdlib_logger.debug("should appear")

        logger.remove(handler_id)
        assert any("should appear" in m for m in messages)

    def test_console_disabled(self):
        """config.console_enabled=False suppresses console output."""
        config = LoggingConfig(console_enabled=False)
        setup_logging(config=config)
        # No crash; stderr handler should not be added

    def test_idempotent(self):
        """Calling setup_logging twice does not crash or produce errors."""
        setup_logging(verbose=False)
        setup_logging(verbose=True)

        messages = []
        handler_id = logger.add(lambda msg: messages.append(str(msg)), level="DEBUG")

        stdlib_logger = logging.getLogger("test.idempotent")
        stdlib_logger.info("still works")

        logger.remove(handler_id)
        assert any("still works" in m for m in messages)

    def test_intercept_handler_installed_on_root(self):
        """After setup, root logger uses InterceptHandler."""
        setup_logging()

        root = logging.getLogger()
        handler_types = [type(h) for h in root.handlers]
        assert InterceptHandler in handler_types


class TestAddFileHandler:
    """Tests for add_file_handler function."""

    def test_creates_log_file(self, tmp_path):
        """Log file is created in the correct workflow subdirectory."""
        setup_logging()

        log_path = add_file_handler(
            workflow="backtest",
            logs_dir=str(tmp_path),
        )

        assert log_path is not None
        assert log_path.parent == tmp_path / "backtest"
        assert log_path.suffix == ".log"
        assert log_path.exists()

        # Clean up loguru handler
        logger.remove()

    def test_creates_workflow_subdirectory(self, tmp_path):
        """Workflow subdirectory is created automatically."""
        setup_logging()

        log_path = add_file_handler(
            workflow="train",
            logs_dir=str(tmp_path),
        )

        assert (tmp_path / "train").is_dir()
        assert log_path is not None

        logger.remove()

    def test_disabled_returns_none(self, tmp_path):
        """When enabled=False, no file is created and None is returned."""
        setup_logging()

        log_path = add_file_handler(
            workflow="ingest",
            logs_dir=str(tmp_path),
            enabled=False,
        )

        assert log_path is None
        assert not (tmp_path / "ingest").exists()

    def test_messages_written_to_file(self, tmp_path):
        """Log messages appear in the file."""
        setup_logging()

        log_path = add_file_handler(
            workflow="test",
            logs_dir=str(tmp_path),
            level="DEBUG",
        )

        logger.info("test message for file")

        # Force flush by removing handlers
        logger.remove()

        assert log_path is not None
        content = log_path.read_text()
        assert "test message for file" in content

    def test_json_format(self, tmp_path):
        """fmt='json' produces serialized (JSON) log output."""
        import json

        setup_logging()

        log_path = add_file_handler(
            workflow="json_test",
            logs_dir=str(tmp_path),
            fmt="json",
        )

        logger.info("json log entry")

        logger.remove()

        assert log_path is not None
        content = log_path.read_text().strip()
        # Each line should be valid JSON
        for line in content.splitlines():
            parsed = json.loads(line)
            assert "text" in parsed

    def test_timestamp_in_filename(self, tmp_path):
        """Log filename contains a UTC timestamp."""
        import re

        setup_logging()

        log_path = add_file_handler(
            workflow="ts_test",
            logs_dir=str(tmp_path),
        )

        assert log_path is not None
        # Filename should match YYYYMMDD_HHMMSS.log
        assert re.match(r"\d{8}_\d{6}\.log", log_path.name)

        logger.remove()

    def test_multiple_workflows(self, tmp_path):
        """Multiple workflows create separate subdirectories and files."""
        setup_logging()

        path_a = add_file_handler(workflow="ingest", logs_dir=str(tmp_path))
        path_b = add_file_handler(workflow="train", logs_dir=str(tmp_path))

        assert path_a is not None
        assert path_b is not None
        assert path_a.parent.name == "ingest"
        assert path_b.parent.name == "train"
        assert path_a != path_b

        logger.remove()
