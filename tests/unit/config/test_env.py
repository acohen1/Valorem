"""Unit tests for config/env.py - Trading environment configuration."""

import os
from unittest.mock import patch

import pytest

from src.config.env import (
    EnvironmentConfig,
    TradingMode,
    print_validation_results,
    validate_cli_config,
)


class TestTradingMode:
    """Tests for TradingMode enum."""

    def test_mode_values(self):
        """Should have expected mode values."""
        assert TradingMode.MOCK.value == "mock"
        assert TradingMode.PAPER_LIVE.value == "paper_live"
        assert TradingMode.PAPER_DB.value == "paper_db"
        assert TradingMode.LIVE.value == "live"

    def test_mode_from_string(self):
        """Should create mode from string value."""
        assert TradingMode("mock") == TradingMode.MOCK
        assert TradingMode("paper_live") == TradingMode.PAPER_LIVE
        assert TradingMode("paper_db") == TradingMode.PAPER_DB
        assert TradingMode("live") == TradingMode.LIVE

    def test_invalid_mode_raises(self):
        """Should raise ValueError for invalid mode."""
        with pytest.raises(ValueError):
            TradingMode("invalid")


class TestEnvironmentConfigInit:
    """Tests for EnvironmentConfig initialization."""

    def test_default_mock_mode(self):
        """Should accept MOCK mode with minimal config."""
        config = EnvironmentConfig(mode=TradingMode.MOCK)
        assert config.mode == TradingMode.MOCK
        assert config.databento_api_key is None
        assert config.database_url is None

    def test_paper_live_mode(self):
        """Should accept PAPER_LIVE mode with API key."""
        config = EnvironmentConfig(
            mode=TradingMode.PAPER_LIVE,
            databento_api_key="test-key",
        )
        assert config.mode == TradingMode.PAPER_LIVE
        assert config.databento_api_key == "test-key"

    def test_paper_db_mode(self):
        """Should accept PAPER_DB mode with database URL."""
        config = EnvironmentConfig(
            mode=TradingMode.PAPER_DB,
            database_url="postgresql://localhost/valorem",
        )
        assert config.mode == TradingMode.PAPER_DB
        assert config.database_url == "postgresql://localhost/valorem"

    def test_live_mode(self):
        """Should accept LIVE mode with API key and broker config."""
        config = EnvironmentConfig(
            mode=TradingMode.LIVE,
            databento_api_key="test-key",
            ibkr_host="127.0.0.1",
            ibkr_port=7497,
            ibkr_client_id=1,
        )
        assert config.mode == TradingMode.LIVE
        assert config.databento_api_key == "test-key"
        assert config.ibkr_host == "127.0.0.1"
        assert config.ibkr_port == 7497


class TestEnvironmentConfigFromEnv:
    """Tests for EnvironmentConfig.from_env()."""

    def test_default_mode(self):
        """Should use default mode when no env vars set."""
        with patch.dict(os.environ, {}, clear=True):
            with patch("os.getenv", side_effect=lambda k, d=None: d):
                config = EnvironmentConfig.from_env(mode="mock")
                assert config.mode == TradingMode.MOCK

    def test_mode_from_env_override(self):
        """Should override mode from VALOREM_MODE env var."""
        with patch.dict(os.environ, {"VALOREM_MODE": "paper_live"}):
            config = EnvironmentConfig.from_env(mode="mock")
            assert config.mode == TradingMode.PAPER_LIVE

    def test_api_key_from_env(self):
        """Should load API key from environment."""
        with patch.dict(os.environ, {"DATABENTO_API_KEY": "db-test-key"}):
            config = EnvironmentConfig.from_env()
            assert config.databento_api_key == "db-test-key"

    def test_database_url_from_env(self):
        """Should load database URL from environment."""
        with patch.dict(os.environ, {"VALOREM_DATABASE_URL": "postgresql://test"}):
            config = EnvironmentConfig.from_env()
            assert config.database_url == "postgresql://test"

    def test_log_level_from_env(self):
        """Should load log level from environment."""
        with patch.dict(os.environ, {"VALOREM_LOG_LEVEL": "DEBUG"}):
            config = EnvironmentConfig.from_env()
            assert config.log_level == "DEBUG"

    def test_invalid_mode_from_env_uses_default(self):
        """Should use default mode if env var is invalid."""
        with patch.dict(os.environ, {"VALOREM_MODE": "invalid_mode"}):
            config = EnvironmentConfig.from_env(mode="mock")
            assert config.mode == TradingMode.MOCK

    def test_accepts_string_mode(self):
        """Should accept string for mode parameter."""
        config = EnvironmentConfig.from_env(mode="paper_db")
        assert config.mode == TradingMode.PAPER_DB


class TestEnvironmentConfigFromDict:
    """Tests for EnvironmentConfig.from_dict()."""

    def test_from_dict_basic(self):
        """Should create config from dictionary."""
        data = {
            "mode": "paper_live",
            "databento_api_key": "test-key",
        }
        config = EnvironmentConfig.from_dict(data)
        assert config.mode == TradingMode.PAPER_LIVE
        assert config.databento_api_key == "test-key"

    def test_from_dict_defaults(self):
        """Should use defaults for missing keys."""
        data = {}
        config = EnvironmentConfig.from_dict(data)
        assert config.mode == TradingMode.MOCK
        assert config.log_level == "INFO"

    def test_from_dict_falls_back_to_env(self):
        """Should fall back to env vars if dict values are None."""
        data = {"mode": "paper_live"}
        with patch.dict(os.environ, {"DATABENTO_API_KEY": "env-key"}):
            config = EnvironmentConfig.from_dict(data)
            assert config.databento_api_key == "env-key"


class TestEnvironmentConfigValidate:
    """Tests for EnvironmentConfig.validate()."""

    def test_mock_mode_no_validation(self):
        """MOCK mode should pass without API keys."""
        config = EnvironmentConfig(mode=TradingMode.MOCK)
        # Should not raise
        config.validate()

    def test_paper_live_requires_api_key(self):
        """PAPER_LIVE mode should require Databento API key."""
        config = EnvironmentConfig(mode=TradingMode.PAPER_LIVE)

        with pytest.raises(ValueError, match="DATABENTO_API_KEY required"):
            config.validate()

    def test_paper_live_with_api_key(self):
        """PAPER_LIVE mode should pass with API key."""
        config = EnvironmentConfig(
            mode=TradingMode.PAPER_LIVE,
            databento_api_key="test-key",
        )
        # Should not raise
        config.validate()

    def test_paper_db_requires_database_url(self):
        """PAPER_DB mode should require database URL."""
        config = EnvironmentConfig(mode=TradingMode.PAPER_DB)

        with pytest.raises(ValueError, match="VALOREM_DATABASE_URL required"):
            config.validate()

    def test_paper_db_with_database_url(self):
        """PAPER_DB mode should pass with database URL."""
        config = EnvironmentConfig(
            mode=TradingMode.PAPER_DB,
            database_url="postgresql://localhost/valorem",
        )
        # Should not raise
        config.validate()

    def test_live_requires_api_key(self):
        """LIVE mode should require Databento API key."""
        config = EnvironmentConfig(mode=TradingMode.LIVE)

        with pytest.raises(ValueError, match="DATABENTO_API_KEY required"):
            config.validate()

    def test_live_with_api_key(self):
        """LIVE mode should pass with API key."""
        config = EnvironmentConfig(
            mode=TradingMode.LIVE,
            databento_api_key="test-key",
        )
        # Should not raise
        config.validate()


class TestEnvironmentConfigProperties:
    """Tests for EnvironmentConfig properties."""

    def test_is_live_data_mock(self):
        """MOCK mode should not be live data."""
        config = EnvironmentConfig(mode=TradingMode.MOCK)
        assert config.is_live_data is False

    def test_is_live_data_paper_live(self):
        """PAPER_LIVE mode should be live data."""
        config = EnvironmentConfig(mode=TradingMode.PAPER_LIVE)
        assert config.is_live_data is True

    def test_is_live_data_paper_db(self):
        """PAPER_DB mode should not be live data."""
        config = EnvironmentConfig(mode=TradingMode.PAPER_DB)
        assert config.is_live_data is False

    def test_is_live_data_live(self):
        """LIVE mode should be live data."""
        config = EnvironmentConfig(mode=TradingMode.LIVE)
        assert config.is_live_data is True

    def test_is_simulated_execution(self):
        """Simulated execution for all modes except LIVE."""
        assert EnvironmentConfig(mode=TradingMode.MOCK).is_simulated_execution is True
        assert EnvironmentConfig(mode=TradingMode.PAPER_LIVE).is_simulated_execution is True
        assert EnvironmentConfig(mode=TradingMode.PAPER_DB).is_simulated_execution is True
        assert EnvironmentConfig(mode=TradingMode.LIVE).is_simulated_execution is False

    def test_is_real_execution(self):
        """Only LIVE mode uses real broker execution."""
        assert EnvironmentConfig(mode=TradingMode.MOCK).is_real_execution is False
        assert EnvironmentConfig(mode=TradingMode.PAPER_LIVE).is_real_execution is False
        assert EnvironmentConfig(mode=TradingMode.PAPER_DB).is_real_execution is False
        assert EnvironmentConfig(mode=TradingMode.LIVE).is_real_execution is True

    def test_requires_api_key(self):
        """PAPER_LIVE and LIVE require API key."""
        assert EnvironmentConfig(mode=TradingMode.MOCK).requires_api_key is False
        assert EnvironmentConfig(mode=TradingMode.PAPER_LIVE).requires_api_key is True
        assert EnvironmentConfig(mode=TradingMode.PAPER_DB).requires_api_key is False
        assert EnvironmentConfig(mode=TradingMode.LIVE).requires_api_key is True

    def test_requires_broker(self):
        """Only LIVE mode requires broker connection."""
        assert EnvironmentConfig(mode=TradingMode.MOCK).requires_broker is False
        assert EnvironmentConfig(mode=TradingMode.PAPER_LIVE).requires_broker is False
        assert EnvironmentConfig(mode=TradingMode.PAPER_DB).requires_broker is False
        assert EnvironmentConfig(mode=TradingMode.LIVE).requires_broker is True


class TestEnvironmentConfigSerialization:
    """Tests for EnvironmentConfig serialization."""

    def test_to_dict_masks_api_key(self):
        """to_dict should mask sensitive API key."""
        config = EnvironmentConfig(
            mode=TradingMode.PAPER_LIVE,
            databento_api_key="secret-key",
        )
        result = config.to_dict()
        assert result["databento_api_key"] == "***"

    def test_to_dict_masks_database_url(self):
        """to_dict should mask password in database URL."""
        config = EnvironmentConfig(
            mode=TradingMode.PAPER_DB,
            database_url="postgresql://user:secret_password@localhost/db",
        )
        result = config.to_dict()
        assert "secret_password" not in result["database_url"]
        assert "***" in result["database_url"]

    def test_to_dict_handles_none(self):
        """to_dict should handle None values."""
        config = EnvironmentConfig(mode=TradingMode.MOCK)
        result = config.to_dict()
        assert result["databento_api_key"] is None
        assert result["database_url"] is None

    def test_get_summary(self):
        """get_summary should return readable string."""
        config = EnvironmentConfig(
            mode=TradingMode.PAPER_LIVE,
            databento_api_key="test-key",
        )
        summary = config.get_summary()
        assert "paper_live" in summary
        assert "configured" in summary


class TestValidateCliConfig:
    """Tests for validate_cli_config function."""

    def test_valid_mock_config(self):
        """Should validate valid mock config."""
        results = validate_cli_config(mode="mock")
        assert results["valid"] is True
        assert results["config"] is not None
        assert results["config"].mode == TradingMode.MOCK
        assert len(results["errors"]) == 0

    def test_invalid_mode(self):
        """Should fail for invalid mode."""
        results = validate_cli_config(mode="invalid")
        assert results["valid"] is False
        assert results["config"] is None
        assert len(results["errors"]) > 0

    def test_paper_live_missing_api_key(self):
        """Should fail for paper_live without API key."""
        with patch.dict(os.environ, {}, clear=True):
            with patch("os.getenv", return_value=None):
                results = validate_cli_config(mode="paper_live")
                assert results["valid"] is False
                assert any("DATABENTO_API_KEY" in e for e in results["errors"])

    def test_missing_symbols_file_warning(self):
        """Should warn when symbols file missing for paper_live."""
        with patch.dict(os.environ, {"DATABENTO_API_KEY": "test-key"}):
            results = validate_cli_config(mode="paper_live")
            assert results["valid"] is True
            assert any("symbols" in w.lower() for w in results["warnings"])

    def test_nonexistent_symbols_file_error(self, tmp_path):
        """Should error when symbols file doesn't exist."""
        results = validate_cli_config(
            mode="mock",
            symbols_file="/nonexistent/symbols.json",
        )
        assert results["valid"] is False
        assert any("Symbols file not found" in e for e in results["errors"])

    def test_nonexistent_checkpoint_error(self):
        """Should error when checkpoint doesn't exist."""
        results = validate_cli_config(
            mode="mock",
            checkpoint="/nonexistent/model.pt",
        )
        assert results["valid"] is False
        assert any("checkpoint not found" in e for e in results["errors"])


class TestPrintValidationResults:
    """Tests for print_validation_results function."""

    def test_print_valid_results(self, capsys):
        """Should print success for valid config."""
        results = {
            "valid": True,
            "config": EnvironmentConfig(mode=TradingMode.MOCK),
            "warnings": [],
            "errors": [],
        }
        print_validation_results(results)
        captured = capsys.readouterr()
        assert "Configuration valid" in captured.out
        assert "mock" in captured.out

    def test_print_invalid_results(self, capsys):
        """Should print errors for invalid config."""
        results = {
            "valid": False,
            "config": None,
            "warnings": [],
            "errors": ["Missing API key"],
        }
        print_validation_results(results)
        captured = capsys.readouterr()
        assert "Configuration invalid" in captured.out
        assert "Missing API key" in captured.out

    def test_print_warnings(self, capsys):
        """Should print warnings."""
        results = {
            "valid": True,
            "config": EnvironmentConfig(mode=TradingMode.MOCK),
            "warnings": ["No symbols file"],
            "errors": [],
        }
        print_validation_results(results)
        captured = capsys.readouterr()
        assert "Warning" in captured.out
        assert "No symbols file" in captured.out
