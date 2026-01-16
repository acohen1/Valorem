"""Unit tests for environment utilities."""

from pathlib import Path
from unittest.mock import patch

import pytest

from src.config.environments import (
    Environment,
    get_env_config_path,
    get_environment,
    validate_environment_transition,
)


class TestGetEnvironment:
    """Test get_environment function."""

    def test_get_environment_default(self):
        """Test getting environment with default value."""
        with patch.dict("os.environ", {}, clear=True):
            env = get_environment()
            assert env == Environment.DEV

    def test_get_environment_from_env_var(self):
        """Test getting environment from RHUBARB_ENV."""
        with patch.dict("os.environ", {"RHUBARB_ENV": "backtest"}):
            env = get_environment()
            assert env == Environment.BACKTEST

    def test_get_environment_case_insensitive(self):
        """Test that environment variable is case insensitive."""
        with patch.dict("os.environ", {"RHUBARB_ENV": "PAPER"}):
            env = get_environment()
            assert env == Environment.PAPER

    def test_get_environment_invalid_value(self):
        """Test that invalid environment defaults to DEV."""
        with patch.dict("os.environ", {"RHUBARB_ENV": "invalid"}):
            env = get_environment()
            assert env == Environment.DEV

    def test_get_environment_all_valid_values(self):
        """Test all valid environment values."""
        for env_type in Environment:
            with patch.dict("os.environ", {"RHUBARB_ENV": env_type.value}):
                env = get_environment()
                assert env == env_type


class TestGetEnvConfigPath:
    """Test get_env_config_path function."""

    def test_get_env_config_path_exists(self, tmp_path):
        """Test getting environment config path when file exists."""
        # Create environment config file
        env_dir = tmp_path / "environments"
        env_dir.mkdir()
        dev_config = env_dir / "dev.yaml"
        dev_config.write_text("test: true")

        result = get_env_config_path(tmp_path, Environment.DEV)
        assert result == dev_config
        assert result.exists()

    def test_get_env_config_path_not_exists(self, tmp_path):
        """Test getting environment config path when file doesn't exist."""
        result = get_env_config_path(tmp_path, Environment.DEV)
        assert result is None

    def test_get_env_config_path_uses_current_env(self, tmp_path):
        """Test that get_env_config_path uses current environment when None."""
        # Create backtest config
        env_dir = tmp_path / "environments"
        env_dir.mkdir()
        backtest_config = env_dir / "backtest.yaml"
        backtest_config.write_text("test: true")

        with patch.dict("os.environ", {"RHUBARB_ENV": "backtest"}):
            result = get_env_config_path(tmp_path, env=None)
            assert result == backtest_config


class TestValidateEnvironmentTransition:
    """Test validate_environment_transition function."""

    def test_valid_transition_dev_to_dev(self):
        """Test valid transition from dev to dev."""
        # Should not raise
        validate_environment_transition(Environment.DEV, Environment.DEV)

    def test_valid_transition_dev_to_backtest(self):
        """Test valid transition from dev to backtest."""
        # Should not raise
        validate_environment_transition(Environment.DEV, Environment.BACKTEST)

    def test_valid_transition_backtest_to_paper(self):
        """Test valid transition from backtest to paper."""
        # Should not raise
        validate_environment_transition(Environment.BACKTEST, Environment.PAPER)

    def test_valid_transition_paper_to_live(self):
        """Test valid transition from paper to live."""
        # Should not raise
        validate_environment_transition(Environment.PAPER, Environment.LIVE)

    def test_valid_transition_live_to_live(self):
        """Test valid transition from live to live."""
        # Should not raise
        validate_environment_transition(Environment.LIVE, Environment.LIVE)

    def test_invalid_transition_dev_to_paper(self):
        """Test invalid transition from dev to paper."""
        with pytest.raises(ValueError, match="not allowed"):
            validate_environment_transition(Environment.DEV, Environment.PAPER)

    def test_invalid_transition_dev_to_live(self):
        """Test invalid transition from dev to live."""
        with pytest.raises(ValueError, match="not allowed"):
            validate_environment_transition(Environment.DEV, Environment.LIVE)

    def test_invalid_transition_backtest_to_dev(self):
        """Test invalid transition from backtest to dev."""
        with pytest.raises(ValueError, match="not allowed"):
            validate_environment_transition(Environment.BACKTEST, Environment.DEV)

    def test_invalid_transition_live_to_paper(self):
        """Test invalid transition from live to paper."""
        with pytest.raises(ValueError, match="not allowed"):
            validate_environment_transition(Environment.LIVE, Environment.PAPER)

    def test_invalid_transition_live_to_dev(self):
        """Test invalid transition from live to dev."""
        with pytest.raises(ValueError, match="not allowed"):
            validate_environment_transition(Environment.LIVE, Environment.DEV)
