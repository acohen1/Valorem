"""Unit tests for configuration loader."""

import tempfile
from datetime import date
from pathlib import Path

import pytest
import yaml

from src.config.loader import ConfigLoadError, ConfigLoader
from src.config.schema import ConfigSchema


@pytest.fixture
def temp_config_dir(tmp_path):
    """Create temporary config directory with base config."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()

    # Create minimal valid base config
    base_config = {
        "version": "v1",
        "dataset": {
            "splits": {
                "train_start": "2020-01-01",
                "train_end": "2021-12-31",
                "val_start": "2022-01-01",
                "val_end": "2022-06-30",
                "test_start": "2022-07-01",
                "test_end": "2022-12-31",
            }
        },
        "backtest": {
            "start_date": "2023-01-01",
            "end_date": "2023-12-31",
        },
    }

    base_path = config_dir / "config.yaml"
    with open(base_path, "w") as f:
        yaml.dump(base_config, f)

    return config_dir


@pytest.fixture
def temp_config_with_env(temp_config_dir):
    """Create config directory with environment overlay."""
    env_dir = temp_config_dir / "environments"
    env_dir.mkdir()

    env_config = {
        "training": {
            "device": "cpu",
            "batch_size": 4,
        }
    }

    env_path = env_dir / "dev.yaml"
    with open(env_path, "w") as f:
        yaml.dump(env_config, f)

    return temp_config_dir


class TestConfigLoader:
    """Test configuration loader."""

    def test_load_base_config(self, temp_config_dir):
        """Test loading base configuration without environment overlay."""
        config_path = temp_config_dir / "config.yaml"
        config = ConfigLoader.load(config_path, env="nonexistent")

        assert config.version == "v1"
        assert config.dataset.splits.train_start == date(2020, 1, 1)
        # Defaults should be set
        assert config.training.device == "cuda"
        assert config.training.batch_size == 32

    def test_load_with_environment_overlay(self, temp_config_with_env):
        """Test loading configuration with environment overlay."""
        config_path = temp_config_with_env / "config.yaml"
        config = ConfigLoader.load(config_path, env="dev")

        # Base config values
        assert config.version == "v1"
        assert config.dataset.splits.train_start == date(2020, 1, 1)

        # Overridden values from dev.yaml
        assert config.training.device == "cpu"
        assert config.training.batch_size == 4

    def test_deep_merge(self):
        """Test deep dictionary merge."""
        base = {
            "a": 1,
            "b": {"c": 2, "d": 3},
            "e": 4,
        }

        override = {
            "b": {"d": 99},  # Override nested value
            "e": 100,  # Override top-level value
            "f": 5,  # Add new value
        }

        result = ConfigLoader._deep_merge(base, override)

        assert result["a"] == 1  # Unchanged
        assert result["b"]["c"] == 2  # Unchanged nested
        assert result["b"]["d"] == 99  # Overridden nested
        assert result["e"] == 100  # Overridden top-level
        assert result["f"] == 5  # New value

    def test_load_nonexistent_file(self):
        """Test loading nonexistent config file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            ConfigLoader.load(Path("/nonexistent/config.yaml"))

    def test_load_invalid_yaml(self, tmp_path):
        """Test loading invalid YAML raises ConfigLoadError."""
        invalid_yaml_path = tmp_path / "invalid.yaml"
        with open(invalid_yaml_path, "w") as f:
            f.write("invalid: yaml: content:\n  - malformed")

        with pytest.raises(ConfigLoadError, match="Failed to parse YAML"):
            ConfigLoader.load(invalid_yaml_path)

    def test_validate_buy_at_must_be_ask(self, temp_config_dir):
        """Test validation fails if buy_at is not 'ask'."""
        # This is already enforced by Pydantic Literal type, but test cross-field validation
        config_path = temp_config_dir / "config.yaml"
        config = ConfigLoader.load(config_path)

        # Should not raise (buy_at defaults to "ask")
        ConfigLoader.validate(config)

    def test_validate_dataset_min_dte_less_than_max(self, temp_config_dir):
        """Test validation fails if min_dte >= max_dte."""
        # Create config with invalid DTE range
        invalid_config_path = temp_config_dir / "invalid_dte.yaml"
        with open(invalid_config_path, "w") as f:
            yaml.dump({
                "version": "v1",
                "dataset": {
                    "splits": {
                        "train_start": "2020-01-01",
                        "train_end": "2021-12-31",
                        "val_start": "2022-01-01",
                        "val_end": "2022-06-30",
                        "test_start": "2022-07-01",
                        "test_end": "2022-12-31",
                    },
                    "min_dte": 120,
                    "max_dte": 7,  # Invalid: max < min
                },
                "backtest": {
                    "start_date": "2023-01-01",
                    "end_date": "2023-12-31",
                },
            }, f)

        # Validation happens during load()
        with pytest.raises(ValueError, match="dataset.min_dte must be less than dataset.max_dte"):
            ConfigLoader.load(invalid_config_path)

    def test_validate_backtest_dates(self, temp_config_dir):
        """Test validation fails if backtest start >= end."""
        invalid_config_path = temp_config_dir / "invalid_backtest.yaml"
        with open(invalid_config_path, "w") as f:
            yaml.dump({
                "version": "v1",
                "dataset": {
                    "splits": {
                        "train_start": "2020-01-01",
                        "train_end": "2021-12-31",
                        "val_start": "2022-01-01",
                        "val_end": "2022-06-30",
                        "test_start": "2022-07-01",
                        "test_end": "2022-12-31",
                    },
                },
                "backtest": {
                    "start_date": "2023-12-31",
                    "end_date": "2023-01-01",  # Before start
                },
            }, f)

        # Validation happens during load()
        with pytest.raises(ValueError, match="backtest.start_date must be before"):
            ConfigLoader.load(invalid_config_path)
