"""Integration tests for configuration system.

Tests the complete workflow of loading, validating, and using configuration.
"""

from pathlib import Path

import pytest

from src.config.environments import Environment, get_env_config_path
from src.config.loader import ConfigLoader
from src.config.paths import PathResolver


class TestConfigurationIntegration:
    """Integration tests for configuration loading and usage."""

    def test_load_production_config(self):
        """Test loading actual production configuration files."""
        # Get path to actual config files
        repo_root = Path(__file__).parent.parent.parent.parent
        config_path = repo_root / "config" / "config.yaml"

        # Ensure config file exists
        if not config_path.exists():
            pytest.skip("Production config file not found")

        # Load config with dev environment
        config = ConfigLoader.load(config_path, env="dev")

        # Validate config
        ConfigLoader.validate(config)

        # Verify config loaded successfully
        assert config.version == "v1"
        assert config.project.name == "valorem"
        assert config.universe.underlying == "SPY"

        # Verify dev overrides applied
        assert config.training.device != "cuda"  # Overridden from dev.yaml
        assert config.training.batch_size == 4   # Overridden from dev.yaml
        assert config.logging.level == "DEBUG"   # From base config

    def test_path_resolution_with_real_config(self):
        """Test path resolution with actual configuration."""
        repo_root = Path(__file__).parent.parent.parent.parent
        config_path = repo_root / "config" / "config.yaml"

        if not config_path.exists():
            pytest.skip("Production config file not found")

        config = ConfigLoader.load(config_path, env="dev")

        # Create path resolver
        resolver = PathResolver(repo_root, config.paths)

        # Test resolving paths
        db_path = resolver.resolve("db_path")
        assert db_path == repo_root / "data" / "db.sqlite"

        checkpoints_dir = resolver.resolve("checkpoints_dir")
        assert checkpoints_dir == repo_root / "artifacts" / "checkpoints"

    def test_environment_overlay_merging(self):
        """Test that environment overlays correctly merge with base config."""
        repo_root = Path(__file__).parent.parent.parent.parent
        config_path = repo_root / "config" / "config.yaml"

        if not config_path.exists():
            pytest.skip("Production config file not found")

        # Load with base config (no environment)
        base_config = ConfigLoader.load(config_path, env="nonexistent")

        # Load with dev environment
        dev_config = ConfigLoader.load(config_path, env="dev")

        # Base config should use production-tuned defaults
        assert base_config.training.device == "auto"
        assert base_config.training.batch_size == 64

        # Dev config should have overrides
        assert dev_config.training.device == "mps"
        assert dev_config.training.batch_size == 4

        # Non-overridden values should be same
        assert base_config.universe.underlying == dev_config.universe.underlying
        assert base_config.surface.delta_buckets.ATM == dev_config.surface.delta_buckets.ATM

    def test_end_to_end_config_usage(self):
        """Test complete workflow: load config, resolve paths, use settings."""
        repo_root = Path(__file__).parent.parent.parent.parent
        config_path = repo_root / "config" / "config.yaml"

        if not config_path.exists():
            pytest.skip("Production config file not found")

        # 1. Load config
        config = ConfigLoader.load(config_path, env="dev")

        # 2. Validate config
        ConfigLoader.validate(config)

        # 3. Create path resolver
        resolver = PathResolver(repo_root, config.paths)

        # 4. Use configuration values
        assert config.universe.underlying == "SPY"
        assert config.surface.tenor_bins.bins == [7, 14, 30, 60, 90, 120]
        assert config.labels.horizons == [5, 10, 21]

        # 5. Resolve and ensure paths exist
        data_dir = resolver.ensure_exists("data_dir", is_file=False)
        assert data_dir.is_dir()

        # 6. Check model config
        assert config.model.patchtst.patch_len == 6
        assert config.model.gnn.model_type == "GAT"

        # 7. Check risk caps
        assert config.risk.caps.max_abs_delta > 0
        assert config.risk.caps.max_total_notional_usd > 0

    def test_get_environment_config_path(self):
        """Test getting environment config path."""
        repo_root = Path(__file__).parent.parent.parent.parent
        config_dir = repo_root / "config"

        if not config_dir.exists():
            pytest.skip("Config directory not found")

        # Test getting dev environment path
        dev_path = get_env_config_path(config_dir, Environment.DEV)

        if dev_path is not None:
            assert dev_path.exists()
            assert dev_path.name == "dev.yaml"
