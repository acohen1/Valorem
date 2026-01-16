"""Unit tests for path resolution."""

from pathlib import Path

import pytest

from src.config.paths import PathResolver
from src.config.schema import PathsConfig


@pytest.fixture
def temp_root(tmp_path):
    """Create temporary repository root."""
    return tmp_path


@pytest.fixture
def paths_config():
    """Create default paths configuration."""
    return PathsConfig()


class TestPathResolver:
    """Test path resolver."""

    def test_resolve_returns_absolute_path(self, temp_root, paths_config):
        """Test that resolve returns absolute paths."""
        resolver = PathResolver(temp_root, paths_config)

        db_path = resolver.resolve("db_path")
        assert db_path.is_absolute()
        assert db_path == temp_root / "data" / "db.sqlite"

    def test_resolve_all_configured_paths(self, temp_root, paths_config):
        """Test resolving all paths from config."""
        resolver = PathResolver(temp_root, paths_config)

        # Test a few key paths
        assert resolver.resolve("data_dir") == temp_root / "data"
        assert resolver.resolve("parquet_dir") == temp_root / "data" / "parquet"
        assert resolver.resolve("checkpoints_dir") == temp_root / "artifacts" / "checkpoints"
        assert resolver.resolve("logs_dir") == temp_root / "logs"

    def test_resolve_invalid_key_raises_error(self, temp_root, paths_config):
        """Test that resolving invalid key raises KeyError."""
        resolver = PathResolver(temp_root, paths_config)

        with pytest.raises(KeyError, match="Path key 'nonexistent' not found"):
            resolver.resolve("nonexistent")

    def test_get_root(self, temp_root, paths_config):
        """Test getting repository root."""
        resolver = PathResolver(temp_root, paths_config)

        root = resolver.root
        assert root.is_absolute()
        assert root == temp_root.resolve()

    def test_ensure_exists_creates_directory(self, temp_root, paths_config):
        """Test that ensure_exists creates directories."""
        resolver = PathResolver(temp_root, paths_config)

        # Ensure data directory exists
        data_dir = resolver.ensure_exists("data_dir", is_file=False)

        assert data_dir.exists()
        assert data_dir.is_dir()
        assert data_dir == temp_root / "data"

    def test_ensure_exists_creates_parent_for_file(self, temp_root, paths_config):
        """Test that ensure_exists creates parent directory for files."""
        resolver = PathResolver(temp_root, paths_config)

        # Ensure db_path parent directory exists
        db_path = resolver.ensure_exists("db_path", is_file=True)

        # Parent should exist, but file should not be created
        assert db_path.parent.exists()
        assert db_path.parent.is_dir()
        assert not db_path.exists()  # File itself not created

    def test_get_all_paths(self, temp_root, paths_config):
        """Test getting all paths as dictionary."""
        resolver = PathResolver(temp_root, paths_config)

        all_paths = resolver.get_all_paths()

        # Should have all configured paths
        assert "data_dir" in all_paths
        assert "db_path" in all_paths
        assert "checkpoints_dir" in all_paths

        # All paths should be absolute
        for path in all_paths.values():
            assert path.is_absolute()

    def test_resolver_handles_relative_root(self, tmp_path, paths_config):
        """Test that resolver converts relative root to absolute."""
        # Create a relative path (though it will still be resolved)
        relative_root = Path(".")
        resolver = PathResolver(relative_root, paths_config)

        # Root should be absolute
        assert resolver.root.is_absolute()

        # Resolved paths should be absolute
        db_path = resolver.resolve("db_path")
        assert db_path.is_absolute()
