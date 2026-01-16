"""Path resolution utilities.

This module provides utilities to resolve logical path keys to absolute filesystem paths.
"""

import logging
from pathlib import Path

from src.config.schema import PathsConfig

logger = logging.getLogger(__name__)


class PathResolver:
    """Resolves logical path keys to absolute filesystem paths."""

    def __init__(self, root: Path, paths_config: PathsConfig):
        """Initialize path resolver.

        Args:
            root: Repository root directory (absolute path)
            paths_config: Paths configuration from ConfigSchema
        """
        self._root = root.resolve()  # Ensure absolute path
        self._config = paths_config
        logger.debug(f"PathResolver initialized with root: {self._root}")

    @property
    def root(self) -> Path:
        """Get repository root path.

        Returns:
            Absolute path to repository root
        """
        return self._root

    def resolve(self, key: str) -> Path:
        """Resolve logical path key to absolute path.

        Args:
            key: Logical path key (e.g., "db_path", "checkpoints_dir")

        Returns:
            Absolute path

        Raises:
            KeyError: If key is not found in paths config
        """
        if not hasattr(self._config, key):
            raise KeyError(
                f"Path key '{key}' not found in paths config. "
                f"Available keys: {list(PathsConfig.model_fields.keys())}"
            )

        relative_path = getattr(self._config, key)
        absolute_path = (self._root / relative_path).resolve()

        logger.debug(f"Resolved '{key}': {relative_path} -> {absolute_path}")
        return absolute_path

    def ensure_exists(self, key: str, is_file: bool = False) -> Path:
        """Resolve path and ensure it exists (create if needed).

        Args:
            key: Logical path key
            is_file: If True, create parent directory; if False, create directory itself

        Returns:
            Absolute path (guaranteed to exist)
        """
        path = self.resolve(key)

        if is_file:
            # Create parent directory for files
            path.parent.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Ensured parent directory exists: {path.parent}")
        else:
            # Create directory itself
            path.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Ensured directory exists: {path}")

        return path

    def get_all_paths(self) -> dict[str, Path]:
        """Get all configured paths as absolute paths.

        Returns:
            Dictionary mapping path keys to absolute paths
        """
        return {
            key: self.resolve(key) for key in PathsConfig.model_fields.keys()
        }
