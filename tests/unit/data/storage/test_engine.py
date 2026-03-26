"""Unit tests for database engine."""

from pathlib import Path

import pytest

from src.data.storage.engine import DatabaseEngine, create_engine


class TestCreateEngine:
    """Test database engine creation."""

    def test_create_in_memory_engine(self):
        """Test creating in-memory database engine."""
        db_engine = create_engine(in_memory=True)

        assert db_engine is not None
        assert isinstance(db_engine, DatabaseEngine)
        assert str(db_engine.engine.url) == "sqlite:///:memory:"

    def test_create_file_based_engine(self, tmp_path):
        """Test creating file-based database engine."""
        db_path = tmp_path / "test.db"
        db_engine = create_engine(db_path=db_path)

        assert db_engine is not None
        assert isinstance(db_engine, DatabaseEngine)
        assert "test.db" in str(db_engine.engine.url)

    def test_create_engine_without_path_raises_error(self):
        """Test that creating engine without path raises ValueError."""
        with pytest.raises(ValueError, match="db_path is required"):
            create_engine(in_memory=False)

    def test_create_engine_creates_parent_directory(self, tmp_path):
        """Test that engine creation creates parent directory."""
        db_path = tmp_path / "subdir" / "test.db"
        db_engine = create_engine(db_path=db_path)

        assert db_path.parent.exists()


class TestDatabaseEngine:
    """Test DatabaseEngine class."""

    @pytest.fixture
    def db_engine(self):
        """Create in-memory database engine for testing."""
        return create_engine(in_memory=True)

    def test_create_tables(self, db_engine):
        """Test creating database tables."""
        db_engine.create_tables()

        # Check that tables were created
        table_names = db_engine.get_table_names()
        assert "raw_underlying_bars" in table_names
        assert "raw_option_quotes" in table_names
        assert "raw_fred_series" in table_names
        assert "surface_snapshots" in table_names
        assert "node_panel" in table_names

    def test_table_exists(self, db_engine):
        """Test checking if table exists."""
        # Before creating tables
        assert not db_engine.table_exists("raw_underlying_bars")

        # After creating tables
        db_engine.create_tables()
        assert db_engine.table_exists("raw_underlying_bars")
        assert db_engine.table_exists("surface_snapshots")

        # Non-existent table
        assert not db_engine.table_exists("nonexistent_table")

    def test_get_table_names(self, db_engine):
        """Test getting list of table names."""
        # Before creating tables
        assert len(db_engine.get_table_names()) == 0

        # After creating tables
        db_engine.create_tables()
        table_names = db_engine.get_table_names()

        assert len(table_names) == 8
        assert "raw_underlying_bars" in table_names
        assert "raw_option_quotes" in table_names
        assert "raw_option_statistics" in table_names

    def test_drop_tables(self, db_engine):
        """Test dropping database tables."""
        # Create tables
        db_engine.create_tables()
        assert len(db_engine.get_table_names()) > 0

        # Drop tables
        db_engine.drop_tables()
        assert len(db_engine.get_table_names()) == 0

    def test_create_tables_is_idempotent(self, db_engine):
        """Test that calling create_tables multiple times is safe."""
        db_engine.create_tables()
        initial_tables = set(db_engine.get_table_names())

        # Call again
        db_engine.create_tables()
        final_tables = set(db_engine.get_table_names())

        # Should have same tables
        assert initial_tables == final_tables
