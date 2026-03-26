"""Database connection management.

This module provides utilities for creating and managing database connections.
"""

import logging
import sqlite3
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional

from sqlalchemy import Engine, create_engine as sa_create_engine, event
from sqlalchemy.pool import StaticPool

from src.data.storage.schema import Base, metadata

logger = logging.getLogger(__name__)

# Python 3.12 fix: Register explicit datetime adapters to replace deprecated defaults
# This prevents deprecation warnings while maintaining proper datetime handling
def _adapt_date_iso(val):
    """Adapt datetime.date to ISO 8601 date string."""
    return val.isoformat()


def _adapt_datetime_iso(val):
    """Adapt datetime.datetime to SQLite-compatible datetime string.

    Normalizes to UTC before writing so that all stored timestamps are
    in a consistent timezone. Uses space separator and fixed microsecond
    precision to match the format that pandas to_sql() produces,
    ensuring string comparisons in WHERE clauses work correctly.
    """
    if val.tzinfo is not None:
        val = val.astimezone(timezone.utc).replace(tzinfo=None)
    return val.strftime("%Y-%m-%d %H:%M:%S.%f")


def _convert_date(val):
    """Convert ISO 8601 date string to datetime.date."""
    return date.fromisoformat(val.decode())


def _convert_timestamp(val):
    """Convert ISO 8601 datetime string to datetime.datetime.

    All stored timestamps are UTC (normalized on write by _adapt_datetime_iso).
    We return naive datetimes to avoid tz-aware/naive mismatches in pandas
    merge_asof and other operations — the convention is that all timestamps
    in the system are implicitly UTC.
    """
    dt = datetime.fromisoformat(val.decode())
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


# Register adapters (Python -> SQLite)
sqlite3.register_adapter(date, _adapt_date_iso)
sqlite3.register_adapter(datetime, _adapt_datetime_iso)

# Register converters (SQLite -> Python)
sqlite3.register_converter("date", _convert_date)
sqlite3.register_converter("timestamp", _convert_timestamp)


class DatabaseEngine:
    """Manages database connection and lifecycle."""

    def __init__(self, engine: Engine):
        """Initialize database engine wrapper.

        Args:
            engine: SQLAlchemy engine instance
        """
        self._engine = engine
        logger.debug(f"DatabaseEngine initialized with {engine.url}")

    @property
    def engine(self) -> Engine:
        """Get underlying SQLAlchemy engine.

        Returns:
            SQLAlchemy Engine instance
        """
        return self._engine

    def create_tables(self) -> None:
        """Create all tables defined in schema.

        This creates tables that don't already exist. Existing tables are not modified.
        """
        logger.debug("Creating database tables")
        Base.metadata.create_all(self._engine)
        logger.debug(f"Created {len(Base.metadata.tables)} tables")

    def drop_tables(self) -> None:
        """Drop all tables defined in schema.

        Warning: This will delete all data!
        """
        logger.warning("Dropping all database tables")
        Base.metadata.drop_all(self._engine)
        logger.info("Dropped all tables")

    def table_exists(self, table_name: str) -> bool:
        """Check if a table exists in the database.

        Args:
            table_name: Name of table to check

        Returns:
            True if table exists, False otherwise
        """
        from sqlalchemy import inspect

        inspector = inspect(self._engine)
        return table_name in inspector.get_table_names()

    def get_table_names(self) -> list[str]:
        """Get list of all tables in the database.

        Returns:
            List of table names
        """
        from sqlalchemy import inspect

        inspector = inspect(self._engine)
        return inspector.get_table_names()


def create_engine(
    db_path: Optional[Path] = None,
    echo: bool = False,
    in_memory: bool = False,
) -> DatabaseEngine:
    """Create database engine with appropriate configuration.

    Args:
        db_path: Path to SQLite database file (required unless in_memory=True)
        echo: If True, log all SQL statements (useful for debugging)
        in_memory: If True, create in-memory database (overrides db_path)

    Returns:
        DatabaseEngine instance

    Raises:
        ValueError: If db_path is None and in_memory is False
    """
    if in_memory:
        # In-memory database (useful for testing)
        logger.info("Creating in-memory database")
        url = "sqlite:///:memory:"
        engine = sa_create_engine(
            url,
            echo=echo,
            connect_args={
                "check_same_thread": False,
            },
            poolclass=StaticPool,  # Required for in-memory with multiple threads
        )
    else:
        if db_path is None:
            raise ValueError("db_path is required when in_memory=False")

        # File-based database
        db_path = Path(db_path).resolve()
        db_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Creating database engine for {db_path}")
        url = f"sqlite:///{db_path}"
        engine = sa_create_engine(
            url,
            echo=echo,
            connect_args={
                "check_same_thread": False,
            },
        )

    # Enable foreign keys for SQLite
    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_conn, connection_record):
        cursor = dbapi_conn.cursor()
        try:
            cursor.execute("PRAGMA foreign_keys=ON")
        finally:
            cursor.close()

    return DatabaseEngine(engine)
