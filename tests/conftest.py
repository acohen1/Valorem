"""Root test configuration — session-scoped fixtures shared across all tests.

Provides a single in-memory SQLite engine for the entire test session.
Schema (6 tables, indexes, constraints) is created once at session start,
eliminating ~130 per-test DDL operations.

Individual test modules use module-scoped fixtures that reference this
engine, with DELETE-based cleanup between tests.
"""

import pytest
from sqlalchemy import create_engine as sa_create_engine, event
from sqlalchemy.pool import StaticPool

import src.data.storage.engine  # noqa: F401 — registers sqlite3 datetime adapters
from src.data.storage.schema import Base


@pytest.fixture(scope="session")
def db_engine_session():
    """Single in-memory SQLite engine for the entire test session.

    Schema is created once. Individual tests use module-scoped
    connections with DELETE-based cleanup.
    """
    engine = sa_create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_conn, connection_record):
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(engine)
    yield engine
    engine.dispose()
