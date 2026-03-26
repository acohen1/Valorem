"""Shared fixtures for surface integration tests.

Provides:
- Module-scoped DB engine (reuses session-scoped ORM engine)
- Autouse DELETE-based cleanup between tests
- raw_repo / derived_repo fixtures

The module-scoped engine avoids per-test DDL overhead. Schema is
created once at session start via the root conftest; individual
tests get automatic DELETE-based cleanup.
"""

import pytest
from sqlalchemy import text

from src.data.storage.repository import DerivedRepository, RawRepository
from src.data.storage.schema import Base


@pytest.fixture(scope="module")
def db_engine(db_engine_session):
    """Module-scoped engine — schema already exists from session fixture."""
    return db_engine_session


@pytest.fixture(autouse=True)
def clean_tables(db_engine):
    """Delete all rows between tests (faster than recreating schema)."""
    yield
    with db_engine.connect() as conn:
        for table in reversed(Base.metadata.sorted_tables):
            conn.execute(text(f"DELETE FROM {table.name}"))
        conn.commit()


@pytest.fixture
def raw_repo(db_engine):
    """Create RawRepository backed by the shared engine."""
    return RawRepository(db_engine)


@pytest.fixture
def derived_repo(db_engine):
    """Create DerivedRepository backed by the shared engine."""
    return DerivedRepository(db_engine)
