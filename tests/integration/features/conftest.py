"""Shared fixtures for features integration tests.

Provides:
- Module-scoped DatabaseEngine wrapper (reuses session-scoped ORM engine)
- Autouse DELETE-based cleanup between tests
- raw_repo / derived_repo fixtures

Note: db_engine returns a DatabaseEngine wrapper (not a raw SA engine)
because FeatureEngine.__init__ expects a DatabaseEngine. Repos use
db_engine.engine for the underlying SA engine.
"""

import pytest
from sqlalchemy import text

from src.data.storage.engine import DatabaseEngine
from src.data.storage.repository import DerivedRepository, RawRepository
from src.data.storage.schema import Base


@pytest.fixture(scope="module")
def db_engine(db_engine_session):
    """Module-scoped DatabaseEngine wrapper — schema already exists.

    Returns a DatabaseEngine wrapper because FeatureEngine expects it.
    """
    return DatabaseEngine(db_engine_session)


@pytest.fixture(autouse=True)
def clean_tables(db_engine):
    """Delete all rows between tests (faster than recreating schema)."""
    yield
    with db_engine.engine.connect() as conn:
        for table in reversed(Base.metadata.sorted_tables):
            conn.execute(text(f"DELETE FROM {table.name}"))
        conn.commit()


@pytest.fixture
def raw_repo(db_engine):
    """Create RawRepository backed by the shared engine."""
    return RawRepository(db_engine.engine)


@pytest.fixture
def derived_repo(db_engine):
    """Create DerivedRepository backed by the shared engine."""
    return DerivedRepository(db_engine.engine)
