"""Shared fixtures for integration/data/storage tests.

Uses the session-scoped ORM engine from root conftest.
Integration tests provide DataFrames with all required columns.
"""

import pytest
from sqlalchemy import text

from src.data.storage.schema import Base


@pytest.fixture(scope="module")
def db_engine(db_engine_session):
    """Module-scoped raw SA engine — ORM schema already exists."""
    return db_engine_session


@pytest.fixture(autouse=True)
def clean_tables(db_engine):
    """Delete all rows between tests (faster than recreating schema)."""
    yield
    with db_engine.connect() as conn:
        for table in reversed(Base.metadata.sorted_tables):
            conn.execute(text(f"DELETE FROM {table.name}"))
        conn.commit()
