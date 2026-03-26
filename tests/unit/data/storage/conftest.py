"""Shared fixtures for unit/data/storage tests.

Provides:
- Module-scoped DB engine with simplified schema for repository unit tests
  (test_repository.py uses DataFrames without provider-metadata columns like
  dataset/schema/stype_in, so we use a schema that matches those DataFrames)
- Autouse cleanup between tests (DELETE FROM all tables)
- Shared raw_repo / derived_repo fixtures

The module-scoped engine creates DDL once per test module instead of once
per test (~70x reduction for test_repository.py).

Note: test_engine.py tests the DatabaseEngine wrapper itself and keeps its
own fixtures — it does not use these shared fixtures.
"""

import pytest
from sqlalchemy import create_engine as sa_create_engine, event, text
from sqlalchemy.pool import StaticPool

from src.data.storage.repository import DerivedRepository, RawRepository

# Table names for cleanup — must match the DDL below
_TABLE_NAMES = [
    "node_panel",
    "surface_snapshots",
    "raw_ingestion_log",
    "raw_fred_series",
    "raw_option_quotes",
    "raw_underlying_bars",
]

# Simplified DDL matching the DataFrames used in repository unit tests.
# Integration tests use the full ORM schema via the session-scoped engine.
_DDL = """
CREATE TABLE raw_underlying_bars (
    ts_utc TIMESTAMP NOT NULL,
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    open REAL NOT NULL,
    high REAL NOT NULL,
    low REAL NOT NULL,
    close REAL NOT NULL,
    volume INTEGER NOT NULL,
    ingest_run_id TEXT NOT NULL,
    source_ingested_at TIMESTAMP NOT NULL,
    UNIQUE(symbol, ts_utc, timeframe)
);

CREATE TABLE raw_option_quotes (
    ts_utc TIMESTAMP NOT NULL,
    option_symbol TEXT NOT NULL,
    exp_date DATE NOT NULL,
    strike REAL NOT NULL,
    right TEXT NOT NULL,
    bid REAL NOT NULL,
    ask REAL NOT NULL,
    ingest_run_id TEXT NOT NULL,
    source_ingested_at TIMESTAMP NOT NULL,
    UNIQUE(option_symbol, ts_utc)
);

CREATE TABLE raw_fred_series (
    series_id TEXT NOT NULL,
    obs_date DATE NOT NULL,
    value REAL NOT NULL,
    release_datetime_utc TIMESTAMP NOT NULL,
    source_ingested_at TIMESTAMP NOT NULL,
    UNIQUE(series_id, obs_date, release_datetime_utc)
);

CREATE TABLE raw_ingestion_log (
    ingest_run_id TEXT PRIMARY KEY,
    dataset TEXT NOT NULL,
    schema TEXT NOT NULL,
    stype_in TEXT NOT NULL,
    symbols TEXT NOT NULL,
    start_date DATE NOT NULL,
    end_date DATE NOT NULL,
    row_count INTEGER NOT NULL,
    source_ingested_at TIMESTAMP NOT NULL
);

CREATE TABLE surface_snapshots (
    ts_utc TIMESTAMP NOT NULL,
    option_symbol TEXT NOT NULL,
    delta_bucket TEXT NOT NULL,
    implied_vol REAL NOT NULL,
    build_run_id TEXT NOT NULL,
    snapshot_version TEXT NOT NULL,
    source_created_at TIMESTAMP NOT NULL
);

CREATE TABLE node_panel (
    ts_utc TIMESTAMP NOT NULL,
    tenor_days INTEGER,
    delta_bucket TEXT NOT NULL,
    option_symbol TEXT,
    iv_mid REAL,
    iv_bid REAL,
    iv_ask REAL,
    spread_pct REAL,
    delta REAL,
    gamma REAL,
    vega REAL,
    theta REAL,
    iv_change_1d REAL,
    iv_change_5d REAL,
    skew_slope REAL,
    term_slope REAL,
    curvature REAL,
    oi_change_5d REAL,
    volume_ratio REAL,
    underlying_rv_5d REAL,
    underlying_rv_10d REAL,
    underlying_rv_21d REAL,
    sofr_level REAL,
    sofr_change_1w REAL,
    feature_version TEXT NOT NULL,
    is_masked INTEGER NOT NULL,
    mask_reason TEXT
);
"""


@pytest.fixture(scope="module")
def db_engine():
    """Module-scoped in-memory SQLite engine with simplified schema.

    Schema is created once per module. Individual tests get automatic
    DELETE-based cleanup via the clean_tables autouse fixture.
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

    # Execute DDL via raw DBAPI connection (supports multiple statements)
    raw_conn = engine.raw_connection()
    raw_conn.executescript(_DDL)
    raw_conn.close()

    yield engine
    engine.dispose()


@pytest.fixture(autouse=True)
def clean_tables(db_engine):
    """Delete all rows between tests (faster than recreating schema).

    Skips cleanup if db_engine is not a raw SQLAlchemy engine
    (e.g., test_engine.py provides a DatabaseEngine wrapper).
    """
    yield
    if not hasattr(db_engine, "connect"):
        return
    with db_engine.connect() as conn:
        for table_name in _TABLE_NAMES:
            conn.execute(text(f"DELETE FROM {table_name}"))
        conn.commit()


@pytest.fixture
def raw_repo(db_engine):
    """Create RawRepository backed by the shared engine."""
    return RawRepository(db_engine)


@pytest.fixture
def derived_repo(db_engine):
    """Create DerivedRepository backed by the shared engine."""
    return DerivedRepository(db_engine)
