from __future__ import annotations

"""SQLite feature-store helpers.

Thin wrapper for a local SQLite backing store that powers Valorem's feature
pipelines.

Key responsibilities
-------------------
1. Create (if absent) and configure the DB (WAL mode).
2. **Upsert** pandas DataFrames efficiently via *INSERT OR REPLACE* keyed on the
   date index.
3. Provide a convenience *load* helper that returns a DataFrame slice for model
   consumption.

Dependencies: `pip install sqlalchemy pandas`.
"""

from contextlib import contextmanager
from pathlib import Path
import os
import sqlite3
import logging
from datetime import datetime
from typing import Any, Iterator

import json
import numbers
import numpy as np

import pandas as pd
from sqlalchemy import create_engine, event, inspect, text
from sqlalchemy.engine import Engine

__all__ = [
    "get_engine",
    "create_table_if_absent",
    "upsert",
    "load",
]

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Connection helpers
# ---------------------------------------------------------------------------
_repo_root = Path(__file__).resolve().parents[2]
_DB_PATH = Path(os.getenv("VALOREM_DB", _repo_root / "valorem.db")).expanduser().resolve()


def get_engine(**kwargs: Any) -> Engine:
    """Return a SQLAlchemy Engine bound to Valorem’s SQLite DB."""
    uri = f"sqlite:///{_DB_PATH}"
    engine = create_engine(uri, echo=kwargs.pop("echo", False), future=True, **kwargs)

    # Switch SQLite to WAL & NORMAL sync for concurrent reads with minimal perf hit
    @event.listens_for(engine, "connect")
    def _set_sqlite_pragma(dbapi_conn, _):  # noqa: D401
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA journal_mode=WAL;")
        cursor.execute("PRAGMA synchronous=NORMAL;")
        cursor.close()

    return engine


@contextmanager
def _txn(engine: Engine) -> Iterator[sqlite3.Connection]:
    """Yield a raw sqlite3 connection inside a SQLAlchemy transaction."""
    with engine.begin() as conn:  # SQLAlchemy Connection (tx started)
        yield conn.connection


# ---------------------------------------------------------------------------
# Table creation
# ---------------------------------------------------------------------------

def create_table_if_absent(table: str, df: pd.DataFrame, engine: Engine | None = None) -> None:
    """Create *table* (if missing) with columns derived from *df*."""
    if engine is None:
        engine = get_engine()

    insp = inspect(engine)
    if insp.has_table(table):
        return  # already present

    cols_sql = ["date TEXT PRIMARY KEY"] + [f"{col} REAL" for col in df.columns]
    ddl = f"CREATE TABLE IF NOT EXISTS {table} ({', '.join(cols_sql)});"

    logger.info("Creating table %s", table)
    with engine.begin() as conn:
        conn.execute(text(ddl))


# ---------------------------------------------------------------------------
# Upsert helper
# ---------------------------------------------------------------------------
def _scalarize(val: Any) -> Any:
    """Convert complex Python objects into SQLite-safe scalars."""
    if val is None or isinstance(val, (str, bytes, numbers.Number, bool)):
        return val
    # lists / dicts / numpy arrays → JSON text
    if isinstance(val, (list, dict, np.ndarray)):
        return json.dumps(val, separators=(",", ":"))
    # fallback: string‐ify
    return str(val)

def upsert(
    df: pd.DataFrame,
    table: str,
    *,
    engine: Engine | None = None,
    chunk: int = 5000,
) -> None:
    """Insert or replace *df* into *table* keyed on the datetime index."""
    if engine is None:
        engine = get_engine()

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DatetimeIndex for upsert().")

    # sanitize → all scalar / JSON-text
    sanitized = df.copy()
    for col in sanitized.columns:
        sanitized[col] = sanitized[col].map(_scalarize)

    create_table_if_absent(table, sanitized, engine)

    records = [
        (dt.strftime("%Y-%m-%d %H:%M:%S.%f"), *row)
        for dt, row in zip(sanitized.index.to_pydatetime(), sanitized.to_numpy())
    ]

    placeholders = ",".join(["?"] * (1 + len(df.columns)))
    columns_sql  = ",".join(["date"] + list(df.columns))
    sql = f"INSERT OR REPLACE INTO {table} ({columns_sql}) VALUES ({placeholders});"

    with _txn(engine) as conn:
        cursor = conn.cursor()
        for i in range(0, len(records), chunk):
            cursor.executemany(sql, records[i : i + chunk])
        conn.commit()

    logger.info("Upserted %d rows into %s", len(df), table)


# ---------------------------------------------------------------------------
# Load helper
# ---------------------------------------------------------------------------

def load(
    table: str,
    *,
    start: datetime | str | None = None,
    end: datetime | str | None = None,
    engine: Engine | None = None,
) -> pd.DataFrame:
    """Fetch a DataFrame slice from *table* between *start* and *end*."""
    if engine is None:
        engine = get_engine()

    query = f"SELECT * FROM {table}"
    clauses: list[str] = []
    params: list[str] = []

    if start is not None:
        clauses.append("date >= ?")
        params.append(pd.to_datetime(start).strftime("%Y-%m-%d"))
    if end is not None:
        clauses.append("date <= ?")
        params.append(pd.to_datetime(end).strftime("%Y-%m-%d"))

    if clauses:
        query += " WHERE " + " AND ".join(clauses)
    query += " ORDER BY date;"

    # ---
    # pandas→SQLAlchemy API quirk: params must be a *sequence representing a single
    # positional-parameter set* (tuple) or a *mapping*—**NOT** a list of scalars.
    # Convert to tuple for single execution.
    # ---
    params_tuple = tuple(params) if params else None

    df = pd.read_sql_query(
        query,
        engine,
        params=params_tuple,
        parse_dates=["date"],
        index_col="date",
    )
    return df
