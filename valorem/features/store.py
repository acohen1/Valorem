from __future__ import annotations
"""
SQLite feature-store helpers for Valorem.

Key responsibilities
--------------------
1. Create (if absent) and configure the DB (WAL mode).
2. **Upsert** pandas DataFrames via “INSERT OR IGNORE” (or `REPLACE`)
   keyed on the timestamp index so duplicates never crash the pipeline.
3. Provide fast helpers to check whether a table already holds data for a
   given calendar day (`table_has_date`) and to load slices back into pandas.

Dependencies
------------
pip install sqlalchemy pandas
"""
from contextlib import contextmanager
from pathlib import Path
from datetime import datetime
from typing import Any, Iterator

import json
import numbers
import os
import sqlite3
import logging

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, event, inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import OperationalError

__all__ = [
    "get_engine",
    "create_table_if_absent",
    "upsert",
    "load",
    "table_has_date",
]

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Connection helpers
# ---------------------------------------------------------------------------
_repo_root = Path(__file__).resolve().parents[2]
_DB_PATH = Path(os.getenv("VALOREM_DB", _repo_root / "valorem.db")).expanduser().resolve()


def get_engine(**kwargs: Any) -> Engine:
    """Return a SQLAlchemy Engine bound to Valorem's SQLite DB."""
    engine = create_engine(
        f"sqlite:///{_DB_PATH}",
        echo=kwargs.pop("echo", False),
        future=True,
        **kwargs,
    )

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
    with engine.begin() as conn:  # SQLAlchemy Connection
        yield conn.connection


# ---------------------------------------------------------------------------
# Table creation
# ---------------------------------------------------------------------------
def create_table_if_absent(
    table: str,
    df: pd.DataFrame,
    engine: Engine | None = None,
) -> None:
    """
    Create *table* (if missing) with columns derived from `df`.

    Columns are declared REAL; you can ALTER later if you need TEXT/BLOB.
    """
    engine = engine or get_engine()
    insp = inspect(engine)
    if insp.has_table(table):
        # If table exists, check for missing columns and migrate schema if needed
        existing = {col["name"] for col in insp.get_columns(table)}
        missing  = [c for c in df.columns if c not in existing]
        if missing:
            with engine.begin() as conn:
                for col in missing:
                    conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {col} REAL;"))
                    logger.info("Added column %s to %s", col, table)
        return                                   # now table is up-to-date

    cols_sql = ["date TEXT PRIMARY KEY"] + [f"{c} REAL" for c in df.columns]
    ddl = f"CREATE TABLE {table} ({', '.join(cols_sql)});"

    logger.info("Creating table %s", table)
    with engine.begin() as conn:
        conn.execute(text(ddl))


# ---------------------------------------------------------------------------
# Scalar-conversion helper
# ---------------------------------------------------------------------------
def _scalarize(val: Any) -> Any:
    """Convert complex Python objects into SQLite-safe scalars."""
    if val is None or isinstance(val, (str, bytes, numbers.Number, bool)):
        return val
    if isinstance(val, (list, dict, np.ndarray)):
        return json.dumps(val, separators=(",", ":"))
    return str(val)


# ---------------------------------------------------------------------------
# Upsert helper
# ---------------------------------------------------------------------------
def upsert(
    df: pd.DataFrame,
    table: str,
    *,
    replace: bool = False,
    engine: Engine | None = None,
    chunk: int = 5_000,
) -> None:
    """
    Insert `df` into `table`.

    Parameters
    ----------
    replace : bool, default False
        • False → `INSERT OR IGNORE` (keeps existing rows, ideal for immutable ticks).  
        • True  → `INSERT OR REPLACE` (overwrite on duplicate timestamp, useful for
          revised macro series).
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DatetimeIndex for upsert().")

    engine = engine or get_engine()
    create_table_if_absent(table, df, engine)

    sanitized = df.copy()
    for col in sanitized.columns:
        sanitized[col] = sanitized[col].map(_scalarize)

    records = [
        (ts.strftime("%Y-%m-%d %H:%M:%S.%f"), *row)
        for ts, row in zip(sanitized.index.to_pydatetime(), sanitized.to_numpy())
    ]

    verb = "REPLACE" if replace else "IGNORE"
    placeholders = ",".join("?" for _ in range(1 + len(df.columns)))
    columns_sql = ",".join(["date"] + list(df.columns))
    sql = f"INSERT OR {verb} INTO {table} ({columns_sql}) VALUES ({placeholders});"

    with _txn(engine) as conn:
        cur = conn.cursor()
        for i in range(0, len(records), chunk):
            cur.executemany(sql, records[i : i + chunk])
        conn.commit()

    logger.info("Upserted %d rows into %s (replace=%s)", len(df), table, replace)


# ---------------------------------------------------------------------------
# Fast “do we have this date?” helper
# ---------------------------------------------------------------------------
def table_has_date(
    table: str,
    date: str | datetime,
    *,
    engine: Engine | None = None,
) -> bool:
    """
    Return True if *table* already contains at least one row for the given
    UTC calendar day (YYYY-MM-DD).  If the table does not yet exist, False.
    """
    engine = engine or get_engine()
    day = pd.to_datetime(date).strftime("%Y-%m-%d")

    sql = text(f"SELECT 1 FROM {table} WHERE date LIKE :day LIMIT 1")
    try:
        with engine.connect() as conn:
            return conn.execute(sql, {"day": f"{day}%"}).first() is not None
    except OperationalError:
        # table doesn't exist yet
        return False


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
    """Return a DataFrame slice between *start* and *end* (inclusive)."""
    engine = engine or get_engine()

    query = f"SELECT * FROM {table}"
    params: list[str] = []
    if start is not None:
        params.append(pd.to_datetime(start).strftime("%Y-%m-%d"))
        query += " WHERE date >= ?"
    if end is not None:
        params.append(pd.to_datetime(end).strftime("%Y-%m-%d"))
        query += (" AND" if "WHERE" in query else " WHERE") + " date <= ?"
    query += " ORDER BY date;"

    return pd.read_sql_query(
        query,
        engine,
        params=tuple(params) if params else None,
        parse_dates=["date"],
        index_col="date",
    )
