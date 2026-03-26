"""Repository pattern for database access.

This module implements the repository pattern for raw and derived data access.
All database operations go through these repositories for proper abstraction.
"""

from datetime import UTC, datetime

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.engine import Engine
from sqlalchemy.exc import OperationalError

from src.exceptions import DataReadError, DataWriteError


def _sqlite_upsert(table, conn, keys, data_iter):
    """Insert rows, skipping any that violate unique constraints.

    Used for raw (immutable) tables where existing data should never be overwritten.
    """
    data = [dict(zip(keys, row)) for row in data_iter]
    if not data:
        return
    stmt = sqlite_insert(table.table).values(data).on_conflict_do_nothing()
    conn.execute(stmt)


def _sqlite_upsert_replace(table, conn, keys, data_iter):
    """Insert rows, replacing all non-key columns on conflict.

    Used for derived (recomputable) tables where fresh data should overwrite stale values.
    """
    data = [dict(zip(keys, row)) for row in data_iter]
    if not data:
        return
    stmt = sqlite_insert(table.table).values(data)
    # Update all columns provided in the data (except auto-increment id)
    update_cols = {k: stmt.excluded[k] for k in keys if k != "id"}
    stmt = stmt.on_conflict_do_update(set_=update_cols)
    conn.execute(stmt)


class RawRepository:
    """Data access layer for raw tables (immutable, append-only)."""

    def __init__(self, engine: Engine):
        """Initialize repository with database engine.

        Args:
            engine: SQLAlchemy engine for database connection
        """
        self._engine = engine

    def write_underlying_bars(self, df: pd.DataFrame, run_id: str) -> None:
        """Write underlying bars to raw table.

        Args:
            df: DataFrame with columns [ts_utc, symbol, timeframe, open, high, low, close, volume, ...]
            run_id: Unique ingestion run identifier

        Raises:
            DataWriteError: If database write fails
        """
        if df.empty:
            return

        # Add ingestion metadata
        df = df.copy()
        df["ingest_run_id"] = run_id
        df["source_ingested_at"] = datetime.now(UTC)

        # Write to database with transaction (auto-rollback on exception)
        # Uses upsert to skip duplicates on re-runs
        try:
            with self._engine.begin() as conn:
                df.to_sql(
                    "raw_underlying_bars",
                    conn,
                    if_exists="append",
                    index=False,
                    chunksize=500,
                    method=_sqlite_upsert,
                )
        except Exception as e:
            raise DataWriteError(f"Failed to write underlying bars: {e}") from e

    def read_underlying_bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: str = "1m",
    ) -> pd.DataFrame:
        """Read underlying bars for date range.

        Args:
            symbol: Underlying symbol (e.g., "SPY")
            start: Start datetime (inclusive)
            end: End datetime (exclusive)
            timeframe: Bar timeframe filter (default: "1m")

        Returns:
            DataFrame with columns from raw_underlying_bars table

        Raises:
            DataReadError: If database read fails
        """
        query = text("""
            SELECT * FROM raw_underlying_bars
            WHERE symbol = :symbol
              AND timeframe = :timeframe
              AND ts_utc >= :start
              AND ts_utc < :end
            ORDER BY ts_utc
        """)

        try:
            return pd.read_sql(
                query,
                self._engine,
                params={
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "start": start,
                    "end": end,
                },
            )
        except Exception as e:
            raise DataReadError(f"Failed to read underlying bars: {e}") from e

    def write_option_quotes(self, df: pd.DataFrame, run_id: str) -> None:
        """Write option quotes to raw table.

        Args:
            df: DataFrame with columns [ts_utc, option_symbol, exp_date, strike, right, bid, ask, ...]
            run_id: Unique ingestion run identifier

        Raises:
            DataWriteError: If database write fails
        """
        if df.empty:
            return

        # Add ingestion metadata
        df = df.copy()
        df["ingest_run_id"] = run_id
        df["source_ingested_at"] = datetime.now(UTC)

        # Write to database with transaction (auto-rollback on exception)
        # Uses upsert to skip duplicates on re-runs
        try:
            with self._engine.begin() as conn:
                df.to_sql(
                    "raw_option_quotes",
                    conn,
                    if_exists="append",
                    index=False,
                    chunksize=500,
                    method=_sqlite_upsert,
                )
        except Exception as e:
            raise DataWriteError(f"Failed to write option quotes: {e}") from e

    def read_option_quotes(
        self,
        start: datetime,
        end: datetime,
        option_symbols: list[str] | None = None,
    ) -> pd.DataFrame:
        """Read option quotes for date range.

        Args:
            start: Start datetime (inclusive)
            end: End datetime (exclusive)
            option_symbols: Optional list of option symbols to filter

        Returns:
            DataFrame with columns from raw_option_quotes table

        Raises:
            DataReadError: If database read fails
        """
        query = """
            SELECT * FROM raw_option_quotes
            WHERE ts_utc >= :start
              AND ts_utc < :end
        """

        params = {"start": start, "end": end}

        if option_symbols:
            placeholders = ",".join([f":sym{i}" for i in range(len(option_symbols))])
            query += f" AND option_symbol IN ({placeholders})"
            for i, symbol in enumerate(option_symbols):
                params[f"sym{i}"] = symbol

        query += " ORDER BY ts_utc, option_symbol"

        try:
            return pd.read_sql(text(query), self._engine, params=params)
        except Exception as e:
            raise DataReadError(f"Failed to read option quotes: {e}") from e

    def write_fred_series(self, df: pd.DataFrame) -> None:
        """Write FRED series data to raw table.

        Args:
            df: DataFrame with columns [series_id, obs_date, value, release_datetime_utc]

        Raises:
            DataWriteError: If database write fails
        """
        if df.empty:
            return

        # Add ingestion metadata
        df = df.copy()
        df["source_ingested_at"] = datetime.now(UTC)

        # Write to database with transaction (auto-rollback on exception)
        try:
            with self._engine.begin() as conn:
                df.to_sql(
                    "raw_fred_series",
                    conn,
                    if_exists="append",
                    index=False,
                    chunksize=500,
                    method=_sqlite_upsert,
                )
        except Exception as e:
            raise DataWriteError(f"Failed to write FRED series: {e}") from e

    def read_fred_series(
        self,
        series_id: str,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        """Read FRED series for date range.

        Args:
            series_id: FRED series identifier (e.g., "DGS10")
            start: Start datetime (inclusive)
            end: End datetime (exclusive)

        Returns:
            DataFrame with columns from raw_fred_series table

        Raises:
            DataReadError: If database read fails
        """
        query = text("""
            SELECT * FROM raw_fred_series
            WHERE series_id = :series_id
              AND obs_date >= :start
              AND obs_date < :end
            ORDER BY obs_date
        """)

        try:
            return pd.read_sql(
                query,
                self._engine,
                params={"series_id": series_id, "start": start.date(), "end": end.date()},
            )
        except Exception as e:
            raise DataReadError(f"Failed to read FRED series: {e}") from e

    def count_underlying_bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: str = "1m",
    ) -> int:
        """Count existing underlying bar rows for a symbol/date range.

        Args:
            symbol: Underlying symbol (e.g., "SPY")
            start: Start datetime (inclusive)
            end: End datetime (exclusive)
            timeframe: Bar timeframe filter (default: "1m")

        Returns:
            Number of existing rows
        """
        query = text(
            "SELECT COUNT(*) FROM raw_underlying_bars "
            "WHERE symbol = :symbol AND timeframe = :timeframe "
            "AND ts_utc >= :start AND ts_utc < :end"
        )
        try:
            with self._engine.connect() as conn:
                result = conn.execute(
                    query,
                    {
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "start": start,
                        "end": end,
                    },
                )
                return result.scalar() or 0
        except OperationalError:
            return 0

    def count_option_quotes(
        self,
        start: datetime,
        end: datetime,
    ) -> int:
        """Count existing option quote rows for a date range.

        Args:
            start: Start datetime (inclusive)
            end: End datetime (exclusive)

        Returns:
            Number of existing rows
        """
        query = text(
            "SELECT COUNT(*) FROM raw_option_quotes "
            "WHERE ts_utc >= :start AND ts_utc < :end"
        )
        try:
            with self._engine.connect() as conn:
                result = conn.execute(query, {"start": start, "end": end})
                return result.scalar() or 0
        except OperationalError:
            return 0

    def write_option_bars(self, df: pd.DataFrame, run_id: str) -> None:
        """Write option OHLCV bars to raw table.

        Args:
            df: DataFrame with columns [ts_utc, option_symbol, timeframe, open, high, low, close, volume]
            run_id: Unique ingestion run identifier

        Raises:
            DataWriteError: If database write fails
        """
        if df.empty:
            return

        df = df.copy()
        df["ingest_run_id"] = run_id
        df["source_ingested_at"] = datetime.now(UTC)

        try:
            with self._engine.begin() as conn:
                df.to_sql(
                    "raw_option_bars",
                    conn,
                    if_exists="append",
                    index=False,
                    chunksize=500,
                    method=_sqlite_upsert,
                )
        except Exception as e:
            raise DataWriteError(f"Failed to write option bars: {e}") from e

    def read_option_bars(
        self,
        start: datetime,
        end: datetime,
        option_symbols: list[str] | None = None,
        timeframe: str = "1d",
    ) -> pd.DataFrame:
        """Read option OHLCV bars for date range.

        Args:
            start: Start datetime (inclusive)
            end: End datetime (exclusive)
            option_symbols: Optional list of option symbols to filter
            timeframe: Bar timeframe (default: "1d")

        Returns:
            DataFrame with columns from raw_option_bars table

        Raises:
            DataReadError: If database read fails
        """
        query = """
            SELECT * FROM raw_option_bars
            WHERE ts_utc >= :start
              AND ts_utc < :end
              AND timeframe = :timeframe
        """

        params: dict = {"start": start, "end": end, "timeframe": timeframe}

        if option_symbols:
            placeholders = ",".join([f":sym{i}" for i in range(len(option_symbols))])
            query += f" AND option_symbol IN ({placeholders})"
            for i, symbol in enumerate(option_symbols):
                params[f"sym{i}"] = symbol

        query += " ORDER BY ts_utc, option_symbol"

        try:
            return pd.read_sql(text(query), self._engine, params=params)
        except Exception as e:
            raise DataReadError(f"Failed to read option bars: {e}") from e

    def write_option_statistics(self, df: pd.DataFrame, run_id: str) -> None:
        """Write option statistics (e.g., daily OI) to raw table.

        Args:
            df: DataFrame with columns [ts_ref, ts_event, option_symbol, stat_type, quantity, price]
            run_id: Unique ingestion run identifier

        Raises:
            DataWriteError: If database write fails
        """
        if df.empty:
            return

        df = df.copy()
        df["ingest_run_id"] = run_id
        df["source_ingested_at"] = datetime.now(UTC)

        try:
            with self._engine.begin() as conn:
                df.to_sql(
                    "raw_option_statistics",
                    conn,
                    if_exists="append",
                    index=False,
                    chunksize=500,
                    method=_sqlite_upsert,
                )
        except Exception as e:
            raise DataWriteError(f"Failed to write option statistics: {e}") from e

    def read_option_statistics(
        self,
        start: datetime,
        end: datetime,
        stat_type: int = 9,
        option_symbols: list[str] | None = None,
    ) -> pd.DataFrame:
        """Read option statistics for date range.

        Args:
            start: Start datetime (inclusive)
            end: End datetime (exclusive)
            stat_type: Statistic type filter (default: 9 = open interest)
            option_symbols: Optional list of option symbols to filter

        Returns:
            DataFrame with columns from raw_option_statistics table

        Raises:
            DataReadError: If database read fails
        """
        query = """
            SELECT * FROM raw_option_statistics
            WHERE ts_ref >= :start
              AND ts_ref < :end
              AND stat_type = :stat_type
        """

        params: dict = {"start": start, "end": end, "stat_type": stat_type}

        if option_symbols:
            placeholders = ",".join([f":sym{i}" for i in range(len(option_symbols))])
            query += f" AND option_symbol IN ({placeholders})"
            for i, symbol in enumerate(option_symbols):
                params[f"sym{i}"] = symbol

        query += " ORDER BY ts_ref, option_symbol"

        try:
            return pd.read_sql(text(query), self._engine, params=params)
        except Exception as e:
            raise DataReadError(f"Failed to read option statistics: {e}") from e

    def count_option_statistics(
        self,
        start: datetime,
        end: datetime,
    ) -> int:
        """Count existing option statistics rows for a date range.

        Args:
            start: Start datetime (inclusive)
            end: End datetime (exclusive)

        Returns:
            Number of existing rows
        """
        query = text(
            "SELECT COUNT(*) FROM raw_option_statistics "
            "WHERE ts_ref >= :start AND ts_ref < :end"
        )
        try:
            with self._engine.connect() as conn:
                result = conn.execute(query, {"start": start, "end": end})
                return result.scalar() or 0
        except OperationalError:
            return 0

    def count_option_bars(
        self,
        start: datetime,
        end: datetime,
    ) -> int:
        """Count existing option bar rows for a date range.

        Args:
            start: Start datetime (inclusive)
            end: End datetime (exclusive)

        Returns:
            Number of existing rows
        """
        query = text(
            "SELECT COUNT(*) FROM raw_option_bars "
            "WHERE ts_utc >= :start AND ts_utc < :end"
        )
        try:
            with self._engine.connect() as conn:
                result = conn.execute(query, {"start": start, "end": end})
                return result.scalar() or 0
        except OperationalError:
            return 0

    def count_fred_series(
        self,
        series_id: str,
        start: datetime,
        end: datetime,
    ) -> int:
        """Count existing FRED rows for a series/date range.

        Args:
            series_id: FRED series identifier (e.g., "DGS10")
            start: Start datetime (inclusive)
            end: End datetime (exclusive)

        Returns:
            Number of existing rows
        """
        query = text(
            "SELECT COUNT(*) FROM raw_fred_series "
            "WHERE series_id = :series_id AND obs_date >= :start AND obs_date < :end"
        )
        try:
            with self._engine.connect() as conn:
                result = conn.execute(
                    query,
                    {"series_id": series_id, "start": start.date(), "end": end.date()},
                )
                return result.scalar() or 0
        except OperationalError:
            return 0

    def write_ingestion_log(self, log_entry: dict) -> None:
        """Write ingestion log entry.

        Args:
            log_entry: Dictionary with ingestion metadata
                Required keys: ingest_run_id, dataset, schema, stype_in, symbols,
                              start_date, end_date, row_count, source_ingested_at

        Raises:
            DataWriteError: If database write fails
        """
        df = pd.DataFrame([log_entry])

        # Write to database with transaction (auto-rollback on exception)
        try:
            with self._engine.begin() as conn:
                df.to_sql(
                    "raw_ingestion_log",
                    conn,
                    if_exists="append",
                    index=False,
                )
        except Exception as e:
            raise DataWriteError(f"Failed to write ingestion log: {e}") from e

    def read_ingestion_log(
        self,
        dataset: str | None = None,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> pd.DataFrame:
        """Read ingestion log entries.

        Args:
            dataset: Optional dataset filter (e.g., "GLBX.MDP3")
            start_date: Optional start date filter
            end_date: Optional end date filter

        Returns:
            DataFrame with columns from raw_ingestion_log table

        Raises:
            DataReadError: If database read fails
        """
        query = "SELECT * FROM raw_ingestion_log WHERE 1=1"
        params = {}

        if dataset:
            query += " AND dataset = :dataset"
            params["dataset"] = dataset

        if start_date:
            query += " AND end_date >= :start_date"
            params["start_date"] = start_date.date()

        if end_date:
            query += " AND start_date < :end_date"
            params["end_date"] = end_date.date()

        query += " ORDER BY source_ingested_at DESC"

        try:
            return pd.read_sql(text(query), self._engine, params=params)
        except Exception as e:
            raise DataReadError(f"Failed to read ingestion log: {e}") from e


class DerivedRepository:
    """Data access layer for derived tables (versioned, reproducible)."""

    def __init__(self, engine: Engine):
        """Initialize repository with database engine.

        Args:
            engine: SQLAlchemy engine for database connection
        """
        self._engine = engine

    def write_surface_snapshots(
        self,
        df: pd.DataFrame,
        build_run_id: str,
        version: str,
    ) -> None:
        """Write surface snapshots with version.

        Args:
            df: DataFrame with surface snapshot columns
            build_run_id: Unique build run identifier
            version: Surface version identifier (e.g., "v1.0")

        Raises:
            DataWriteError: If database write fails
        """
        if df.empty:
            return

        # Add build metadata
        df = df.copy()
        df["build_run_id"] = build_run_id
        df["snapshot_version"] = version
        df["source_created_at"] = datetime.now(UTC)

        # Sanitize inf → NaN before DB write (SQLite doesn't support inf)
        numeric_cols = df.select_dtypes(include="number").columns
        df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)

        # Write to database with transaction (auto-rollback on exception)
        # Uses upsert-replace so recomputed data overwrites stale values
        try:
            with self._engine.begin() as conn:
                df.to_sql(
                    "surface_snapshots",
                    conn,
                    if_exists="append",
                    index=False,
                    chunksize=500,
                    method=_sqlite_upsert_replace,
                )
        except Exception as e:
            raise DataWriteError(f"Failed to write surface snapshots: {e}") from e

    def read_surface_snapshots(
        self,
        start: datetime,
        end: datetime,
        version: str,
        delta_buckets: list[str] | None = None,
    ) -> pd.DataFrame:
        """Read surface snapshots for analysis.

        Args:
            start: Start datetime (inclusive)
            end: End datetime (exclusive)
            version: Surface version identifier
            delta_buckets: Optional list of delta buckets to filter (e.g., ["ATM", "P25"])

        Returns:
            DataFrame with columns from surface_snapshots table

        Raises:
            DataReadError: If database read fails
        """
        query = """
            SELECT * FROM surface_snapshots
            WHERE ts_utc >= :start
              AND ts_utc < :end
              AND snapshot_version = :version
        """

        params = {"start": start, "end": end, "version": version}

        if delta_buckets:
            placeholders = ",".join([f":bucket{i}" for i in range(len(delta_buckets))])
            query += f" AND delta_bucket IN ({placeholders})"
            for i, bucket in enumerate(delta_buckets):
                params[f"bucket{i}"] = bucket

        query += " ORDER BY ts_utc, option_symbol"

        try:
            return pd.read_sql(text(query), self._engine, params=params)
        except Exception as e:
            raise DataReadError(f"Failed to read surface snapshots: {e}") from e

    def write_node_panel(
        self,
        df: pd.DataFrame,
        feature_version: str,
    ) -> None:
        """Write node panel with features.

        Args:
            df: DataFrame with node panel columns
            feature_version: Feature version identifier (e.g., "v1.0")

        Raises:
            DataWriteError: If database write fails
        """
        if df.empty:
            return

        # Add version metadata
        df = df.copy()
        df["feature_version"] = feature_version

        # Set required columns with defaults if not present
        if "is_masked" not in df.columns:
            df["is_masked"] = False
        if "mask_reason" not in df.columns:
            df["mask_reason"] = None

        # Filter to columns that exist in the node_panel schema
        # This allows writing flexible feature DataFrames while only
        # persisting the columns defined in the schema
        valid_columns = [
            # Node identification
            "ts_utc", "tenor_days", "delta_bucket", "option_symbol",
            # Label-only (NOT a model feature — used for DHR label construction)
            "mid_price",
            # Core features (iv_mid/iv_bid/iv_ask excluded — live in surface_snapshots)
            "spread_pct", "delta", "gamma",
            "vega", "theta",
            # Derived features
            "iv_change_1d", "iv_change_5d",
            "iv_vol_5d", "iv_vol_10d", "iv_vol_21d",
            "iv_zscore_5d", "iv_zscore_10d", "iv_zscore_21d",
            "skew_slope", "term_slope",
            "curvature", "oi_change_5d", "volume_ratio_5d", "log_volume",
            "log_oi",
            # Global features (denormalized)
            "underlying_rv_5d", "underlying_rv_10d", "underlying_rv_21d",
            # Macro features (denormalized — must match FRED series IDs)
            "DGS10_level", "DGS10_change_1w",
            "DGS2_level", "DGS2_change_1w",
            "VIXCLS_level", "VIXCLS_change_1w",
            # Metadata
            "feature_version", "is_masked", "mask_reason",
        ]
        columns_to_write = [c for c in valid_columns if c in df.columns]
        df_filtered = df[columns_to_write].copy()

        # Sanitize inf → NaN before writing (SQLite stores inf inconsistently)
        numeric_cols = df_filtered.select_dtypes(include="number").columns
        df_filtered[numeric_cols] = df_filtered[numeric_cols].replace(
            [np.inf, -np.inf], np.nan
        )

        # Write to database with transaction (auto-rollback on exception)
        # Uses upsert-replace so recomputed data overwrites stale values
        try:
            with self._engine.begin() as conn:
                df_filtered.to_sql(
                    "node_panel",
                    conn,
                    if_exists="append",
                    index=False,
                    chunksize=500,
                    method=_sqlite_upsert_replace,
                )
        except Exception as e:
            raise DataWriteError(f"Failed to write node panel: {e}") from e

    def read_node_panel(
        self,
        feature_version: str,
        start: datetime | None = None,
        end: datetime | None = None,
        delta_buckets: list[str] | None = None,
    ) -> pd.DataFrame:
        """Read node panel for analysis.

        Args:
            feature_version: Feature version identifier
            start: Optional start datetime (inclusive)
            end: Optional end datetime (exclusive)
            delta_buckets: Optional list of delta buckets to filter

        Returns:
            DataFrame with columns from node_panel table

        Raises:
            DataReadError: If database read fails
        """
        query = "SELECT * FROM node_panel WHERE feature_version = :version"
        params = {"version": feature_version}

        if start:
            query += " AND ts_utc >= :start"
            params["start"] = start

        if end:
            query += " AND ts_utc < :end"
            params["end"] = end

        if delta_buckets:
            placeholders = ",".join([f":bucket{i}" for i in range(len(delta_buckets))])
            query += f" AND delta_bucket IN ({placeholders})"
            for i, bucket in enumerate(delta_buckets):
                params[f"bucket{i}"] = bucket

        query += " ORDER BY ts_utc, delta_bucket, tenor_days"

        try:
            return pd.read_sql(text(query), self._engine, params=params)
        except Exception as e:
            raise DataReadError(f"Failed to read node panel: {e}") from e
