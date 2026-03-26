"""Unit tests for database schema definitions."""

from datetime import date, datetime

import pytest
from sqlalchemy import inspect

from src.data.storage.engine import create_engine
from src.data.storage.schema import (
    Base,
    NodePanel,
    RawFredSeries,
    RawIngestionLog,
    RawOptionBar,
    RawOptionQuote,
    RawUnderlyingBar,
    SurfaceSnapshot,
)


class TestSchemaDefinitions:
    """Test SQLAlchemy schema definitions."""

    def test_all_tables_defined(self):
        """Test that all required tables are defined."""
        table_names = set(Base.metadata.tables.keys())

        expected_tables = {
            "raw_underlying_bars",
            "raw_option_quotes",
            "raw_option_bars",
            "raw_option_statistics",
            "raw_fred_series",
            "raw_ingestion_log",
            "surface_snapshots",
            "node_panel",
        }

        assert table_names == expected_tables

    def test_raw_underlying_bars_columns(self):
        """Test raw_underlying_bars table has all required columns."""
        table = Base.metadata.tables["raw_underlying_bars"]
        column_names = {col.name for col in table.columns}

        required_columns = {
            "id",
            "dataset",
            "schema",
            "stype_in",
            "instrument_id",
            "publisher_id",
            "ts_utc",
            "ts_recv_utc",
            "symbol",
            "timeframe",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "source_ingested_at",
            "ingest_run_id",
        }

        assert required_columns.issubset(column_names)

    def test_raw_option_quotes_columns(self):
        """Test raw_option_quotes table has all required columns."""
        table = Base.metadata.tables["raw_option_quotes"]
        column_names = {col.name for col in table.columns}

        required_columns = {
            "id",
            "dataset",
            "schema",
            "stype_in",
            "ts_utc",
            "option_symbol",
            "exp_date",
            "strike",
            "right",
            "bid",
            "ask",
            "volume",
            "open_interest",
            "source_ingested_at",
            "ingest_run_id",
        }

        assert required_columns.issubset(column_names)

    def test_raw_fred_series_columns(self):
        """Test raw_fred_series table has all required columns."""
        table = Base.metadata.tables["raw_fred_series"]
        column_names = {col.name for col in table.columns}

        required_columns = {
            "id",
            "series_id",
            "obs_date",
            "value",
            "release_datetime_utc",
            "source_ingested_at",
        }

        assert required_columns.issubset(column_names)

    def test_surface_snapshots_columns(self):
        """Test surface_snapshots table has all required columns."""
        table = Base.metadata.tables["surface_snapshots"]
        column_names = {col.name for col in table.columns}

        required_columns = {
            "id",
            "ts_utc",
            "option_symbol",
            "exp_date",
            "strike",
            "right",
            "bid",
            "ask",
            "mid_price",
            "tte_years",
            "tenor_days",
            "underlying_price",
            "iv_mid",
            "delta",
            "gamma",
            "vega",
            "theta",
            "delta_bucket",
            "flags",
            "snapshot_version",
            "build_run_id",
        }

        assert required_columns.issubset(column_names)

    def test_node_panel_columns(self):
        """Test node_panel table has all required columns."""
        table = Base.metadata.tables["node_panel"]
        column_names = {col.name for col in table.columns}

        required_columns = {
            "id",
            "ts_utc",
            "tenor_days",
            "delta_bucket",
            "option_symbol",
            "delta",
            "feature_version",
            "is_masked",
        }

        assert required_columns.issubset(column_names)


class TestSchemaConstraints:
    """Test database constraints and indexes."""

    def test_raw_underlying_bars_unique_constraint(self):
        """Test raw_underlying_bars has unique constraint on (symbol, ts_utc, timeframe)."""
        table = Base.metadata.tables["raw_underlying_bars"]

        # Find unique constraints
        unique_constraints = [c for c in table.constraints if hasattr(c, "columns")]
        constraint_columns = [
            tuple(sorted([col.name for col in c.columns])) for c in unique_constraints
        ]

        expected = tuple(sorted(["symbol", "ts_utc", "timeframe"]))
        assert expected in constraint_columns

    def test_raw_option_quotes_unique_constraint(self):
        """Test raw_option_quotes has unique constraint on (option_symbol, ts_utc)."""
        table = Base.metadata.tables["raw_option_quotes"]

        unique_constraints = [c for c in table.constraints if hasattr(c, "columns")]
        constraint_columns = [
            tuple(sorted([col.name for col in c.columns])) for c in unique_constraints
        ]

        expected = tuple(sorted(["option_symbol", "ts_utc"]))
        assert expected in constraint_columns

    def test_raw_fred_series_unique_constraint(self):
        """Test raw_fred_series has unique constraint on (series_id, obs_date)."""
        table = Base.metadata.tables["raw_fred_series"]

        unique_constraints = [c for c in table.constraints if hasattr(c, "columns")]
        constraint_columns = [
            tuple(sorted([col.name for col in c.columns])) for c in unique_constraints
        ]

        expected = tuple(sorted(["series_id", "obs_date"]))
        assert expected in constraint_columns

    def test_raw_underlying_bars_indexes(self):
        """Test raw_underlying_bars has required indexes."""
        table = Base.metadata.tables["raw_underlying_bars"]
        index_names = {idx.name for idx in table.indexes}

        assert "idx_bars_ts" in index_names
        assert "idx_bars_symbol" in index_names

    def test_raw_option_quotes_indexes(self):
        """Test raw_option_quotes has required indexes."""
        table = Base.metadata.tables["raw_option_quotes"]
        index_names = {idx.name for idx in table.indexes}

        assert "idx_quotes_ts" in index_names
        assert "idx_quotes_exp" in index_names

    def test_surface_snapshots_indexes(self):
        """Test surface_snapshots has required indexes."""
        table = Base.metadata.tables["surface_snapshots"]
        index_names = {idx.name for idx in table.indexes}

        assert "idx_snapshots_ts_bucket" in index_names
        assert "idx_snapshots_version" in index_names


class TestORMModels:
    """Test ORM model instantiation."""

    def test_raw_underlying_bar_model(self):
        """Test creating RawUnderlyingBar instance."""
        bar = RawUnderlyingBar(
            dataset="GLBX.MDP3",
            schema="ohlcv-1m",
            stype_in="raw_symbol",
            ts_utc=datetime(2023, 1, 1, 9, 30),
            symbol="SPY",
            timeframe="1m",
            open=400.0,
            high=401.0,
            low=399.0,
            close=400.5,
            volume=1000,
            source_ingested_at=datetime(2023, 1, 1, 10, 0),
            ingest_run_id="run_123",
        )

        assert bar.symbol == "SPY"
        assert bar.open == 400.0
        assert bar.volume == 1000

    def test_raw_option_quote_model(self):
        """Test creating RawOptionQuote instance."""
        quote = RawOptionQuote(
            dataset="OPRA",
            schema="cbbo-1m",
            stype_in="raw_symbol",
            ts_utc=datetime(2023, 1, 1, 9, 30),
            option_symbol="SPY230120C00400000",
            exp_date=date(2023, 1, 20),
            strike=400.0,
            right="C",
            bid=5.0,
            ask=5.1,
            bid_size=10,
            ask_size=5,
            volume=100,
            open_interest=500,
            source_ingested_at=datetime(2023, 1, 1, 10, 0),
            ingest_run_id="run_123",
        )

        assert quote.option_symbol == "SPY230120C00400000"
        assert quote.strike == 400.0
        assert quote.right == "C"

    def test_surface_snapshot_model(self):
        """Test creating SurfaceSnapshot instance."""
        snapshot = SurfaceSnapshot(
            ts_utc=datetime(2023, 1, 1, 9, 30),
            option_symbol="SPY230120C00400000",
            exp_date=date(2023, 1, 20),
            strike=400.0,
            right="C",
            bid=5.0,
            ask=5.1,
            mid_price=5.05,
            spread=0.1,
            spread_pct=1.98,
            tte_years=0.05,
            tenor_days=19,
            underlying_price=400.0,
            rf_rate=0.05,
            dividend_yield=0.01,
            iv_mid=0.20,
            delta=0.50,
            gamma=0.02,
            vega=10.0,
            theta=-5.0,
            delta_bucket="ATM",
            flags=0,
            snapshot_version="v1",
            build_run_id="build_123",
            source_created_at=datetime(2023, 1, 1, 10, 0),
        )

        assert snapshot.delta_bucket == "ATM"
        assert snapshot.iv_mid == 0.20
        assert snapshot.delta == 0.50
