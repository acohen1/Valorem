"""Integration tests for database operations.

Tests the complete workflow of creating database, inserting data, and querying.
"""

from datetime import date, datetime

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.data.storage.schema import (
    RawFredSeries,
    RawIngestionLog,
    RawOptionQuote,
    RawUnderlyingBar,
    SurfaceSnapshot,
)


class TestDatabaseIntegration:
    """Integration tests for database operations."""

    def test_insert_and_query_underlying_bars(self, db_engine):
        """Test inserting and querying underlying bars."""
        with Session(db_engine) as session:
            # Insert data
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
            session.add(bar)
            session.commit()

            # Query data
            stmt = select(RawUnderlyingBar).where(RawUnderlyingBar.symbol == "SPY")
            result = session.execute(stmt).scalar_one()

            assert result.symbol == "SPY"
            assert result.open == 400.0
            assert result.close == 400.5

    def test_insert_and_query_option_quotes(self, db_engine):
        """Test inserting and querying option quotes."""
        with Session(db_engine) as session:
            # Insert data
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
                volume=100,
                open_interest=500,
                source_ingested_at=datetime(2023, 1, 1, 10, 0),
                ingest_run_id="run_123",
            )
            session.add(quote)
            session.commit()

            # Query data
            stmt = select(RawOptionQuote).where(RawOptionQuote.strike == 400.0)
            result = session.execute(stmt).scalar_one()

            assert result.option_symbol == "SPY230120C00400000"
            assert result.bid == 5.0
            assert result.ask == 5.1

    def test_insert_and_query_fred_series(self, db_engine):
        """Test inserting and querying FRED series."""
        with Session(db_engine) as session:
            # Insert data
            series = RawFredSeries(
                series_id="DGS10",
                obs_date=date(2023, 1, 1),
                value=4.5,
                release_datetime_utc=datetime(2023, 1, 1, 15, 0),
                source_ingested_at=datetime(2023, 1, 1, 16, 0),
            )
            session.add(series)
            session.commit()

            # Query data
            stmt = select(RawFredSeries).where(RawFredSeries.series_id == "DGS10")
            result = session.execute(stmt).scalar_one()

            assert result.series_id == "DGS10"
            assert result.value == 4.5

    def test_insert_and_query_surface_snapshots(self, db_engine):
        """Test inserting and querying surface snapshots."""
        with Session(db_engine) as session:
            # Insert data
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
            session.add(snapshot)
            session.commit()

            # Query data
            stmt = select(SurfaceSnapshot).where(SurfaceSnapshot.delta_bucket == "ATM")
            result = session.execute(stmt).scalar_one()

            assert result.iv_mid == 0.20
            assert result.delta == 0.50

    def test_unique_constraint_enforcement(self, db_engine):
        """Test that unique constraints are enforced."""
        from sqlalchemy.exc import IntegrityError

        with Session(db_engine) as session:
            # Insert first bar
            bar1 = RawUnderlyingBar(
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
                source_ingested_at=datetime(2023, 1, 1, 10, 0),
                ingest_run_id="run_123",
            )
            session.add(bar1)
            session.commit()

            # Try to insert duplicate (same symbol, ts_utc, ingest_run_id)
            bar2 = RawUnderlyingBar(
                dataset="GLBX.MDP3",
                schema="ohlcv-1m",
                stype_in="raw_symbol",
                ts_utc=datetime(2023, 1, 1, 9, 30),
                symbol="SPY",
                timeframe="1m",
                open=401.0,
                high=402.0,
                low=400.0,
                close=401.5,
                source_ingested_at=datetime(2023, 1, 1, 10, 0),
                ingest_run_id="run_123",  # Same run_id
            )
            session.add(bar2)

            with pytest.raises(IntegrityError):
                session.commit()

    def test_indexes_created(self, db_engine):
        """Test that indexes are created correctly."""
        from sqlalchemy import inspect

        inspector = inspect(db_engine)

        # Check indexes on raw_underlying_bars
        indexes = inspector.get_indexes("raw_underlying_bars")
        index_names = {idx["name"] for idx in indexes}

        assert "idx_bars_ts" in index_names
        assert "idx_bars_symbol" in index_names

        # Check indexes on raw_option_quotes
        indexes = inspector.get_indexes("raw_option_quotes")
        index_names = {idx["name"] for idx in indexes}

        assert "idx_quotes_ts" in index_names
        assert "idx_quotes_exp" in index_names

    def test_duplicate_bars_rejected(self, db_engine):
        """Test that duplicate bars (same symbol, ts_utc, timeframe) are rejected."""
        from sqlalchemy.exc import IntegrityError

        with Session(db_engine) as session:
            # Insert bars from run_1
            bar1 = RawUnderlyingBar(
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
                source_ingested_at=datetime(2023, 1, 1, 10, 0),
                ingest_run_id="run_1",
            )
            session.add(bar1)
            session.commit()

        with Session(db_engine) as session:
            # Attempt duplicate with different run_id - should fail
            bar2 = RawUnderlyingBar(
                dataset="GLBX.MDP3",
                schema="ohlcv-1m",
                stype_in="raw_symbol",
                ts_utc=datetime(2023, 1, 1, 9, 30),
                symbol="SPY",
                timeframe="1m",
                open=401.0,
                high=402.0,
                low=400.0,
                close=401.5,
                source_ingested_at=datetime(2023, 1, 1, 11, 0),
                ingest_run_id="run_2",
            )
            session.add(bar2)
            with pytest.raises(IntegrityError):
                session.commit()
            session.rollback()

        # Verify only original data exists
        with Session(db_engine) as session:
            stmt = select(RawUnderlyingBar).where(RawUnderlyingBar.symbol == "SPY")
            results = session.execute(stmt).scalars().all()
            assert len(results) == 1
            assert results[0].ingest_run_id == "run_1"

    def test_different_timeframes_can_coexist(self, db_engine):
        """Test that same symbol/timestamp with different timeframes are allowed."""
        with Session(db_engine) as session:
            bar_1m = RawUnderlyingBar(
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
                source_ingested_at=datetime(2023, 1, 1, 10, 0),
                ingest_run_id="run_1",
            )
            bar_1h = RawUnderlyingBar(
                dataset="GLBX.MDP3",
                schema="ohlcv-1h",
                stype_in="raw_symbol",
                ts_utc=datetime(2023, 1, 1, 9, 30),
                symbol="SPY",
                timeframe="1h",
                open=400.0,
                high=402.0,
                low=398.0,
                close=401.0,
                source_ingested_at=datetime(2023, 1, 1, 10, 0),
                ingest_run_id="run_1",
            )
            session.add_all([bar_1m, bar_1h])
            session.commit()

            stmt = select(RawUnderlyingBar).where(RawUnderlyingBar.symbol == "SPY")
            results = session.execute(stmt).scalars().all()
            assert len(results) == 2
            assert {r.timeframe for r in results} == {"1m", "1h"}
