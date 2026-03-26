"""Unit tests for repository pattern implementation."""

from datetime import UTC, datetime
from unittest.mock import Mock, patch

import pandas as pd
import pytest
from sqlalchemy import text
from sqlalchemy.engine import Engine

from src.data.storage.repository import DerivedRepository, RawRepository
from src.exceptions import DataWriteError


class TestRawRepositoryInitialization:
    """Test RawRepository initialization."""

    def test_initialization_with_engine(self, db_engine):
        """Test repository initialization with SQLAlchemy engine."""
        repo = RawRepository(db_engine)
        assert repo._engine is not None
        assert isinstance(repo._engine, Engine)


class TestRawRepositoryUnderlyingBars:
    """Test RawRepository underlying bars methods."""

    @pytest.fixture
    def repo(self, raw_repo):
        """Alias for shared raw_repo fixture."""
        return raw_repo

    def test_write_underlying_bars(self, repo):
        """Test writing underlying bars to database."""
        df = pd.DataFrame(
            {
                "ts_utc": pd.date_range("2023-01-01 09:30", periods=3, freq="1min"),
                "symbol": ["SPY"] * 3,
                "timeframe": ["1m"] * 3,
                "open": [400.0, 400.5, 401.0],
                "high": [400.5, 401.0, 401.5],
                "low": [399.5, 400.0, 400.5],
                "close": [400.2, 400.8, 401.2],
                "volume": [1000, 1500, 1200],
            }
        )

        repo.write_underlying_bars(df, run_id="test-run-001")

        # Verify data was written
        result = pd.read_sql("SELECT * FROM raw_underlying_bars", repo._engine)
        assert len(result) == 3
        assert result["ingest_run_id"].iloc[0] == "test-run-001"
        assert "source_ingested_at" in result.columns

    def test_write_underlying_bars_empty_df(self, repo):
        """Test that empty DataFrame is handled gracefully."""
        df = pd.DataFrame(
            columns=[
                "ts_utc",
                "symbol",
                "timeframe",
                "open",
                "high",
                "low",
                "close",
                "volume",
            ]
        )

        # Should not raise error
        repo.write_underlying_bars(df, run_id="test-run-002")

        # Verify no data was written
        result = pd.read_sql("SELECT * FROM raw_underlying_bars", repo._engine)
        assert len(result) == 0


    def test_read_underlying_bars(self, repo):
        """Test reading underlying bars from database."""
        # Write test data
        df = pd.DataFrame(
            {
                "ts_utc": pd.date_range("2023-01-01 09:30", periods=5, freq="1min"),
                "symbol": ["SPY"] * 5,
                "timeframe": ["1m"] * 5,
                "open": [400.0, 400.5, 401.0, 401.5, 402.0],
                "high": [400.5, 401.0, 401.5, 402.0, 402.5],
                "low": [399.5, 400.0, 400.5, 401.0, 401.5],
                "close": [400.2, 400.8, 401.2, 401.8, 402.2],
                "volume": [1000, 1500, 1200, 1300, 1100],
            }
        )
        repo.write_underlying_bars(df, run_id="test-run-004")

        # Read subset of data
        start = datetime(2023, 1, 1, 9, 31)
        end = datetime(2023, 1, 1, 9, 34)

        result = repo.read_underlying_bars("SPY", start, end)

        assert len(result) == 3
        assert result["symbol"].iloc[0] == "SPY"
        assert result["open"].iloc[0] == 400.5

    def test_count_underlying_bars(self, repo):
        """Test counting existing underlying bar rows."""
        df = pd.DataFrame(
            {
                "ts_utc": pd.date_range("2023-01-01 09:30", periods=5, freq="1min"),
                "symbol": ["SPY"] * 5,
                "timeframe": ["1m"] * 5,
                "open": [400.0] * 5,
                "high": [401.0] * 5,
                "low": [399.0] * 5,
                "close": [400.5] * 5,
                "volume": [1000] * 5,
            }
        )
        repo.write_underlying_bars(df, run_id="count-test")

        count = repo.count_underlying_bars(
            "SPY",
            datetime(2023, 1, 1, 9, 30),
            datetime(2023, 1, 1, 9, 35),
        )
        assert count == 5

    def test_count_underlying_bars_no_data(self, repo):
        """Test counting returns 0 when no data exists."""
        count = repo.count_underlying_bars(
            "AAPL",
            datetime(2023, 1, 1, 9, 30),
            datetime(2023, 1, 1, 10, 30),
        )
        assert count == 0

    def test_read_underlying_bars_no_data(self, repo):
        """Test reading when no data exists."""
        start = datetime(2023, 1, 1, 9, 30)
        end = datetime(2023, 1, 1, 10, 30)

        result = repo.read_underlying_bars("AAPL", start, end)

        assert len(result) == 0

    def test_read_underlying_bars_time_filtering(self, repo):
        """Test that time filtering is inclusive of start, exclusive of end."""
        df = pd.DataFrame(
            {
                "ts_utc": pd.to_datetime(
                    [
                        "2023-01-01 09:30:00",
                        "2023-01-01 09:31:00",
                        "2023-01-01 09:32:00",
                    ]
                ),
                "symbol": ["SPY"] * 3,
                "timeframe": ["1m"] * 3,
                "open": [400.0, 400.5, 401.0],
                "high": [400.5, 401.0, 401.5],
                "low": [399.5, 400.0, 400.5],
                "close": [400.2, 400.8, 401.2],
                "volume": [1000, 1500, 1200],
            }
        )
        repo.write_underlying_bars(df, run_id="test-run-005")

        # Query with start = 09:30:00 (inclusive), end = 09:32:00 (exclusive)
        start = datetime(2023, 1, 1, 9, 30, 0)
        end = datetime(2023, 1, 1, 9, 32, 0)

        result = repo.read_underlying_bars("SPY", start, end)

        assert len(result) == 2
        # Convert to datetime if needed for comparison
        ts_0 = pd.to_datetime(result["ts_utc"].iloc[0])
        ts_1 = pd.to_datetime(result["ts_utc"].iloc[1])
        assert ts_0.strftime("%H:%M:%S") == "09:30:00"
        assert ts_1.strftime("%H:%M:%S") == "09:31:00"


class TestRawRepositoryOptionQuotes:
    """Test RawRepository option quotes methods."""

    @pytest.fixture
    def repo(self, raw_repo):
        """Alias for shared raw_repo fixture."""
        return raw_repo

    def test_write_option_quotes(self, repo):
        """Test writing option quotes to database."""
        df = pd.DataFrame(
            {
                "ts_utc": pd.date_range("2023-01-01 09:30", periods=3, freq="1min"),
                "option_symbol": ["SPY230120C00400000"] * 3,
                "exp_date": [datetime(2023, 1, 20).date()] * 3,
                "strike": [400.0] * 3,
                "right": ["C"] * 3,
                "bid": [5.0, 5.1, 5.05],
                "ask": [5.1, 5.2, 5.15],
            }
        )

        repo.write_option_quotes(df, run_id="test-run-006")

        # Verify data was written
        result = pd.read_sql("SELECT * FROM raw_option_quotes", repo._engine)
        assert len(result) == 3
        assert result["ingest_run_id"].iloc[0] == "test-run-006"
        assert "source_ingested_at" in result.columns

    def test_write_option_quotes_empty_df(self, repo):
        """Test that empty DataFrame is handled gracefully."""
        df = pd.DataFrame(
            columns=[
                "ts_utc",
                "option_symbol",
                "exp_date",
                "strike",
                "right",
                "bid",
                "ask",
            ]
        )

        repo.write_option_quotes(df, run_id="test-run-007")

        result = pd.read_sql("SELECT * FROM raw_option_quotes", repo._engine)
        assert len(result) == 0

    def test_read_option_quotes(self, repo):
        """Test reading option quotes from database."""
        df = pd.DataFrame(
            {
                "ts_utc": pd.date_range("2023-01-01 09:30", periods=5, freq="1min"),
                "option_symbol": ["SPY230120C00400000"] * 5,
                "exp_date": [datetime(2023, 1, 20).date()] * 5,
                "strike": [400.0] * 5,
                "right": ["C"] * 5,
                "bid": [5.0, 5.1, 5.05, 5.15, 5.2],
                "ask": [5.1, 5.2, 5.15, 5.25, 5.3],
            }
        )
        repo.write_option_quotes(df, run_id="test-run-008")

        start = datetime(2023, 1, 1, 9, 31)
        end = datetime(2023, 1, 1, 9, 34)

        result = repo.read_option_quotes(start, end)

        assert len(result) == 3
        assert result["option_symbol"].iloc[0] == "SPY230120C00400000"

    def test_read_option_quotes_with_symbol_filter(self, repo):
        """Test reading option quotes with symbol filtering."""
        df = pd.DataFrame(
            {
                "ts_utc": pd.date_range("2023-01-01 09:30", periods=4, freq="1min"),
                "option_symbol": [
                    "SPY230120C00400000",
                    "SPY230120P00400000",
                    "SPY230120C00400000",
                    "SPY230120P00400000",
                ],
                "exp_date": [datetime(2023, 1, 20).date()] * 4,
                "strike": [400.0] * 4,
                "right": ["C", "P", "C", "P"],
                "bid": [5.0, 4.9, 5.1, 4.95],
                "ask": [5.1, 5.0, 5.2, 5.05],
            }
        )
        repo.write_option_quotes(df, run_id="test-run-009")

        start = datetime(2023, 1, 1, 9, 30)
        end = datetime(2023, 1, 1, 9, 35)

        # Filter for calls only
        result = repo.read_option_quotes(
            start, end, option_symbols=["SPY230120C00400000"]
        )

        assert len(result) == 2
        assert all(result["option_symbol"] == "SPY230120C00400000")

    def test_read_option_quotes_multiple_symbols(self, repo):
        """Test reading option quotes with multiple symbol filters."""
        df = pd.DataFrame(
            {
                "ts_utc": pd.date_range("2023-01-01 09:30", periods=6, freq="1min"),
                "option_symbol": [
                    "SPY230120C00400000",
                    "SPY230120P00400000",
                    "AAPL230120C00150000",
                ] * 2,
                "exp_date": [datetime(2023, 1, 20).date()] * 6,
                "strike": [400.0, 400.0, 150.0] * 2,
                "right": ["C", "P", "C"] * 2,
                "bid": [5.0, 4.9, 3.0, 5.1, 4.95, 3.05],
                "ask": [5.1, 5.0, 3.1, 5.2, 5.05, 3.15],
            }
        )
        repo.write_option_quotes(df, run_id="test-run-010")

        start = datetime(2023, 1, 1, 9, 30)
        end = datetime(2023, 1, 1, 9, 36)

        # Filter for SPY options only
        result = repo.read_option_quotes(
            start, end, option_symbols=["SPY230120C00400000", "SPY230120P00400000"]
        )

        assert len(result) == 4
        assert "AAPL230120C00150000" not in result["option_symbol"].values

    def test_count_option_quotes(self, repo):
        """Test counting existing option quote rows."""
        df = pd.DataFrame(
            {
                "ts_utc": pd.date_range("2023-01-01 09:30", periods=3, freq="1min"),
                "option_symbol": ["SPY230120C00400000"] * 3,
                "exp_date": [datetime(2023, 1, 20).date()] * 3,
                "strike": [400.0] * 3,
                "right": ["C"] * 3,
                "bid": [5.0, 5.1, 5.05],
                "ask": [5.1, 5.2, 5.15],
            }
        )
        repo.write_option_quotes(df, run_id="count-test")

        count = repo.count_option_quotes(
            datetime(2023, 1, 1, 9, 30),
            datetime(2023, 1, 1, 9, 33),
        )
        assert count == 3

    def test_count_option_quotes_no_data(self, repo):
        """Test counting returns 0 when no data exists."""
        count = repo.count_option_quotes(
            datetime(2023, 6, 1),
            datetime(2023, 6, 30),
        )
        assert count == 0

    def test_read_option_quotes_no_symbol_filter(self, repo):
        """Test reading all option quotes when no filter provided."""
        df = pd.DataFrame(
            {
                "ts_utc": pd.date_range("2023-01-01 09:30", periods=2, freq="1min"),
                "option_symbol": ["SPY230120C00400000", "SPY230120P00400000"],
                "exp_date": [datetime(2023, 1, 20).date()] * 2,
                "strike": [400.0] * 2,
                "right": ["C", "P"],
                "bid": [5.0, 4.9],
                "ask": [5.1, 5.0],
            }
        )
        repo.write_option_quotes(df, run_id="test-run-011")

        start = datetime(2023, 1, 1, 9, 30)
        end = datetime(2023, 1, 1, 9, 32)

        result = repo.read_option_quotes(start, end, option_symbols=None)

        assert len(result) == 2


class TestRawRepositoryFredSeries:
    """Test RawRepository FRED series methods."""

    @pytest.fixture
    def repo(self, raw_repo):
        """Alias for shared raw_repo fixture."""
        return raw_repo

    def test_write_fred_series(self, repo):
        """Test writing FRED series to database."""
        df = pd.DataFrame(
            {
                "series_id": ["DGS10"] * 3,
                "obs_date": pd.to_datetime(
                    ["2023-01-01", "2023-01-02", "2023-01-03"]
                ).date,
                "value": [0.045, 0.046, 0.044],
                "release_datetime_utc": pd.to_datetime(
                    [
                        "2023-01-02 00:00:00",
                        "2023-01-03 00:00:00",
                        "2023-01-04 00:00:00",
                    ]
                ),
            }
        )

        repo.write_fred_series(df)

        result = pd.read_sql("SELECT * FROM raw_fred_series", repo._engine)
        assert len(result) == 3
        assert result["series_id"].iloc[0] == "DGS10"
        assert "source_ingested_at" in result.columns

    def test_write_fred_series_empty_df(self, repo):
        """Test that empty DataFrame is handled gracefully."""
        df = pd.DataFrame(
            columns=["series_id", "obs_date", "value", "release_datetime_utc"]
        )

        repo.write_fred_series(df)

        result = pd.read_sql("SELECT * FROM raw_fred_series", repo._engine)
        assert len(result) == 0

    def test_write_fred_series_duplicate_upsert(self, repo):
        """Test that duplicate writes are silently skipped via upsert."""
        df = pd.DataFrame(
            {
                "series_id": ["DGS10"],
                "obs_date": [datetime(2023, 1, 1).date()],
                "value": [0.045],
                "release_datetime_utc": [datetime(2023, 1, 2, 0, 0, 0)],
            }
        )

        # Write once
        repo.write_fred_series(df)

        # Write duplicate — upsert silently skips
        repo.write_fred_series(df)

        # Verify only one row exists (duplicate was ignored)
        result = pd.read_sql("SELECT * FROM raw_fred_series", repo._engine)
        assert len(result) == 1

    def test_read_fred_series(self, repo):
        """Test reading FRED series from database."""
        df = pd.DataFrame(
            {
                "series_id": ["DGS10"] * 5,
                "obs_date": pd.to_datetime(
                    [
                        "2023-01-01",
                        "2023-01-02",
                        "2023-01-03",
                        "2023-01-04",
                        "2023-01-05",
                    ]
                ).date,
                "value": [0.045, 0.046, 0.044, 0.047, 0.043],
                "release_datetime_utc": pd.to_datetime(
                    [
                        "2023-01-02 00:00:00",
                        "2023-01-03 00:00:00",
                        "2023-01-04 00:00:00",
                        "2023-01-05 00:00:00",
                        "2023-01-06 00:00:00",
                    ]
                ),
            }
        )
        repo.write_fred_series(df)

        start = datetime(2023, 1, 2)
        end = datetime(2023, 1, 5)

        result = repo.read_fred_series("DGS10", start, end)

        assert len(result) == 3
        assert result["series_id"].iloc[0] == "DGS10"
        assert result["value"].iloc[0] == 0.046

    def test_count_fred_series(self, repo):
        """Test counting existing FRED series rows."""
        df = pd.DataFrame(
            {
                "series_id": ["DGS10"] * 3,
                "obs_date": pd.to_datetime(
                    ["2023-01-01", "2023-01-02", "2023-01-03"]
                ).date,
                "value": [0.045, 0.046, 0.044],
                "release_datetime_utc": pd.to_datetime(
                    [
                        "2023-01-02 00:00:00",
                        "2023-01-03 00:00:00",
                        "2023-01-04 00:00:00",
                    ]
                ),
            }
        )
        repo.write_fred_series(df)

        count = repo.count_fred_series(
            "DGS10",
            datetime(2023, 1, 1),
            datetime(2023, 1, 4),
        )
        assert count == 3

    def test_count_fred_series_no_data(self, repo):
        """Test counting returns 0 when no data exists."""
        count = repo.count_fred_series(
            "VIX",
            datetime(2023, 1, 1),
            datetime(2023, 1, 31),
        )
        assert count == 0

    def test_read_fred_series_no_data(self, repo):
        """Test reading when no data exists."""
        start = datetime(2023, 1, 1)
        end = datetime(2023, 1, 31)

        result = repo.read_fred_series("VIX", start, end)

        assert len(result) == 0


class TestRawRepositoryIngestionLog:
    """Test RawRepository ingestion log methods."""

    @pytest.fixture
    def repo(self, raw_repo):
        """Alias for shared raw_repo fixture."""
        return raw_repo

    def test_write_ingestion_log(self, repo):
        """Test writing ingestion log entry."""
        log_entry = {
            "ingest_run_id": "test-run-012",
            "dataset": "GLBX.MDP3",
            "schema": "ohlcv-1m",
            "stype_in": "raw_symbol",
            "symbols": "SPY",
            "start_date": datetime(2023, 1, 1).date(),
            "end_date": datetime(2023, 1, 31).date(),
            "row_count": 1000,
            "source_ingested_at": datetime.now(UTC),
        }

        repo.write_ingestion_log(log_entry)

        result = pd.read_sql("SELECT * FROM raw_ingestion_log", repo._engine)
        assert len(result) == 1
        assert result["ingest_run_id"].iloc[0] == "test-run-012"
        assert result["row_count"].iloc[0] == 1000

    def test_read_ingestion_log_no_filters(self, repo):
        """Test reading all ingestion log entries."""
        log_entries = [
            {
                "ingest_run_id": f"test-run-{i:03d}",
                "dataset": "GLBX.MDP3",
                "schema": "ohlcv-1m",
                "stype_in": "raw_symbol",
                "symbols": "SPY",
                "start_date": datetime(2023, 1, 1).date(),
                "end_date": datetime(2023, 1, 31).date(),
                "row_count": 1000 + i,
                "source_ingested_at": datetime.now(UTC),
            }
            for i in range(3)
        ]

        for entry in log_entries:
            repo.write_ingestion_log(entry)

        result = repo.read_ingestion_log()

        assert len(result) == 3

    def test_read_ingestion_log_with_dataset_filter(self, repo):
        """Test reading ingestion log with dataset filter."""
        log_entries = [
            {
                "ingest_run_id": "test-run-013",
                "dataset": "GLBX.MDP3",
                "schema": "ohlcv-1m",
                "stype_in": "raw_symbol",
                "symbols": "SPY",
                "start_date": datetime(2023, 1, 1).date(),
                "end_date": datetime(2023, 1, 31).date(),
                "row_count": 1000,
                "source_ingested_at": datetime.now(UTC),
            },
            {
                "ingest_run_id": "test-run-014",
                "dataset": "OPRA.PILLAR",
                "schema": "cbbo-1m",
                "stype_in": "raw_symbol",
                "symbols": "SPY230120C00400000",
                "start_date": datetime(2023, 1, 1).date(),
                "end_date": datetime(2023, 1, 31).date(),
                "row_count": 5000,
                "source_ingested_at": datetime.now(UTC),
            },
        ]

        for entry in log_entries:
            repo.write_ingestion_log(entry)

        result = repo.read_ingestion_log(dataset="GLBX.MDP3")

        assert len(result) == 1
        assert result["dataset"].iloc[0] == "GLBX.MDP3"

    def test_read_ingestion_log_with_date_filters(self, repo):
        """Test reading ingestion log with date filters."""
        log_entries = [
            {
                "ingest_run_id": "test-run-015",
                "dataset": "GLBX.MDP3",
                "schema": "ohlcv-1m",
                "stype_in": "raw_symbol",
                "symbols": "SPY",
                "start_date": datetime(2023, 1, 1).date(),
                "end_date": datetime(2023, 1, 15).date(),
                "row_count": 500,
                "source_ingested_at": datetime.now(UTC),
            },
            {
                "ingest_run_id": "test-run-016",
                "dataset": "GLBX.MDP3",
                "schema": "ohlcv-1m",
                "stype_in": "raw_symbol",
                "symbols": "SPY",
                "start_date": datetime(2023, 2, 1).date(),
                "end_date": datetime(2023, 2, 28).date(),
                "row_count": 1000,
                "source_ingested_at": datetime.now(UTC),
            },
        ]

        for entry in log_entries:
            repo.write_ingestion_log(entry)

        result = repo.read_ingestion_log(
            start_date=datetime(2023, 1, 10), end_date=datetime(2023, 1, 20)
        )

        assert len(result) == 1
        assert result["ingest_run_id"].iloc[0] == "test-run-015"


class TestDerivedRepositoryInitialization:
    """Test DerivedRepository initialization."""

    def test_initialization_with_engine(self, db_engine):
        """Test repository initialization with SQLAlchemy engine."""
        repo = DerivedRepository(db_engine)
        assert repo._engine is not None
        assert isinstance(repo._engine, Engine)


class TestDerivedRepositorySurfaceSnapshots:
    """Test DerivedRepository surface snapshots methods."""

    @pytest.fixture
    def repo(self, derived_repo):
        """Alias for shared derived_repo fixture."""
        return derived_repo

    def test_write_surface_snapshots(self, repo):
        """Test writing surface snapshots to database."""
        df = pd.DataFrame(
            {
                "ts_utc": pd.date_range("2023-01-01 09:30", periods=3, freq="1min"),
                "option_symbol": ["SPY230120C00400000"] * 3,
                "delta_bucket": ["ATM"] * 3,
                "implied_vol": [0.15, 0.16, 0.155],
            }
        )

        repo.write_surface_snapshots(df, build_run_id="build-001", version="v1.0")

        result = pd.read_sql("SELECT * FROM surface_snapshots", repo._engine)
        assert len(result) == 3
        assert result["build_run_id"].iloc[0] == "build-001"
        assert result["snapshot_version"].iloc[0] == "v1.0"
        assert "source_created_at" in result.columns

    def test_write_surface_snapshots_empty_df(self, repo):
        """Test that empty DataFrame is handled gracefully."""
        df = pd.DataFrame(
            columns=["ts_utc", "option_symbol", "delta_bucket", "implied_vol"]
        )

        repo.write_surface_snapshots(df, build_run_id="build-002", version="v1.0")

        result = pd.read_sql("SELECT * FROM surface_snapshots", repo._engine)
        assert len(result) == 0

    def test_read_surface_snapshots(self, repo):
        """Test reading surface snapshots from database."""
        df = pd.DataFrame(
            {
                "ts_utc": pd.date_range("2023-01-01 09:30", periods=5, freq="1min"),
                "option_symbol": ["SPY230120C00400000"] * 5,
                "delta_bucket": ["ATM"] * 5,
                "implied_vol": [0.15, 0.16, 0.155, 0.17, 0.165],
            }
        )
        repo.write_surface_snapshots(df, build_run_id="build-003", version="v1.0")

        start = datetime(2023, 1, 1, 9, 31)
        end = datetime(2023, 1, 1, 9, 34)

        result = repo.read_surface_snapshots(start, end, version="v1.0")

        assert len(result) == 3
        assert result["implied_vol"].iloc[0] == 0.16

    def test_read_surface_snapshots_with_delta_bucket_filter(self, repo):
        """Test reading surface snapshots with delta bucket filtering."""
        df = pd.DataFrame(
            {
                "ts_utc": pd.date_range("2023-01-01 09:30", periods=4, freq="1min"),
                "option_symbol": ["SPY230120C00400000"] * 4,
                "delta_bucket": ["ATM", "P25", "ATM", "C25"],
                "implied_vol": [0.15, 0.18, 0.16, 0.14],
            }
        )
        repo.write_surface_snapshots(df, build_run_id="build-004", version="v1.0")

        start = datetime(2023, 1, 1, 9, 30)
        end = datetime(2023, 1, 1, 9, 35)

        result = repo.read_surface_snapshots(
            start, end, version="v1.0", delta_buckets=["ATM"]
        )

        assert len(result) == 2
        assert all(result["delta_bucket"] == "ATM")

    def test_read_surface_snapshots_multiple_delta_buckets(self, repo):
        """Test reading surface snapshots with multiple delta bucket filters."""
        df = pd.DataFrame(
            {
                "ts_utc": pd.date_range("2023-01-01 09:30", periods=6, freq="1min"),
                "option_symbol": ["SPY230120C00400000"] * 6,
                "delta_bucket": ["ATM", "P25", "P10", "ATM", "C25", "C10"],
                "implied_vol": [0.15, 0.18, 0.22, 0.16, 0.14, 0.20],
            }
        )
        repo.write_surface_snapshots(df, build_run_id="build-005", version="v1.0")

        start = datetime(2023, 1, 1, 9, 30)
        end = datetime(2023, 1, 1, 9, 36)

        result = repo.read_surface_snapshots(
            start, end, version="v1.0", delta_buckets=["ATM", "P25"]
        )

        assert len(result) == 3
        assert set(result["delta_bucket"].unique()) == {"ATM", "P25"}

    def test_read_surface_snapshots_version_filtering(self, repo):
        """Test that version filtering works correctly."""
        df_v1 = pd.DataFrame(
            {
                "ts_utc": pd.date_range("2023-01-01 09:30", periods=2, freq="1min"),
                "option_symbol": ["SPY230120C00400000"] * 2,
                "delta_bucket": ["ATM"] * 2,
                "implied_vol": [0.15, 0.16],
            }
        )
        df_v2 = pd.DataFrame(
            {
                "ts_utc": pd.date_range("2023-01-01 09:30", periods=2, freq="1min"),
                "option_symbol": ["SPY230120C00400000"] * 2,
                "delta_bucket": ["ATM"] * 2,
                "implied_vol": [0.18, 0.19],
            }
        )

        repo.write_surface_snapshots(df_v1, build_run_id="build-006", version="v1.0")
        repo.write_surface_snapshots(df_v2, build_run_id="build-007", version="v2.0")

        start = datetime(2023, 1, 1, 9, 30)
        end = datetime(2023, 1, 1, 9, 32)

        result = repo.read_surface_snapshots(start, end, version="v1.0")

        assert len(result) == 2
        assert result["implied_vol"].iloc[0] == 0.15


class TestDerivedRepositoryNodePanel:
    """Test DerivedRepository node panel methods."""

    @pytest.fixture
    def repo(self, derived_repo):
        """Alias for shared derived_repo fixture."""
        return derived_repo

    def test_write_node_panel(self, repo):
        """Test writing node panel to database."""
        df = pd.DataFrame(
            {
                "ts_utc": pd.date_range("2023-01-01 09:30", periods=3, freq="1min"),
                "delta_bucket": ["ATM"] * 3,
                "tenor_days": [30] * 3,
                "spread_pct": [0.05, 0.06, 0.055],
            }
        )

        repo.write_node_panel(df, feature_version="v1.0")

        result = pd.read_sql("SELECT * FROM node_panel", repo._engine)
        assert len(result) == 3
        assert result["feature_version"].iloc[0] == "v1.0"
        assert result["is_masked"].iloc[0] == 0  # Default value

    def test_write_node_panel_empty_df(self, repo):
        """Test that empty DataFrame is handled gracefully."""
        df = pd.DataFrame(columns=["ts_utc", "delta_bucket", "tenor_days", "spread_pct"])

        repo.write_node_panel(df, feature_version="v1.0")

        result = pd.read_sql("SELECT * FROM node_panel", repo._engine)
        assert len(result) == 0

    def test_read_node_panel_by_version(self, repo):
        """Test reading node panel by feature version."""
        df_v1 = pd.DataFrame(
            {
                "ts_utc": pd.date_range("2023-01-01 09:30", periods=2, freq="1min"),
                "delta_bucket": ["ATM"] * 2,
                "tenor_days": [30] * 2,
                "spread_pct": [0.05, 0.06],
            }
        )
        df_v2 = pd.DataFrame(
            {
                "ts_utc": pd.date_range("2023-01-01 09:30", periods=2, freq="1min"),
                "delta_bucket": ["ATM"] * 2,
                "tenor_days": [30] * 2,
                "spread_pct": [0.08, 0.09],
            }
        )

        repo.write_node_panel(df_v1, feature_version="v1.0")
        repo.write_node_panel(df_v2, feature_version="v2.0")

        result = repo.read_node_panel(feature_version="v1.0")

        assert len(result) == 2
        assert result["spread_pct"].iloc[0] == 0.05

    def test_read_node_panel_with_time_filters(self, repo):
        """Test reading node panel with time filters."""
        df = pd.DataFrame(
            {
                "ts_utc": pd.date_range("2023-01-01 09:30", periods=5, freq="1min"),
                "delta_bucket": ["ATM"] * 5,
                "tenor_days": [30] * 5,
                "spread_pct": [0.05, 0.06, 0.055, 0.07, 0.065],
            }
        )
        repo.write_node_panel(df, feature_version="v1.0")

        start = datetime(2023, 1, 1, 9, 31)
        end = datetime(2023, 1, 1, 9, 34)

        result = repo.read_node_panel(feature_version="v1.0", start=start, end=end)

        assert len(result) == 3
        assert result["spread_pct"].iloc[0] == 0.06

    def test_read_node_panel_with_delta_bucket_filter(self, repo):
        """Test reading node panel with delta bucket filtering."""
        df = pd.DataFrame(
            {
                "ts_utc": pd.date_range("2023-01-01 09:30", periods=4, freq="1min"),
                "delta_bucket": ["ATM", "P25", "ATM", "C25"],
                "tenor_days": [30, 30, 30, 30],
                "spread_pct": [0.05, 0.08, 0.06, 0.04],
            }
        )
        repo.write_node_panel(df, feature_version="v1.0")

        result = repo.read_node_panel(feature_version="v1.0", delta_buckets=["ATM"])

        assert len(result) == 2
        assert all(result["delta_bucket"] == "ATM")

    def test_read_node_panel_ordering(self, repo):
        """Test that results are ordered correctly."""
        df = pd.DataFrame(
            {
                "ts_utc": pd.to_datetime(
                    ["2023-01-01 09:30", "2023-01-01 09:30", "2023-01-01 09:31"]
                ),
                "delta_bucket": ["P25", "ATM", "ATM"],
                "tenor_days": [7, 30, 7],
                "spread_pct": [0.08, 0.05, 0.06],
            }
        )
        repo.write_node_panel(df, feature_version="v1.0")

        result = repo.read_node_panel(feature_version="v1.0")

        # Verify ordering: ts_utc, delta_bucket, tenor_days
        assert len(result) == 3
        assert result.iloc[0]["delta_bucket"] == "ATM"
        assert result.iloc[0]["tenor_days"] == 30
        assert result.iloc[1]["delta_bucket"] == "P25"
        assert result.iloc[1]["tenor_days"] == 7
        assert result.iloc[2]["delta_bucket"] == "ATM"
        assert result.iloc[2]["tenor_days"] == 7


class TestRoundTripIntegration:
    """Integration tests for write → read round-trips."""

    def test_underlying_bars_round_trip(self, raw_repo):
        """Test write → read round-trip for underlying bars."""
        original_df = pd.DataFrame(
            {
                "ts_utc": pd.date_range("2023-01-01 09:30", periods=10, freq="1min"),
                "symbol": ["SPY"] * 10,
                "timeframe": ["1m"] * 10,
                "open": [400.0 + i * 0.5 for i in range(10)],
                "high": [400.5 + i * 0.5 for i in range(10)],
                "low": [399.5 + i * 0.5 for i in range(10)],
                "close": [400.2 + i * 0.5 for i in range(10)],
                "volume": [1000 + i * 100 for i in range(10)],
            }
        )

        # Write
        raw_repo.write_underlying_bars(original_df, run_id="round-trip-001")

        # Read
        start = datetime(2023, 1, 1, 9, 30)
        end = datetime(2023, 1, 1, 9, 40)
        result_df = raw_repo.read_underlying_bars("SPY", start, end)

        # Verify
        assert len(result_df) == len(original_df)
        assert list(result_df["symbol"].unique()) == ["SPY"]
        assert result_df["open"].iloc[0] == 400.0
        assert result_df["close"].iloc[-1] == 404.7

    def test_option_quotes_round_trip(self, raw_repo):
        """Test write → read round-trip for option quotes."""
        original_df = pd.DataFrame(
            {
                "ts_utc": pd.date_range("2023-01-01 09:30", periods=10, freq="1min"),
                "option_symbol": ["SPY230120C00400000"] * 10,
                "exp_date": [datetime(2023, 1, 20).date()] * 10,
                "strike": [400.0] * 10,
                "right": ["C"] * 10,
                "bid": [5.0 + i * 0.1 for i in range(10)],
                "ask": [5.1 + i * 0.1 for i in range(10)],
            }
        )

        # Write
        raw_repo.write_option_quotes(original_df, run_id="round-trip-002")

        # Read
        start = datetime(2023, 1, 1, 9, 30)
        end = datetime(2023, 1, 1, 9, 40)
        result_df = raw_repo.read_option_quotes(start, end)

        # Verify
        assert len(result_df) == len(original_df)
        assert result_df["option_symbol"].iloc[0] == "SPY230120C00400000"
        assert result_df["bid"].iloc[0] == 5.0
        assert result_df["ask"].iloc[-1] == 6.0

    def test_surface_snapshots_round_trip(self, derived_repo):
        """Test write → read round-trip for surface snapshots."""
        original_df = pd.DataFrame(
            {
                "ts_utc": pd.date_range("2023-01-01 09:30", periods=10, freq="1min"),
                "option_symbol": ["SPY230120C00400000"] * 10,
                "delta_bucket": ["ATM"] * 10,
                "implied_vol": [0.15 + i * 0.01 for i in range(10)],
            }
        )

        # Write
        derived_repo.write_surface_snapshots(
            original_df, build_run_id="round-trip-003", version="v1.0"
        )

        # Read
        start = datetime(2023, 1, 1, 9, 30)
        end = datetime(2023, 1, 1, 9, 40)
        result_df = derived_repo.read_surface_snapshots(start, end, version="v1.0")

        # Verify
        assert len(result_df) == len(original_df)
        assert result_df["implied_vol"].iloc[0] == 0.15
        assert result_df["implied_vol"].iloc[-1] == 0.24

    def test_node_panel_round_trip(self, derived_repo):
        """Test write → read round-trip for node panel."""
        original_df = pd.DataFrame(
            {
                "ts_utc": pd.date_range("2023-01-01 09:30", periods=10, freq="1min"),
                "delta_bucket": ["ATM"] * 10,
                "tenor_days": [30] * 10,
                "spread_pct": [0.05 + i * 0.01 for i in range(10)],
            }
        )

        # Write
        derived_repo.write_node_panel(original_df, feature_version="v1.0")

        # Read
        result_df = derived_repo.read_node_panel(feature_version="v1.0")

        # Verify
        assert len(result_df) == len(original_df)
        assert result_df["spread_pct"].iloc[0] == 0.05
        assert result_df["spread_pct"].iloc[-1] == pytest.approx(0.14)


class TestTransactionSafety:
    """Test transaction safety - rollback on failure, no partial writes."""

    @pytest.fixture
    def standalone_engine(self):
        """Standalone engine for the rollback test which drops/recreates tables.

        This test modifies the schema (adds a CHECK constraint), so it
        cannot use the shared module-scoped engine.
        """
        from sqlalchemy import create_engine as sa_create_engine
        from sqlalchemy.pool import StaticPool

        engine = sa_create_engine(
            "sqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        raw_conn = engine.raw_connection()
        raw_conn.executescript("""
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
        """)
        raw_conn.close()
        yield engine
        engine.dispose()

    def test_underlying_bars_rollback_on_failure(self, standalone_engine):
        """Test that failed write leaves no partial data."""
        repo = RawRepository(standalone_engine)

        # Write initial data successfully
        df_good = pd.DataFrame(
            {
                "ts_utc": pd.date_range("2023-01-01 09:30", periods=2, freq="1min"),
                "symbol": ["SPY"] * 2,
                "timeframe": ["1m"] * 2,
                "open": [400.0, 400.5],
                "high": [400.5, 401.0],
                "low": [399.5, 400.0],
                "close": [400.2, 400.8],
                "volume": [1000, 1500],
            }
        )
        repo.write_underlying_bars(df_good, run_id="good-run")

        # Verify initial data
        result = pd.read_sql("SELECT * FROM raw_underlying_bars", standalone_engine)
        assert len(result) == 2

        # Drop the table and recreate with strict schema to force failure
        with standalone_engine.connect() as conn:
            conn.execute(text("DROP TABLE raw_underlying_bars"))
            conn.execute(
                text(
                    """
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
                    UNIQUE(symbol, ts_utc, timeframe),
                    CHECK (open > 0)
                )
            """
                )
            )
            conn.commit()

        # Write good data first
        repo.write_underlying_bars(df_good, run_id="good-run-2")

        # Create data that violates constraint
        df_violates = pd.DataFrame(
            {
                "ts_utc": pd.date_range("2023-01-02 09:30", periods=3, freq="1min"),
                "symbol": ["SPY"] * 3,
                "timeframe": ["1m"] * 3,
                "open": [-1.0, 401.5, 402.0],  # First value violates CHECK constraint
                "high": [401.5, 402.0, 402.5],
                "low": [400.5, 401.0, 401.5],
                "close": [401.2, 401.8, 402.2],
                "volume": [1100, 1200, 1300],
            }
        )

        # This should fail and rollback
        with pytest.raises(DataWriteError, match="Failed to write underlying bars"):
            repo.write_underlying_bars(df_violates, run_id="bad-run")

        # Verify only the good data remains (no partial write from failed transaction)
        result = pd.read_sql("SELECT * FROM raw_underlying_bars", standalone_engine)
        assert len(result) == 2
        assert all(result["ingest_run_id"] == "good-run-2")

    def test_option_quotes_transaction_commits_on_success(self, raw_repo):
        """Test that successful writes commit correctly."""
        df = pd.DataFrame(
            {
                "ts_utc": pd.date_range("2023-01-01 09:30", periods=3, freq="1min"),
                "option_symbol": ["SPY230120C00400000"] * 3,
                "exp_date": [datetime(2023, 1, 20).date()] * 3,
                "strike": [400.0] * 3,
                "right": ["C"] * 3,
                "bid": [5.0, 5.1, 5.05],
                "ask": [5.1, 5.2, 5.15],
            }
        )

        raw_repo.write_option_quotes(df, run_id="commit-test")

        # Verify data was committed
        result = pd.read_sql("SELECT * FROM raw_option_quotes", raw_repo._engine)
        assert len(result) == 3
        assert result["ingest_run_id"].iloc[0] == "commit-test"

    def test_surface_snapshots_transaction_commits_on_success(self, derived_repo):
        """Test that successful surface snapshot writes commit correctly."""
        df = pd.DataFrame(
            {
                "ts_utc": pd.date_range("2023-01-01 09:30", periods=3, freq="1min"),
                "option_symbol": ["SPY230120C00400000"] * 3,
                "delta_bucket": ["ATM"] * 3,
                "implied_vol": [0.15, 0.16, 0.155],
            }
        )

        derived_repo.write_surface_snapshots(
            df, build_run_id="commit-test", version="v1.0"
        )

        # Verify data was committed
        result = pd.read_sql("SELECT * FROM surface_snapshots", derived_repo._engine)
        assert len(result) == 3
        assert result["build_run_id"].iloc[0] == "commit-test"

    def test_node_panel_transaction_commits_on_success(self, derived_repo):
        """Test that successful node panel writes commit correctly."""
        df = pd.DataFrame(
            {
                "ts_utc": pd.date_range("2023-01-01 09:30", periods=3, freq="1min"),
                "delta_bucket": ["ATM"] * 3,
                "tenor_days": [30] * 3,
                "spread_pct": [0.05, 0.06, 0.055],
            }
        )

        derived_repo.write_node_panel(df, feature_version="v1.0")

        # Verify data was committed
        result = pd.read_sql("SELECT * FROM node_panel", derived_repo._engine)
        assert len(result) == 3
        assert result["feature_version"].iloc[0] == "v1.0"

    def test_underlying_bars_upsert_on_duplicate(self, raw_repo):
        """Test that duplicate underlying bars are silently skipped via upsert."""
        df = pd.DataFrame(
            {
                "ts_utc": pd.date_range("2023-01-01 09:30", periods=2, freq="1min"),
                "symbol": ["SPY"] * 2,
                "timeframe": ["1m"] * 2,
                "open": [400.0, 400.5],
                "high": [400.5, 401.0],
                "low": [399.5, 400.0],
                "close": [400.2, 400.8],
                "volume": [1000, 1500],
            }
        )

        # First write succeeds
        raw_repo.write_underlying_bars(df, run_id="run-1")

        # Second write — upsert silently skips duplicates
        raw_repo.write_underlying_bars(df, run_id="run-2")

        # Verify only original data exists (duplicates ignored)
        result = pd.read_sql("SELECT * FROM raw_underlying_bars", raw_repo._engine)
        assert len(result) == 2
        assert all(result["ingest_run_id"] == "run-1")

    def test_option_quotes_upsert_on_duplicate(self, raw_repo):
        """Test that duplicate option quotes are silently skipped via upsert."""
        df = pd.DataFrame(
            {
                "ts_utc": pd.date_range("2023-01-01 09:30", periods=2, freq="1min"),
                "option_symbol": ["SPY230120C00400000"] * 2,
                "exp_date": [datetime(2023, 1, 20).date()] * 2,
                "strike": [400.0] * 2,
                "right": ["C"] * 2,
                "bid": [5.0, 5.1],
                "ask": [5.1, 5.2],
            }
        )

        # First write succeeds
        raw_repo.write_option_quotes(df, run_id="run-1")

        # Second write — upsert silently skips duplicates
        raw_repo.write_option_quotes(df, run_id="run-2")

        # Verify only original data exists (duplicates ignored)
        result = pd.read_sql("SELECT * FROM raw_option_quotes", raw_repo._engine)
        assert len(result) == 2
        assert all(result["ingest_run_id"] == "run-1")

    def test_fred_series_upsert_on_duplicate(self, raw_repo):
        """Test that duplicate FRED series are silently skipped via upsert."""
        df = pd.DataFrame(
            {
                "series_id": ["DGS10"],
                "obs_date": [datetime(2023, 1, 1).date()],
                "value": [0.045],
                "release_datetime_utc": [datetime(2023, 1, 2, 0, 0, 0)],
            }
        )

        # First write succeeds
        raw_repo.write_fred_series(df)

        # Second write — upsert silently skips duplicate
        raw_repo.write_fred_series(df)

        # Verify only one row exists (duplicate was ignored)
        result = pd.read_sql("SELECT * FROM raw_fred_series", raw_repo._engine)
        assert len(result) == 1

    def test_ingestion_log_transaction_commits_on_success(self, raw_repo):
        """Test that ingestion log writes commit correctly."""
        log_entry = {
            "ingest_run_id": "tx-test-001",
            "dataset": "GLBX.MDP3",
            "schema": "ohlcv-1m",
            "stype_in": "raw_symbol",
            "symbols": "SPY",
            "start_date": datetime(2023, 1, 1).date(),
            "end_date": datetime(2023, 1, 31).date(),
            "row_count": 1000,
            "source_ingested_at": datetime.now(UTC),
        }

        raw_repo.write_ingestion_log(log_entry)

        # Verify data was committed
        result = pd.read_sql("SELECT * FROM raw_ingestion_log", raw_repo._engine)
        assert len(result) == 1
        assert result["ingest_run_id"].iloc[0] == "tx-test-001"
