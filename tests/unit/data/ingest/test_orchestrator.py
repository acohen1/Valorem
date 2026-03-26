"""Unit tests for IngestionOrchestrator."""

import tempfile
from datetime import date, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.config.schema import (
    ConfigSchema,
    DatabentoIngestionConfig,
    DataConfig,
    DatasetConfig,
    DatasetSplitsConfig,
    FeaturesConfig,
    IngestionConfig,
    MacroFeaturesConfig,
    RetryConfig,
)
from src.data.ingest.manifest import Manifest, ManifestMetadata
from src.data.ingest.orchestrator import (
    CostExceededException,
    IngestionOrchestrator,
    IngestionResult,
)
from src.data.providers.mock import MockMacroDataProvider, MockMarketDataProvider


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def config():
    """Create test configuration.

    Retry is disabled (max_retries=0) by default so that tests expecting
    failures don't trigger real sleep delays.  Retry-specific tests create
    their own config with retries enabled.
    """
    return ConfigSchema(
        data=DataConfig(
            ingestion=IngestionConfig(
                databento=DatabentoIngestionConfig(
                    retry=RetryConfig(max_retries=0),
                ),
            ),
        ),
        dataset=DatasetConfig(
            splits=DatasetSplitsConfig(
                train_start=date(2020, 1, 1),
                train_end=date(2022, 12, 31),
                val_start=date(2023, 1, 1),
                val_end=date(2023, 6, 30),
                test_start=date(2023, 7, 1),
                test_end=date(2023, 12, 31),
            ),
            min_dte=7,
            max_dte=90,
        ),
        backtest={"start_date": date(2023, 7, 1), "end_date": date(2023, 12, 31)},
        features=FeaturesConfig(
            macro=MacroFeaturesConfig(series=["DGS10", "VIXCLS"]),
        ),
    )


@pytest.fixture
def mock_market_provider():
    """Create mock market data provider."""
    return MockMarketDataProvider(seed=42)


@pytest.fixture
def mock_macro_provider():
    """Create mock macro data provider."""
    return MockMacroDataProvider(seed=42)


@pytest.fixture
def mock_repository():
    """Create mock repository."""
    repo = MagicMock()
    repo.write_underlying_bars = MagicMock()
    repo.write_option_quotes = MagicMock()
    repo.write_option_bars = MagicMock()
    repo.write_fred_series = MagicMock()
    repo.write_ingestion_log = MagicMock()
    # Default: no existing data (count methods return 0)
    repo.count_underlying_bars = MagicMock(return_value=0)
    repo.count_option_quotes = MagicMock(return_value=0)
    repo.count_option_bars = MagicMock(return_value=0)
    repo.count_fred_series = MagicMock(return_value=0)
    return repo


@pytest.fixture
def orchestrator(config, mock_market_provider, mock_macro_provider, mock_repository):
    """Create orchestrator with mock providers."""
    return IngestionOrchestrator(
        market_provider=mock_market_provider,
        macro_provider=mock_macro_provider,
        repository=mock_repository,
        config=config,
    )


# ============================================================================
# IngestionResult Tests
# ============================================================================


class TestIngestionResult:
    """Tests for IngestionResult dataclass."""

    def test_total_rows(self):
        """Test total_rows property."""
        result = IngestionResult(
            run_id="test",
            start_time=datetime.now(),
            end_time=datetime.now(),
            underlying_rows=100,
            option_rows=1000,
            macro_rows=50,
        )

        assert result.total_rows == 1150

    def test_success_property(self):
        """Test success property."""
        result = IngestionResult(
            run_id="test",
            start_time=datetime.now(),
            end_time=datetime.now(),
            errors=[],
        )
        assert result.success

    def test_failure_with_errors(self):
        """Test success is False when errors present."""
        result = IngestionResult(
            run_id="test",
            start_time=datetime.now(),
            end_time=datetime.now(),
            errors=["Test error"],
        )
        assert not result.success

    def test_preview_not_success(self):
        """Test preview mode is not considered success."""
        result = IngestionResult(
            run_id="test",
            start_time=datetime.now(),
            end_time=datetime.now(),
            preview_only=True,
        )
        assert not result.success

    def test_summary_success(self):
        """Test summary for successful result."""
        result = IngestionResult(
            run_id="test-run-id-123",
            start_time=datetime.now(),
            end_time=datetime.now(),
            underlying_rows=100,
            option_rows=1000,
            macro_rows=50,
            actual_cost=1.50,
        )
        summary = result.summary()

        assert "SUCCESS" in summary
        assert "test-run" in summary
        assert "1150" in summary

    def test_summary_failure(self):
        """Test summary for failed result."""
        result = IngestionResult(
            run_id="test-run-id",
            start_time=datetime.now(),
            end_time=datetime.now(),
            errors=["Test error"],
        )
        summary = result.summary()

        assert "FAILED" in summary

    def test_summary_preview(self):
        """Test summary for preview result."""
        result = IngestionResult(
            run_id="test",
            start_time=datetime.now(),
            end_time=datetime.now(),
            preview_only=True,
            estimated_cost=5.0,
        )
        summary = result.summary()

        assert "Preview" in summary
        assert "5.00" in summary


# ============================================================================
# Orchestrator Initialization Tests
# ============================================================================


class TestOrchestratorInitialization:
    """Tests for orchestrator initialization."""

    def test_initialization(self, orchestrator):
        """Test orchestrator initializes correctly."""
        assert orchestrator._market is not None
        assert orchestrator._macro is not None
        assert orchestrator._repo is not None
        assert orchestrator._config is not None
        assert orchestrator._validator is not None
        assert orchestrator._manifest_generator is not None


# ============================================================================
# Run ID Generation Tests
# ============================================================================


class TestRunIdGeneration:
    """Tests for run ID generation."""

    def test_generate_run_id_is_uuid(self, orchestrator):
        """Test run ID is a valid UUID."""
        run_id = orchestrator._generate_run_id()

        assert len(run_id) == 36  # UUID format
        assert run_id.count("-") == 4

    def test_generate_run_id_unique(self, orchestrator):
        """Test run IDs are unique."""
        run_ids = [orchestrator._generate_run_id() for _ in range(100)]
        assert len(set(run_ids)) == 100


# ============================================================================
# Cost Estimation Tests
# ============================================================================


# ============================================================================
# Preview Mode Tests
# ============================================================================


class TestPreviewMode:
    """Tests for preview mode."""

    def test_preview_only_returns_estimate(self, orchestrator, mock_repository):
        """Test preview mode returns cost estimate without fetching data."""
        start = datetime(2024, 1, 2)
        end = datetime(2024, 1, 3)

        result = orchestrator.run_ingestion(
            start=start,
            end=end,
            preview_only=True,
        )

        assert result.preview_only
        assert result.estimated_cost >= 0
        assert result.underlying_rows == 0
        assert result.option_rows == 0

        # Should not have written to repository
        mock_repository.write_underlying_bars.assert_not_called()
        mock_repository.write_option_quotes.assert_not_called()
        mock_repository.write_fred_series.assert_not_called()


# ============================================================================
# Cost Limit Tests
# ============================================================================


class TestCostLimits:
    """Tests for cost limit enforcement."""

    def test_cost_exceeds_limit_raises(self, config, mock_macro_provider, mock_repository, tmp_path):
        """Test that exceeding cost limit raises exception."""
        # Isolate manifest directory so no on-disk manifest is loaded
        config.paths.manifest_dir = str(tmp_path / "manifest")

        # Create mock provider that returns high cost
        mock_market = MagicMock()
        mock_market.estimate_cost.return_value = 1000.0  # Way over limit
        mock_market.fetch_underlying_bars.return_value = pd.DataFrame()
        mock_market.resolve_option_symbols.return_value = []

        orchestrator = IngestionOrchestrator(
            market_provider=mock_market,
            macro_provider=mock_macro_provider,
            repository=mock_repository,
            config=config,
        )

        with pytest.raises(CostExceededException) as exc_info:
            orchestrator.run_ingestion(
                start=datetime(2024, 1, 2),
                end=datetime(2024, 1, 3),
            )

        assert "1000.00" in str(exc_info.value)
        assert "exceeds limit" in str(exc_info.value)


# ============================================================================
# Data Validation Tests
# ============================================================================


class TestDataValidation:
    """Tests for data validation."""

    def test_validation_failure_skips_bad_data(self, config, mock_macro_provider, mock_repository):
        """Test that validation failure skips writing bad data (warns, doesn't raise)."""
        # Create mock provider that returns invalid data
        mock_market = MagicMock()
        mock_market.estimate_cost.return_value = 0.0
        mock_market.resolve_option_symbols.return_value = []

        # Create invalid underlying data (high < open)
        invalid_df = pd.DataFrame({
            "ts_utc": [datetime(2024, 1, 2, 9, 30)],
            "open": [100.0],
            "high": [50.0],  # Invalid: high < open
            "low": [99.0],
            "close": [100.0],
            "volume": [1000],
        })
        mock_market.fetch_underlying_bars.return_value = invalid_df

        orchestrator = IngestionOrchestrator(
            market_provider=mock_market,
            macro_provider=mock_macro_provider,
            repository=mock_repository,
            config=config,
        )

        # Orchestrator now warns and skips bad data instead of raising
        result = orchestrator.run_ingestion(
            start=datetime(2024, 1, 2),
            end=datetime(2024, 1, 3),
        )

        assert result is not None

    def test_skip_validation_bypasses_checks(self, config, mock_macro_provider, mock_repository):
        """Test that skip_validation bypasses quality checks."""
        # Create mock provider that returns invalid data
        mock_market = MagicMock()
        mock_market.estimate_cost.return_value = 0.0
        mock_market.resolve_option_symbols.return_value = []

        # Create invalid underlying data
        invalid_df = pd.DataFrame({
            "ts_utc": [datetime(2024, 1, 2, 9, 30)],
            "open": [100.0],
            "high": [50.0],  # Invalid
            "low": [99.0],
            "close": [100.0],
            "volume": [1000],
        })
        mock_market.fetch_underlying_bars.return_value = invalid_df

        orchestrator = IngestionOrchestrator(
            market_provider=mock_market,
            macro_provider=mock_macro_provider,
            repository=mock_repository,
            config=config,
        )

        # Should not raise with skip_validation=True
        result = orchestrator.run_ingestion(
            start=datetime(2024, 1, 2),
            end=datetime(2024, 1, 3),
            skip_validation=True,
        )

        assert result.underlying_rows == 1


# ============================================================================
# Manifest Tests
# ============================================================================


class TestManifestHandling:
    """Tests for manifest loading and generation."""

    def test_generates_manifest_when_not_exists(self, orchestrator):
        """Test manifest is generated when none exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Point config to temp directory
            orchestrator._config.paths.manifest_dir = tmpdir

            result = orchestrator.run_ingestion(
                start=datetime(2024, 1, 2),
                end=datetime(2024, 1, 3),
            )

            # Should have generated a manifest
            manifest_files = list(Path(tmpdir).glob("*.json"))
            assert len(manifest_files) == 1

    def test_loads_existing_manifest(self, orchestrator):
        """Test existing manifest is loaded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            orchestrator._config.paths.manifest_dir = tmpdir

            # Generate manifest on first run
            orchestrator.run_ingestion(
                start=datetime(2024, 1, 2),
                end=datetime(2024, 1, 3),
            )

            # Mock load_manifest to track calls
            original_load = orchestrator._manifest_generator.load_manifest
            orchestrator._manifest_generator.load_manifest = MagicMock(
                side_effect=original_load
            )

            # Run again - should load existing manifest
            orchestrator.run_ingestion(
                start=datetime(2024, 1, 2),
                end=datetime(2024, 1, 3),
            )

            orchestrator._manifest_generator.load_manifest.assert_called()


# ============================================================================
# Full Ingestion Tests
# ============================================================================


class TestFullIngestion:
    """Tests for full ingestion workflow."""

    def test_full_ingestion_success(self, orchestrator, mock_repository):
        """Test successful full ingestion."""
        start = datetime(2024, 1, 2)
        end = datetime(2024, 1, 3)

        result = orchestrator.run_ingestion(
            start=start,
            end=end,
        )

        assert result.success
        assert result.run_id is not None
        assert result.underlying_rows > 0
        assert len(result.errors) == 0

        # Verify repository writes were called
        mock_repository.write_underlying_bars.assert_called_once()
        mock_repository.write_ingestion_log.assert_called_once()

    def test_ingestion_captures_errors(self, config, mock_macro_provider, mock_repository):
        """Test that errors are captured in result."""
        # Create mock provider that fails
        mock_market = MagicMock()
        mock_market.estimate_cost.return_value = 0.0
        mock_market.resolve_option_symbols.return_value = []
        mock_market.fetch_underlying_bars.side_effect = RuntimeError("API error")

        orchestrator = IngestionOrchestrator(
            market_provider=mock_market,
            macro_provider=mock_macro_provider,
            repository=mock_repository,
            config=config,
        )

        result = orchestrator.run_ingestion(
            start=datetime(2024, 1, 2),
            end=datetime(2024, 1, 3),
            skip_validation=True,
        )

        assert not result.success
        assert len(result.errors) > 0
        assert "API error" in result.errors[0]


# ============================================================================
# Data Preparation Tests
# ============================================================================


class TestDataPreparation:
    """Tests for data preparation methods."""

    def test_prepare_underlying_bars_adds_metadata(self, orchestrator):
        """Test prepare_underlying_bars adds required metadata."""
        df = pd.DataFrame({
            "ts_utc": [datetime(2024, 1, 2, 9, 30)],
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.5],
            "volume": [1000],
        })

        prepared = orchestrator._prepare_underlying_bars(df)

        assert "dataset" in prepared.columns
        assert "schema" in prepared.columns
        assert "stype_in" in prepared.columns
        assert "symbol" in prepared.columns
        assert "timeframe" in prepared.columns

    def test_prepare_option_quotes_adds_metadata(self, orchestrator):
        """Test prepare_option_quotes adds required metadata."""
        df = pd.DataFrame({
            "ts_utc": [datetime(2024, 1, 2, 9, 30)],
            "option_symbol": ["SPY240119C00450000"],
            "bid": [5.0],
            "ask": [5.10],
        })

        prepared = orchestrator._prepare_option_quotes(df)

        assert "dataset" in prepared.columns
        assert "schema" in prepared.columns
        assert "stype_in" in prepared.columns


# ============================================================================
# Log Entry Tests
# ============================================================================


class TestLogEntry:
    """Tests for ingestion log entry building."""

    def test_build_log_entry_contains_required_fields(self, orchestrator):
        """Test log entry contains all required fields."""
        from src.data.ingest.manifest import Manifest, ManifestMetadata

        metadata = ManifestMetadata(
            as_of_ts_utc=datetime.now(),
            spot_reference=450.0,
            dte_min=7,
            dte_max=90,
            moneyness_min=0.8,
            moneyness_max=1.2,
            options_per_expiry_side=50,
            underlying="SPY",
            generated_at=datetime.now(),
            config_hash="abc123",
            symbols_count=100,
            expiries_count=10,
        )
        manifest = Manifest(symbols=[], metadata=metadata)

        log_entry = orchestrator._build_log_entry(
            run_id="test-run-id",
            start=datetime(2024, 1, 2),
            end=datetime(2024, 1, 3),
            underlying_rows=100,
            option_rows=1000,
            macro_rows=50,
            cost=1.50,
            manifest=manifest,
        )

        assert log_entry["ingest_run_id"] == "test-run-id"
        assert log_entry["row_count"] == 1150
        assert log_entry["cost_usd"] == 1.50
        assert "git_sha" in log_entry
        assert "config_snapshot" in log_entry
        # Verify config_snapshot contains manifest info
        import json
        config_snapshot = json.loads(log_entry["config_snapshot"])
        assert "manifest_hash" in config_snapshot

    def test_get_git_sha_handles_missing_git(self, orchestrator):
        """Test _get_git_sha handles missing git gracefully."""
        with patch("subprocess.run", side_effect=FileNotFoundError):
            sha = orchestrator._get_git_sha()
            assert sha == "unknown"

    def test_get_git_sha_handles_error(self, orchestrator):
        """Test _get_git_sha handles command errors."""
        mock_result = MagicMock()
        mock_result.returncode = 1

        with patch("subprocess.run", return_value=mock_result):
            sha = orchestrator._get_git_sha()
            assert sha == "unknown"


# ============================================================================
# Edge Cases
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_option_symbols(self, config, mock_macro_provider, mock_repository):
        """Test ingestion with no option symbols."""
        mock_market = MagicMock()
        mock_market.estimate_cost.return_value = 0.0
        mock_market.resolve_option_symbols.return_value = []  # No symbols
        mock_market.fetch_underlying_bars.return_value = pd.DataFrame({
            "ts_utc": [datetime(2024, 1, 2, 9, 30)],
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.5],
            "volume": [1000],
        })
        mock_market.fetch_option_quotes.return_value = pd.DataFrame()

        orchestrator = IngestionOrchestrator(
            market_provider=mock_market,
            macro_provider=mock_macro_provider,
            repository=mock_repository,
            config=config,
        )

        result = orchestrator.run_ingestion(
            start=datetime(2024, 1, 2),
            end=datetime(2024, 1, 3),
        )

        # Should complete successfully with no options
        assert result.success
        assert result.option_rows == 0

    def test_filter_symbols_for_chunk_drops_expired(self, orchestrator):
        """Test that expired symbols are filtered out for later chunks."""
        metadata = ManifestMetadata(
            as_of_ts_utc=datetime(2023, 4, 1),
            spot_reference=400.0,
            dte_min=7,
            dte_max=90,
            moneyness_min=0.8,
            moneyness_max=1.2,
            options_per_expiry_side=50,
            underlying="SPY",
            generated_at=datetime(2023, 4, 1),
            config_hash="test",
            symbols_count=4,
            expiries_count=2,
        )
        manifest = Manifest(
            symbols=[
                "SPY230410C00400000",
                "SPY230410P00400000",
                "SPY230519C00400000",
                "SPY230519P00400000",
            ],
            metadata=metadata,
            symbols_by_expiry={
                "2023-04-10": ["SPY230410C00400000", "SPY230410P00400000"],
                "2023-05-19": ["SPY230519C00400000", "SPY230519P00400000"],
            },
        )

        # April chunk: all symbols should be included
        april_symbols = orchestrator._filter_symbols_for_chunk(manifest, date(2023, 4, 1))
        assert len(april_symbols) == 4

        # May chunk: April-expiry symbols should be dropped
        may_symbols = orchestrator._filter_symbols_for_chunk(manifest, date(2023, 5, 1))
        assert len(may_symbols) == 2
        assert all("230519" in s for s in may_symbols)

    def test_filter_symbols_for_chunk_no_expiry_data(self, orchestrator):
        """Test fallback when symbols_by_expiry is empty."""
        metadata = ManifestMetadata(
            as_of_ts_utc=datetime(2023, 4, 1),
            spot_reference=400.0,
            dte_min=7,
            dte_max=90,
            moneyness_min=0.8,
            moneyness_max=1.2,
            options_per_expiry_side=50,
            underlying="SPY",
            generated_at=datetime(2023, 4, 1),
            config_hash="test",
            symbols_count=2,
            expiries_count=0,
        )
        manifest = Manifest(
            symbols=["SPY230410C00400000", "SPY230519C00400000"],
            metadata=metadata,
            symbols_by_expiry={},
        )

        # With no expiry data, all symbols should be returned (fallback)
        result = orchestrator._filter_symbols_for_chunk(manifest, date(2023, 5, 1))
        assert len(result) == 2

    def test_filter_symbols_for_chunk_all_expired(self, orchestrator):
        """Test that all-expired chunk returns empty list."""
        metadata = ManifestMetadata(
            as_of_ts_utc=datetime(2023, 4, 1),
            spot_reference=400.0,
            dte_min=7,
            dte_max=90,
            moneyness_min=0.8,
            moneyness_max=1.2,
            options_per_expiry_side=50,
            underlying="SPY",
            generated_at=datetime(2023, 4, 1),
            config_hash="test",
            symbols_count=2,
            expiries_count=1,
        )
        manifest = Manifest(
            symbols=["SPY230410C00400000", "SPY230410P00400000"],
            metadata=metadata,
            symbols_by_expiry={
                "2023-04-10": ["SPY230410C00400000", "SPY230410P00400000"],
            },
        )

        # All symbols expired before June
        result = orchestrator._filter_symbols_for_chunk(manifest, date(2023, 6, 1))
        assert len(result) == 0

    def test_skip_existing_underlying_bars(self, config, mock_macro_provider, mock_repository):
        """Test that existing underlying bars are still re-fetched for integrity."""
        mock_market = MagicMock()
        mock_market.estimate_cost.return_value = 0.0
        mock_market.resolve_option_symbols.return_value = []
        mock_market.fetch_underlying_bars.return_value = pd.DataFrame({
            "ts_utc": pd.date_range("2024-01-02", periods=5, freq="min"),
            "open": [100.0] * 5,
            "high": [101.0] * 5,
            "low": [99.0] * 5,
            "close": [100.5] * 5,
            "volume": [1000] * 5,
        })

        # Repo says 1000 rows exist for underlying
        mock_repository.count_underlying_bars.return_value = 1000
        mock_repository.read_underlying_bars.return_value = pd.DataFrame({
            "ts_utc": pd.date_range("2024-01-02", periods=5, freq="min"),
            "open": [100.0] * 5,
            "high": [101.0] * 5,
            "low": [99.0] * 5,
            "close": [100.5] * 5,
            "volume": [1000] * 5,
        })
        mock_repository.count_option_quotes.return_value = 0
        mock_repository.count_fred_series.return_value = 0

        orchestrator = IngestionOrchestrator(
            market_provider=mock_market,
            macro_provider=mock_macro_provider,
            repository=mock_repository,
            config=config,
        )

        result = orchestrator.run_ingestion(
            start=datetime(2024, 1, 2),
            end=datetime(2024, 1, 3),
            skip_validation=True,
        )

        # fetch_underlying_bars is called once by _detect_max_available_date probe,
        # plus once for actual fetch (non-empty chunks are no longer skipped).
        assert mock_market.fetch_underlying_bars.call_count == 2
        mock_repository.read_underlying_bars.assert_not_called()
        assert not result.skipped_underlying
        assert result.underlying_rows == 5

    def test_skip_existing_option_chunks(self, config, mock_macro_provider, mock_repository, tmp_path):
        """Test that option chunks are fetched even when rows already exist."""
        # Use a symbol that expires well after the test range (March 2024)
        mock_market = MagicMock()
        mock_market.estimate_cost.return_value = 0.0
        mock_market.resolve_option_symbols.return_value = ["SPY240315C00450000"]
        mock_market.fetch_underlying_bars.return_value = pd.DataFrame({
            "ts_utc": [datetime(2024, 1, 2)],
            "close": [450.0],
        })
        mock_market.fetch_option_quotes.return_value = pd.DataFrame({
            "ts_utc": [datetime(2024, 2, 1, 9, 30)],
            "option_symbol": ["SPY240315C00450000"],
            "bid": [5.0],
            "ask": [5.10],
        })

        # Existing rows no longer trigger chunk skipping.
        mock_repository.count_underlying_bars.return_value = 0
        mock_repository.count_option_quotes.return_value = 500
        mock_repository.count_fred_series.return_value = 0

        config.paths.manifest_dir = str(tmp_path / "manifest")

        orchestrator = IngestionOrchestrator(
            market_provider=mock_market,
            macro_provider=mock_macro_provider,
            repository=mock_repository,
            config=config,
        )

        result = orchestrator.run_ingestion(
            start=datetime(2024, 1, 1),
            end=datetime(2024, 3, 1),
            skip_validation=True,
        )

        # fetch_option_quotes called for both Jan and Feb chunks.
        assert mock_market.fetch_option_quotes.call_count == 2
        assert result.skipped_chunks == 0

    def test_skip_existing_fred_series(self, config, mock_repository):
        """Test that existing FRED series are skipped."""
        mock_market = MagicMock()
        mock_market.estimate_cost.return_value = 0.0
        mock_market.resolve_option_symbols.return_value = []
        mock_market.fetch_underlying_bars.return_value = pd.DataFrame()

        mock_macro = MagicMock()

        # Both series already exist
        mock_repository.count_underlying_bars.return_value = 0
        mock_repository.count_fred_series.return_value = 100

        orchestrator = IngestionOrchestrator(
            market_provider=mock_market,
            macro_provider=mock_macro,
            repository=mock_repository,
            config=config,
        )

        result = orchestrator.run_ingestion(
            start=datetime(2024, 1, 2),
            end=datetime(2024, 1, 3),
            skip_validation=True,
        )

        # fetch_series should never be called
        mock_macro.fetch_series.assert_not_called()
        assert result.skipped_series == 2  # DGS10 and VIXCLS

    def test_force_overrides_skip(self, config, mock_macro_provider, mock_repository):
        """Test that force=True re-fetches even when data exists."""
        mock_market = MagicMock()
        mock_market.estimate_cost.return_value = 0.0
        mock_market.resolve_option_symbols.return_value = []
        mock_market.fetch_underlying_bars.return_value = pd.DataFrame({
            "ts_utc": [datetime(2024, 1, 2, 9, 30)],
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.5],
            "volume": [1000],
        })

        # Repo says data exists — but force=True should ignore this
        mock_repository.count_underlying_bars.return_value = 1000

        orchestrator = IngestionOrchestrator(
            market_provider=mock_market,
            macro_provider=mock_macro_provider,
            repository=mock_repository,
            config=config,
        )

        result = orchestrator.run_ingestion(
            start=datetime(2024, 1, 2),
            end=datetime(2024, 1, 3),
            skip_validation=True,
            force=True,
        )

        # Should have fetched from API despite existing data
        # fetch_underlying_bars is called three times:
        #   1. _detect_max_available_date probe
        #   2. spot reference during manifest regeneration (interval="1d")
        #   3. actual data fetch (interval="1m")
        assert mock_market.fetch_underlying_bars.call_count == 3
        # count methods should NOT have been called
        mock_repository.count_underlying_bars.assert_not_called()
        assert not result.skipped_underlying

    def test_force_regenerates_manifest(self, config, mock_macro_provider, mock_repository, tmp_path):
        """Test that force=True regenerates the manifest instead of reusing cached."""
        mock_market = MagicMock()
        mock_market.estimate_cost.return_value = 0.0
        # Original manifest had no symbols; regeneration finds new ones
        mock_market.resolve_option_symbols.return_value = ["SPY240315C00450000"]
        mock_market.fetch_underlying_bars.return_value = pd.DataFrame({
            "ts_utc": [datetime(2024, 1, 2, 9, 30)],
            "open": [450.0],
            "high": [451.0],
            "low": [449.0],
            "close": [450.0],
            "volume": [1000],
        })
        mock_market.fetch_option_quotes.return_value = pd.DataFrame()

        config.paths.manifest_dir = str(tmp_path / "manifest")

        orchestrator = IngestionOrchestrator(
            market_provider=mock_market,
            macro_provider=mock_macro_provider,
            repository=mock_repository,
            config=config,
        )

        # First run: generates and caches manifest
        orchestrator.run_ingestion(
            start=datetime(2024, 1, 2),
            end=datetime(2024, 1, 3),
            skip_validation=True,
        )

        # Second run without force: reuses cached manifest, no resolve call
        mock_market.resolve_option_symbols.reset_mock()
        orchestrator.run_ingestion(
            start=datetime(2024, 1, 2),
            end=datetime(2024, 1, 3),
            skip_validation=True,
        )
        mock_market.resolve_option_symbols.assert_not_called()

        # Third run with force: should regenerate manifest
        mock_market.resolve_option_symbols.reset_mock()
        orchestrator.run_ingestion(
            start=datetime(2024, 1, 2),
            end=datetime(2024, 1, 3),
            skip_validation=True,
            force=True,
        )
        mock_market.resolve_option_symbols.assert_called_once()

    def test_skip_summary_message(self):
        """Test summary includes skip information."""
        result = IngestionResult(
            run_id="test-run-id-123",
            start_time=datetime.now(),
            end_time=datetime.now(),
            underlying_rows=100,
            option_rows=0,
            macro_rows=0,
            skipped_underlying=True,
            skipped_chunks=3,
            skipped_series=2,
        )
        summary = result.summary()
        assert "skipped=" in summary
        assert "underlying" in summary
        assert "3 option chunks" in summary
        assert "2 FRED series" in summary

    def test_macro_fetch_failure_captured(self, config, mock_market_provider, mock_repository):
        """Test macro fetch failures are captured."""
        mock_macro = MagicMock()
        mock_macro.fetch_series.side_effect = RuntimeError("FRED API error")

        orchestrator = IngestionOrchestrator(
            market_provider=mock_market_provider,
            macro_provider=mock_macro,
            repository=mock_repository,
            config=config,
        )

        result = orchestrator.run_ingestion(
            start=datetime(2024, 1, 2),
            end=datetime(2024, 1, 3),
            skip_validation=True,
        )

        assert not result.success
        assert any("FRED" in e for e in result.errors)


# ============================================================================
# Retry Tests
# ============================================================================


def _config_with_retries(max_retries=3, base_delay=1.0, max_delay=5.0):
    """Create a config with retry enabled (short delays for testing)."""
    return ConfigSchema(
        data=DataConfig(
            ingestion=IngestionConfig(
                databento=DatabentoIngestionConfig(
                    retry=RetryConfig(
                        max_retries=max_retries,
                        base_delay_seconds=base_delay,
                        max_delay_seconds=max_delay,
                    ),
                ),
            ),
        ),
        dataset=DatasetConfig(
            splits=DatasetSplitsConfig(
                train_start=date(2020, 1, 1),
                train_end=date(2022, 12, 31),
                val_start=date(2023, 1, 1),
                val_end=date(2023, 6, 30),
                test_start=date(2023, 7, 1),
                test_end=date(2023, 12, 31),
            ),
            min_dte=7,
            max_dte=90,
        ),
        backtest={"start_date": date(2023, 7, 1), "end_date": date(2023, 12, 31)},
        features=FeaturesConfig(
            macro=MacroFeaturesConfig(series=["DGS10", "VIXCLS"]),
        ),
    )


class TestRetryLogic:
    """Tests for retry with exponential backoff."""

    def test_retry_succeeds_on_second_attempt(self, mock_repository, tmp_path):
        """Option quotes fetch retries and succeeds on second attempt."""
        config = _config_with_retries(max_retries=2)
        config.paths.manifest_dir = str(tmp_path / "manifest")

        mock_market = MagicMock()
        mock_market.estimate_cost.return_value = 0.0
        mock_market.resolve_option_symbols.return_value = ["SPY240315C00450000"]
        mock_market.fetch_underlying_bars.return_value = pd.DataFrame({
            "ts_utc": [datetime(2024, 1, 2)],
            "close": [450.0],
        })
        # First call raises, second succeeds
        success_df = pd.DataFrame({
            "ts_utc": [datetime(2024, 1, 2, 9, 30)],
            "option_symbol": ["SPY240315C00450000"],
            "bid": [5.0], "ask": [5.10],
        })
        mock_market.fetch_option_quotes.side_effect = [
            RuntimeError("Read timed out"),
            success_df,
        ]

        mock_macro = MagicMock()
        mock_macro.fetch_series.return_value = pd.DataFrame()

        orchestrator = IngestionOrchestrator(
            market_provider=mock_market,
            macro_provider=mock_macro,
            repository=mock_repository,
            config=config,
        )

        with patch("src.data.ingest.orchestrator.time.sleep"):
            result = orchestrator.run_ingestion(
                start=datetime(2024, 1, 2),
                end=datetime(2024, 1, 3),
                skip_validation=True,
            )

        # Should have called fetch_option_quotes twice (1 fail + 1 success)
        assert mock_market.fetch_option_quotes.call_count == 2
        assert result.option_rows == 1
        assert len(result.failed_chunks) == 0
        assert not any("Options chunk error" in e for e in result.errors)

    def test_retry_exhausted_records_failure(self, mock_repository, tmp_path):
        """Chunk is recorded as failed after all retries exhausted."""
        config = _config_with_retries(max_retries=2)
        config.paths.manifest_dir = str(tmp_path / "manifest")

        mock_market = MagicMock()
        mock_market.estimate_cost.return_value = 0.0
        mock_market.resolve_option_symbols.return_value = ["SPY240315C00450000"]
        mock_market.fetch_underlying_bars.return_value = pd.DataFrame({
            "ts_utc": [datetime(2024, 1, 2)],
            "close": [450.0],
        })
        # All attempts fail
        mock_market.fetch_option_quotes.side_effect = RuntimeError("Read timed out")

        mock_macro = MagicMock()
        mock_macro.fetch_series.return_value = pd.DataFrame()

        orchestrator = IngestionOrchestrator(
            market_provider=mock_market,
            macro_provider=mock_macro,
            repository=mock_repository,
            config=config,
        )

        with patch("src.data.ingest.orchestrator.time.sleep"):
            result = orchestrator.run_ingestion(
                start=datetime(2024, 1, 2),
                end=datetime(2024, 1, 3),
                skip_validation=True,
            )

        # 3 total attempts (1 initial + 2 retries)
        assert mock_market.fetch_option_quotes.call_count == 3
        assert result.option_rows == 0
        assert len(result.failed_chunks) == 1
        assert any("Options chunk error" in e for e in result.errors)

    def test_retry_backoff_timing(self, mock_repository, tmp_path):
        """Verify exponential backoff delays: base * 2^attempt."""
        config = _config_with_retries(max_retries=3, base_delay=30.0, max_delay=300.0)
        config.paths.manifest_dir = str(tmp_path / "manifest")

        mock_market = MagicMock()
        mock_market.estimate_cost.return_value = 0.0
        mock_market.resolve_option_symbols.return_value = ["SPY240315C00450000"]
        mock_market.fetch_underlying_bars.return_value = pd.DataFrame({
            "ts_utc": [datetime(2024, 1, 2)],
            "close": [450.0],
        })
        mock_market.fetch_option_quotes.side_effect = RuntimeError("timeout")

        mock_macro = MagicMock()
        mock_macro.fetch_series.return_value = pd.DataFrame()

        orchestrator = IngestionOrchestrator(
            market_provider=mock_market,
            macro_provider=mock_macro,
            repository=mock_repository,
            config=config,
        )

        with patch("src.data.ingest.orchestrator.time.sleep") as mock_sleep:
            orchestrator.run_ingestion(
                start=datetime(2024, 1, 2),
                end=datetime(2024, 1, 3),
                skip_validation=True,
            )

        # 3 retries → 3 sleep calls with exponential delays
        assert mock_sleep.call_count == 3
        delays = [call.args[0] for call in mock_sleep.call_args_list]
        assert delays == [30.0, 60.0, 120.0]

    def test_retry_max_delay_cap(self, mock_repository, tmp_path):
        """Verify delay is capped at max_delay_seconds."""
        config = _config_with_retries(max_retries=3, base_delay=100.0, max_delay=200.0)
        config.paths.manifest_dir = str(tmp_path / "manifest")

        mock_market = MagicMock()
        mock_market.estimate_cost.return_value = 0.0
        mock_market.resolve_option_symbols.return_value = ["SPY240315C00450000"]
        mock_market.fetch_underlying_bars.return_value = pd.DataFrame({
            "ts_utc": [datetime(2024, 1, 2)],
            "close": [450.0],
        })
        mock_market.fetch_option_quotes.side_effect = RuntimeError("timeout")

        mock_macro = MagicMock()
        mock_macro.fetch_series.return_value = pd.DataFrame()

        orchestrator = IngestionOrchestrator(
            market_provider=mock_market,
            macro_provider=mock_macro,
            repository=mock_repository,
            config=config,
        )

        with patch("src.data.ingest.orchestrator.time.sleep") as mock_sleep:
            orchestrator.run_ingestion(
                start=datetime(2024, 1, 2),
                end=datetime(2024, 1, 3),
                skip_validation=True,
            )

        delays = [call.args[0] for call in mock_sleep.call_args_list]
        # 100*1=100, 100*2=200, 100*4=400→capped at 200
        assert delays == [100.0, 200.0, 200.0]

    def test_retry_underlying_bars(self, mock_repository):
        """Underlying bars fetch is also retried."""
        config = _config_with_retries(max_retries=1)
        mock_market = MagicMock()
        mock_market.estimate_cost.return_value = 0.0
        mock_market.resolve_option_symbols.return_value = []

        success_df = pd.DataFrame({
            "ts_utc": [datetime(2024, 1, 2, 9, 30)],
            "open": [450.0], "high": [451.0], "low": [449.0],
            "close": [450.0], "volume": [1000],
        })
        # For underlying: first call is for manifest (1d), fails;
        # but manifest generation is separate from the main fetch.
        # The main underlying fetch at Step 3 will fail once then succeed.
        mock_market.fetch_underlying_bars.side_effect = [
            success_df,  # manifest generation (1d bars) — succeeds
            RuntimeError("connection reset"),  # main fetch attempt 1
            success_df,  # main fetch attempt 2 (retry)
        ]

        mock_macro = MagicMock()
        mock_macro.fetch_series.return_value = pd.DataFrame()

        orchestrator = IngestionOrchestrator(
            market_provider=mock_market,
            macro_provider=mock_macro,
            repository=mock_repository,
            config=config,
        )

        with patch("src.data.ingest.orchestrator.time.sleep"):
            result = orchestrator.run_ingestion(
                start=datetime(2024, 1, 2),
                end=datetime(2024, 1, 3),
                skip_validation=True,
            )

        assert result.underlying_rows == 1
        assert not any("Underlying fetch error" in e for e in result.errors)

    def test_retry_fred_series(self, mock_repository):
        """FRED series fetch is also retried."""
        config = _config_with_retries(max_retries=1)
        mock_market = MagicMock()
        mock_market.estimate_cost.return_value = 0.0
        mock_market.resolve_option_symbols.return_value = []
        mock_market.fetch_underlying_bars.return_value = pd.DataFrame()

        mock_macro = MagicMock()
        fred_df = pd.DataFrame({
            "obs_date": [date(2024, 1, 2)],
            "value": [4.5],
            "release_datetime_utc": [datetime(2024, 1, 3)],
        })
        # DGS10: fail then succeed; VIXCLS: succeed immediately
        mock_macro.fetch_series.side_effect = [
            RuntimeError("FRED timeout"),  # DGS10 attempt 1
            fred_df,                        # DGS10 attempt 2 (retry)
            fred_df,                        # VIXCLS attempt 1
        ]

        orchestrator = IngestionOrchestrator(
            market_provider=mock_market,
            macro_provider=mock_macro,
            repository=mock_repository,
            config=config,
        )

        with patch("src.data.ingest.orchestrator.time.sleep"):
            result = orchestrator.run_ingestion(
                start=datetime(2024, 1, 2),
                end=datetime(2024, 1, 3),
                skip_validation=True,
            )

        assert result.macro_rows == 2  # 1 DGS10 + 1 VIXCLS
        assert not any("FRED" in e for e in result.errors)

    def test_no_retry_when_max_retries_zero(self, mock_repository, tmp_path):
        """With max_retries=0, failure is immediate with no sleep."""
        config = _config_with_retries(max_retries=0)
        config.paths.manifest_dir = str(tmp_path / "manifest")

        mock_market = MagicMock()
        mock_market.estimate_cost.return_value = 0.0
        mock_market.resolve_option_symbols.return_value = ["SPY240315C00450000"]
        mock_market.fetch_underlying_bars.return_value = pd.DataFrame({
            "ts_utc": [datetime(2024, 1, 2)],
            "close": [450.0],
        })
        mock_market.fetch_option_quotes.side_effect = RuntimeError("timeout")

        mock_macro = MagicMock()
        mock_macro.fetch_series.return_value = pd.DataFrame()

        orchestrator = IngestionOrchestrator(
            market_provider=mock_market,
            macro_provider=mock_macro,
            repository=mock_repository,
            config=config,
        )

        with patch("src.data.ingest.orchestrator.time.sleep") as mock_sleep:
            result = orchestrator.run_ingestion(
                start=datetime(2024, 1, 2),
                end=datetime(2024, 1, 3),
                skip_validation=True,
            )

        # Only 1 attempt, no retries, no sleeps
        assert mock_market.fetch_option_quotes.call_count == 1
        mock_sleep.assert_not_called()
        assert len(result.failed_chunks) == 1

    def test_failed_chunks_summary(self):
        """Test summary includes failed chunk count."""
        result = IngestionResult(
            run_id="test-run-id-123",
            start_time=datetime.now(),
            end_time=datetime.now(),
            underlying_rows=100,
            option_rows=500,
            macro_rows=10,
            failed_chunks=["2024-07-01 to 2024-08-01"],
        )
        summary = result.summary()
        assert "failed_chunks=1" in summary
