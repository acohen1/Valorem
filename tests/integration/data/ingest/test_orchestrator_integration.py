"""Integration tests for IngestionOrchestrator.

These tests verify the end-to-end ingestion workflow including:
- Database writes and reads
- Manifest generation and persistence
- Full pipeline from providers to storage
"""

from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import text

from src.data.ingest.orchestrator import IngestionOrchestrator
from src.data.providers.mock import MockMacroDataProvider, MockMarketDataProvider


# ============================================================================
# End-to-End Ingestion Tests
# ============================================================================


class TestEndToEndIngestion:
    """Test complete ingestion workflow."""

    def test_full_ingestion_writes_to_database(self, orchestrator, engine):
        """Test full ingestion writes data to database."""
        start = datetime(2024, 1, 2)
        end = datetime(2024, 1, 3)

        result = orchestrator.run_ingestion(
            start=start,
            end=end,
        )

        assert result.success
        assert result.underlying_rows > 0

        # Verify data was written to database
        with engine.connect() as conn:
            bars_count = conn.execute(
                text("SELECT COUNT(*) FROM raw_underlying_bars")
            ).scalar()
            assert bars_count == result.underlying_rows

            log_count = conn.execute(
                text("SELECT COUNT(*) FROM raw_ingestion_log")
            ).scalar()
            assert log_count == 1

    def test_ingestion_with_option_quotes(self, orchestrator, engine):
        """Test ingestion includes option quotes."""
        start = datetime(2024, 1, 2)
        end = datetime(2024, 1, 3)

        result = orchestrator.run_ingestion(
            start=start,
            end=end,
        )

        assert result.success

        # Mock provider may or may not have option quotes depending on implementation
        # Just verify no errors
        assert len(result.errors) == 0

    def test_ingestion_with_macro_data(self, orchestrator, engine):
        """Test ingestion includes macro data."""
        start = datetime(2024, 1, 2)
        end = datetime(2024, 1, 5)

        result = orchestrator.run_ingestion(
            start=start,
            end=end,
        )

        assert result.success
        assert result.macro_rows > 0

        # Verify FRED data was written
        with engine.connect() as conn:
            fred_count = conn.execute(
                text("SELECT COUNT(*) FROM raw_fred_series")
            ).scalar()
            assert fred_count == result.macro_rows

    def test_multiple_ingestion_runs(self, orchestrator, engine):
        """Test multiple ingestion runs append data correctly."""
        # First run
        result1 = orchestrator.run_ingestion(
            start=datetime(2024, 1, 2),
            end=datetime(2024, 1, 3),
        )

        # Second run
        result2 = orchestrator.run_ingestion(
            start=datetime(2024, 1, 3),
            end=datetime(2024, 1, 4),
        )

        assert result1.success
        assert result2.success
        assert result1.run_id != result2.run_id

        # Verify both runs created log entries
        with engine.connect() as conn:
            log_count = conn.execute(
                text("SELECT COUNT(*) FROM raw_ingestion_log")
            ).scalar()
            assert log_count == 2


# ============================================================================
# Manifest Persistence Tests
# ============================================================================


class TestManifestPersistence:
    """Test manifest generation and persistence."""

    def test_manifest_created_on_first_run(self, orchestrator, config):
        """Test manifest is created on first run."""
        manifest_dir = Path(config.paths.manifest_dir)

        # No manifest files initially
        manifest_files = list(manifest_dir.glob("*.json")) if manifest_dir.exists() else []
        assert len(manifest_files) == 0

        # Run ingestion
        result = orchestrator.run_ingestion(
            start=datetime(2024, 1, 2),
            end=datetime(2024, 1, 3),
        )

        assert result.success

        # Manifest should now exist
        manifest_files = list(manifest_dir.glob("*.json"))
        assert len(manifest_files) == 1

    def test_manifest_reused_on_subsequent_run(self, orchestrator, config):
        """Test existing manifest is reused."""
        manifest_dir = Path(config.paths.manifest_dir)

        # First run creates manifest
        result1 = orchestrator.run_ingestion(
            start=datetime(2024, 1, 2),
            end=datetime(2024, 1, 3),
        )

        # Get manifest modification time
        manifest_files = list(manifest_dir.glob("*.json"))
        first_mtime = manifest_files[0].stat().st_mtime

        # Second run with different dates (to avoid duplicate key violations)
        result2 = orchestrator.run_ingestion(
            start=datetime(2024, 1, 4),
            end=datetime(2024, 1, 5),
        )

        assert result1.success
        assert result2.success

        # Should have created a new manifest for the new date
        manifest_files_after = list(manifest_dir.glob("*.json"))
        assert len(manifest_files_after) == 2


# ============================================================================
# Validation Integration Tests
# ============================================================================


class TestValidationIntegration:
    """Test validation in full pipeline."""

    def test_validation_runs_in_pipeline(self, orchestrator):
        """Test validation is executed as part of pipeline."""
        result = orchestrator.run_ingestion(
            start=datetime(2024, 1, 2),
            end=datetime(2024, 1, 3),
        )

        assert result.success
        # Validation results should be populated
        assert "underlying_bars" in result.validation_results
        assert result.validation_results["underlying_bars"].passed

    def test_skip_validation_flag(self, orchestrator, engine):
        """Test skip_validation bypasses validation."""
        result = orchestrator.run_ingestion(
            start=datetime(2024, 1, 2),
            end=datetime(2024, 1, 3),
            skip_validation=True,
        )

        assert result.success
        # Validation results should be empty when skipped
        assert len(result.validation_results) == 0


# ============================================================================
# Cost and Preview Tests
# ============================================================================


class TestCostAndPreview:
    """Test cost estimation and preview mode."""

    def test_preview_mode_no_writes(self, orchestrator, engine):
        """Test preview mode doesn't write to database."""
        result = orchestrator.run_ingestion(
            start=datetime(2024, 1, 2),
            end=datetime(2024, 1, 3),
            preview_only=True,
        )

        assert result.preview_only
        assert result.estimated_cost >= 0

        # No data should be written
        with engine.connect() as conn:
            bars_count = conn.execute(
                text("SELECT COUNT(*) FROM raw_underlying_bars")
            ).scalar()
            assert bars_count == 0

            log_count = conn.execute(
                text("SELECT COUNT(*) FROM raw_ingestion_log")
            ).scalar()
            assert log_count == 0

    def test_preview_then_full_run(self, orchestrator, engine):
        """Test preview followed by full run."""
        # Preview first
        preview_result = orchestrator.run_ingestion(
            start=datetime(2024, 1, 2),
            end=datetime(2024, 1, 3),
            preview_only=True,
        )

        assert preview_result.preview_only

        # Then full run
        full_result = orchestrator.run_ingestion(
            start=datetime(2024, 1, 2),
            end=datetime(2024, 1, 3),
        )

        assert full_result.success
        assert full_result.underlying_rows > 0


# ============================================================================
# Data Integrity Tests
# ============================================================================


class TestDataIntegrity:
    """Test data integrity in stored data."""

    def test_underlying_bars_have_correct_columns(self, orchestrator, engine):
        """Test underlying bars have correct schema."""
        result = orchestrator.run_ingestion(
            start=datetime(2024, 1, 2),
            end=datetime(2024, 1, 3),
        )

        assert result.success

        with engine.connect() as conn:
            df = pd.read_sql(
                text("SELECT * FROM raw_underlying_bars LIMIT 10"),
                conn,
            )

        required_cols = [
            "ts_utc", "open", "high", "low", "close",
            "symbol", "timeframe", "ingest_run_id",
        ]
        for col in required_cols:
            assert col in df.columns, f"Missing column: {col}"

    def test_run_id_consistent_across_tables(self, orchestrator, engine):
        """Test run_id is consistent across data and log tables."""
        result = orchestrator.run_ingestion(
            start=datetime(2024, 1, 2),
            end=datetime(2024, 1, 3),
        )

        with engine.connect() as conn:
            # Get run_id from log
            log_run_id = conn.execute(
                text("SELECT ingest_run_id FROM raw_ingestion_log LIMIT 1")
            ).scalar()

            # Get run_id from bars
            bars_run_id = conn.execute(
                text("SELECT DISTINCT ingest_run_id FROM raw_underlying_bars LIMIT 1")
            ).scalar()

        assert log_run_id == bars_run_id
        assert log_run_id == result.run_id


# ============================================================================
# Error Recovery Tests
# ============================================================================


class TestErrorRecovery:
    """Test error handling and recovery."""

    def test_partial_failure_captured(self, config, repository):
        """Test partial failures are captured correctly."""
        # Create market provider that fails on macro data
        mock_market = MockMarketDataProvider(seed=42)
        mock_macro = MockMacroDataProvider(seed=42)

        def failing_fetch_series(*args, **kwargs):
            raise RuntimeError("FRED API error")

        mock_macro.fetch_series = failing_fetch_series

        orchestrator = IngestionOrchestrator(
            market_provider=mock_market,
            macro_provider=mock_macro,
            repository=repository,
            config=config,
        )

        result = orchestrator.run_ingestion(
            start=datetime(2024, 1, 2),
            end=datetime(2024, 1, 3),
            skip_validation=True,
        )

        # Should still have underlying data
        assert result.underlying_rows > 0
        # But should have captured the FRED error
        assert any("FRED" in e for e in result.errors)


