"""Shared fixtures for ingest integration tests.

Provides:
- temp_dir: Temporary directory for test files
- config: Test configuration with temp paths
- engine: Database engine with schema (raw SA engine)
- repository: RawRepository backed by the engine
- orchestrator: IngestionOrchestrator with mock providers

These fixtures remain function-scoped because the orchestrator
uses file-based SQLite through config paths (each test gets a
fresh temp directory and database file).
"""

import tempfile
from datetime import date
from pathlib import Path

import pytest

from src.config.schema import (
    ConfigSchema,
    DataConfig,
    DatabentoIngestionConfig,
    DatasetConfig,
    DatasetSplitsConfig,
    FeaturesConfig,
    IngestionConfig,
    MacroFeaturesConfig,
    PathsConfig,
    RetryConfig,
)
from src.data.ingest.orchestrator import IngestionOrchestrator
from src.data.providers.mock import MockMacroDataProvider, MockMarketDataProvider
from src.data.storage.engine import create_engine
from src.data.storage.repository import RawRepository


@pytest.fixture
def temp_dir():
    """Create temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def config(temp_dir):
    """Create test configuration with temp paths."""
    return ConfigSchema(
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
        data=DataConfig(
            ingestion=IngestionConfig(
                databento=DatabentoIngestionConfig(
                    retry=RetryConfig(max_retries=0),
                ),
            ),
        ),
        features=FeaturesConfig(
            macro=MacroFeaturesConfig(series=["DGS10", "VIXCLS"]),
        ),
        paths=PathsConfig(
            data_dir=str(temp_dir / "data"),
            db_path=str(temp_dir / "data" / "test.db"),
            manifest_dir=str(temp_dir / "data" / "manifest"),
        ),
    )


@pytest.fixture
def engine(config):
    """Create database engine with schema."""
    db_engine = create_engine(config.paths.db_path)
    db_engine.create_tables()
    return db_engine.engine


@pytest.fixture
def repository(engine):
    """Create repository with database."""
    return RawRepository(engine)


@pytest.fixture
def orchestrator(config, repository):
    """Create orchestrator with mock providers and real database."""
    return IngestionOrchestrator(
        market_provider=MockMarketDataProvider(seed=42),
        macro_provider=MockMacroDataProvider(seed=42),
        repository=repository,
        config=config,
    )
