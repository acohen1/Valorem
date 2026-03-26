"""Unit tests for IngestionService implementations.

Tests for the data ingestion layer:
- MockIngestionService (synthetic data generation)
- DatabentoIngestionService (live data streaming)
"""

from datetime import date, datetime, timedelta, timezone
from unittest.mock import MagicMock, patch
import threading
import time

import pandas as pd
import pytest

from src.config.schema import (
    BlackScholesConfig,
    DeltaBucketsConfig,
    SurfaceConfig,
    TenorBinsConfig,
    UniverseConfig,
)


@pytest.fixture
def surface_config() -> SurfaceConfig:
    """Create surface configuration for testing."""
    return SurfaceConfig(
        delta_buckets=DeltaBucketsConfig(),
        tenor_bins=TenorBinsConfig(),
        black_scholes=BlackScholesConfig(),
    )


@pytest.fixture
def universe_config() -> UniverseConfig:
    """Create universe configuration for testing."""
    return UniverseConfig(underlying="SPY")


@pytest.fixture
def mock_derived_repo():
    """Create mock DerivedRepository."""
    repo = MagicMock()
    repo.write_surface_snapshots = MagicMock()
    return repo


@pytest.fixture
def mock_raw_repo():
    """Create mock RawRepository."""
    return MagicMock()


class TestMockIngestionService:
    """Tests for MockIngestionService."""

    def test_init(
        self,
        mock_raw_repo,
        mock_derived_repo,
        surface_config,
        universe_config,
    ) -> None:
        """Test initialization."""
        from src.live.ingestion import MockIngestionService

        service = MockIngestionService(
            raw_repo=mock_raw_repo,
            derived_repo=mock_derived_repo,
            surface_config=surface_config,
            universe_config=universe_config,
        )

        assert service._underlying_price == 450.0
        assert service._base_iv == 0.20
        assert not service.is_running
        assert service.generation_count == 0

    def test_init_custom_params(
        self,
        mock_raw_repo,
        mock_derived_repo,
        surface_config,
        universe_config,
    ) -> None:
        """Test initialization with custom parameters."""
        from src.live.ingestion import MockIngestionService

        service = MockIngestionService(
            raw_repo=mock_raw_repo,
            derived_repo=mock_derived_repo,
            surface_config=surface_config,
            universe_config=universe_config,
            underlying_price=500.0,
            base_iv=0.25,
            random_walk=False,
            surface_interval_seconds=1.0,
            seed=42,
        )

        assert service._underlying_price == 500.0
        assert service._base_iv == 0.25
        assert not service._random_walk

    def test_start_stop(
        self,
        mock_raw_repo,
        mock_derived_repo,
        surface_config,
        universe_config,
    ) -> None:
        """Test start and stop lifecycle."""
        from src.live.ingestion import MockIngestionService

        service = MockIngestionService(
            raw_repo=mock_raw_repo,
            derived_repo=mock_derived_repo,
            surface_config=surface_config,
            universe_config=universe_config,
            surface_interval_seconds=0.1,
        )

        assert not service.is_running

        service.start()
        assert service.is_running

        # Wait for at least one generation
        time.sleep(0.2)

        service.stop()
        assert not service.is_running

    def test_generates_surfaces(
        self,
        mock_raw_repo,
        mock_derived_repo,
        surface_config,
        universe_config,
    ) -> None:
        """Test that surfaces are generated and persisted."""
        from src.live.ingestion import MockIngestionService

        service = MockIngestionService(
            raw_repo=mock_raw_repo,
            derived_repo=mock_derived_repo,
            surface_config=surface_config,
            universe_config=universe_config,
            surface_interval_seconds=0.05,
        )

        service.start()
        time.sleep(0.2)  # Wait for a few generations
        service.stop()

        # Verify surfaces were written
        assert mock_derived_repo.write_surface_snapshots.called
        assert service.generation_count > 0

    def test_random_walk_changes_price(
        self,
        mock_raw_repo,
        mock_derived_repo,
        surface_config,
        universe_config,
    ) -> None:
        """Test that random walk changes underlying price."""
        from src.live.ingestion import MockIngestionService

        service = MockIngestionService(
            raw_repo=mock_raw_repo,
            derived_repo=mock_derived_repo,
            surface_config=surface_config,
            universe_config=universe_config,
            underlying_price=450.0,
            random_walk=True,
            surface_interval_seconds=0.05,
            seed=42,
        )

        initial_price = service.underlying_price

        service.start()
        time.sleep(0.15)
        service.stop()

        # Price should have changed due to random walk
        assert service.underlying_price != initial_price

    def test_no_random_walk_price_stable(
        self,
        mock_raw_repo,
        mock_derived_repo,
        surface_config,
        universe_config,
    ) -> None:
        """Test that without random walk, price stays stable."""
        from src.live.ingestion import MockIngestionService

        service = MockIngestionService(
            raw_repo=mock_raw_repo,
            derived_repo=mock_derived_repo,
            surface_config=surface_config,
            universe_config=universe_config,
            underlying_price=450.0,
            random_walk=False,
            surface_interval_seconds=0.05,
        )

        initial_price = service.underlying_price

        service.start()
        time.sleep(0.15)
        service.stop()

        # Price should remain the same
        assert service.underlying_price == initial_price

    def test_double_start_warns(
        self,
        mock_raw_repo,
        mock_derived_repo,
        surface_config,
        universe_config,
        caplog,
    ) -> None:
        """Test that double start logs warning."""
        from src.live.ingestion import MockIngestionService
        import logging

        service = MockIngestionService(
            raw_repo=mock_raw_repo,
            derived_repo=mock_derived_repo,
            surface_config=surface_config,
            universe_config=universe_config,
        )

        service.start()
        with caplog.at_level(logging.WARNING):
            service.start()  # Should warn

        service.stop()

        assert "already running" in caplog.text


class TestDatabentoIngestionServiceInit:
    """Tests for DatabentoIngestionService initialization."""

    def test_init_requires_api_key(
        self,
        mock_raw_repo,
        mock_derived_repo,
        surface_config,
        universe_config,
    ) -> None:
        """Test initialization fails without API key."""
        from src.live.ingestion import DatabentoIngestionService

        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="API key required"):
                DatabentoIngestionService(
                    raw_repo=mock_raw_repo,
                    derived_repo=mock_derived_repo,
                    surface_config=surface_config,
                    universe_config=universe_config,
                    api_key=None,
                )

    def test_init_with_api_key(
        self,
        mock_raw_repo,
        mock_derived_repo,
        surface_config,
        universe_config,
    ) -> None:
        """Test initialization with API key."""
        from src.live.ingestion import DatabentoIngestionService

        service = DatabentoIngestionService(
            raw_repo=mock_raw_repo,
            derived_repo=mock_derived_repo,
            surface_config=surface_config,
            universe_config=universe_config,
            api_key="test_key",
        )

        assert service is not None
        assert not service.is_running

    def test_init_with_symbols(
        self,
        mock_raw_repo,
        mock_derived_repo,
        surface_config,
        universe_config,
    ) -> None:
        """Test initialization with option symbols."""
        from src.live.ingestion import DatabentoIngestionService

        symbols = ["SPY240315C00450000", "SPY240315P00440000"]

        service = DatabentoIngestionService(
            raw_repo=mock_raw_repo,
            derived_repo=mock_derived_repo,
            surface_config=surface_config,
            universe_config=universe_config,
            api_key="test_key",
            option_symbols=symbols,
        )

        assert service._option_symbols == symbols

    def test_set_symbols(
        self,
        mock_raw_repo,
        mock_derived_repo,
        surface_config,
        universe_config,
    ) -> None:
        """Test set_symbols method."""
        from src.live.ingestion import DatabentoIngestionService

        service = DatabentoIngestionService(
            raw_repo=mock_raw_repo,
            derived_repo=mock_derived_repo,
            surface_config=surface_config,
            universe_config=universe_config,
            api_key="test_key",
        )

        symbols = ["SPY240315C00450000"]
        service.set_symbols(symbols)

        assert service._option_symbols == symbols


class TestDatabentoIngestionServiceSymbolParsing:
    """Tests for option symbol parsing in DatabentoIngestionService."""

    def test_parse_call_symbol(
        self,
        mock_raw_repo,
        mock_derived_repo,
        surface_config,
        universe_config,
    ) -> None:
        """Test parsing a call option symbol."""
        from src.live.ingestion import DatabentoIngestionService

        service = DatabentoIngestionService(
            raw_repo=mock_raw_repo,
            derived_repo=mock_derived_repo,
            surface_config=surface_config,
            universe_config=universe_config,
            api_key="test_key",
        )

        result = service._parse_option_symbol("SPY240315C00450000")

        assert result is not None
        assert result["strike"] == 450.0
        assert result["right"] == "C"
        assert result["expiry"] == date(2024, 3, 15)

    def test_parse_put_symbol(
        self,
        mock_raw_repo,
        mock_derived_repo,
        surface_config,
        universe_config,
    ) -> None:
        """Test parsing a put option symbol."""
        from src.live.ingestion import DatabentoIngestionService

        service = DatabentoIngestionService(
            raw_repo=mock_raw_repo,
            derived_repo=mock_derived_repo,
            surface_config=surface_config,
            universe_config=universe_config,
            api_key="test_key",
        )

        result = service._parse_option_symbol("SPY240315P00440000")

        assert result is not None
        assert result["strike"] == 440.0
        assert result["right"] == "P"

    def test_parse_fractional_strike(
        self,
        mock_raw_repo,
        mock_derived_repo,
        surface_config,
        universe_config,
    ) -> None:
        """Test parsing symbol with fractional strike."""
        from src.live.ingestion import DatabentoIngestionService

        service = DatabentoIngestionService(
            raw_repo=mock_raw_repo,
            derived_repo=mock_derived_repo,
            surface_config=surface_config,
            universe_config=universe_config,
            api_key="test_key",
        )

        # 450.50 strike = 00450500
        result = service._parse_option_symbol("SPY240315C00450500")

        assert result is not None
        assert result["strike"] == 450.5

    def test_parse_invalid_symbol_returns_none(
        self,
        mock_raw_repo,
        mock_derived_repo,
        surface_config,
        universe_config,
    ) -> None:
        """Test that invalid symbols return None."""
        from src.live.ingestion import DatabentoIngestionService

        service = DatabentoIngestionService(
            raw_repo=mock_raw_repo,
            derived_repo=mock_derived_repo,
            surface_config=surface_config,
            universe_config=universe_config,
            api_key="test_key",
        )

        result = service._parse_option_symbol("INVALID")

        assert result is None


class TestIngestionServiceProtocol:
    """Tests for IngestionService protocol compliance."""

    def test_mock_implements_protocol(
        self,
        mock_raw_repo,
        mock_derived_repo,
        surface_config,
        universe_config,
    ) -> None:
        """Test MockIngestionService implements IngestionService protocol."""
        from src.live.ingestion import MockIngestionService, IngestionService

        service = MockIngestionService(
            raw_repo=mock_raw_repo,
            derived_repo=mock_derived_repo,
            surface_config=surface_config,
            universe_config=universe_config,
        )

        # Verify protocol methods exist
        assert hasattr(service, "start")
        assert hasattr(service, "stop")
        assert hasattr(service, "is_running")
        assert callable(service.start)
        assert callable(service.stop)

    def test_databento_implements_protocol(
        self,
        mock_raw_repo,
        mock_derived_repo,
        surface_config,
        universe_config,
    ) -> None:
        """Test DatabentoIngestionService implements IngestionService protocol."""
        from src.live.ingestion import DatabentoIngestionService, IngestionService

        service = DatabentoIngestionService(
            raw_repo=mock_raw_repo,
            derived_repo=mock_derived_repo,
            surface_config=surface_config,
            universe_config=universe_config,
            api_key="test_key",
        )

        # Verify protocol methods exist
        assert hasattr(service, "start")
        assert hasattr(service, "stop")
        assert hasattr(service, "is_running")
        assert callable(service.start)
        assert callable(service.stop)
