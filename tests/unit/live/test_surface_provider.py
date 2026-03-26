"""Unit tests for DatabaseSurfaceProvider.

Tests for the database-backed surface provider.
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.exceptions import DataReadError


@pytest.fixture
def mock_derived_repo():
    """Create mock DerivedRepository."""
    return MagicMock()


@pytest.fixture
def sample_surface_df():
    """Create sample surface DataFrame."""
    return pd.DataFrame([
        {
            "ts_utc": datetime.now(timezone.utc),
            "option_symbol": "SPY240315C00450000",
            "tenor_days": 30,
            "delta_bucket": "ATM",
            "strike": 450.0,
            "expiry": datetime.now(timezone.utc).date() + timedelta(days=30),
            "right": "C",
            "bid": 10.0,
            "ask": 10.20,
            "delta": 0.50,
            "gamma": 0.02,
            "vega": 0.30,
            "theta": -0.05,
            "iv_mid": 0.20,
            "underlying_price": 450.0,
        }
    ])


class TestDatabaseSurfaceProvider:
    """Tests for DatabaseSurfaceProvider."""

    def test_init(self, mock_derived_repo) -> None:
        """Test initialization."""
        from src.live.surface_provider import DatabaseSurfaceProvider

        provider = DatabaseSurfaceProvider(
            derived_repo=mock_derived_repo,
            version="live",
        )

        assert provider._derived_repo == mock_derived_repo
        assert provider._version == "live"
        assert provider._lookback_seconds == 60

    def test_init_custom_params(self, mock_derived_repo) -> None:
        """Test initialization with custom parameters."""
        from src.live.surface_provider import DatabaseSurfaceProvider

        provider = DatabaseSurfaceProvider(
            derived_repo=mock_derived_repo,
            version="test_v1",
            underlying_symbol="QQQ",
            lookback_seconds=120,
        )

        assert provider._version == "test_v1"
        assert provider._underlying_symbol == "QQQ"
        assert provider._lookback_seconds == 120

    def test_get_latest_surface_calls_repo(
        self, mock_derived_repo, sample_surface_df
    ) -> None:
        """Test that get_latest_surface queries repository."""
        from src.live.surface_provider import DatabaseSurfaceProvider

        mock_derived_repo.read_surface_snapshots.return_value = sample_surface_df

        provider = DatabaseSurfaceProvider(mock_derived_repo, version="live")
        surface = provider.get_latest_surface()

        mock_derived_repo.read_surface_snapshots.assert_called_once()
        call_args = mock_derived_repo.read_surface_snapshots.call_args
        assert call_args.kwargs["version"] == "live"
        assert len(surface) == 1

    def test_get_latest_surface_filters_latest(self, mock_derived_repo) -> None:
        """Test that only the latest timestamp is returned."""
        from src.live.surface_provider import DatabaseSurfaceProvider

        now = datetime.now(timezone.utc)
        old_ts = now - timedelta(seconds=30)

        df = pd.DataFrame([
            {"ts_utc": old_ts, "option_symbol": "SPY_OLD", "strike": 440.0},
            {"ts_utc": now, "option_symbol": "SPY_NEW", "strike": 450.0},
        ])
        mock_derived_repo.read_surface_snapshots.return_value = df

        provider = DatabaseSurfaceProvider(mock_derived_repo, version="live")
        surface = provider.get_latest_surface()

        assert len(surface) == 1
        assert surface.iloc[0]["option_symbol"] == "SPY_NEW"

    def test_get_latest_surface_empty_raises(self, mock_derived_repo) -> None:
        """Test that empty database raises error."""
        from src.live.surface_provider import DatabaseSurfaceProvider

        mock_derived_repo.read_surface_snapshots.return_value = pd.DataFrame()

        provider = DatabaseSurfaceProvider(mock_derived_repo, version="live")

        with pytest.raises(DataReadError, match="No surface data"):
            provider.get_latest_surface()

    def test_get_surface_at(self, mock_derived_repo, sample_surface_df) -> None:
        """Test get_surface_at method."""
        from src.live.surface_provider import DatabaseSurfaceProvider

        mock_derived_repo.read_surface_snapshots.return_value = sample_surface_df

        provider = DatabaseSurfaceProvider(mock_derived_repo, version="live")
        target_time = datetime.now(timezone.utc)
        surface = provider.get_surface_at(target_time)

        assert len(surface) == 1
        call_args = mock_derived_repo.read_surface_snapshots.call_args
        assert call_args.kwargs["end"] == target_time

    def test_get_surface_history(self, mock_derived_repo, sample_surface_df) -> None:
        """Test get_surface_history method."""
        from src.live.surface_provider import DatabaseSurfaceProvider

        mock_derived_repo.read_surface_snapshots.return_value = sample_surface_df

        provider = DatabaseSurfaceProvider(mock_derived_repo, version="live")
        start = datetime.now(timezone.utc) - timedelta(hours=1)
        end = datetime.now(timezone.utc)
        surface = provider.get_surface_history(start, end)

        assert len(surface) == 1
        call_args = mock_derived_repo.read_surface_snapshots.call_args
        assert call_args.kwargs["start"] == start
        assert call_args.kwargs["end"] == end

    def test_version_property(self, mock_derived_repo) -> None:
        """Test version property getter and setter."""
        from src.live.surface_provider import DatabaseSurfaceProvider

        provider = DatabaseSurfaceProvider(mock_derived_repo, version="v1")
        assert provider.version == "v1"

        provider.version = "v2"
        assert provider.version == "v2"
