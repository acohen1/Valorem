"""Surface provider protocol and database implementation.

This module defines the SurfaceProvider protocol and provides
DatabaseSurfaceProvider, which reads volatility surfaces from the database.
All data (live or mock) flows through the database via IngestionService
before being read by this provider.

Architecture:
    IngestionService ──→ DB ──→ DatabaseSurfaceProvider ──→ TradingLoop
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Protocol, runtime_checkable

import pandas as pd

from src.data.storage.repository import DerivedRepository
from src.exceptions import DataReadError

logger = logging.getLogger(__name__)


@runtime_checkable
class SurfaceProvider(Protocol):
    """Protocol for providing volatility surface data."""

    def get_latest_surface(self) -> pd.DataFrame:
        """Get the latest volatility surface.

        Returns:
            DataFrame with columns: option_symbol, tenor_days, delta_bucket,
            strike, expiry, right, bid, ask, delta, gamma, vega, theta, iv
        """
        ...


class DatabaseSurfaceProvider:
    """Database-backed surface provider.

    Reads volatility surfaces from the database. All surfaces are stored
    via IngestionService, regardless of whether they originated from
    live data (DatabentoIngestionService) or mock data (MockIngestionService).

    Example:
        provider = DatabaseSurfaceProvider(derived_repo, version="live")
        surface = provider.get_latest_surface()
    """

    def __init__(
        self,
        derived_repo: DerivedRepository,
        version: str = "live",
        underlying_symbol: str = "SPY",
        lookback_seconds: int = 60,
    ):
        """Initialize surface provider.

        Args:
            derived_repo: DerivedRepository instance for reading surfaces
            version: Surface version to query (e.g., "live", "mock", "v1.0")
            underlying_symbol: Underlying symbol for filtering
            lookback_seconds: How far back to look for latest surface
        """
        self._derived_repo = derived_repo
        self._version = version
        self._underlying_symbol = underlying_symbol
        self._lookback_seconds = lookback_seconds

    def get_latest_surface(self) -> pd.DataFrame:
        """Get the most recent surface snapshot from database.

        Returns:
            DataFrame with surface data including columns:
            - option_symbol, tenor_days, delta_bucket, strike, expiry, right
            - bid, ask, mid_price, spread, spread_pct
            - delta, gamma, vega, theta, iv_mid
            - underlying_price, flags, ts_utc

        Raises:
            RuntimeError: If no surface available in lookback window
        """
        end = datetime.now(timezone.utc)
        start = end - timedelta(seconds=self._lookback_seconds)

        try:
            surface_df = self._derived_repo.read_surface_snapshots(
                start=start,
                end=end,
                version=self._version,
            )
        except Exception as e:
            raise DataReadError(f"Failed to read surface from database: {e}") from e

        if surface_df.empty:
            raise DataReadError(
                f"No surface data available in database for version '{self._version}' "
                f"in last {self._lookback_seconds} seconds"
            )

        # Get most recent timestamp
        latest_ts = surface_df["ts_utc"].max()
        latest_surface = surface_df[surface_df["ts_utc"] == latest_ts].copy()

        return latest_surface

    def get_surface_at(self, timestamp: datetime) -> pd.DataFrame:
        """Get surface snapshot at or before a specific timestamp.

        Args:
            timestamp: Target timestamp

        Returns:
            DataFrame with surface data closest to (but not after) timestamp

        Raises:
            RuntimeError: If no surface available before timestamp
        """
        start = timestamp - timedelta(seconds=self._lookback_seconds)
        end = timestamp

        try:
            surface_df = self._derived_repo.read_surface_snapshots(
                start=start,
                end=end,
                version=self._version,
            )
        except Exception as e:
            raise DataReadError(f"Failed to read surface from database: {e}") from e

        if surface_df.empty:
            raise DataReadError(
                f"No surface data available before {timestamp} for version '{self._version}'"
            )

        # Get most recent timestamp at or before target
        latest_ts = surface_df["ts_utc"].max()
        surface = surface_df[surface_df["ts_utc"] == latest_ts].copy()

        return surface

    def get_surface_history(
        self,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        """Get all surface snapshots in a time range.

        Args:
            start: Start datetime (inclusive)
            end: End datetime (exclusive)

        Returns:
            DataFrame with all surface snapshots in range
        """
        try:
            return self._derived_repo.read_surface_snapshots(
                start=start,
                end=end,
                version=self._version,
            )
        except Exception as e:
            raise DataReadError(f"Failed to read surface history: {e}") from e

    @property
    def version(self) -> str:
        """Get the surface version being read."""
        return self._version

    @version.setter
    def version(self, value: str) -> None:
        """Set the surface version to read."""
        self._version = value
