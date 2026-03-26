"""FRED (Federal Reserve Economic Data) implementation of MacroDataProvider protocol.

This module provides a concrete implementation of the MacroDataProvider protocol
using the FRED REST API for macroeconomic and financial time series data.
"""

import logging
import os
from datetime import datetime, timedelta

import pandas as pd
import requests

from src.exceptions import ConfigError, ProviderError


class FREDProvider:
    """FRED implementation of MacroDataProvider protocol.

    This provider fetches macroeconomic data from the Federal Reserve Economic Data (FRED)
    API, including interest rates, economic indicators, and other time series data.
    Ensures point-in-time correctness by tracking release timestamps.

    Attributes:
        _api_key: FRED API key for authentication
        _base_url: Base URL for FRED API
        _session: HTTP session for making requests
        _logger: Logger instance for this provider
    """

    def __init__(self, api_key: str | None = None, base_url: str | None = None):
        """Initialize FRED provider.

        Args:
            api_key: FRED API key. If None, loads from FRED_API_KEY env var.
            base_url: FRED API base URL. If None, uses default FRED API endpoint.

        Raises:
            ConfigError: If API key is not provided and not found in environment
        """
        key = api_key or os.getenv("FRED_API_KEY")
        if not key:
            raise ConfigError(
                "FRED API key required. Provide via api_key parameter "
                "or FRED_API_KEY environment variable."
            )

        self._api_key = key
        self._base_url = base_url or "https://api.stlouisfed.org/fred"
        self._session = requests.Session()
        self._logger = logging.getLogger(__name__)

    def fetch_series(
        self,
        series_id: str,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        """Fetch macro time series with release timestamps from FRED.

        Args:
            series_id: FRED series identifier (e.g., "DGS10" for 10-year treasury)
            start: Start datetime (inclusive)
            end: End datetime (exclusive)

        Returns:
            DataFrame with columns: [obs_date, value, release_datetime_utc]
            Index: obs_date (date)

        Raises:
            ValueError: If series_id not found
            RuntimeError: If API request fails

        Note:
            The release_datetime_utc column is critical for point-in-time correctness.
            It indicates when the data became available, not when it was observed.
        """
        if not series_id:
            raise ValueError("Series ID cannot be empty")

        try:
            self._logger.info(
                f"Fetching FRED series {series_id} from {start.date()} to {end.date()}"
            )

            # Fetch observations from FRED API
            url = f"{self._base_url}/series/observations"
            params = {
                "series_id": series_id,
                "api_key": self._api_key,
                "file_type": "json",
                "observation_start": start.strftime("%Y-%m-%d"),
                # FRED API treats observation_end as inclusive; subtract 1 day
                # to match our exclusive-end contract.
                "observation_end": (end - timedelta(days=1)).strftime("%Y-%m-%d"),
                "sort_order": "asc",
            }

            response = self._session.get(url, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()

            if "error_message" in data:
                raise ValueError(f"FRED API error: {data['error_message']}")

            observations = data.get("observations", [])

            if not observations:
                self._logger.warning(f"No data returned for series {series_id}")
                return pd.DataFrame(columns=["obs_date", "value", "release_datetime_utc"])

            # Parse observations into DataFrame
            records = []
            for obs in observations:
                # Skip missing values
                if obs["value"] == ".":
                    continue

                try:
                    value = float(obs["value"])

                    # Convert percent values to decimal (e.g., 5.25 -> 0.0525)
                    # FRED typically returns rates as percentages
                    if self._is_percent_series(series_id):
                        value = value / 100.0

                    records.append(
                        {
                            "obs_date": pd.to_datetime(obs["date"]).date(),
                            "value": value,
                            # FRED observations endpoint does not expose precise
                            # release timestamps; use market-close UTC on obs_date
                            # as a conservative availability proxy.
                            "release_datetime_utc": (
                                pd.to_datetime(obs["date"]) + timedelta(hours=21)
                            ),
                        }
                    )
                except (ValueError, KeyError) as e:
                    self._logger.warning(f"Skipping invalid observation: {e}")
                    continue

            if not records:
                return pd.DataFrame(columns=["obs_date", "value", "release_datetime_utc"])

            df = pd.DataFrame(records)
            self._logger.info(f"Fetched {len(df)} observations for {series_id}")

            return df

        except ValueError:
            # Re-raise ValueError (e.g., from FRED API error_message)
            raise
        except requests.exceptions.RequestException as e:
            self._logger.error(f"Failed to fetch series {series_id}: {e}")
            raise ProviderError(f"FRED API request failed: {e}") from e
        except Exception as e:
            self._logger.error(f"Unexpected error fetching {series_id}: {e}")
            raise ProviderError(f"FRED API error: {e}") from e

    def get_latest_value(
        self,
        series_id: str,
        as_of: datetime,
    ) -> tuple[datetime, float]:
        """Get latest released value as of timestamp from FRED.

        This method ensures point-in-time correctness by only returning values
        that were released before or at the as_of timestamp.

        Args:
            series_id: FRED series identifier
            as_of: Reference timestamp

        Returns:
            Tuple of (release_datetime_utc, value)

        Raises:
            ValueError: If series_id not found or no data before as_of
            RuntimeError: If API request fails

        Example:
            >>> provider = FREDProvider()
            >>> dt, rate = provider.get_latest_value("DGS10", datetime(2023, 1, 15, 10, 0))
            >>> # Returns the 10Y treasury rate that was released before Jan 15, 10am
        """
        if not series_id:
            raise ValueError("Series ID cannot be empty")

        try:
            # Fetch recent data (last 90 days before as_of)
            start = as_of - pd.Timedelta(days=90)
            df = self.fetch_series(series_id, start, as_of)

            if df.empty:
                raise ValueError(f"No data available for {series_id} before {as_of}")

            # Filter to only releases before as_of
            df = df[df["release_datetime_utc"] <= as_of]

            if df.empty:
                raise ValueError(f"No data released for {series_id} before {as_of}")

            # Get the latest release
            latest = df.iloc[-1]
            return latest["release_datetime_utc"], float(latest["value"])

        except Exception as e:
            if isinstance(e, ValueError):
                raise
            self._logger.error(f"Failed to get latest value for {series_id}: {e}")
            raise ProviderError(f"FRED API error: {e}") from e

    def _is_percent_series(self, series_id: str) -> bool:
        """Check if a series returns values as percentages.

        Args:
            series_id: FRED series identifier

        Returns:
            True if series values should be converted from percent to decimal
        """
        # Common rate/percentage series patterns
        percent_prefixes = [
            "DGS",  # Treasury rates (DGS10, DGS2, etc.)
            "DFII",  # Treasury inflation-indexed
            "DFF",  # Federal funds rate
            "EFFR",  # Effective federal funds rate
            "FEDFUNDS",  # Federal funds effective rate
            "MORTGAGE",  # Mortgage rates
            "CORESTICKM",  # Core inflation (month-over-month %)
            # Note: CPIAUCSL is an index level (~300), NOT a percentage
        ]

        # VIX and similar are already in percentage form but represent index values
        # Don't convert these
        if series_id in ["VIXCLS", "VIX"]:
            return False

        return any(series_id.startswith(prefix) for prefix in percent_prefixes)
