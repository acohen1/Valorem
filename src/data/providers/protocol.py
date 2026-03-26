"""Abstract protocols for data providers.

This module defines abstract interfaces (Protocols) for market data and macro data providers.
These protocols enable dependency injection and swapping implementations without tight coupling.
"""

from datetime import datetime
from typing import Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class MarketDataProvider(Protocol):
    """Abstract interface for market data vendors.

    This protocol defines the contract that all market data providers must satisfy.
    Implementations include Databento, IBKR, polygon.io, or mock providers for testing.
    """

    def fetch_underlying_bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        interval: str = "1m",
        suppress_error_log: bool = False,
    ) -> pd.DataFrame:
        """Fetch OHLCV bars for underlying asset.

        Args:
            symbol: Underlying symbol (e.g., "SPY")
            start: Start datetime (inclusive)
            end: End datetime (exclusive)
            interval: Bar interval (e.g., "1m", "5m", "1h", "1d")
            suppress_error_log: If True, suppress logging of expected errors

        Returns:
            DataFrame with columns: [ts_utc, open, high, low, close, volume]
            Index: ts_utc (datetime)

        Raises:
            ValueError: If symbol not found or interval invalid
            RuntimeError: If API request fails
        """
        ...

    def fetch_option_quotes(
        self,
        symbols: list[str],
        start: datetime,
        end: datetime,
        schema: str = "cbbo-1m",
    ) -> pd.DataFrame:
        """Fetch option quote data.

        Args:
            symbols: List of option symbols (e.g., ["SPY230120C00400000"])
            start: Start datetime (inclusive)
            end: End datetime (exclusive)
            schema: Data schema (e.g., "cbbo-1m", "mbp-1", "trades")

        Returns:
            DataFrame with columns: [ts_utc, option_symbol, bid, ask, bid_size, ask_size, ...]
            Index: ts_utc (datetime)

        Raises:
            ValueError: If symbols invalid or schema unsupported
            RuntimeError: If API request fails
        """
        ...

    def fetch_option_bars(
        self,
        symbols: list[str],
        start: datetime,
        end: datetime,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Fetch OHLCV bars for option contracts.

        Args:
            symbols: List of option symbols (e.g., ["SPY230120C00400000"])
            start: Start datetime (inclusive)
            end: End datetime (exclusive)
            interval: Bar interval (e.g., "1m", "1h", "1d")

        Returns:
            DataFrame with columns: [ts_utc, option_symbol, open, high, low, close, volume]

        Raises:
            ValueError: If symbols invalid or interval unsupported
            RuntimeError: If API request fails
        """
        ...

    def fetch_option_statistics(
        self,
        symbols: list[str],
        start: datetime,
        end: datetime,
        stat_types: list[int] | None = None,
    ) -> pd.DataFrame:
        """Fetch option statistics (e.g., daily open interest).

        Args:
            symbols: List of option symbols (e.g., ["SPY230120C00400000"])
            start: Start datetime (inclusive)
            end: End datetime (exclusive)
            stat_types: Stat type filter (default: [9] = open interest).
                        See databento_dbn.StatType for values.

        Returns:
            DataFrame with columns: [ts_ref, ts_event, option_symbol, stat_type, quantity, price]

        Raises:
            ValueError: If symbols invalid
            RuntimeError: If API request fails
        """
        ...

    def estimate_cost(
        self,
        dataset: str,
        schema: str,
        symbols: list[str],
        start: datetime,
        end: datetime,
    ) -> float:
        """Estimate USD cost before fetching data.

        This method helps prevent accidental expensive API calls by providing
        cost estimates before execution.

        Args:
            dataset: Dataset identifier (e.g., "GLBX.MDP3", "OPRA")
            schema: Schema identifier (e.g., "ohlcv-1m", "cbbo-1m")
            symbols: List of symbols to fetch
            start: Start datetime
            end: End datetime

        Returns:
            Estimated cost in USD

        Raises:
            ValueError: If dataset/schema combination invalid
        """
        ...

    def resolve_option_symbols(
        self,
        parent: str,
        as_of: datetime,
        dte_min: int,
        dte_max: int,
        moneyness_min: float,
        moneyness_max: float,
        max_available_date: datetime | None = None,
    ) -> list[str]:
        """Resolve available option symbols for given criteria.

        This method queries the option chain to find symbols matching the specified
        filters. Useful for backtesting where you need to discover what options
        were available at a historical point in time.

        Args:
            parent: Parent symbol (e.g., "SPY")
            as_of: Reference datetime for filtering
            dte_min: Minimum days to expiration
            dte_max: Maximum days to expiration
            moneyness_min: Minimum strike/spot ratio
            moneyness_max: Maximum strike/spot ratio
            max_available_date: Maximum date that data is available (used to cap query range)

        Returns:
            List of option symbols matching criteria

        Raises:
            ValueError: If parent symbol invalid
            RuntimeError: If resolution fails
        """
        ...


@runtime_checkable
class MacroDataProvider(Protocol):
    """Abstract interface for macro/fundamental data vendors.

    This protocol defines the contract for providers of macroeconomic data
    like interest rates, VIX, economic indicators, etc.
    Implementations include FRED API, Bloomberg, or mock providers.
    """

    def fetch_series(
        self,
        series_id: str,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        """Fetch macro time series with release timestamps.

        Args:
            series_id: Series identifier (e.g., "DGS10" for 10-year treasury)
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
        ...

    def get_latest_value(
        self,
        series_id: str,
        as_of: datetime,
    ) -> tuple[datetime, float]:
        """Get latest released value as of timestamp.

        This method ensures point-in-time correctness by only returning values
        that were released before or at the as_of timestamp.

        Args:
            series_id: Series identifier
            as_of: Reference timestamp

        Returns:
            Tuple of (release_datetime_utc, value)

        Raises:
            ValueError: If series_id not found or no data before as_of
            RuntimeError: If API request fails

        Example:
            >>> provider = FredProvider()
            >>> dt, rate = provider.get_latest_value("DGS10", datetime(2023, 1, 15, 10, 0))
            >>> # Returns the 10Y treasury rate that was released before Jan 15, 10am
        """
        ...
