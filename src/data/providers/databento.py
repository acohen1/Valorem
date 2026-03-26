"""Databento implementation of MarketDataProvider protocol.

This module provides a concrete implementation of the MarketDataProvider protocol
using the Databento Historical API for market data ingestion.
"""

import logging
import os
from datetime import datetime, timedelta

import databento as db
import pandas as pd

from src.exceptions import ConfigError, ProviderError


class DatabentoProvider:
    """Databento implementation of MarketDataProvider protocol.

    This provider fetches market data from Databento's Historical API,
    normalizes column names to match our schema, and handles API errors gracefully.

    Attributes:
        _client: Databento Historical API client
        _logger: Logger instance for this provider
    """

    def __init__(
        self,
        api_key: str | None = None,
        dataset_equities: str = "DBEQ.BASIC",
        dataset_options: str = "OPRA.PILLAR",
        definition_query_days: int | None = None,
    ):
        """Initialize Databento provider.

        Args:
            api_key: Databento API key. If None, loads from DATABENTO_API_KEY env var.
            dataset_equities: Databento dataset for equity data (default: DBEQ.BASIC).
            dataset_options: Databento dataset for options data (default: OPRA.PILLAR).
            definition_query_days: Days to query for option definitions. If None,
                auto-calculates based on dte_max (dte_max // 8, bounded by [7, 30]).

        Raises:
            ConfigError: If API key is not provided and not found in environment
        """
        key = api_key or os.getenv("DATABENTO_API_KEY")
        if not key:
            raise ConfigError(
                "Databento API key required. Provide via api_key parameter "
                "or DATABENTO_API_KEY environment variable."
            )

        self._client = db.Historical(key=key)
        self._dataset_equities = dataset_equities
        self._dataset_options = dataset_options
        self._definition_query_days_override = definition_query_days
        self._logger = logging.getLogger(__name__)

    def fetch_underlying_bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        interval: str = "1m",
        suppress_error_log: bool = False,
    ) -> pd.DataFrame:
        """Fetch OHLCV bars for underlying asset from Databento.

        Args:
            symbol: Underlying symbol (e.g., "SPY")
            start: Start datetime (inclusive)
            end: End datetime (exclusive)
            interval: Bar interval (e.g., "1s", "1m", "1h", "1d")
            suppress_error_log: If True, suppress logging of data availability errors
                (data_end_after_available_end, data_start_after_available_end,
                license_not_found_unauthorized). Other errors are always logged.

        Returns:
            DataFrame with columns: [ts_utc, open, high, low, close, volume]
            Index: ts_utc (datetime)

        Raises:
            ValueError: If symbol not found or interval invalid
            RuntimeError: If API request fails
        """
        # Map interval to Databento schema (validate before try block)
        # Note: Databento only supports 1s, 1m, 1h, 1d (no 5m)
        schema_map = {
            "1s": "ohlcv-1s",
            "1m": "ohlcv-1m",
            "1h": "ohlcv-1h",
            "1d": "ohlcv-1d",
        }

        if interval not in schema_map:
            raise ValueError(
                f"Unsupported interval: {interval}. "
                f"Supported: {list(schema_map.keys())}"
            )

        schema = schema_map[interval]

        try:

            self._logger.debug(
                f"Fetching {schema} bars for {symbol} from {start} to {end}"
            )

            # Fetch data from Databento
            data = self._client.timeseries.get_range(
                dataset=self._dataset_equities,
                schema=schema,
                symbols=[symbol],
                stype_in="raw_symbol",
                start=start.isoformat(),
                end=end.isoformat(),
            )

            # Convert to DataFrame and normalize
            df = data.to_df()

            if df.empty:
                self._logger.warning(f"No data returned for {symbol}")
                return pd.DataFrame(
                    columns=["ts_utc", "open", "high", "low", "close", "volume"]
                )

            return self._normalize_bars(df)

        except Exception as e:
            error_str = str(e)
            # Only suppress logging for expected data availability errors during probing
            is_availability_error = (
                "data_end_after_available_end" in error_str or
                "data_start_after_available_end" in error_str or
                "license_not_found_unauthorized" in error_str or
                "live data license is required" in error_str
            )

            # Log all errors except availability errors when suppress_error_log=True
            if not (suppress_error_log and is_availability_error):
                self._logger.error(f"Failed to fetch bars for {symbol}: {e}")

            raise ProviderError(f"Databento API error: {e}") from e

    def fetch_option_quotes(
        self,
        symbols: list[str],
        start: datetime,
        end: datetime,
        schema: str = "cbbo-1m",
    ) -> pd.DataFrame:
        """Fetch option quote data from Databento.

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
        if not symbols:
            return pd.DataFrame(
                columns=["ts_utc", "option_symbol", "bid", "ask", "bid_size", "ask_size"]
            )

        try:
            self._logger.debug(
                f"Fetching {schema} quotes for {len(symbols)} symbols from {start} to {end}"
            )

            # Fetch data from Databento
            data = self._client.timeseries.get_range(
                dataset=self._dataset_options,
                schema=schema,
                symbols=symbols,
                stype_in="raw_symbol",
                start=start.isoformat(),
                end=end.isoformat(),
            )

            # Convert to DataFrame and normalize
            df = data.to_df()

            if df.empty:
                self._logger.warning(f"No data returned for {len(symbols)} symbols")
                return pd.DataFrame(
                    columns=["ts_utc", "option_symbol", "bid", "ask", "bid_size", "ask_size"]
                )

            return self._normalize_quotes(df)

        except Exception as e:
            self._logger.error(f"Failed to fetch option quotes: {e}")
            raise ProviderError(f"Databento API error: {e}") from e

    def fetch_option_bars(
        self,
        symbols: list[str],
        start: datetime,
        end: datetime,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Fetch OHLCV bars for option contracts from Databento.

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
        schema_map = {
            "1m": "ohlcv-1m",
            "1h": "ohlcv-1h",
            "1d": "ohlcv-1d",
        }

        if interval not in schema_map:
            raise ValueError(
                f"Unsupported interval: {interval}. "
                f"Supported: {list(schema_map.keys())}"
            )

        schema = schema_map[interval]

        if not symbols:
            return pd.DataFrame(
                columns=["ts_utc", "option_symbol", "open", "high", "low", "close", "volume"]
            )

        try:
            self._logger.debug(
                f"Fetching {schema} option bars for {len(symbols)} symbols from {start} to {end}"
            )

            data = self._client.timeseries.get_range(
                dataset=self._dataset_options,
                schema=schema,
                symbols=symbols,
                stype_in="raw_symbol",
                start=start.isoformat(),
                end=end.isoformat(),
            )

            df = data.to_df()

            if df.empty:
                self._logger.warning(f"No option bar data returned for {len(symbols)} symbols")
                return pd.DataFrame(
                    columns=["ts_utc", "option_symbol", "open", "high", "low", "close", "volume"]
                )

            return self._normalize_option_bars(df)

        except Exception as e:
            self._logger.error(f"Failed to fetch option bars: {e}")
            raise ProviderError(f"Databento API error: {e}") from e

    def fetch_option_statistics(
        self,
        symbols: list[str],
        start: datetime,
        end: datetime,
        stat_types: list[int] | None = None,
    ) -> pd.DataFrame:
        """Fetch option statistics (e.g., daily open interest) from Databento.

        Args:
            symbols: List of option symbols (e.g., ["SPY230120C00400000"])
            start: Start datetime (inclusive)
            end: End datetime (exclusive)
            stat_types: Stat type filter (default: [9] = open interest only).
                        See databento_dbn.StatType for values.

        Returns:
            DataFrame with columns: [ts_ref, ts_event, option_symbol, stat_type, quantity, price]

        Raises:
            ValueError: If symbols invalid
            RuntimeError: If API request fails
        """
        if stat_types is None:
            stat_types = [9]  # OPEN_INTEREST

        empty_df = pd.DataFrame(
            columns=["ts_ref", "ts_event", "option_symbol", "stat_type", "quantity", "price"]
        )

        if not symbols:
            return empty_df

        try:
            self._logger.debug(
                f"Fetching statistics for {len(symbols)} symbols from {start} to {end} "
                f"(stat_types={stat_types})"
            )

            data = self._client.timeseries.get_range(
                dataset=self._dataset_options,
                schema="statistics",
                symbols=symbols,
                stype_in="raw_symbol",
                start=start.isoformat(),
                end=end.isoformat(),
            )

            df = data.to_df()

            if df.empty:
                self._logger.warning(f"No statistics returned for {len(symbols)} symbols")
                return empty_df

            return self._normalize_statistics(df, stat_types)

        except Exception as e:
            self._logger.error(f"Failed to fetch option statistics: {e}")
            raise ProviderError(f"Databento API error: {e}") from e

    def estimate_cost(
        self,
        dataset: str,
        schema: str,
        symbols: list[str],
        start: datetime,
        end: datetime,
    ) -> float:
        """Estimate USD cost before fetching data from Databento.

        Args:
            dataset: Dataset identifier (e.g., "GLBX.MDP3", "OPRA.PILLAR")
            schema: Schema identifier (e.g., "ohlcv-1m", "cbbo-1m")
            symbols: List of symbols to fetch
            start: Start datetime
            end: End datetime

        Returns:
            Estimated cost in USD

        Raises:
            ValueError: If dataset/schema combination invalid
            RuntimeError: If cost estimation fails
        """
        try:
            self._logger.debug(
                f"Estimating cost for {dataset}/{schema} with {len(symbols)} symbols"
            )

            # Use Databento's cost estimation API
            # Note: get_cost returns float directly, not a metadata object
            cost = self._client.metadata.get_cost(
                dataset=dataset,
                schema=schema,
                symbols=symbols,
                stype_in="raw_symbol",
                start=start.isoformat(),
                end=end.isoformat(),
            )

            self._logger.debug(f"Estimated cost: ${cost:.2f}")

            return cost

        except Exception as e:
            self._logger.error(f"Failed to estimate cost: {e}")
            raise ProviderError(f"Databento API error: {e}") from e

    def _calculate_query_days(self, dte_max: int) -> int:
        """Calculate optimal definition query window based on DTE range.

        Options are typically listed over a rolling window. For longer DTE ranges,
        we need a wider query window to capture all relevant listings.

        The formula scales with max_dte: options expiring 120 days out may still
        be getting listed/updated over a ~15 day period (weekly options, new
        strikes, etc.). A wider DTE range means options listed later in the month
        are still relevant.

        Args:
            dte_max: Maximum days to expiration from config

        Returns:
            Number of days to query (7-30)
        """
        # Check for explicit override first
        if self._definition_query_days_override is not None:
            return self._definition_query_days_override

        # Auto-calculate: query roughly 1/8 of max_dte, bounded by [7, 30]
        # Examples:
        #   dte_max=30  -> query 7 days  (30//8 = 3, clamped to 7)
        #   dte_max=60  -> query 7 days  (60//8 = 7)
        #   dte_max=120 -> query 15 days (120//8 = 15)
        #   dte_max=240 -> query 30 days (240//8 = 30)
        calculated = dte_max // 8
        return min(max(7, calculated), 30)

    def resolve_option_symbols(
        self,
        parent: str,
        as_of: datetime,
        dte_min: int = 0,
        dte_max: int = 365,
        moneyness_min: float = 0.0,
        moneyness_max: float = float("inf"),
        max_available_date: datetime | None = None,
    ) -> list[str]:
        """Resolve available option symbols for a parent underlying.

        Uses Databento's parent symbology to fetch all option contracts
        for a given underlying symbol. Returns instrument definitions
        which contain the raw option symbols.

        The provider fetches definitions over a dynamically calculated window
        based on dte_max. Options with longer DTEs require wider query windows
        to capture all listing events (weekly options, new strikes, etc.).
        The query is capped to max_available_date to avoid querying beyond available data.

        Args:
            parent: Parent symbol (e.g., "SPY")
            as_of: Reference datetime for symbol resolution
            dte_min: Minimum days to expiration (used by manifest generator)
            dte_max: Maximum days to expiration (used to calculate query window)
            moneyness_min: Minimum strike/spot ratio (used by manifest generator)
            moneyness_max: Maximum strike/spot ratio (used by manifest generator)
            max_available_date: Maximum date that data is available (caps query range)

        Returns:
            List of available option symbols

        Raises:
            RuntimeError: If API request fails
        """
        try:
            # Calculate optimal query window based on DTE range
            query_days = self._calculate_query_days(dte_max)

            self._logger.info(
                f"Resolving option symbols for {parent} as of {as_of.date()} "
                f"(query window: {query_days} days)"
            )

            # Use parent symbology to get all options for the underlying
            # Parent symbol format: {underlying}.OPT (e.g., "SPY.OPT")
            parent_symbol = f"{parent}.OPT"

            # Fetch instrument definitions using parent symbology
            # Query window scales with dte_max: options expiring further out
            # may be listed over a longer period (weekly options, new strikes, etc.)
            query_end = as_of + timedelta(days=query_days)

            # Cap to max available date if provided (prevents querying beyond available data)
            if max_available_date and query_end > max_available_date:
                original_query_end = query_end
                query_end = max_available_date
                self._logger.debug(
                    f"Capping definition query end from {original_query_end.date()} "
                    f"to {max_available_date.date()} (data availability limit)"
                )

            data = self._client.timeseries.get_range(
                dataset=self._dataset_options,
                schema="definition",
                stype_in="parent",
                symbols=[parent_symbol],
                start=as_of.date().isoformat(),
                end=query_end.date().isoformat(),
            )

            # Convert to DataFrame and extract symbols
            df = data.to_df()

            if df.empty:
                self._logger.warning(f"No option symbols found for {parent}")
                return []

            # Extract raw_symbol column which contains the option symbols
            if "raw_symbol" in df.columns:
                symbols = df["raw_symbol"].dropna().unique().tolist()
            elif "symbol" in df.columns:
                symbols = df["symbol"].dropna().unique().tolist()
            else:
                # Try to get from index if symbol is the index
                symbols = df.index.unique().tolist()

            self._logger.info(f"Resolved {len(symbols)} option symbols for {parent}")
            return sorted(set(str(s) for s in symbols))

        except Exception as e:
            self._logger.error(f"Failed to resolve option symbols for {parent}: {e}")
            raise ProviderError(f"Databento API error: {e}") from e

    def fetch_option_definitions(
        self,
        parent: str,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        """Fetch option instrument definitions for a date range.

        Returns full instrument definition data including expiry dates,
        strikes, and other contract specifications.

        Args:
            parent: Parent symbol (e.g., "SPY")
            start: Start datetime
            end: End datetime

        Returns:
            DataFrame with instrument definitions

        Raises:
            RuntimeError: If API request fails
        """
        try:
            self._logger.info(
                f"Fetching option definitions for {parent} from {start.date()} to {end.date()}"
            )

            # Use parent symbology to get all options for the underlying
            parent_symbol = f"{parent}.OPT"

            data = self._client.timeseries.get_range(
                dataset=self._dataset_options,
                schema="definition",
                stype_in="parent",
                symbols=[parent_symbol],
                start=start.isoformat(),
                end=end.isoformat(),
            )

            df = data.to_df()

            if df.empty:
                self._logger.warning(f"No option definitions found for {parent}")
                return pd.DataFrame()

            self._logger.info(f"Fetched {len(df)} option definitions for {parent}")
            return df

        except Exception as e:
            self._logger.error(f"Failed to fetch option definitions: {e}")
            raise ProviderError(f"Databento API error: {e}") from e

    def _normalize_bars(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize Databento OHLCV data to standard schema.

        Args:
            df: Raw DataFrame from Databento

        Returns:
            Normalized DataFrame with standard column names
        """
        # ts_event may be the index; reset to make it a column
        if df.index.name == "ts_event":
            df = df.reset_index()

        # Databento column mapping
        column_map = {
            "ts_event": "ts_utc",
            # Databento already uses: open, high, low, close, volume
        }

        df = df.rename(columns=column_map)

        # Ensure ts_utc is datetime
        if "ts_utc" in df.columns:
            df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True)

        # Select and order columns
        standard_columns = ["ts_utc", "open", "high", "low", "close", "volume"]
        available_columns = [col for col in standard_columns if col in df.columns]
        df = df[available_columns].reset_index(drop=True)

        # Deduplicate: DBEQ.BASIC returns multiple rows per timestamp from
        # different publishers/exchanges. Aggregate to a single OHLCV bar.
        if "ts_utc" in df.columns and df.duplicated(subset=["ts_utc"]).any():
            agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
            if "volume" in df.columns:
                agg["volume"] = "sum"
            df = df.groupby("ts_utc", as_index=False).agg(agg)

        # Downcast uint64 to int64 (SQLite doesn't support unsigned 64-bit)
        for col in df.select_dtypes(include=["uint64"]).columns:
            df[col] = df[col].astype("int64")

        return df

    def _normalize_option_bars(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize Databento option OHLCV data to standard schema.

        Args:
            df: Raw DataFrame from Databento

        Returns:
            Normalized DataFrame with standard column names
        """
        # ts_event may be the index; reset to make it a column
        if df.index.name == "ts_event":
            df = df.reset_index()

        # Databento column mapping
        column_map = {
            "ts_event": "ts_utc",
            "symbol": "option_symbol",
        }

        df = df.rename(columns=column_map)

        # Ensure ts_utc is datetime
        if "ts_utc" in df.columns:
            df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True)

        # Select and order columns
        standard_columns = ["ts_utc", "option_symbol", "open", "high", "low", "close", "volume"]
        available_columns = [col for col in standard_columns if col in df.columns]
        df = df[available_columns].reset_index(drop=True)

        # Drop rows with null timestamps
        if "ts_utc" in df.columns:
            df = df.dropna(subset=["ts_utc"]).reset_index(drop=True)

        # Deduplicate: aggregate by (ts_utc, option_symbol) with OHLCV semantics
        key_cols = ["ts_utc", "option_symbol"]
        if all(c in df.columns for c in key_cols) and df.duplicated(subset=key_cols).any():
            agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
            if "volume" in df.columns:
                agg["volume"] = "sum"
            df = df.groupby(key_cols, as_index=False).agg(agg)

        # Downcast uint64 to int64 (SQLite doesn't support unsigned 64-bit)
        for col in df.select_dtypes(include=["uint64"]).columns:
            df[col] = df[col].astype("int64")

        return df

    def _normalize_statistics(self, df: pd.DataFrame, stat_types: list[int]) -> pd.DataFrame:
        """Normalize Databento statistics data to standard schema.

        Args:
            df: Raw DataFrame from Databento (StatMsg records)
            stat_types: List of stat_type values to keep

        Returns:
            Normalized DataFrame with columns: [ts_ref, ts_event, option_symbol, stat_type, quantity, price]
        """
        # ts_event may be the index; reset to make it a column
        if df.index.name == "ts_event":
            df = df.reset_index()

        # Rename symbol column
        if "symbol" in df.columns:
            df = df.rename(columns={"symbol": "option_symbol"})

        # Convert timestamps
        for ts_col in ("ts_event", "ts_ref"):
            if ts_col in df.columns:
                df[ts_col] = pd.to_datetime(df[ts_col], utc=True)

        # Filter to requested stat types
        if "stat_type" in df.columns:
            df = df[df["stat_type"].isin(stat_types)].copy()

        # Note: to_df() with default PriceType.FLOAT already converts
        # fixed-point prices to float dollars — no manual scaling needed.

        # Select standard columns
        standard_columns = ["ts_ref", "ts_event", "option_symbol", "stat_type", "quantity", "price"]
        available_columns = [col for col in standard_columns if col in df.columns]
        df = df[available_columns].reset_index(drop=True)

        # Fill missing ts_ref from ts_event (OPRA statistics often has NaT ts_ref)
        if "ts_ref" in df.columns and "ts_event" in df.columns:
            df["ts_ref"] = df["ts_ref"].fillna(df["ts_event"])

        # Deduplicate: keep last per (option_symbol, ts_ref, stat_type) — latest publication wins
        key_cols = ["option_symbol", "ts_ref", "stat_type"]
        if all(c in df.columns for c in key_cols) and df.duplicated(subset=key_cols).any():
            df = df.drop_duplicates(subset=key_cols, keep="last")

        # Downcast uint64 to int64 (SQLite doesn't support unsigned 64-bit)
        for col in df.select_dtypes(include=["uint64"]).columns:
            df[col] = df[col].astype("int64")

        return df

    def _normalize_quotes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize Databento option quote data to standard schema.

        Args:
            df: Raw DataFrame from Databento

        Returns:
            Normalized DataFrame with standard column names
        """
        # ts_recv or ts_event may be the index; reset to make it a column
        if df.index.name in ("ts_event", "ts_recv"):
            df = df.reset_index()

        # Databento CBBO column mapping
        # CBBO-1m uses bid_px_00/ask_px_00 (level 0 of consolidated book)
        column_map = {
            "ts_event": "ts_utc",
            "symbol": "option_symbol",
            # CBBO-1m format (level-0 suffixed)
            "bid_px_00": "bid",
            "ask_px_00": "ask",
            "bid_sz_00": "bid_size",
            "ask_sz_00": "ask_size",
            # Fallback for other schemas (mbp-1, etc.)
            "bid_px": "bid",
            "ask_px": "ask",
            "bid_sz": "bid_size",
            "ask_sz": "ask_size",
        }

        df = df.rename(columns=column_map)

        # Ensure ts_utc is datetime
        if "ts_utc" in df.columns:
            df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True)

        # Select and order columns
        standard_columns = [
            "ts_utc",
            "option_symbol",
            "bid",
            "ask",
            "bid_size",
            "ask_size",
        ]
        available_columns = [col for col in standard_columns if col in df.columns]
        df = df[available_columns].reset_index(drop=True)

        # Drop rows with null timestamps (Databento sentinel values)
        if "ts_utc" in df.columns:
            df = df.dropna(subset=["ts_utc"]).reset_index(drop=True)

        # Deduplicate: keep first occurrence per (ts_utc, option_symbol)
        key_cols = ["ts_utc", "option_symbol"]
        if all(c in df.columns for c in key_cols) and df.duplicated(subset=key_cols).any():
            df = df.drop_duplicates(subset=key_cols, keep="first")

        # Downcast uint64 to int64 (SQLite doesn't support unsigned 64-bit)
        for col in df.select_dtypes(include=["uint64"]).columns:
            df[col] = df[col].astype("int64")

        return df
