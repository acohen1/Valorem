"""Mock implementations of data provider protocols for testing.

These mock providers generate realistic synthetic data for unit and integration tests.
They implement the same protocols as real providers, enabling seamless testing.
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd


class MockMarketDataProvider:
    """Mock market data provider for testing.

    Generates synthetic but realistic OHLCV bars and option quotes.
    Useful for unit tests and integration tests without external API dependencies.
    """

    def __init__(self, seed: int = 42):
        """Initialize mock provider.

        Args:
            seed: Random seed for reproducible data generation
        """
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    def fetch_underlying_bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        interval: str = "1m",
        suppress_error_log: bool = False,
    ) -> pd.DataFrame:
        """Fetch synthetic OHLCV bars for underlying asset.

        Args:
            symbol: Underlying symbol (e.g., "SPY")
            start: Start datetime (inclusive)
            end: End datetime (exclusive)
            interval: Bar interval (e.g., "1m", "5m", "1h", "1d")
            suppress_error_log: If True, suppress logging of expected errors

        Returns:
            DataFrame with columns: [ts_utc, open, high, low, close, volume]

        Raises:
            ValueError: If interval is not supported
        """
        if interval not in ["1m", "5m", "1h", "1d"]:
            raise ValueError(f"Unsupported interval: {interval}")

        # Handle empty range
        if start >= end:
            return pd.DataFrame(columns=["ts_utc", "open", "high", "low", "close", "volume"])

        # Generate time index
        freq_map = {"1m": "1min", "5m": "5min", "1h": "1h", "1d": "1D"}
        freq = freq_map[interval]
        ts_index = pd.date_range(start, end, freq=freq, inclusive="left")

        if len(ts_index) == 0:
            return pd.DataFrame(columns=["ts_utc", "open", "high", "low", "close", "volume"])

        # Generate realistic price series
        # Start at 400 for SPY, scale for other symbols
        base_price = 400.0 if symbol == "SPY" else 100.0
        n_bars = len(ts_index)

        # Generate random walk for close prices
        returns = self._rng.normal(0.0, 0.001, n_bars)  # ~0.1% volatility per bar
        prices = base_price * np.exp(np.cumsum(returns))

        # Generate OHLC from close prices
        close = prices
        open_ = np.roll(close, 1)
        open_[0] = base_price

        # High/low with realistic intrabar movement
        intrabar_range = self._rng.uniform(0.0005, 0.002, n_bars)  # 0.05-0.2% range
        high = close * (1 + intrabar_range)
        low = close * (1 - intrabar_range)

        # Ensure OHLC consistency
        high = np.maximum(high, np.maximum(open_, close))
        low = np.minimum(low, np.minimum(open_, close))

        # Generate volume
        base_volume = 1_000_000 if symbol == "SPY" else 100_000
        volume = self._rng.integers(
            int(base_volume * 0.5), int(base_volume * 1.5), n_bars
        )

        df = pd.DataFrame(
            {
                "ts_utc": ts_index,
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            }
        )

        return df

    def fetch_option_quotes(
        self,
        symbols: list[str],
        start: datetime,
        end: datetime,
        schema: str = "cbbo-1m",
    ) -> pd.DataFrame:
        """Fetch synthetic option quote data.

        Args:
            symbols: List of option symbols (e.g., ["SPY230120C00400000"])
            start: Start datetime (inclusive)
            end: End datetime (exclusive)
            schema: Data schema (e.g., "cbbo-1m")

        Returns:
            DataFrame with columns: [ts_utc, option_symbol, bid, ask, bid_size, ask_size]

        Raises:
            ValueError: If schema is not supported
        """
        if schema not in ["cbbo-1m", "mbp-1"]:
            raise ValueError(f"Unsupported schema: {schema}")

        if not symbols:
            return pd.DataFrame(
                columns=["ts_utc", "option_symbol", "bid", "ask", "bid_size", "ask_size"]
            )

        # Handle empty range
        if start >= end:
            return pd.DataFrame(
                columns=["ts_utc", "option_symbol", "bid", "ask", "bid_size", "ask_size"]
            )

        # Generate time index
        ts_index = pd.date_range(start, end, freq="1min", inclusive="left")

        if len(ts_index) == 0:
            return pd.DataFrame(
                columns=["ts_utc", "option_symbol", "bid", "ask", "bid_size", "ask_size"]
            )

        # Generate quotes for each symbol
        dfs = []
        for symbol in symbols:
            n_quotes = len(ts_index)

            # Base option price (simplified)
            base_price = 5.0
            mid_prices = base_price + self._rng.normal(0, 0.1, n_quotes)
            mid_prices = np.maximum(mid_prices, 0.01)  # Floor at $0.01

            # Generate bid/ask spread (1-2%)
            spread_pct = self._rng.uniform(0.01, 0.02, n_quotes)
            bid = mid_prices * (1 - spread_pct / 2)
            ask = mid_prices * (1 + spread_pct / 2)

            # Generate sizes
            bid_size = self._rng.integers(10, 100, n_quotes)
            ask_size = self._rng.integers(10, 100, n_quotes)

            df = pd.DataFrame(
                {
                    "ts_utc": ts_index,
                    "option_symbol": symbol,
                    "bid": bid,
                    "ask": ask,
                    "bid_size": bid_size,
                    "ask_size": ask_size,
                }
            )
            dfs.append(df)

        return pd.concat(dfs, ignore_index=True)

    def fetch_option_bars(
        self,
        symbols: list[str],
        start: datetime,
        end: datetime,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Fetch synthetic OHLCV bars for option contracts.

        Args:
            symbols: List of option symbols
            start: Start datetime (inclusive)
            end: End datetime (exclusive)
            interval: Bar interval (e.g., "1m", "1h", "1d")

        Returns:
            DataFrame with columns: [ts_utc, option_symbol, open, high, low, close, volume]
        """
        if interval not in ["1m", "1h", "1d"]:
            raise ValueError(f"Unsupported interval: {interval}")

        if not symbols:
            return pd.DataFrame(
                columns=["ts_utc", "option_symbol", "open", "high", "low", "close", "volume"]
            )

        if start >= end:
            return pd.DataFrame(
                columns=["ts_utc", "option_symbol", "open", "high", "low", "close", "volume"]
            )

        freq_map = {"1m": "1min", "1h": "1h", "1d": "1D"}
        ts_index = pd.date_range(start, end, freq=freq_map[interval], inclusive="left")

        if len(ts_index) == 0:
            return pd.DataFrame(
                columns=["ts_utc", "option_symbol", "open", "high", "low", "close", "volume"]
            )

        dfs = []
        for symbol in symbols:
            n_bars = len(ts_index)
            base_price = 5.0
            mid_prices = base_price + self._rng.normal(0, 0.1, n_bars)
            mid_prices = np.maximum(mid_prices, 0.01)

            intrabar_range = self._rng.uniform(0.005, 0.02, n_bars)
            high = mid_prices * (1 + intrabar_range)
            low = mid_prices * (1 - intrabar_range)
            open_ = mid_prices + self._rng.normal(0, 0.05, n_bars)
            close = mid_prices + self._rng.normal(0, 0.05, n_bars)

            high = np.maximum(high, np.maximum(open_, close))
            low = np.minimum(low, np.minimum(open_, close))

            volume = self._rng.integers(100, 10000, n_bars)

            df = pd.DataFrame({
                "ts_utc": ts_index,
                "option_symbol": symbol,
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            })
            dfs.append(df)

        return pd.concat(dfs, ignore_index=True)

    def fetch_option_statistics(
        self,
        symbols: list[str],
        start: datetime,
        end: datetime,
        stat_types: list[int] | None = None,
    ) -> pd.DataFrame:
        """Fetch synthetic option statistics (daily OI).

        Args:
            symbols: List of option symbols
            start: Start datetime (inclusive)
            end: End datetime (exclusive)
            stat_types: Stat type filter (default: [9] = open interest)

        Returns:
            DataFrame with columns: [ts_ref, ts_event, option_symbol, stat_type, quantity, price]
        """
        if stat_types is None:
            stat_types = [9]

        if not symbols or start >= end:
            return pd.DataFrame(
                columns=["ts_ref", "ts_event", "option_symbol", "stat_type", "quantity", "price"]
            )

        ts_index = pd.date_range(start, end, freq="1D", inclusive="left")
        if len(ts_index) == 0:
            return pd.DataFrame(
                columns=["ts_ref", "ts_event", "option_symbol", "stat_type", "quantity", "price"]
            )

        dfs = []
        for symbol in symbols:
            for stat_type in stat_types:
                n = len(ts_index)
                quantity = self._rng.integers(100, 50000, n) if stat_type == 9 else None
                price = self._rng.uniform(0.5, 10.0, n) if stat_type != 9 else None

                df = pd.DataFrame({
                    "ts_ref": ts_index,
                    "ts_event": ts_index,
                    "option_symbol": symbol,
                    "stat_type": stat_type,
                    "quantity": quantity,
                    "price": price,
                })
                dfs.append(df)

        return pd.concat(dfs, ignore_index=True)

    def estimate_cost(
        self,
        dataset: str,
        schema: str,
        symbols: list[str],
        start: datetime,
        end: datetime,
    ) -> float:
        """Estimate USD cost (always $0 for mock).

        Args:
            dataset: Dataset identifier
            schema: Schema identifier
            symbols: List of symbols
            start: Start datetime
            end: End datetime

        Returns:
            Estimated cost in USD (always 0.0 for mock)
        """
        return 0.0

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
        """Resolve synthetic option symbols.

        Generates realistic option symbols matching the criteria.

        Args:
            parent: Parent symbol (e.g., "SPY")
            as_of: Reference datetime
            dte_min: Minimum days to expiration
            dte_max: Maximum days to expiration
            moneyness_min: Minimum strike/spot ratio
            moneyness_max: Maximum strike/spot ratio
            max_available_date: Maximum date that data is available (unused in mock)

        Returns:
            List of synthetic option symbols

        Raises:
            ValueError: If parent symbol is invalid
        """
        if not parent:
            raise ValueError("Parent symbol cannot be empty")

        # Generate expiration dates
        expirations = []
        current_date = as_of.date()
        for days in range(dte_min, dte_max + 1, 7):  # Weekly expirations
            exp_date = current_date + timedelta(days=days)
            expirations.append(exp_date)

        # Generate strikes (assume spot = 400 for SPY)
        spot = 400.0 if parent == "SPY" else 100.0
        strikes = []
        for moneyness in np.arange(moneyness_min, moneyness_max + 0.05, 0.05):
            strike = round(spot * moneyness)
            if strike not in strikes:
                strikes.append(strike)

        # Generate option symbols
        symbols = []
        for exp_date in expirations:
            exp_str = exp_date.strftime("%y%m%d")
            for strike in strikes:
                # Format: SPY230120C00400000 (Call) and SPY230120P00400000 (Put)
                for right in ["C", "P"]:
                    symbol = f"{parent}{exp_str}{right}{int(strike * 1000):08d}"
                    symbols.append(symbol)

        return symbols[:50]  # Limit to 50 symbols for testing


class MockMacroDataProvider:
    """Mock macro data provider for testing.

    Generates synthetic but realistic macro time series data.
    """

    def __init__(self, seed: int = 42):
        """Initialize mock provider.

        Args:
            seed: Random seed for reproducible data generation
        """
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    def fetch_series(
        self,
        series_id: str,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        """Fetch synthetic macro time series.

        Args:
            series_id: Series identifier (e.g., "DGS10")
            start: Start datetime (inclusive)
            end: End datetime (exclusive)

        Returns:
            DataFrame with columns: [obs_date, value, release_datetime_utc]

        Raises:
            ValueError: If series_id is invalid
        """
        if not series_id:
            raise ValueError("Series ID cannot be empty")

        # Handle empty range
        if start >= end:
            return pd.DataFrame(columns=["obs_date", "value", "release_datetime_utc"])

        # Generate daily observations
        date_range = pd.date_range(start.date(), end.date(), freq="D", inclusive="left")

        if len(date_range) == 0:
            return pd.DataFrame(columns=["obs_date", "value", "release_datetime_utc"])

        n_obs = len(date_range)

        # Generate realistic values based on series type
        if series_id.startswith("DGS"):  # Treasury rates
            base_value = 4.5  # Base rate around 4.5%
            values = base_value + self._rng.normal(0, 0.1, n_obs)
            values = np.maximum(values, 0.01)  # Floor at 0.01%
        elif series_id == "VIXCLS":  # VIX
            base_value = 20.0
            values = base_value + self._rng.normal(0, 2.0, n_obs)
            values = np.maximum(values, 10.0)  # Floor at 10
        else:
            # Generic series
            values = 100.0 + self._rng.normal(0, 5.0, n_obs)

        # Release datetime is typically same day at 15:00 UTC
        release_times = [
            datetime.combine(d, datetime.min.time()).replace(hour=15, tzinfo=None)
            for d in date_range
        ]

        df = pd.DataFrame(
            {
                "obs_date": date_range,
                "value": values,
                "release_datetime_utc": release_times,
            }
        )

        return df

    def get_latest_value(
        self,
        series_id: str,
        as_of: datetime,
    ) -> tuple[datetime, float]:
        """Get latest synthetic value as of timestamp.

        Args:
            series_id: Series identifier
            as_of: Reference timestamp

        Returns:
            Tuple of (release_datetime_utc, value)

        Raises:
            ValueError: If series_id is invalid or no data before as_of
        """
        if not series_id:
            raise ValueError("Series ID cannot be empty")

        # Mock data only available from 2020 onwards
        min_date = datetime(2020, 1, 1)
        if as_of < min_date:
            raise ValueError(f"No data available for {series_id} before {as_of}")

        # Fetch data up to as_of
        start = max(as_of - timedelta(days=30), min_date)  # Look back 30 days
        df = self.fetch_series(series_id, start, as_of)

        if df.empty:
            raise ValueError(f"No data available for {series_id} before {as_of}")

        # Filter to only releases before as_of
        df = df[df["release_datetime_utc"] <= as_of]

        if df.empty:
            raise ValueError(f"No data released for {series_id} before {as_of}")

        # Return the latest release
        latest = df.iloc[-1]
        return latest["release_datetime_utc"], float(latest["value"])
