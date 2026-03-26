"""Realized volatility feature generator.

Computes realized variance, volatility-of-volatility, and drawdown features
from underlying price data.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class RealizedVolConfig:
    """Configuration for realized volatility features.

    Attributes:
        variance_windows: Windows for realized variance (in days).
        vol_of_vol_window: Window for volatility-of-volatility.
        drawdown_window: Window for max drawdown calculation.
        annualization_factor: Factor to annualize variance (252 trading days).
        min_periods: Minimum periods for rolling calculations.
    """

    variance_windows: list[int] = field(default_factory=lambda: [5, 10, 21])
    vol_of_vol_window: int = 21
    drawdown_window: int = 252
    annualization_factor: int = 252
    min_periods: int = 2


class RealizedVolGenerator:
    """Generate realized volatility features from underlying price data.

    Features generated:
    - rv_{n}d: Realized variance (annualized) over n days
    - realized_vol_{n}d: Realized volatility (sqrt of variance)
    - vol_of_vol_{n}d: Rolling std of rolling std (volatility of volatility)
    - drawdown: Current drawdown from rolling max
    - max_drawdown_{n}d: Maximum drawdown over n-day window

    Example:
        >>> generator = RealizedVolGenerator()
        >>> result = generator.generate(underlying_df)
        >>> print(result["rv_21d"].mean())
        0.045  # ~4.5% annualized variance
    """

    def __init__(self, config: RealizedVolConfig | None = None):
        """Initialize realized volatility generator.

        Args:
            config: Realized volatility configuration. Uses defaults if None.
        """
        self._config = config or RealizedVolConfig()

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate realized volatility features from underlying bars.

        Args:
            df: DataFrame with underlying bars. Must have columns:
                - ts_utc: Timestamp
                - close: Closing price

        Returns:
            DataFrame with original columns plus volatility features.

        Raises:
            ValueError: If required columns are missing.
        """
        if df.empty:
            return df.copy()

        required_cols = {"ts_utc", "close"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        result = df.copy()
        result = result.sort_values("ts_utc").reset_index(drop=True)
        steps_per_day = self._infer_steps_per_day(result["ts_utc"])

        # Compute 1-trading-day returns (needed for variance calculation).
        # shift(1) excludes the current bar's return from rolling windows,
        # consistent with IV features (iv_features.py) and microstructure
        # features (microstructure.py). Without the shift, rv_5d at time t
        # includes the return from close[t-1] to close[t], leaking
        # same-bar information into the feature.
        result["_returns_1d"] = result["close"].pct_change(steps_per_day).shift(1)

        # Realized variance for each window
        for window in self._config.variance_windows:
            result = self._add_realized_variance(result, window, steps_per_day)

        # Volatility of volatility
        result = self._add_vol_of_vol(result, steps_per_day)

        # Drawdown features
        result = self._add_drawdown(result, steps_per_day)

        # Drop temporary columns
        result = result.drop(columns=["_returns_1d"])

        return result

    def _add_realized_variance(
        self, df: pd.DataFrame, window: int, steps_per_day: int = 1
    ) -> pd.DataFrame:
        """Add realized variance and volatility for a window.

        Realized variance is computed as the rolling variance of returns,
        annualized by multiplying by the annualization factor.

        Args:
            df: DataFrame with _returns_1d column.
            window: Rolling window size.

        Returns:
            DataFrame with rv_{window}d and realized_vol_{window}d columns.
        """
        window_steps = window * steps_per_day

        # Realized variance (annualized)
        df[f"rv_{window}d"] = (
            df["_returns_1d"]
            .rolling(window=window_steps, min_periods=self._config.min_periods)
            .var()
            * self._config.annualization_factor
        )

        # Realized volatility (annualized)
        df[f"realized_vol_{window}d"] = np.sqrt(df[f"rv_{window}d"])

        return df

    def _add_vol_of_vol(self, df: pd.DataFrame, steps_per_day: int = 1) -> pd.DataFrame:
        """Add volatility-of-volatility feature.

        Vol-of-vol is computed as the rolling std of the rolling std of returns.
        This captures changes in volatility regime.

        Args:
            df: DataFrame with _returns_1d column.

        Returns:
            DataFrame with vol_of_vol column.
        """
        window_days = self._config.vol_of_vol_window
        window_steps = window_days * steps_per_day
        min_periods = self._config.min_periods

        # First compute rolling volatility
        rolling_vol = df["_returns_1d"].rolling(window=window_steps, min_periods=min_periods).std()

        # Then compute volatility of that volatility
        df[f"vol_of_vol_{window_days}d"] = rolling_vol.rolling(
            window=window_steps, min_periods=min_periods
        ).std()

        return df

    def _add_drawdown(self, df: pd.DataFrame, steps_per_day: int = 1) -> pd.DataFrame:
        """Add drawdown features.

        Drawdown is computed as the current price relative to the rolling maximum.

        Args:
            df: DataFrame with close column.

        Returns:
            DataFrame with drawdown and max_drawdown columns.
        """
        window_days = self._config.drawdown_window
        window_steps = window_days * steps_per_day

        # Rolling maximum price
        rolling_max = df["close"].rolling(window=window_steps, min_periods=1).max()

        # Current drawdown (negative value, 0 = at peak)
        df["drawdown"] = df["close"] / rolling_max - 1

        # Maximum drawdown over the window (most negative point)
        df[f"max_drawdown_{window_days}d"] = df["drawdown"].rolling(
            window=window_steps, min_periods=1
        ).min()

        return df

    def compute_realized_variance(
        self,
        returns: pd.Series,
        window: int,
        annualize: bool = True,
    ) -> pd.Series:
        """Compute realized variance from returns.

        Standalone utility method.

        Args:
            returns: Series of returns.
            window: Rolling window size.
            annualize: Whether to annualize the variance.

        Returns:
            Series of realized variance values.
        """
        variance = returns.rolling(
            window=window, min_periods=self._config.min_periods
        ).var()

        if annualize:
            variance = variance * self._config.annualization_factor

        return variance

    def compute_drawdown(self, prices: pd.Series, window: int | None = None) -> pd.Series:
        """Compute drawdown from prices.

        Standalone utility method.

        Args:
            prices: Series of prices.
            window: Rolling window for max. Uses config default if None.

        Returns:
            Series of drawdown values (negative, 0 = at peak).
        """
        window = window or self._config.drawdown_window
        rolling_max = prices.rolling(window=window, min_periods=1).max()
        return prices / rolling_max - 1

    @staticmethod
    def _infer_steps_per_day(ts: pd.Series) -> int:
        """Infer median observations per trading day."""
        if ts.empty:
            return 1
        counts = pd.to_datetime(ts).dt.normalize().value_counts()
        if counts.empty:
            return 1
        return max(1, int(np.median(counts.values)))
