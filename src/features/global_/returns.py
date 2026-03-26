"""Underlying returns feature generator.

Computes simple and log returns at various horizons from underlying price data.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class ReturnsConfig:
    """Configuration for returns feature generation.

    Attributes:
        periods: List of return periods (in days).
        include_log_returns: Whether to include log returns.
        include_simple_returns: Whether to include simple returns.
    """

    periods: list[int] = field(default_factory=lambda: [1, 5, 10, 21])
    include_log_returns: bool = True
    include_simple_returns: bool = True


class ReturnsGenerator:
    """Generate return features from underlying price data.

    Features generated:
    - returns_{n}d: Simple returns over n days (pct_change)
    - log_returns_{n}d: Log returns over n days

    Example:
        >>> generator = ReturnsGenerator()
        >>> result = generator.generate(underlying_df)
        >>> print(result.columns)
        Index(['ts_utc', 'close', 'returns_1d', 'returns_5d', ...])
    """

    def __init__(self, config: ReturnsConfig | None = None):
        """Initialize returns generator.

        Args:
            config: Returns configuration. Uses defaults if None.
        """
        self._config = config or ReturnsConfig()

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate return features from underlying bars.

        Args:
            df: DataFrame with underlying bars. Must have columns:
                - ts_utc: Timestamp
                - close: Closing price

        Returns:
            DataFrame with original columns plus return features.

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

        # Compute returns for each period
        for period in self._config.periods:
            period_steps = period * steps_per_day
            shifted = result["close"].shift(period_steps)

            if self._config.include_simple_returns:
                with np.errstate(divide="ignore", invalid="ignore"):
                    simple = result["close"].diff(period_steps) / shifted
                result[f"returns_{period}d"] = simple.replace(
                    [np.inf, -np.inf], np.nan
                )

            if self._config.include_log_returns:
                ratio = result["close"] / shifted
                result[f"log_returns_{period}d"] = np.where(
                    ratio > 0, np.log(ratio), np.nan
                )

        return result

    def compute_returns(
        self,
        prices: pd.Series,
        periods: list[int] | None = None,
    ) -> pd.DataFrame:
        """Compute returns for a price series.

        Standalone utility method for computing returns.

        Args:
            prices: Series of prices.
            periods: List of return periods. Uses config if None.

        Returns:
            DataFrame with return columns.
        """
        periods = periods or self._config.periods
        result = pd.DataFrame(index=prices.index)

        for period in periods:
            shifted = prices.shift(period)

            if self._config.include_simple_returns:
                with np.errstate(divide="ignore", invalid="ignore"):
                    simple = prices.diff(period) / shifted
                result[f"returns_{period}d"] = simple.replace(
                    [np.inf, -np.inf], np.nan
                )

            if self._config.include_log_returns:
                ratio = prices / shifted
                result[f"log_returns_{period}d"] = np.where(
                    ratio > 0, np.log(ratio), np.nan
                )

        return result

    @staticmethod
    def _infer_steps_per_day(ts: pd.Series) -> int:
        """Infer median observations per trading day."""
        if ts.empty:
            return 1
        counts = pd.to_datetime(ts).dt.normalize().value_counts()
        if counts.empty:
            return 1
        return max(1, int(np.median(counts.values)))
