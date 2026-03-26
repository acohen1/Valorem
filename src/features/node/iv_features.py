"""IV feature generation for node-level analysis.

This module computes implied volatility features from surface snapshots:
- IV changes over various horizons (1d, 5d, etc.)
- IV volatility (rolling standard deviation)
- IV momentum and mean reversion signals
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class IVFeatureConfig:
    """Configuration for IV feature generation.

    Attributes:
        change_periods: Periods for IV change calculation (in observations)
        rolling_windows: Windows for IV volatility (in observations)
        min_periods: Minimum observations for rolling calculations
    """

    change_periods: list[int] = field(default_factory=lambda: [1, 5])
    rolling_windows: list[int] = field(default_factory=lambda: [5, 10, 21])
    min_periods: int | None = None


class IVFeatureGenerator:
    """Generate IV-based features from surface snapshots.

    Computes features for each node (tenor, delta_bucket) time series:
    - iv_change_{n}d: Change in IV over n periods
    - iv_vol_{n}d: Rolling std of IV over n periods
    - iv_zscore_{n}d: Z-score of IV relative to rolling mean/std

    All features are computed using only past data (no lookahead).

    Example:
        config = IVFeatureConfig(change_periods=[1, 5], rolling_windows=[5, 21])
        generator = IVFeatureGenerator(config)
        features_df = generator.generate(surface_df)
    """

    def __init__(self, config: IVFeatureConfig | None = None):
        """Initialize IV feature generator.

        Args:
            config: Feature configuration. Uses defaults if None.
        """
        self._config = config or IVFeatureConfig()

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate IV features for all nodes.

        Args:
            df: Surface snapshot DataFrame with columns:
                - ts_utc: Timestamp
                - tenor_days: Tenor in days
                - delta_bucket: Delta bucket name
                - iv_mid: Mid IV value

        Returns:
            DataFrame with original columns plus IV features
        """
        if df.empty:
            return df.copy()

        # Validate required columns
        required_cols = ["ts_utc", "tenor_days", "delta_bucket", "iv_mid"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Sort for proper time ordering within each node
        df = df.sort_values(["tenor_days", "delta_bucket", "ts_utc"]).copy()

        # Group by node and compute features
        result_dfs = []
        for (tenor, bucket), group in df.groupby(["tenor_days", "delta_bucket"]):
            node_features = self._compute_node_features(group)
            result_dfs.append(node_features)

        if not result_dfs:
            return df

        return pd.concat(result_dfs, ignore_index=True)

    def _compute_node_features(self, node_df: pd.DataFrame) -> pd.DataFrame:
        """Compute IV features for a single node time series.

        Args:
            node_df: DataFrame for a single (tenor, delta_bucket) node

        Returns:
            DataFrame with IV features added
        """
        node_df = node_df.copy()
        iv_series = node_df["iv_mid"]
        steps_per_day = self._infer_steps_per_day(node_df["ts_utc"])

        # IV changes
        for period in self._config.change_periods:
            col_name = f"iv_change_{period}d"
            node_df[col_name] = iv_series.diff(period * steps_per_day)

        # IV volatility (rolling std) — shift(1) excludes current observation
        iv_shifted = iv_series.shift(1)
        for window in self._config.rolling_windows:
            window_steps = window * steps_per_day
            col_name = f"iv_vol_{window}d"
            node_df[col_name] = iv_shifted.rolling(
                window=window_steps,
                min_periods=self._config.min_periods,
            ).std()

        # IV z-score relative to rolling mean/std — shift(1) excludes current observation
        for window in self._config.rolling_windows:
            window_steps = window * steps_per_day
            rolling_mean = iv_shifted.rolling(
                window=window_steps,
                min_periods=self._config.min_periods,
            ).mean()
            rolling_std = iv_shifted.rolling(
                window=window_steps,
                min_periods=self._config.min_periods,
            ).std()

            # Avoid division by zero
            col_name = f"iv_zscore_{window}d"
            with np.errstate(divide="ignore", invalid="ignore"):
                zscore = (iv_series - rolling_mean) / rolling_std
                node_df[col_name] = zscore.replace([np.inf, -np.inf], np.nan)

        return node_df

    def compute_iv_changes(
        self, df: pd.DataFrame, periods: list[int] | None = None
    ) -> pd.DataFrame:
        """Compute only IV change features (standalone utility).

        Args:
            df: DataFrame with iv_mid column, pre-sorted by time within node
            periods: Periods to compute. Uses config default if None.

        Returns:
            DataFrame with IV change columns added
        """
        periods = periods or self._config.change_periods
        df = df.copy()
        steps_per_day = self._infer_steps_per_day(df["ts_utc"]) if "ts_utc" in df.columns else 1

        for period in periods:
            df[f"iv_change_{period}d"] = df["iv_mid"].diff(period * steps_per_day)

        return df

    def compute_iv_volatility(
        self, df: pd.DataFrame, windows: list[int] | None = None
    ) -> pd.DataFrame:
        """Compute only IV volatility features (standalone utility).

        Args:
            df: DataFrame with iv_mid column, pre-sorted by time within node
            windows: Rolling windows to use. Uses config default if None.

        Returns:
            DataFrame with IV volatility columns added
        """
        windows = windows or self._config.rolling_windows
        df = df.copy()
        steps_per_day = self._infer_steps_per_day(df["ts_utc"]) if "ts_utc" in df.columns else 1

        iv_shifted = df["iv_mid"].shift(1)
        for window in windows:
            window_steps = window * steps_per_day
            df[f"iv_vol_{window}d"] = iv_shifted.rolling(
                window=window_steps,
                min_periods=self._config.min_periods,
            ).std()

        return df

    @staticmethod
    def _infer_steps_per_day(ts: pd.Series) -> int:
        """Infer median observations per trading day."""
        if ts.empty:
            return 1
        counts = pd.to_datetime(ts).dt.normalize().value_counts()
        if counts.empty:
            return 1
        return max(1, int(np.median(counts.values)))
