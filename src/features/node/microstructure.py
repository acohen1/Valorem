"""Microstructure feature generation for node-level analysis.

This module computes market microstructure features from surface snapshots:
- Spread dynamics (rolling averages, changes)
- Volume patterns (ratios, momentum)
- Open interest changes
- Quote stability metrics
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class MicrostructureConfig:
    """Configuration for microstructure feature generation.

    Attributes:
        volume_ratio_windows: Windows for volume ratio calculation
        oi_change_periods: Periods for OI change calculation
        min_periods: Minimum observations for rolling calculations
    """

    volume_ratio_windows: list[int] = field(default_factory=lambda: [5])
    oi_change_periods: list[int] = field(default_factory=lambda: [5])
    min_periods: int | None = None


class MicrostructureFeatureGenerator:
    """Generate microstructure features from surface snapshots.

    Computes features for each node (tenor, delta_bucket) time series:
    - log_volume: Log-transformed volume
    - volume_ratio_{n}d: Current volume / rolling mean volume
    - log_oi: Log-transformed open interest
    - oi_change_{n}d: Percent change in open interest

    All features are computed using only past data (no lookahead).

    Example:
        config = MicrostructureConfig()
        generator = MicrostructureFeatureGenerator(config)
        features_df = generator.generate(surface_df)
    """

    def __init__(self, config: MicrostructureConfig | None = None):
        """Initialize microstructure feature generator.

        Args:
            config: Feature configuration. Uses defaults if None.
        """
        self._config = config or MicrostructureConfig()

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate microstructure features for all nodes.

        Args:
            df: Surface snapshot DataFrame with columns:
                - ts_utc: Timestamp
                - tenor_days: Tenor in days
                - delta_bucket: Delta bucket name
                - volume: Trading volume (optional)
                - open_interest: Open interest (optional)

        Returns:
            DataFrame with original columns plus microstructure features
        """
        if df.empty:
            return df.copy()

        # Validate required columns
        required_cols = ["ts_utc", "tenor_days", "delta_bucket"]
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
        """Compute microstructure features for a single node time series.

        Args:
            node_df: DataFrame for a single (tenor, delta_bucket) node

        Returns:
            DataFrame with microstructure features added
        """
        node_df = node_df.copy()
        steps_per_day = self._infer_steps_per_day(node_df["ts_utc"])

        # Volume features (if available)
        if "volume" in node_df.columns:
            self._add_volume_features(node_df, steps_per_day)

        # Open interest features (if available)
        if "open_interest" in node_df.columns:
            self._add_oi_features(node_df, steps_per_day)

        return node_df

    def _add_volume_features(self, df: pd.DataFrame, steps_per_day: int = 1) -> None:
        """Add volume-related features to DataFrame in-place.

        Args:
            df: DataFrame with volume column
        """
        volume = df["volume"].astype(float)

        # shift(1) excludes current observation from rolling stats
        volume_shifted = volume.shift(1)
        for window in self._config.volume_ratio_windows:
            window_steps = window * steps_per_day
            # Volume ratio (current / rolling mean of prior observations)
            rolling_mean = volume_shifted.rolling(
                window=window_steps,
                min_periods=self._config.min_periods,
            ).mean()

            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = volume / rolling_mean
                df[f"volume_ratio_{window}d"] = ratio.replace([np.inf, -np.inf], np.nan)

        # Log volume for scale-invariance
        df["log_volume"] = np.log1p(volume)

    def _add_oi_features(self, df: pd.DataFrame, steps_per_day: int = 1) -> None:
        """Add open interest features to DataFrame in-place.

        Args:
            df: DataFrame with open_interest column
        """
        oi = df["open_interest"].astype(float)

        # OI percent changes (clipped to [-1, 10] to prevent extreme values
        # when OI jumps from near-zero to thousands)
        for period in self._config.oi_change_periods:
            period_steps = period * steps_per_day
            with np.errstate(divide="ignore", invalid="ignore"):
                pct_change = oi.diff(period_steps) / oi.shift(period_steps)
                pct_change = pct_change.replace([np.inf, -np.inf], np.nan)
                df[f"oi_change_{period}d"] = pct_change.clip(lower=-1.0, upper=10.0)

        # Log OI for scale-invariance
        df["log_oi"] = np.log1p(oi)

    def compute_volume_features(
        self, df: pd.DataFrame, windows: list[int] | None = None
    ) -> pd.DataFrame:
        """Compute only volume features (standalone utility).

        Args:
            df: DataFrame with volume column, pre-sorted by time
            windows: Rolling windows to use. Uses config default if None.

        Returns:
            DataFrame with volume feature columns added
        """
        if "volume" not in df.columns:
            raise ValueError("DataFrame must have 'volume' column")

        original_windows = self._config.volume_ratio_windows
        if windows is not None:
            self._config.volume_ratio_windows = windows

        df = df.copy()
        steps_per_day = self._infer_steps_per_day(df["ts_utc"]) if "ts_utc" in df.columns else 1
        self._add_volume_features(df, steps_per_day)

        self._config.volume_ratio_windows = original_windows
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
