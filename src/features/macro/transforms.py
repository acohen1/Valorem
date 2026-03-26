"""Macro feature transforms.

Computes level, change, and z-score features from FRED series data.
All features respect release-time alignment to prevent data leakage.
"""

from dataclasses import dataclass, field
from typing import Literal

import pandas as pd

from src.config.constants import MarketConstants
from src.features.macro.alignment import AlignmentConfig, ReleaseTimeAligner


@dataclass
class MacroTransformConfig:
    """Configuration for macro transforms.

    Attributes:
        include_level: Include raw level feature.
        include_change_1w: Include 1-week change feature.
        include_change_1m: Include 1-month change feature.
        include_zscore: Include rolling z-score feature.
        zscore_window: Rolling window for z-score calculation.
        percent_to_decimal: Whether to convert percent values to decimal.
        percent_series: Series IDs that should be converted from percent.
        alignment: Release-time alignment configuration.
    """

    include_level: bool = True
    include_change_1w: bool = True
    include_change_1m: bool = True
    include_zscore: bool = True
    zscore_window: int = 252
    percent_to_decimal: bool = False  # FRED provider already converts % to decimal
    percent_series: list[str] = field(
        default_factory=lambda: [s for s in MarketConstants.EXTENDED_FRED_SERIES if s != "VIXCLS"]
    )
    alignment: AlignmentConfig = field(default_factory=AlignmentConfig)


class MacroTransformGenerator:
    """Generate transformed features from FRED series.

    Features generated for each series:
    - {series}_level: Raw value (optionally converted from percent)
    - {series}_change_1w: 1-week absolute change
    - {series}_change_1m: 1-month absolute change
    - {series}_zscore: Rolling z-score (standardized value)

    All features respect release-time alignment to prevent lookahead bias.

    Example:
        >>> generator = MacroTransformGenerator()
        >>> result = generator.generate(fred_df, "DGS10")
        >>> print(result.columns)
        Index(['ts_utc', 'DGS10_level', 'DGS10_change_1w', ...])
    """

    # Change periods in days
    CHANGE_PERIODS = {
        "1w": 7,
        "1m": 30,
    }

    def __init__(self, config: MacroTransformConfig | None = None):
        """Initialize macro transform generator.

        Args:
            config: Transform configuration. Uses defaults if None.
        """
        self._config = config or MacroTransformConfig()
        self._aligner = ReleaseTimeAligner(self._config.alignment)

    def generate(
        self,
        df: pd.DataFrame,
        series_id: str,
    ) -> pd.DataFrame:
        """Generate all transform features for a FRED series.

        Args:
            df: DataFrame with FRED series data. Must have columns:
                - obs_date: Observation date
                - value: Series value
                - release_datetime_utc (optional): Release timestamp

        Returns:
            DataFrame with ts_utc and all transform columns.

        Raises:
            ValueError: If required columns are missing.
        """
        if df.empty:
            return pd.DataFrame(columns=["ts_utc"])

        required_cols = {"obs_date", "value"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Align to release times
        aligned = self._aligner.align(df)

        # Convert percent to decimal if applicable
        if self._config.percent_to_decimal and series_id in self._config.percent_series:
            aligned["value"] = aligned["value"] / 100.0

        result = aligned[["ts_utc"]].copy()

        # Generate each transform
        if self._config.include_level:
            result = self._add_level(result, aligned, series_id)

        if self._config.include_change_1w:
            result = self._add_change(result, aligned, series_id, "1w")

        if self._config.include_change_1m:
            result = self._add_change(result, aligned, series_id, "1m")

        if self._config.include_zscore:
            result = self._add_zscore(result, aligned, series_id)

        return result

    def generate_multi(
        self,
        series_data: dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        """Generate transforms for multiple FRED series and merge.

        Args:
            series_data: Dict mapping series_id to DataFrame.

        Returns:
            Merged DataFrame with all series transforms.
        """
        if not series_data:
            return pd.DataFrame(columns=["ts_utc"])

        results = []
        for series_id, df in series_data.items():
            if not df.empty:
                result = self.generate(df, series_id)
                results.append(result)

        if not results:
            return pd.DataFrame(columns=["ts_utc"])

        # Merge all results on ts_utc
        merged = results[0]
        for result in results[1:]:
            merged = merged.merge(result, on="ts_utc", how="outer")

        return merged.sort_values("ts_utc").reset_index(drop=True)

    def _add_level(
        self,
        result: pd.DataFrame,
        aligned: pd.DataFrame,
        series_id: str,
    ) -> pd.DataFrame:
        """Add level (raw value) feature.

        Args:
            result: Result DataFrame to add column to.
            aligned: Aligned DataFrame with value column.
            series_id: FRED series ID for column naming.

        Returns:
            DataFrame with {series_id}_level column.
        """
        result[f"{series_id}_level"] = aligned["value"].values
        return result

    def _add_change(
        self,
        result: pd.DataFrame,
        aligned: pd.DataFrame,
        series_id: str,
        period: Literal["1w", "1m"],
    ) -> pd.DataFrame:
        """Add change feature for a period.

        Uses calendar-day lookup: for each row, finds the most recent prior
        observation at least ``lag_days`` calendar days earlier, then diffs.
        This gives correct 1w/1m semantics regardless of the publication
        frequency of the underlying FRED series.

        Args:
            result: Result DataFrame to add column to.
            aligned: Aligned DataFrame with value column.
            series_id: FRED series ID for column naming.
            period: Change period ("1w" or "1m").

        Returns:
            DataFrame with {series_id}_change_{period} column.
        """
        import numpy as np

        lag_days = self.CHANGE_PERIODS[period]
        ts = pd.to_datetime(aligned["ts_utc"])
        values = aligned["value"].values

        # Vectorised calendar-day lag via searchsorted.
        # For each ts[i], find the rightmost ts <= ts[i] - lag_days.
        target_ts = ts - pd.Timedelta(days=lag_days)
        lag_pos = np.searchsorted(ts.values, target_ts.values, side="right") - 1

        changes = np.full(len(values), np.nan)
        valid = lag_pos >= 0
        changes[valid] = values[valid] - values[lag_pos[valid]]

        result[f"{series_id}_change_{period}"] = changes
        return result

    def _add_zscore(
        self,
        result: pd.DataFrame,
        aligned: pd.DataFrame,
        series_id: str,
    ) -> pd.DataFrame:
        """Add rolling z-score feature.

        Z-score = (value - rolling_mean) / rolling_std

        Args:
            result: Result DataFrame to add column to.
            aligned: Aligned DataFrame with value column.
            series_id: FRED series ID for column naming.

        Returns:
            DataFrame with {series_id}_zscore column.
        """
        window = self._config.zscore_window
        values = aligned["value"]

        # shift(1) excludes current observation from its own z-score stats.
        # min_periods=max(10, window//10) avoids degenerate early z-scores
        # (with min_periods=2, the first z-score is always ±0.707).
        shifted = values.shift(1)
        safe_min_periods = min(window, max(10, window // 10))
        rolling_mean = shifted.rolling(window=window, min_periods=safe_min_periods).mean()
        rolling_std = shifted.rolling(window=window, min_periods=safe_min_periods).std()

        # Avoid division by zero
        zscore = (values - rolling_mean) / rolling_std.replace(0, float("nan"))

        result[f"{series_id}_zscore"] = zscore.values
        return result

    def compute_change(
        self,
        values: pd.Series,
        period: Literal["1w", "1m"],
    ) -> pd.Series:
        """Compute change for a series.

        Standalone utility method.

        Args:
            values: Series of values.
            period: Change period.

        Returns:
            Series of changes.
        """
        lag_days = self.CHANGE_PERIODS[period]
        return values.diff(lag_days)

    def compute_zscore(
        self,
        values: pd.Series,
        window: int | None = None,
    ) -> pd.Series:
        """Compute rolling z-score for a series.

        Standalone utility method.

        Args:
            values: Series of values.
            window: Rolling window. Uses config if None.

        Returns:
            Series of z-scores.
        """
        window = window or self._config.zscore_window
        safe_min_periods = min(window, max(10, window // 10))

        shifted = values.shift(1)
        rolling_mean = shifted.rolling(window=window, min_periods=safe_min_periods).mean()
        rolling_std = shifted.rolling(window=window, min_periods=safe_min_periods).std()

        return (values - rolling_mean) / rolling_std.replace(0, float("nan"))
