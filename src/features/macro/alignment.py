"""Release-time alignment for macro features.

This module handles the complex timing of macroeconomic data releases.
FRED data has a publication lag - data for a given observation date is
only available after the release date.

Two alignment modes are supported:
- strict: Use actual release timestamps (requires release_datetime_utc column)
- conservative: Assume data is available N days after observation date
"""

from dataclasses import dataclass
from datetime import timedelta
from typing import Literal

import pandas as pd


@dataclass
class AlignmentConfig:
    """Configuration for release-time alignment.

    Attributes:
        mode: Alignment mode ('strict' or 'conservative').
        conservative_delay_days: Days to delay in conservative mode.
        time_column: Column name for output timestamps.
    """

    mode: Literal["strict", "conservative"] = "conservative"
    conservative_delay_days: int = 2
    time_column: str = "ts_utc"


class ReleaseTimeAligner:
    """Align macro data to respect release times.

    FRED data is published with a lag. For example, weekly jobless claims
    from Thursday are released the following Thursday. This aligner ensures
    features don't leak future information by respecting release timing.

    Modes:
    - strict: Use the actual release_datetime_utc from FRED API.
        Only rows with valid release timestamps are kept.
    - conservative: Shift observation dates by a fixed delay.
        All rows are kept, assuming data is available after delay.

    Example:
        >>> aligner = ReleaseTimeAligner(AlignmentConfig(mode='conservative'))
        >>> aligned = aligner.align(fred_df)
        >>> # Data is now timestamped by when it was available, not observed
    """

    def __init__(self, config: AlignmentConfig | None = None):
        """Initialize release-time aligner.

        Args:
            config: Alignment configuration. Uses defaults if None.
        """
        self._config = config or AlignmentConfig()

    def align(self, df: pd.DataFrame) -> pd.DataFrame:
        """Align DataFrame to release times.

        Args:
            df: DataFrame with FRED series data. Expected columns:
                - obs_date: Observation date
                - value: Series value
                - release_datetime_utc (optional): Release timestamp

        Returns:
            DataFrame with aligned timestamps in ts_utc column.

        Raises:
            ValueError: If required columns are missing.
        """
        if df.empty:
            return df.copy()

        required_cols = {"obs_date", "value"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        result = df.copy()

        if self._config.mode == "strict":
            result = self._align_strict(result)
        else:
            result = self._align_conservative(result)

        return result.sort_values(self._config.time_column).reset_index(drop=True)

    def _align_strict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Align using actual release timestamps.

        Only keeps rows where release_datetime_utc is available.

        Args:
            df: DataFrame with release_datetime_utc column.

        Returns:
            DataFrame with ts_utc from release timestamps.
        """
        if "release_datetime_utc" not in df.columns:
            raise ValueError(
                "Strict mode requires 'release_datetime_utc' column. "
                "Use conservative mode if release times are not available."
            )

        # Filter to rows with valid release timestamps
        result = df[df["release_datetime_utc"].notna()].copy()

        # Use release timestamp as the effective time
        result[self._config.time_column] = pd.to_datetime(result["release_datetime_utc"])

        return result

    def _align_conservative(self, df: pd.DataFrame) -> pd.DataFrame:
        """Align using conservative delay from observation date.

        Args:
            df: DataFrame with obs_date column.

        Returns:
            DataFrame with ts_utc shifted from obs_date.
        """
        result = df.copy()

        # Ensure obs_date is datetime
        obs_date = pd.to_datetime(result["obs_date"])

        # Shift by conservative delay
        delay = timedelta(days=self._config.conservative_delay_days)
        result[self._config.time_column] = obs_date + delay

        return result

    def get_effective_time(
        self,
        obs_date: pd.Timestamp,
        release_datetime: pd.Timestamp | None = None,
    ) -> pd.Timestamp:
        """Get effective time for a single observation.

        Utility method for getting the effective timestamp.

        Args:
            obs_date: Observation date.
            release_datetime: Actual release timestamp (optional).

        Returns:
            Effective timestamp when data is considered available.
        """
        if self._config.mode == "strict":
            if release_datetime is not None and pd.notna(release_datetime):
                return release_datetime
            raise ValueError("Strict mode requires release_datetime")

        return obs_date + timedelta(days=self._config.conservative_delay_days)

    def validate_no_leakage(
        self,
        df: pd.DataFrame,
        reference_times: pd.Series,
    ) -> bool:
        """Validate that aligned data doesn't leak future information.

        For each reference time in the feature panel, verifies that the
        most recent aligned macro observation has ts_utc <= reference_time.
        Uses merge_asof with direction='backward' to find what each
        reference time would see, then checks that no obs_date exceeds
        the reference (accounting for publication delay).

        Args:
            df: Aligned DataFrame with ts_utc and obs_date columns.
            reference_times: Series of reference timestamps to check.

        Returns:
            True if no leakage detected, False otherwise.
        """
        if df.empty or reference_times.empty:
            return True

        ts_col = self._config.time_column
        if ts_col not in df.columns:
            raise ValueError(f"DataFrame missing '{ts_col}' column")
        if "obs_date" not in df.columns:
            return True

        aligned_ts = pd.to_datetime(df[ts_col]).sort_values()
        ref_ts = pd.to_datetime(reference_times).sort_values()

        # For each reference time, the latest aligned observation must
        # have ts_utc <= reference_time (backward merge semantics).
        # Additionally, obs_date must be strictly before reference_time
        # to ensure the observation was knowable.
        ref_df = pd.DataFrame({"ref_ts": ref_ts})
        macro_df = df[["obs_date", ts_col]].copy()
        macro_df[ts_col] = pd.to_datetime(macro_df[ts_col])
        macro_df["obs_date"] = pd.to_datetime(macro_df["obs_date"])
        macro_df = macro_df.sort_values(ts_col)

        merged = pd.merge_asof(
            ref_df, macro_df, left_on="ref_ts", right_on=ts_col, direction="backward"
        )

        # Check: no observation date should be on or after the reference time
        valid = merged.dropna(subset=["obs_date"])
        if valid.empty:
            return True

        leaking = valid["obs_date"] >= valid["ref_ts"]
        return not leaking.any()
