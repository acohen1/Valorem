"""Quality filtering for option quotes.

This module provides quality assessment for option quotes using
bitfield-encoded flags. Poor quality quotes can be excluded from
surface construction to improve accuracy.
"""

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd


@dataclass
class QualityConfig:
    """Configuration for quality filtering.

    Attributes:
        allow_crossed_quotes: If True, allow bid > ask quotes (unusual but possible)
        max_spread_pct: Maximum bid-ask spread as percentage of mid price
        min_volume: Minimum daily volume (None to disable check)
        min_open_interest: Minimum open interest (None to disable check)
        eod_max_staleness_days: Maximum days since last quote update
    """

    allow_crossed_quotes: bool = False
    max_spread_pct: float = 0.50  # 50% spread
    min_volume: int | None = 10
    min_open_interest: int | None = 100
    eod_max_staleness_days: int = 1


class QualityFilter:
    """Compute quality flags for option quotes.

    Quality flags are encoded as a bitfield, allowing multiple flags
    to be set simultaneously. This enables efficient filtering and
    aggregation of quality issues.

    Flag encoding:
    - FLAG_CROSSED (bit 0, value 1): Bid > Ask (crossed quote)
    - FLAG_STALE (bit 1, value 2): Quote older than max staleness
    - FLAG_WIDE_SPREAD (bit 2, value 4): Spread exceeds max percentage
    - FLAG_LOW_VOLUME (bit 3, value 8): Volume below minimum
    - FLAG_LOW_OI (bit 4, value 16): Open interest below minimum

    Example:
        flags = 0b00101 (value 5) means FLAG_CROSSED | FLAG_WIDE_SPREAD

    Usage:
        config = QualityConfig(max_spread_pct=0.30)
        filter = QualityFilter(config)
        flags = filter.compute_flags(df)

        # Filter out low quality options
        good_quality = df[flags == 0]

        # Check specific flag
        crossed = (flags & QualityFilter.FLAG_CROSSED) != 0
    """

    # Flag constants (bitfield positions)
    FLAG_CROSSED = 1 << 0  # 0b00001 = 1
    FLAG_STALE = 1 << 1  # 0b00010 = 2
    FLAG_WIDE_SPREAD = 1 << 2  # 0b00100 = 4
    FLAG_LOW_VOLUME = 1 << 3  # 0b01000 = 8
    FLAG_LOW_OI = 1 << 4  # 0b10000 = 16

    def __init__(self, config: QualityConfig):
        """Initialize quality filter with configuration.

        Args:
            config: Quality filtering configuration
        """
        self._config = config

    def compute_flags(
        self,
        df: pd.DataFrame,
        reference_time: datetime | None = None,
    ) -> pd.Series:
        """Compute quality flags for each option quote.

        Args:
            df: DataFrame with option quote data. Expected columns:
                - bid: Bid price
                - ask: Ask price
                - spread_pct: Bid-ask spread as percentage (optional, computed if missing)
                - volume: Daily volume (optional)
                - open_interest: Open interest (optional)
                - ts_utc: Quote timestamp (for staleness check)
            reference_time: Reference time for staleness calculation.
                If None, uses max(ts_utc) from the data.

        Returns:
            Series of integer flags with same index as input DataFrame.
            Value 0 means no quality issues detected.
        """
        flags = pd.Series(0, index=df.index, dtype=int)

        # Crossed quotes: bid > ask
        if not self._config.allow_crossed_quotes and "bid" in df.columns and "ask" in df.columns:
            crossed = df["ask"] < df["bid"]
            flags = flags | (crossed.astype(int) * self.FLAG_CROSSED)

        # Wide spread check
        if "spread_pct" in df.columns:
            spread_pct = df["spread_pct"]
        elif "bid" in df.columns and "ask" in df.columns:
            # Compute spread percentage from bid/ask
            mid = (df["bid"] + df["ask"]) / 2
            spread_pct = (df["ask"] - df["bid"]) / np.maximum(mid, 1e-10)
        else:
            spread_pct = None

        if spread_pct is not None:
            wide = spread_pct > self._config.max_spread_pct
            flags = flags | (wide.astype(int) * self.FLAG_WIDE_SPREAD)

        # Staleness check
        if "ts_utc" in df.columns:
            if reference_time is None:
                reference_time = pd.to_datetime(df["ts_utc"]).max()

            quote_dates = pd.to_datetime(df["ts_utc"]).dt.date
            ref_date = pd.to_datetime(reference_time).date()
            staleness_days = (ref_date - quote_dates).apply(lambda x: x.days if hasattr(x, 'days') else 0)
            stale = staleness_days > self._config.eod_max_staleness_days
            flags = flags | (stale.astype(int) * self.FLAG_STALE)

        # Low volume check
        if self._config.min_volume is not None and "volume" in df.columns:
            low_vol = pd.to_numeric(df["volume"], errors="coerce").fillna(0) < self._config.min_volume
            flags = flags | (low_vol.astype(int) * self.FLAG_LOW_VOLUME)

        # Low open interest check
        if self._config.min_open_interest is not None and "open_interest" in df.columns:
            low_oi = pd.to_numeric(df["open_interest"], errors="coerce").fillna(0) < self._config.min_open_interest
            flags = flags | (low_oi.astype(int) * self.FLAG_LOW_OI)

        return flags

    def is_good_quality(self, flags: pd.Series) -> pd.Series:
        """Check if quotes pass all quality checks.

        Args:
            flags: Series of quality flags

        Returns:
            Boolean Series (True = good quality, no flags set)
        """
        return flags == 0

    def has_flag(self, flags: pd.Series, flag: int) -> pd.Series:
        """Check if a specific flag is set.

        Args:
            flags: Series of quality flags
            flag: Flag constant to check (e.g., FLAG_CROSSED)

        Returns:
            Boolean Series (True = flag is set)
        """
        return (flags & flag) != 0

    def describe_flags(self, flag_value: int) -> list[str]:
        """Get human-readable description of flags set in a value.

        Args:
            flag_value: Integer flag value

        Returns:
            List of flag names that are set

        Example:
            >>> filter.describe_flags(5)
            ['CROSSED', 'WIDE_SPREAD']
        """
        descriptions = []
        if flag_value & self.FLAG_CROSSED:
            descriptions.append("CROSSED")
        if flag_value & self.FLAG_STALE:
            descriptions.append("STALE")
        if flag_value & self.FLAG_WIDE_SPREAD:
            descriptions.append("WIDE_SPREAD")
        if flag_value & self.FLAG_LOW_VOLUME:
            descriptions.append("LOW_VOLUME")
        if flag_value & self.FLAG_LOW_OI:
            descriptions.append("LOW_OI")
        return descriptions
