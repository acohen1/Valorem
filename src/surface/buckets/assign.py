"""Delta bucket assignment for option classification.

This module assigns options to standardized delta buckets based on their
calculated delta values. Buckets enable consistent comparison across
different strikes and expirations.
"""

import numpy as np
import pandas as pd


class DeltaBucketAssigner:
    """Assign options to delta buckets based on their delta values.

    Delta buckets provide a standardized way to classify options by their
    moneyness (delta). This enables:
    - Consistent volatility surface construction
    - Cross-expiration comparison
    - Standardized feature engineering

    Standard bucket naming convention:
    - P10, P25, P40: Put-side buckets (negative delta)
    - ATM: At-the-money (|delta| near 0.5)
    - C40, C25, C10: Call-side buckets (positive delta)

    The ATM bucket is special: it uses absolute delta to capture both
    calls and puts near the money.
    """

    def __init__(self, bucket_config: dict[str, list[float]]):
        """Initialize bucket assigner with configuration.

        Args:
            bucket_config: Dictionary mapping bucket names to delta ranges.
                Standard buckets have 2-element lists [min, max].
                ATM bucket has 4 elements: [neg_max, neg_min, pos_min, pos_max]
                which defines |delta| range as [pos_min, pos_max].

        Raises:
            ValueError: If bucket configuration is invalid

        Example:
            config = {
                "P10": [-1.0, -0.15],
                "P25": [-0.15, -0.35],
                "ATM": [-0.55, -0.45, 0.45, 0.55],
                "C25": [0.15, 0.35],
                "C10": [0.0, 0.15],
            }
            assigner = DeltaBucketAssigner(config)
        """
        if not bucket_config:
            raise ValueError("bucket_config cannot be empty")

        self._buckets = self._parse_bucket_config(bucket_config)
        self._bucket_names = [name for name, _, _ in self._buckets]

    def _parse_bucket_config(
        self, config: dict[str, list[float]]
    ) -> list[tuple[str, float, float]]:
        """Parse bucket configuration into (name, min, max) tuples.

        Args:
            config: Raw bucket configuration dictionary

        Returns:
            List of (bucket_name, min_delta, max_delta) tuples

        Raises:
            ValueError: If any bucket configuration is invalid
        """
        buckets = []

        for name, bounds in config.items():
            if name == "ATM":
                # ATM encoded as 4 numbers: [neg_max, neg_min, pos_min, pos_max]
                # Means |delta| in [pos_min, pos_max]
                if len(bounds) != 4:
                    raise ValueError(
                        f"ATM bucket must have exactly 4 elements, got {len(bounds)}"
                    )
                # Use positive side for ATM range
                buckets.append((name, bounds[2], bounds[3]))
            else:
                # Standard 2-element bucket
                if len(bounds) != 2:
                    raise ValueError(
                        f"Bucket {name} must have exactly 2 elements, got {len(bounds)}"
                    )
                min_delta, max_delta = bounds[0], bounds[1]
                # Ensure min <= max (swap if needed for negative buckets)
                if min_delta > max_delta:
                    min_delta, max_delta = max_delta, min_delta
                buckets.append((name, min_delta, max_delta))

        return buckets

    def assign(self, deltas: pd.Series) -> pd.Series:
        """Assign each delta value to a bucket.

        Args:
            deltas: Series of delta values (typically from Greeks calculator)
                Call deltas are positive [0, 1], put deltas are negative [-1, 0]

        Returns:
            Series of bucket names (str) with same index as input.
            Options not falling into any bucket will have None.
        """
        bucket_names = pd.Series(
            [None] * len(deltas), index=deltas.index, dtype=object
        )

        for name, min_delta, max_delta in self._buckets:
            if name == "ATM":
                # ATM: check |delta| is in range
                mask = (np.abs(deltas) >= min_delta) & (np.abs(deltas) <= max_delta)
            else:
                # Directional buckets: check delta is in range
                mask = (deltas >= min_delta) & (deltas <= max_delta)

            # Only assign if not already assigned (first match wins)
            unassigned = bucket_names.isna()
            bucket_names.loc[mask & unassigned] = name

        return bucket_names

    def get_bucket_names(self) -> list[str]:
        """Get list of all bucket names in configuration order.

        Returns:
            List of bucket name strings
        """
        return self._bucket_names.copy()

    def get_bucket_bounds(self, bucket_name: str) -> tuple[float, float]:
        """Get the delta bounds for a specific bucket.

        Args:
            bucket_name: Name of the bucket

        Returns:
            Tuple of (min_delta, max_delta)

        Raises:
            KeyError: If bucket name not found
        """
        for name, min_delta, max_delta in self._buckets:
            if name == bucket_name:
                return (min_delta, max_delta)
        raise KeyError(f"Bucket '{bucket_name}' not found")
