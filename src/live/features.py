"""Feature provider protocol and implementations for live trading.

This module defines the FeatureProvider protocol and provides implementations
for computing features from volatility surface data in real-time.

IMPORTANT: The model requires all 29 DEFAULT_FEATURE_COLS (defined in
src/models/dataset.py). The DatabaseFeatureProvider reads these from the
pre-computed node_panel. The RollingFeatureProvider is a lightweight fallback
that computes only a subset of surface-derived features — it CANNOT produce
a full model-compatible feature set and should only be used for warmup or
when the database is unavailable.
"""

import logging
from datetime import datetime, timezone
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd

from ..config.constants import SurfaceConstants

logger = logging.getLogger(__name__)


@runtime_checkable
class FeatureProvider(Protocol):
    """Protocol for computing features from surface data.

    Implementations maintain historical state and compute time-series
    features (z-scores, changes, slopes) from rolling surface data.
    """

    def get_features(self, surface: pd.DataFrame) -> pd.DataFrame:
        """Compute features from current surface.

        Args:
            surface: Current volatility surface

        Returns:
            DataFrame with feature columns per node
        """
        ...

    def update(self, surface: pd.DataFrame) -> None:
        """Update internal state with new surface data.

        Args:
            surface: Current surface to add to history
        """
        ...

    def reset(self) -> None:
        """Clear all historical state."""
        ...


class RollingFeatureProvider:
    """Lightweight feature provider using rolling window of historical surfaces.

    WARNING: This provider computes only a subset of surface-derived features.
    It CANNOT produce the full 29-feature set required by the trained model
    (missing: greeks, realized vol, macro features, volume features). Use
    DatabaseFeatureProvider for model inference. This class is suitable only
    for warmup, monitoring, or degraded-mode operation.

    Features computed:
    - iv_change_1d, iv_change_5d: IV changes
    - iv_vol_5d, iv_vol_10d, iv_vol_21d: Rolling IV volatility (shift(1))
    - iv_zscore_5d, iv_zscore_10d, iv_zscore_21d: IV z-scores (shift(1))
    - spread_pct: Current bid-ask spread percentage
    - skew_slope, term_slope, curvature: Surface shape features

    Features NOT available (must come from database/external):
    - delta, gamma, vega, theta (require Black-Scholes + greeks computation)
    - underlying_rv_5d/10d/21d (require underlying price history)
    - VIXCLS_level/change_1w, DGS10_level/change_1w, DGS2_level/change_1w (FRED)
    - log_volume, volume_ratio_5d, log_oi, oi_change_5d (require OHLCV bar data)
    """

    # Standard node structure aligned with model graph topology.
    TENORS = list(SurfaceConstants.TENOR_DAYS_DEFAULT)
    BUCKETS = list(SurfaceConstants.DELTA_BUCKETS_GRAPH)

    def __init__(
        self,
        lookback_days: int = 22,
        min_history: int = 5,
    ):
        """Initialize rolling feature provider.

        Args:
            lookback_days: Number of surfaces to maintain in buffer
            min_history: Minimum surfaces required before computing features
        """
        self._lookback_days = lookback_days
        self._min_history = min_history

        # Rolling buffer of historical surfaces
        # Each entry is a dict: {(tenor, bucket): {iv, spread_pct, ...}}
        self._history: list[dict[tuple[int, str], dict]] = []
        self._timestamps: list[datetime] = []

    def update(self, surface: pd.DataFrame) -> None:
        """Add surface to rolling history buffer.

        Args:
            surface: Current volatility surface DataFrame
        """
        if surface.empty:
            logger.debug("Empty surface, skipping update")
            return

        # Extract node data from surface
        node_data = self._extract_node_data(surface)
        self._history.append(node_data)
        self._timestamps.append(datetime.now(timezone.utc))

        # Trim to lookback window
        if len(self._history) > self._lookback_days:
            self._history = self._history[-self._lookback_days:]
            self._timestamps = self._timestamps[-self._lookback_days:]

        logger.debug(f"Feature history: {len(self._history)}/{self._lookback_days}")

    def get_features(self, surface: pd.DataFrame) -> pd.DataFrame:
        """Compute features from history + current surface.

        Args:
            surface: Current volatility surface

        Returns:
            DataFrame with one row per node and feature columns.
            Returns empty DataFrame if insufficient history.
        """
        if len(self._history) < self._min_history:
            logger.debug(
                f"Insufficient history: {len(self._history)}/{self._min_history}"
            )
            return pd.DataFrame()

        # Build feature records for each node
        records = []
        current_data = self._extract_node_data(surface)

        for tenor in self.TENORS:
            for bucket in self.BUCKETS:
                node_key = (tenor, bucket)
                current = current_data.get(node_key, {})

                if not current:
                    continue

                # Get historical IV values for this node (excludes current)
                iv_history = [
                    h.get(node_key, {}).get("iv")
                    for h in self._history
                ]
                iv_history = [v for v in iv_history if v is not None]

                record = {
                    "tenor_days": tenor,
                    "delta_bucket": bucket,
                    "spread_pct": current.get("spread_pct", 0.0),
                }

                current_iv = current.get("iv")

                # IV changes (current minus N periods ago)
                record["iv_change_1d"] = self._compute_change(current_iv, iv_history, 1)
                record["iv_change_5d"] = self._compute_change(current_iv, iv_history, 5)

                # IV volatility and z-scores at matching training windows
                for window in [5, 10, 21]:
                    vol, zscore = self._compute_vol_and_zscore(
                        current_iv, iv_history, window
                    )
                    record[f"iv_vol_{window}d"] = vol
                    record[f"iv_zscore_{window}d"] = zscore

                # Surface shape features
                record["term_slope"] = self._compute_term_slope(current_data, bucket)
                record["skew_slope"] = self._compute_skew_slope(current_data, tenor)
                record["curvature"] = self._compute_curvature(current_data, tenor)

                records.append(record)

        return pd.DataFrame(records)

    def reset(self) -> None:
        """Clear all historical state."""
        self._history = []
        self._timestamps = []
        logger.info("Feature history reset")

    @property
    def history_count(self) -> int:
        """Return number of surfaces in history buffer."""
        return len(self._history)

    @property
    def history_length(self) -> int:
        """Get current history buffer length."""
        return len(self._history)

    @property
    def is_ready(self) -> bool:
        """Check if enough history for feature computation."""
        return len(self._history) >= self._min_history

    def _extract_node_data(
        self, surface: pd.DataFrame
    ) -> dict[tuple[int, str], dict]:
        """Extract node-level data from surface DataFrame."""
        result = {}

        for _, row in surface.iterrows():
            tenor = row.get("tenor_days")
            bucket = row.get("delta_bucket")

            if pd.isna(tenor) or pd.isna(bucket):
                continue

            key = (int(tenor), str(bucket))

            iv = row.get("iv_mid") or row.get("iv")
            bid = row.get("bid", 0)
            ask = row.get("ask", 0)
            mid = (bid + ask) / 2 if bid and ask else 0
            spread_pct = (ask - bid) / mid if mid > 0 else 0

            result[key] = {
                "iv": float(iv) if pd.notna(iv) else None,
                "spread_pct": float(spread_pct),
                "bid": float(bid) if pd.notna(bid) else None,
                "ask": float(ask) if pd.notna(ask) else None,
            }

        return result

    @staticmethod
    def _compute_change(
        current: float | None,
        history: list[float],
        lookback: int,
    ) -> float:
        """Compute change from N periods ago."""
        if current is None or len(history) < lookback:
            return float("nan")
        past_value = history[-lookback]
        if past_value is None:
            return float("nan")
        return current - past_value

    @staticmethod
    def _compute_vol_and_zscore(
        current: float | None,
        history: list[float],
        window: int,
    ) -> tuple[float, float]:
        """Compute rolling volatility and z-score from history (excludes current).

        Returns:
            (iv_vol, iv_zscore) tuple
        """
        if current is None or len(history) < 2:
            return float("nan"), float("nan")

        # Use at most `window` prior observations
        prior = history[-window:] if len(history) >= window else history
        if len(prior) < 2:
            return float("nan"), float("nan")

        vol = float(np.std(prior, ddof=1))
        mean = float(np.mean(prior))

        zscore = (current - mean) / vol if vol > 1e-10 else 0.0
        return vol, zscore

    def _compute_term_slope(
        self,
        current_data: dict[tuple[int, str], dict],
        bucket: str,
    ) -> float:
        """Compute term structure slope (90d - 7d IV at same bucket)."""
        short_iv = current_data.get((7, bucket), {}).get("iv")
        long_iv = current_data.get((90, bucket), {}).get("iv")
        if short_iv is None or long_iv is None:
            return 0.0
        return long_iv - short_iv

    @staticmethod
    def _compute_skew_slope(
        current_data: dict[tuple[int, str], dict],
        tenor: int,
    ) -> float:
        """Compute skew slope via linear regression across delta buckets."""
        delta_values = SurfaceConstants.DELTA_VALUES
        xs, ys = [], []
        for bucket in SurfaceConstants.DELTA_BUCKETS_GRAPH:
            node = current_data.get((tenor, bucket), {})
            iv = node.get("iv")
            if iv is not None and bucket in delta_values:
                xs.append(delta_values[bucket])
                ys.append(iv)
        if len(xs) < 3:
            return 0.0
        coeffs = np.polyfit(xs, ys, deg=1)
        return float(coeffs[0])

    @staticmethod
    def _compute_curvature(
        current_data: dict[tuple[int, str], dict],
        tenor: int,
    ) -> float:
        """Compute curvature (quadratic coefficient) across delta buckets."""
        delta_values = SurfaceConstants.DELTA_VALUES
        xs, ys = [], []
        for bucket in SurfaceConstants.DELTA_BUCKETS_GRAPH:
            node = current_data.get((tenor, bucket), {})
            iv = node.get("iv")
            if iv is not None and bucket in delta_values:
                xs.append(delta_values[bucket])
                ys.append(iv)
        if len(xs) < 4:
            return 0.0
        coeffs = np.polyfit(xs, ys, deg=2)
        return float(coeffs[0])


class DatabaseFeatureProvider:
    """Feature provider that reads pre-computed features from database.

    This is the primary provider for live model inference. It reads from the
    node_panel table which contains ALL 29 DEFAULT_FEATURE_COLS as computed
    by the batch feature engine (same code path as training).

    Args:
        derived_repo: DerivedRepository instance for reading features
        feature_version: Feature version to query (must match training).
            Loaded from checkpoint metadata, not hardcoded.
        lookback_days: How far back to query for features
    """

    def __init__(
        self,
        derived_repo,
        feature_version: str = "v1.0",
        lookback_days: int = 1,
    ):
        """Initialize database feature provider.

        Args:
            derived_repo: DerivedRepository instance for reading features
            feature_version: Feature version to query (should match checkpoint)
            lookback_days: How far back to query for features
        """
        self._derived_repo = derived_repo
        self._feature_version = feature_version
        self._lookback_days = lookback_days

    def update(self, surface: pd.DataFrame) -> None:
        """No-op for database provider (features are pre-computed)."""
        pass

    def get_features(self, surface: pd.DataFrame) -> pd.DataFrame:
        """Query latest features from database.

        Returns the raw node_panel columns — no renames, no subsetting.
        The signal generator is responsible for selecting the feature columns
        it needs (from checkpoint metadata).

        Args:
            surface: Current surface (used to determine query time)

        Returns:
            DataFrame with pre-computed features
        """
        from datetime import timedelta

        end = datetime.now(timezone.utc)
        start = end - timedelta(days=self._lookback_days)

        try:
            features_df = self._derived_repo.read_node_panel(
                feature_version=self._feature_version, start=start, end=end,
            )
        except Exception as e:
            logger.error(f"Failed to read features from database: {e}")
            return pd.DataFrame()

        if features_df.empty:
            logger.warning(
                f"No features found in database for version={self._feature_version}"
            )
            return pd.DataFrame()

        # Return the most recent timestamp's data — no column renames.
        # Column names in node_panel already match DEFAULT_FEATURE_COLS.
        latest_ts = features_df["ts_utc"].max()
        latest_features = features_df[features_df["ts_utc"] == latest_ts].copy()

        return latest_features

    def reset(self) -> None:
        """No-op for database provider."""
        pass


class MockFeatureProvider:
    """Mock feature provider for testing.

    Generates synthetic features with configurable values.
    Uses DEFAULT_FEATURE_COLS names to match training expectations.
    """

    def __init__(
        self,
        default_zscore: float = 0.0,
        default_change: float = 0.0,
        ready_after: int = 0,
    ):
        """Initialize mock feature provider.

        Args:
            default_zscore: Default z-score value to return
            default_change: Default change value to return
            ready_after: Number of updates before returning features
        """
        self._default_zscore = default_zscore
        self._default_change = default_change
        self._ready_after = ready_after
        self._update_count = 0

    def update(self, surface: pd.DataFrame) -> None:
        """Track update calls."""
        self._update_count += 1

    def get_features(self, surface: pd.DataFrame) -> pd.DataFrame:
        """Generate mock features matching DEFAULT_FEATURE_COLS names.

        Args:
            surface: Surface DataFrame (used for node keys)

        Returns:
            DataFrame with mock feature values
        """
        if self._update_count < self._ready_after:
            return pd.DataFrame()

        if surface.empty:
            return pd.DataFrame()

        records = []
        for _, row in surface.iterrows():
            tenor = row.get("tenor_days")
            bucket = row.get("delta_bucket")

            if pd.isna(tenor) or pd.isna(bucket):
                continue

            records.append({
                "tenor_days": int(tenor),
                "delta_bucket": str(bucket),
                # Match DEFAULT_FEATURE_COLS names
                "delta": 0.5,
                "gamma": 0.01,
                "vega": 0.1,
                "theta": -0.01,
                "spread_pct": 0.02,
                "skew_slope": 0.0,
                "term_slope": 0.0,
                "curvature": 0.0,
                "iv_change_1d": self._default_change,
                "iv_change_5d": self._default_change,
                "iv_vol_5d": 0.01,
                "iv_vol_10d": 0.01,
                "iv_vol_21d": 0.01,
                "iv_zscore_5d": self._default_zscore,
                "iv_zscore_10d": self._default_zscore,
                "iv_zscore_21d": self._default_zscore,
                "underlying_rv_5d": 0.15,
                "underlying_rv_10d": 0.15,
                "underlying_rv_21d": 0.15,
                "VIXCLS_level": 20.0,
                "VIXCLS_change_1w": 0.0,
                "DGS10_level": 0.04,
                "DGS10_change_1w": 0.0,
                "DGS2_level": 0.04,
                "DGS2_change_1w": 0.0,
                "log_volume": 5.0,
                "volume_ratio_5d": 1.0,
                "log_oi": 5.0,
                "oi_change_5d": 0.0,
            })

        return pd.DataFrame(records)

    def reset(self) -> None:
        """Reset update counter."""
        self._update_count = 0
