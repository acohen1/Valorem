"""Cross-sectional surface feature generation.

This module computes features requiring cross-sectional surface structure:
- Skew slope: IV gradient across delta buckets (for fixed tenor)
- Term slope: IV gradient across tenors (for fixed delta bucket)
- Curvature: Second derivative of IV surface (smile curvature)
- ATM spread: Distance from ATM IV

These features capture the shape of the volatility surface at each point in time.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ...config.constants import SurfaceConstants

# Delta bucket ordering for skew slope calculation — use all 7 graph buckets
# (including P40/C40 deep wings) for more informative skew regression.
DELTA_BUCKET_ORDER = list(SurfaceConstants.DELTA_BUCKETS_GRAPH)

# Numeric delta values for regression
DELTA_BUCKET_VALUES = {
    bucket: SurfaceConstants.DELTA_VALUES[bucket]
    for bucket in DELTA_BUCKET_ORDER
}


@dataclass
class SurfaceFeatureConfig:
    """Configuration for surface feature generation.

    Attributes:
        delta_bucket_order: Ordered list of delta buckets for skew calculation
        delta_bucket_values: Numeric delta values for each bucket
        min_buckets_for_skew: Minimum buckets needed to compute skew slope
        min_tenors_for_term: Minimum tenors needed to compute term slope
    """

    delta_bucket_order: list[str] = field(
        default_factory=lambda: DELTA_BUCKET_ORDER.copy()
    )
    delta_bucket_values: dict[str, float] = field(
        default_factory=lambda: DELTA_BUCKET_VALUES.copy()
    )
    min_buckets_for_skew: int = 3
    min_tenors_for_term: int = 2


class SurfaceFeatureGenerator:
    """Generate cross-sectional surface features.

    Computes features that require looking across the surface:
    - skew_slope: Linear regression slope of IV vs delta (per tenor)
    - skew_convexity: Quadratic coefficient of IV smile
    - term_slope: Linear regression slope of IV vs tenor (per delta bucket)
    - atm_spread: Difference from ATM IV at same tenor
    - curvature: Second derivative proxy (reuses skew_convexity)

    These features are computed cross-sectionally (across the surface at each
    timestamp), not longitudinally.

    Example:
        config = SurfaceFeatureConfig()
        generator = SurfaceFeatureGenerator(config)
        features_df = generator.generate(surface_df)
    """

    def __init__(self, config: SurfaceFeatureConfig | None = None):
        """Initialize surface feature generator.

        Args:
            config: Feature configuration. Uses defaults if None.
        """
        self._config = config or SurfaceFeatureConfig()

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate cross-sectional surface features.

        Args:
            df: Surface snapshot DataFrame with columns:
                - ts_utc: Timestamp
                - tenor_days: Tenor in days
                - delta_bucket: Delta bucket name
                - iv_mid: Mid IV value
                - delta: Actual delta value (optional, for more precise regression)

        Returns:
            DataFrame with original columns plus surface features
        """
        if df.empty:
            return df.copy()

        # Validate required columns
        required_cols = ["ts_utc", "tenor_days", "delta_bucket", "iv_mid"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        df = df.copy()

        # Compute skew slope (per timestamp, per tenor)
        skew_df = self.compute_skew_slope(df)
        df = df.merge(
            skew_df[["ts_utc", "tenor_days", "delta_bucket", "skew_slope", "skew_convexity"]],
            on=["ts_utc", "tenor_days", "delta_bucket"],
            how="left",
        )

        # Compute term slope (per timestamp, per delta bucket)
        term_df = self.compute_term_slope(df)
        df = df.merge(
            term_df[["ts_utc", "tenor_days", "delta_bucket", "term_slope"]],
            on=["ts_utc", "tenor_days", "delta_bucket"],
            how="left",
        )

        # Compute ATM spread
        df = self.compute_atm_spread(df)

        # Compute curvature (second derivative proxy)
        df = self.compute_curvature(df)

        return df

    def compute_skew_slope(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute skew slope (IV gradient across delta buckets).

        For each (ts_utc, tenor_days), fits a linear regression of IV vs delta
        to quantify the skew.

        Args:
            df: Surface DataFrame with iv_mid and delta_bucket columns

        Returns:
            DataFrame with skew_slope column added
        """
        results = []

        for (ts, tenor), group in df.groupby(["ts_utc", "tenor_days"]):
            # Get delta values for each bucket
            group = group.copy()

            # Use actual delta if available, otherwise bucket center
            if "delta" in group.columns and group["delta"].notna().any():
                x = group["delta"].values
            else:
                x = group["delta_bucket"].map(self._config.delta_bucket_values).values

            y = group["iv_mid"].values

            # Filter out NaN values
            mask = ~(np.isnan(x) | np.isnan(y))
            x_valid = x[mask]
            y_valid = y[mask]

            # Compute slope if enough points
            if len(x_valid) >= self._config.min_buckets_for_skew:
                # Linear fit: IV = slope * delta + intercept
                coeffs = np.polyfit(x_valid, y_valid, deg=1)
                slope = coeffs[0]

                # Quadratic fit for convexity if enough points
                if len(x_valid) >= 4:
                    coeffs_quad = np.polyfit(x_valid, y_valid, deg=2)
                    convexity = coeffs_quad[0]  # Coefficient of x^2
                else:
                    convexity = np.nan
            else:
                slope = np.nan
                convexity = np.nan

            # Add to all rows for this group
            group["skew_slope"] = slope
            group["skew_convexity"] = convexity
            results.append(group)

        if not results:
            df = df.copy()
            df["skew_slope"] = np.nan
            df["skew_convexity"] = np.nan
            return df

        return pd.concat(results, ignore_index=True)

    def compute_term_slope(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute term slope (IV gradient across tenors).

        For each (ts_utc, delta_bucket), fits a linear regression of IV vs tenor
        to quantify the term structure.

        Args:
            df: Surface DataFrame with iv_mid and tenor_days columns

        Returns:
            DataFrame with term_slope column added
        """
        results = []

        for (ts, bucket), group in df.groupby(["ts_utc", "delta_bucket"]):
            group = group.copy()

            # Use log-tenor for more stable regression (term structure is often log-linear)
            x = np.log(group["tenor_days"].values.astype(float))
            y = group["iv_mid"].values

            # Filter out NaN/inf values
            mask = ~(np.isnan(x) | np.isnan(y) | np.isinf(x))
            x_valid = x[mask]
            y_valid = y[mask]

            # Compute slope if enough points
            if len(x_valid) >= self._config.min_tenors_for_term:
                coeffs = np.polyfit(x_valid, y_valid, deg=1)
                slope = coeffs[0]
            else:
                slope = np.nan

            group["term_slope"] = slope
            results.append(group)

        if not results:
            df = df.copy()
            df["term_slope"] = np.nan
            return df

        return pd.concat(results, ignore_index=True)

    def compute_atm_spread(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute spread from ATM IV.

        For each (ts_utc, tenor_days), computes the difference between
        each bucket's IV and the ATM IV.

        Args:
            df: Surface DataFrame

        Returns:
            DataFrame with atm_spread column added
        """
        df = df.copy()

        # Get ATM IV for each (ts, tenor)
        atm_mask = df["delta_bucket"] == "ATM"
        atm_ivs = df.loc[atm_mask, ["ts_utc", "tenor_days", "iv_mid"]].copy()
        atm_ivs = atm_ivs.rename(columns={"iv_mid": "atm_iv"})

        # Merge back
        df = df.merge(atm_ivs, on=["ts_utc", "tenor_days"], how="left")

        # Compute spread
        df["atm_spread"] = df["iv_mid"] - df["atm_iv"]

        # Drop helper column
        df = df.drop(columns=["atm_iv"])

        return df

    def compute_curvature(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute smile curvature (second derivative proxy).

        Curvature measures how much the smile curves. Computed as the
        average second difference across adjacent delta buckets.

        Args:
            df: Surface DataFrame

        Returns:
            DataFrame with curvature column added
        """
        df = df.copy()

        # Use skew_convexity if already computed
        if "skew_convexity" in df.columns:
            df["curvature"] = df["skew_convexity"]
        else:
            # Compute from skew slope
            skew_df = self.compute_skew_slope(df)
            df = df.merge(
                skew_df[["ts_utc", "tenor_days", "delta_bucket", "skew_convexity"]],
                on=["ts_utc", "tenor_days", "delta_bucket"],
                how="left",
            )
            df["curvature"] = df["skew_convexity"]

        return df

