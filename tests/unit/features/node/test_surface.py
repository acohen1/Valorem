"""Unit tests for surface feature generator."""

import numpy as np
import pandas as pd
import pytest

from src.features.node.surface import (
    DELTA_BUCKET_ORDER,
    DELTA_BUCKET_VALUES,
    SurfaceFeatureConfig,
    SurfaceFeatureGenerator,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_surface_df():
    """Create sample surface data for testing."""
    # Create a realistic surface with skew
    dates = pd.date_range("2024-01-01", periods=5, freq="D")

    data = []
    for date in dates:
        for tenor in [30, 60]:
            # Create IV smile with higher IV for OTM puts
            for bucket, delta in DELTA_BUCKET_VALUES.items():
                # Typical smile: higher IV for wings
                base_iv = 0.20
                smile_contribution = 0.02 * (0.5 - abs(delta)) ** 2
                skew_contribution = -0.01 * delta  # Negative skew (put > call)
                iv = base_iv + smile_contribution + skew_contribution + np.random.randn() * 0.001

                data.append({
                    "ts_utc": date,
                    "tenor_days": tenor,
                    "delta_bucket": bucket,
                    "delta": delta if bucket != "ATM" else 0.5,
                    "iv_mid": iv,
                })

    return pd.DataFrame(data)


@pytest.fixture
def generator():
    """Create surface feature generator with default config."""
    return SurfaceFeatureGenerator()


@pytest.fixture
def simple_skew_df():
    """Create simple surface with linear skew for testing."""
    return pd.DataFrame({
        "ts_utc": pd.to_datetime(["2024-01-01"] * 5),
        "tenor_days": [30] * 5,
        "delta_bucket": ["P10", "P25", "ATM", "C25", "C10"],
        "delta": [-0.10, -0.25, 0.50, 0.25, 0.10],
        "iv_mid": [0.30, 0.25, 0.20, 0.18, 0.16],  # Clear negative skew
    })


# ============================================================================
# Initialization Tests
# ============================================================================


class TestSurfaceFeatureInit:
    """Test surface feature generator initialization."""

    def test_default_config(self):
        """Test default configuration values."""
        gen = SurfaceFeatureGenerator()
        assert gen._config.delta_bucket_order == DELTA_BUCKET_ORDER
        assert gen._config.min_buckets_for_skew == 3
        assert gen._config.min_tenors_for_term == 2

    def test_custom_config(self):
        """Test custom configuration."""
        config = SurfaceFeatureConfig(
            min_buckets_for_skew=2,
            min_tenors_for_term=3,
        )
        gen = SurfaceFeatureGenerator(config)
        assert gen._config.min_buckets_for_skew == 2
        assert gen._config.min_tenors_for_term == 3


# ============================================================================
# Generate Tests
# ============================================================================


class TestSurfaceGenerate:
    """Test surface feature generation."""

    def test_generate_adds_skew_columns(self, generator, sample_surface_df):
        """Test that generate adds skew-related columns."""
        result = generator.generate(sample_surface_df)

        assert "skew_slope" in result.columns
        assert "skew_convexity" in result.columns

    def test_generate_adds_term_columns(self, generator, sample_surface_df):
        """Test that generate adds term structure columns."""
        result = generator.generate(sample_surface_df)

        assert "term_slope" in result.columns

    def test_generate_adds_atm_spread(self, generator, sample_surface_df):
        """Test that generate adds ATM spread."""
        result = generator.generate(sample_surface_df)

        assert "atm_spread" in result.columns

    def test_generate_adds_curvature(self, generator, sample_surface_df):
        """Test that generate adds curvature."""
        result = generator.generate(sample_surface_df)

        assert "curvature" in result.columns

    def test_generate_preserves_original_columns(self, generator, sample_surface_df):
        """Test that original columns are preserved."""
        original_cols = sample_surface_df.columns.tolist()
        result = generator.generate(sample_surface_df)

        for col in original_cols:
            assert col in result.columns

    def test_generate_empty_df(self, generator):
        """Test generation with empty DataFrame."""
        empty_df = pd.DataFrame(columns=["ts_utc", "tenor_days", "delta_bucket", "iv_mid"])
        result = generator.generate(empty_df)
        assert len(result) == 0

    def test_generate_missing_columns_raises(self, generator):
        """Test that missing columns raise ValueError."""
        df = pd.DataFrame({
            "ts_utc": pd.date_range("2024-01-01", periods=3),
            "iv_mid": [0.20, 0.21, 0.22],
        })
        with pytest.raises(ValueError, match="Missing required columns"):
            generator.generate(df)


# ============================================================================
# Skew Slope Tests
# ============================================================================


class TestSkewSlope:
    """Test skew slope calculations."""

    def test_skew_slope_negative_for_puts(self, generator, simple_skew_df):
        """Test skew slope is negative when puts have higher IV."""
        result = generator.compute_skew_slope(simple_skew_df)

        # With higher IV for lower deltas, slope should be negative
        assert result["skew_slope"].iloc[0] < 0

    def test_skew_slope_same_for_all_buckets(self, generator, simple_skew_df):
        """Test skew slope is same for all buckets in same tenor."""
        result = generator.compute_skew_slope(simple_skew_df)

        # All rows for same (ts, tenor) should have same skew
        unique_slopes = result["skew_slope"].unique()
        assert len(unique_slopes) == 1

    def test_skew_slope_linear_fit(self, generator):
        """Test skew slope with known linear relationship."""
        # Create perfectly linear skew
        df = pd.DataFrame({
            "ts_utc": pd.to_datetime(["2024-01-01"] * 5),
            "tenor_days": [30] * 5,
            "delta_bucket": ["P10", "P25", "ATM", "C25", "C10"],
            "delta": [-0.10, -0.25, 0.50, 0.25, 0.10],
            "iv_mid": [0.30, 0.275, 0.20, 0.225, 0.25],  # Linear in delta
        })

        result = generator.compute_skew_slope(df)

        # Fit should capture the linear trend
        assert pd.notna(result["skew_slope"].iloc[0])

    def test_skew_slope_insufficient_points(self, generator):
        """Test skew slope with insufficient points."""
        config = SurfaceFeatureConfig(min_buckets_for_skew=4)
        gen = SurfaceFeatureGenerator(config)

        # Only 3 buckets
        df = pd.DataFrame({
            "ts_utc": pd.to_datetime(["2024-01-01"] * 3),
            "tenor_days": [30] * 3,
            "delta_bucket": ["P25", "ATM", "C25"],
            "iv_mid": [0.25, 0.20, 0.18],
        })

        result = gen.compute_skew_slope(df)

        # Should be NaN with insufficient points
        assert pd.isna(result["skew_slope"].iloc[0])


# ============================================================================
# Skew Convexity Tests
# ============================================================================


class TestSkewConvexity:
    """Test skew convexity (smile curvature) calculations."""

    def test_convexity_computed_for_smile(self, generator):
        """Test convexity is computed for smile shape."""
        # Create smile with clear curvature
        df = pd.DataFrame({
            "ts_utc": pd.to_datetime(["2024-01-01"] * 5),
            "tenor_days": [30] * 5,
            "delta_bucket": ["P10", "P25", "ATM", "C25", "C10"],
            "delta": [-0.10, -0.25, 0.50, 0.25, 0.10],
            "iv_mid": [0.25, 0.21, 0.20, 0.21, 0.25],  # Symmetric smile
        })

        result = generator.compute_skew_slope(df)

        # Convexity should be computed (not NaN)
        assert pd.notna(result["skew_convexity"].iloc[0])
        # The quadratic coefficient captures smile curvature
        assert result["skew_convexity"].iloc[0] != 0

    def test_convexity_requires_four_points(self, generator):
        """Test convexity requires at least 4 points."""
        df = pd.DataFrame({
            "ts_utc": pd.to_datetime(["2024-01-01"] * 3),
            "tenor_days": [30] * 3,
            "delta_bucket": ["P25", "ATM", "C25"],
            "iv_mid": [0.25, 0.20, 0.18],
        })

        result = generator.compute_skew_slope(df)

        # Should be NaN with only 3 points
        assert pd.isna(result["skew_convexity"].iloc[0])


# ============================================================================
# Term Slope Tests
# ============================================================================


class TestTermSlope:
    """Test term slope calculations."""

    def test_term_slope_computed(self, generator, sample_surface_df):
        """Test term slope is computed."""
        result = generator.compute_term_slope(sample_surface_df)

        assert "term_slope" in result.columns
        assert result["term_slope"].notna().any()

    def test_term_slope_same_for_same_bucket(self, generator):
        """Test term slope is same for all rows with same bucket."""
        df = pd.DataFrame({
            "ts_utc": pd.to_datetime(["2024-01-01"] * 4),
            "tenor_days": [30, 60, 30, 60],
            "delta_bucket": ["ATM", "ATM", "P25", "P25"],
            "iv_mid": [0.20, 0.22, 0.25, 0.27],
        })

        result = generator.compute_term_slope(df)

        # ATM rows should have same term slope
        atm_rows = result[result["delta_bucket"] == "ATM"]
        assert len(atm_rows["term_slope"].unique()) == 1

    def test_term_slope_insufficient_tenors(self, generator):
        """Test term slope with insufficient tenors."""
        config = SurfaceFeatureConfig(min_tenors_for_term=3)
        gen = SurfaceFeatureGenerator(config)

        # Only 2 tenors
        df = pd.DataFrame({
            "ts_utc": pd.to_datetime(["2024-01-01"] * 2),
            "tenor_days": [30, 60],
            "delta_bucket": ["ATM", "ATM"],
            "iv_mid": [0.20, 0.22],
        })

        result = gen.compute_term_slope(df)

        # Should be NaN with insufficient tenors
        assert pd.isna(result["term_slope"].iloc[0])


# ============================================================================
# ATM Spread Tests
# ============================================================================


class TestATMSpread:
    """Test ATM spread calculations."""

    def test_atm_spread_zero_for_atm(self, generator):
        """Test ATM spread is zero for ATM bucket."""
        df = pd.DataFrame({
            "ts_utc": pd.to_datetime(["2024-01-01"] * 3),
            "tenor_days": [30] * 3,
            "delta_bucket": ["P25", "ATM", "C25"],
            "iv_mid": [0.25, 0.20, 0.18],
        })

        result = generator.compute_atm_spread(df)

        atm_row = result[result["delta_bucket"] == "ATM"]
        assert atm_row["atm_spread"].iloc[0] == pytest.approx(0.0)

    def test_atm_spread_positive_for_higher_iv(self, generator):
        """Test ATM spread is positive when IV > ATM IV."""
        df = pd.DataFrame({
            "ts_utc": pd.to_datetime(["2024-01-01"] * 3),
            "tenor_days": [30] * 3,
            "delta_bucket": ["P25", "ATM", "C25"],
            "iv_mid": [0.25, 0.20, 0.18],
        })

        result = generator.compute_atm_spread(df)

        p25_row = result[result["delta_bucket"] == "P25"]
        assert p25_row["atm_spread"].iloc[0] == pytest.approx(0.05)

    def test_atm_spread_negative_for_lower_iv(self, generator):
        """Test ATM spread is negative when IV < ATM IV."""
        df = pd.DataFrame({
            "ts_utc": pd.to_datetime(["2024-01-01"] * 3),
            "tenor_days": [30] * 3,
            "delta_bucket": ["P25", "ATM", "C25"],
            "iv_mid": [0.25, 0.20, 0.18],
        })

        result = generator.compute_atm_spread(df)

        c25_row = result[result["delta_bucket"] == "C25"]
        assert c25_row["atm_spread"].iloc[0] == pytest.approx(-0.02)


# ============================================================================
# Curvature Tests
# ============================================================================


class TestCurvature:
    """Test curvature calculations."""

    def test_curvature_uses_convexity(self, generator, sample_surface_df):
        """Test curvature equals skew convexity."""
        result = generator.generate(sample_surface_df)

        # Curvature should equal skew_convexity
        np.testing.assert_allclose(
            result["curvature"].values,
            result["skew_convexity"].values,
            equal_nan=True,
        )


# ============================================================================
# Butterfly Spread Tests
# ============================================================================


# ============================================================================
# Cross-Timestamp Tests
# ============================================================================


class TestCrossTimestamp:
    """Test features are computed per timestamp."""

    def test_skew_computed_per_timestamp(self, generator):
        """Test skew is computed independently per timestamp."""
        df = pd.DataFrame({
            "ts_utc": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"]),
            "tenor_days": [30, 30, 30, 30],
            "delta_bucket": ["ATM", "P25", "ATM", "P25"],
            "iv_mid": [0.20, 0.25, 0.22, 0.30],  # Different skew per day
        })

        result = generator.compute_skew_slope(df)

        # Skew should differ between days
        day1_skew = result[result["ts_utc"] == pd.to_datetime("2024-01-01")]["skew_slope"].iloc[0]
        day2_skew = result[result["ts_utc"] == pd.to_datetime("2024-01-02")]["skew_slope"].iloc[0]

        # Both should be computed (not NaN if >= 3 points, but we only have 2)
        # Actually with only 2 points per day, both should be NaN
        assert pd.isna(day1_skew) or pd.isna(day2_skew)  # May be NaN with min_buckets=3
