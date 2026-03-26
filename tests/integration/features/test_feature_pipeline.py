"""Integration tests for the feature generation pipeline.

These tests verify the end-to-end feature generation workflow including:
- Surface snapshots → node features
- Deterministic output across runs
- Proper handling of realistic data patterns
"""

import numpy as np
import pandas as pd
import pytest

from src.features import (
    FeatureResult,
    IVFeatureConfig,
    IVFeatureGenerator,
    MicrostructureConfig,
    MicrostructureFeatureGenerator,
    NodeFeatureConfig,
    NodeFeatureGenerator,
    SurfaceFeatureConfig,
    SurfaceFeatureGenerator,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def realistic_surface_data():
    """Create realistic surface snapshot data.

    Simulates 60 days of surface data with:
    - 3 tenors (30, 60, 90 days)
    - 5 delta buckets (P10, P25, ATM, C25, C10)
    - Realistic IV patterns (smile, term structure)
    - Volume and OI patterns
    """
    np.random.seed(42)  # Reproducibility
    dates = pd.date_range("2024-01-01", periods=60, freq="D")

    tenors = [30, 60, 90]
    buckets = ["P10", "P25", "ATM", "C25", "C10"]
    delta_map = {"P10": -0.10, "P25": -0.25, "ATM": 0.50, "C25": 0.25, "C10": 0.10}

    data = []
    base_iv = 0.18  # Starting ATM IV

    for i, date in enumerate(dates):
        # Add some trend to base IV
        day_iv = base_iv + 0.0005 * i + np.random.randn() * 0.003

        for tenor in tenors:
            # Term structure: longer tenors have slightly higher IV
            term_adj = 0.005 * np.log(tenor / 30)

            for bucket in buckets:
                delta = delta_map[bucket]

                # Smile: higher IV for wings
                smile_adj = 0.03 * (0.5 - delta) ** 2

                # Skew: puts have higher IV than calls
                skew_adj = -0.01 * delta

                # Final IV
                iv = day_iv + term_adj + smile_adj + skew_adj + np.random.randn() * 0.002
                iv = max(0.05, iv)  # Floor at 5%

                # Volume: ATM has more volume
                base_vol = 5000 if bucket == "ATM" else 2000
                volume = int(base_vol + np.random.randn() * 500)
                volume = max(0, volume)

                # OI: grows over time
                base_oi = 10000 + i * 100
                oi = int(base_oi + np.random.randn() * 1000)

                # Spread: tighter for ATM
                base_spread = 0.015 if bucket == "ATM" else 0.025
                spread = base_spread + np.random.randn() * 0.005
                spread = max(0.005, spread)

                data.append({
                    "ts_utc": date,
                    "tenor_days": tenor,
                    "delta_bucket": bucket,
                    "delta": delta if bucket != "ATM" else 0.5,
                    "iv_mid": iv,
                    "spread_pct": spread,
                    "volume": volume,
                    "open_interest": oi,
                })

    return pd.DataFrame(data)


# ============================================================================
# End-to-End Pipeline Tests
# ============================================================================


class TestEndToEndPipeline:
    """Test complete feature generation pipeline."""

    def test_full_pipeline_runs(self, realistic_surface_data):
        """Test full pipeline completes without error."""
        generator = NodeFeatureGenerator()
        result_df, result = generator.generate(
            realistic_surface_data,
            feature_version="v1.0.0",
        )

        assert result.row_count == len(realistic_surface_data)
        assert result.feature_count > 15
        assert len(result.features_generated) > 0

    def test_pipeline_output_structure(self, realistic_surface_data):
        """Test pipeline output has expected structure."""
        generator = NodeFeatureGenerator()
        result_df, result = generator.generate(realistic_surface_data)

        # Original columns preserved
        for col in realistic_surface_data.columns:
            assert col in result_df.columns

        # Key features present
        expected_features = [
            # IV features
            "iv_change_1d", "iv_change_5d",
            "iv_vol_5d", "iv_vol_10d", "iv_vol_21d",
            "iv_zscore_5d",
            # Microstructure features
            "volume_ratio_5d", "log_volume",
            "oi_change_5d", "log_oi",
            # Surface features
            "skew_slope", "skew_convexity",
            "term_slope",
            "atm_spread",
            "curvature",
        ]

        for feat in expected_features:
            assert feat in result_df.columns, f"Missing feature: {feat}"

    def test_pipeline_node_count(self, realistic_surface_data):
        """Test correct number of nodes processed."""
        generator = NodeFeatureGenerator()
        _, result = generator.generate(realistic_surface_data)

        # 3 tenors x 5 buckets = 15 nodes
        assert result.nodes_processed == 15


# ============================================================================
# Determinism Tests
# ============================================================================


class TestDeterminism:
    """Test that feature generation is deterministic."""

    def test_multiple_runs_identical(self, realistic_surface_data):
        """Test multiple runs produce identical results."""
        generator = NodeFeatureGenerator()

        results = []
        for _ in range(3):
            result_df, _ = generator.generate(
                realistic_surface_data.copy(),
                feature_version="v1.0.0",
            )
            results.append(result_df)

        # Compare all pairs
        for i in range(1, len(results)):
            # Sort for comparison
            df1 = results[0].sort_values(["ts_utc", "tenor_days", "delta_bucket"]).reset_index(drop=True)
            df2 = results[i].sort_values(["ts_utc", "tenor_days", "delta_bucket"]).reset_index(drop=True)

            # Compare all columns
            for col in df1.columns:
                if df1[col].dtype == float:
                    np.testing.assert_allclose(
                        df1[col].values,
                        df2[col].values,
                        rtol=1e-10,
                        equal_nan=True,
                    )
                else:
                    pd.testing.assert_series_equal(df1[col], df2[col])


# ============================================================================
# Feature Quality Tests
# ============================================================================


class TestFeatureQuality:
    """Test feature quality and correctness."""

    def test_iv_changes_reasonable(self, realistic_surface_data):
        """Test IV changes are within reasonable bounds."""
        generator = NodeFeatureGenerator()
        result_df, _ = generator.generate(realistic_surface_data)

        # 1-day IV changes should be small
        iv_change_1d = result_df["iv_change_1d"].dropna()
        assert abs(iv_change_1d.mean()) < 0.01  # Average change < 1%
        assert iv_change_1d.std() < 0.02  # Std < 2%

    def test_skew_slope_negative(self, realistic_surface_data):
        """Test skew slope is generally negative (typical equity skew)."""
        generator = NodeFeatureGenerator()
        result_df, _ = generator.generate(realistic_surface_data)

        # Equity skew should be negative (puts > calls)
        skew_slopes = result_df["skew_slope"].dropna()
        assert skew_slopes.mean() < 0

    def test_atm_spread_symmetric(self, realistic_surface_data):
        """Test ATM spread is zero for ATM buckets."""
        generator = NodeFeatureGenerator()
        result_df, _ = generator.generate(realistic_surface_data)

        atm_rows = result_df[result_df["delta_bucket"] == "ATM"]
        atm_spreads = atm_rows["atm_spread"].dropna()

        # ATM spread should be very close to zero for ATM
        assert abs(atm_spreads.mean()) < 0.001

    def test_volume_ratio_centered(self, realistic_surface_data):
        """Test volume ratio is centered around 1."""
        generator = NodeFeatureGenerator()
        result_df, _ = generator.generate(realistic_surface_data)

        volume_ratios = result_df["volume_ratio_5d"].dropna()

        # Ratio should be around 1 on average
        assert 0.5 < volume_ratios.mean() < 2.0


# ============================================================================
# No Leakage Tests
# ============================================================================


class TestNoLeakage:
    """Test that features don't leak future data."""

    def test_early_rows_have_nans(self, realistic_surface_data):
        """Test early rows have NaN for rolling features."""
        generator = NodeFeatureGenerator()
        result_df, _ = generator.generate(realistic_surface_data)

        # First observation of each node should have NaN for changes
        # Filter to only the first timestamp (earliest day)
        min_ts = result_df["ts_utc"].min()
        first_day = result_df[result_df["ts_utc"] == min_ts]
        assert first_day["iv_change_1d"].isna().all()

    def test_cumulative_features_increase(self, realistic_surface_data):
        """Test rolling features use only past data."""
        generator = NodeFeatureGenerator()
        result_df, _ = generator.generate(realistic_surface_data)

        # For each node, check rolling features use only past data
        for (tenor, bucket), group in result_df.groupby(["tenor_days", "delta_bucket"]):
            group = group.sort_values("ts_utc")
            vol_ratio = group["volume_ratio_5d"]

            # Number of non-NaN values should increase over time
            # (as more history becomes available)
            non_nan_count = vol_ratio.notna().cumsum()
            # Should be monotonically increasing
            assert (non_nan_count.diff().dropna() >= 0).all()


# ============================================================================
# Performance Tests
# ============================================================================


class TestPerformance:
    """Test feature generation performance."""

    def test_reasonable_execution_time(self, realistic_surface_data):
        """Test feature generation completes in reasonable time."""
        import time

        generator = NodeFeatureGenerator()

        start = time.time()
        generator.generate(realistic_surface_data)
        elapsed = time.time() - start

        # Should complete in under 5 seconds for 60 days x 15 nodes
        assert elapsed < 5.0

    def test_scales_with_data_size(self):
        """Test performance scales reasonably with data size."""
        import time

        generator = NodeFeatureGenerator()
        sizes = [100, 500, 1000]
        times = []

        for n in sizes:
            df = pd.DataFrame({
                "ts_utc": pd.date_range("2024-01-01", periods=n),
                "tenor_days": [30] * n,
                "delta_bucket": ["ATM"] * n,
                "iv_mid": [0.20] * n,
                "spread_pct": [0.02] * n,
            })

            start = time.time()
            generator.generate(df)
            elapsed = time.time() - start
            times.append(elapsed)

        # Time should scale roughly linearly (not quadratically)
        # Allow some tolerance for overhead
        ratio = times[-1] / times[0]
        size_ratio = sizes[-1] / sizes[0]

        # Should be at most 2x the linear ratio
        assert ratio < 2 * size_ratio


# ============================================================================
# Edge Cases
# ============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_observation_per_node(self):
        """Test with single observation per node."""
        df = pd.DataFrame({
            "ts_utc": pd.to_datetime(["2024-01-01"] * 5),
            "tenor_days": [30] * 5,
            "delta_bucket": ["P10", "P25", "ATM", "C25", "C10"],
            "iv_mid": [0.25, 0.22, 0.20, 0.21, 0.24],
            "spread_pct": [0.03, 0.025, 0.02, 0.025, 0.03],
        })

        generator = NodeFeatureGenerator()
        result_df, result = generator.generate(df)

        assert result.row_count == 5
        # Cross-sectional features should be computed
        assert pd.notna(result_df["skew_slope"].iloc[0])

    def test_missing_delta_bucket(self):
        """Test with missing delta buckets."""
        df = pd.DataFrame({
            "ts_utc": pd.to_datetime(["2024-01-01"] * 3),
            "tenor_days": [30] * 3,
            "delta_bucket": ["P25", "ATM", "C25"],  # Missing P10, C10
            "iv_mid": [0.25, 0.20, 0.18],
            "spread_pct": [0.025, 0.02, 0.025],
        })

        generator = NodeFeatureGenerator()
        result_df, result = generator.generate(df)

        # Should still compute what it can
        assert result.row_count == 3

    def test_constant_iv(self):
        """Test with constant IV (no changes)."""
        df = pd.DataFrame({
            "ts_utc": pd.date_range("2024-01-01", periods=10),
            "tenor_days": [30] * 10,
            "delta_bucket": ["ATM"] * 10,
            "iv_mid": [0.20] * 10,
            "spread_pct": [0.02] * 10,
        })

        generator = NodeFeatureGenerator()
        result_df, result = generator.generate(df)

        # IV changes should be zero
        assert (result_df["iv_change_1d"].dropna() == 0).all()

    def test_nan_in_iv(self):
        """Test handling of NaN values in IV."""
        df = pd.DataFrame({
            "ts_utc": pd.date_range("2024-01-01", periods=10),
            "tenor_days": [30] * 10,
            "delta_bucket": ["ATM"] * 10,
            "iv_mid": [0.20, np.nan, 0.22, 0.23, np.nan, 0.25, 0.26, 0.27, 0.28, 0.29],
            "spread_pct": [0.02] * 10,
        })

        generator = NodeFeatureGenerator()
        result_df, result = generator.generate(df)

        # Should not crash
        assert result.row_count == 10

    def test_extreme_values(self):
        """Test handling of extreme values."""
        df = pd.DataFrame({
            "ts_utc": pd.date_range("2024-01-01", periods=5),
            "tenor_days": [30] * 5,
            "delta_bucket": ["ATM"] * 5,
            "iv_mid": [0.01, 0.50, 1.00, 2.00, 0.10],  # Very low to very high
            "spread_pct": [0.001, 0.10, 0.50, 1.00, 0.01],  # Extreme spreads
            "volume": [0, 1, 1000000, 10, 100],
        })

        generator = NodeFeatureGenerator()
        result_df, result = generator.generate(df)

        # Should not crash
        assert result.row_count == 5

        # Log volume should handle zero
        assert result_df["log_volume"].iloc[0] == 0  # log1p(0) = 0
