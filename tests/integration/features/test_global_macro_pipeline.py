"""Integration tests for global and macro feature generation.

These tests verify the end-to-end feature generation workflow including:
- Underlying bars → global features
- FRED series → macro features
- Proper handling of realistic data patterns
- No future data leakage
"""

import numpy as np
import pandas as pd
import pytest

from src.features.global_.realized_vol import RealizedVolGenerator
from src.features.global_.returns import ReturnsGenerator
from src.features.macro.alignment import AlignmentConfig
from src.features.macro.transforms import MacroTransformConfig, MacroTransformGenerator


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def realistic_underlying_data():
    """Create realistic underlying bar data.

    Simulates 252 trading days (1 year) of SPY-like data with:
    - Trending behavior
    - Volatility clustering
    - Occasional large moves
    """
    np.random.seed(42)
    n = 252

    # Generate realistic returns with volatility clustering
    base_vol = 0.01  # ~16% annualized
    volatility = base_vol * np.ones(n)

    # Add volatility clustering
    for i in range(1, n):
        volatility[i] = 0.9 * volatility[i - 1] + 0.1 * base_vol + 0.05 * abs(np.random.randn()) * base_vol

    returns = np.random.randn(n) * volatility
    prices = 400 * np.exp(np.cumsum(returns))  # Start at ~400 (SPY-like)

    return pd.DataFrame({
        "ts_utc": pd.date_range("2024-01-01", periods=n, freq="B"),  # Business days
        "open": prices * (1 + np.random.randn(n) * 0.001),
        "high": prices * (1 + abs(np.random.randn(n) * 0.005)),
        "low": prices * (1 - abs(np.random.randn(n) * 0.005)),
        "close": prices,
        "volume": np.random.randint(50000000, 100000000, n),
    })


@pytest.fixture
def realistic_fred_data():
    """Create realistic FRED series data.

    Simulates weekly treasury rate data with:
    - Slow trend
    - Mean reversion
    - Occasional jumps (Fed meetings)
    """
    np.random.seed(42)
    n = 104  # 2 years of weekly data

    # Treasury rate simulation (starting around 4%)
    rate = 4.0
    rates = [rate]

    for _ in range(n - 1):
        # Mean reversion to 4%
        drift = 0.01 * (4.0 - rate)
        # Random shock
        shock = np.random.randn() * 0.05
        # Occasional Fed move
        if np.random.rand() < 0.1:
            shock += np.random.choice([-0.25, 0.25])

        rate = rate + drift + shock
        rate = max(0.5, min(8.0, rate))  # Bounds
        rates.append(rate)

    obs_dates = pd.date_range("2022-01-01", periods=n, freq="W")

    return pd.DataFrame({
        "obs_date": obs_dates,
        "value": rates,
        # Release 4 days after observation (typical for FRED)
        "release_datetime_utc": obs_dates + pd.Timedelta(days=4),
    })


# ============================================================================
# End-to-End Global Features Tests
# ============================================================================


class TestGlobalFeaturesPipeline:
    """Test complete global feature generation pipeline."""

    def test_full_global_pipeline(self, realistic_underlying_data):
        """Test full global features pipeline."""
        returns_gen = ReturnsGenerator()
        rv_gen = RealizedVolGenerator()

        # Generate returns
        with_returns = returns_gen.generate(realistic_underlying_data)

        # Generate realized vol (uses returns internally)
        result = rv_gen.generate(with_returns)

        # Check all expected features present
        expected_features = [
            "returns_1d", "returns_5d", "returns_10d", "returns_21d",
            "log_returns_1d", "log_returns_5d",
            "rv_5d", "rv_10d", "rv_21d",
            "realized_vol_5d", "realized_vol_10d", "realized_vol_21d",
            "vol_of_vol_21d",
            "drawdown", "max_drawdown_252d",
        ]

        for feat in expected_features:
            assert feat in result.columns, f"Missing feature: {feat}"

    def test_global_features_reasonable_values(self, realistic_underlying_data):
        """Test global features have reasonable values."""
        returns_gen = ReturnsGenerator()
        rv_gen = RealizedVolGenerator()

        with_returns = returns_gen.generate(realistic_underlying_data)
        result = rv_gen.generate(with_returns)

        # Daily returns should be small (typically < 5%)
        returns_1d = result["returns_1d"].dropna()
        assert abs(returns_1d.mean()) < 0.01
        assert returns_1d.std() < 0.05

        # Annualized realized vol should be in reasonable range (5-50%)
        rv_21d = result["rv_21d"].dropna()
        realized_vol = np.sqrt(rv_21d)
        assert realized_vol.mean() > 0.05
        assert realized_vol.mean() < 0.50

        # Drawdown should be non-positive
        assert (result["drawdown"] <= 0).all()

    def test_global_features_no_leakage(self, realistic_underlying_data):
        """Test global features don't leak future data."""
        returns_gen = ReturnsGenerator()

        result = returns_gen.generate(realistic_underlying_data)

        # First row should have NaN for returns
        assert pd.isna(result["returns_1d"].iloc[0])

        # First 5 rows should have NaN for 5d returns
        assert result["returns_5d"].iloc[:5].isna().all()

        # Values at row N should only depend on rows <= N
        # (Verified by the fact that rolling operations work correctly)


# ============================================================================
# End-to-End Macro Features Tests
# ============================================================================


class TestMacroFeaturesPipeline:
    """Test complete macro feature generation pipeline."""

    def test_full_macro_pipeline(self, realistic_fred_data):
        """Test full macro features pipeline."""
        generator = MacroTransformGenerator()

        result = generator.generate(realistic_fred_data, "DGS10")

        # Check all expected features present
        expected_features = [
            "ts_utc",
            "DGS10_level",
            "DGS10_change_1w",
            "DGS10_change_1m",
            "DGS10_zscore",
        ]

        for feat in expected_features:
            assert feat in result.columns, f"Missing feature: {feat}"

    def test_macro_percent_conversion(self, realistic_fred_data):
        """Test percent to decimal conversion."""
        config = MacroTransformConfig(percent_to_decimal=True)
        generator = MacroTransformGenerator(config)

        result = generator.generate(realistic_fred_data, "DGS10")

        # Original values are ~4%, converted should be ~0.04
        level = result["DGS10_level"]
        assert level.mean() < 0.1  # Should be decimal, not percent

    def test_macro_multi_series(self, realistic_fred_data):
        """Test multi-series macro generation."""
        # Create VIX-like data (no percent conversion)
        np.random.seed(123)
        n = len(realistic_fred_data)
        vix_data = pd.DataFrame({
            "obs_date": realistic_fred_data["obs_date"],
            "value": 15 + np.random.randn(n) * 3,
        })

        config = MacroTransformConfig(percent_to_decimal=True)
        generator = MacroTransformGenerator(config)
        series_data = {
            "DGS10": realistic_fred_data,
            "VIXCLS": vix_data,
        }

        result = generator.generate_multi(series_data)

        assert "DGS10_level" in result.columns
        assert "VIXCLS_level" in result.columns

        # DGS10 should be converted (small values)
        assert result["DGS10_level"].mean() < 0.1

        # VIX should NOT be converted (larger values)
        assert result["VIXCLS_level"].mean() > 10

    def test_macro_alignment_respects_release(self, realistic_fred_data):
        """Test macro features respect release time alignment."""
        config = MacroTransformConfig(
            alignment=AlignmentConfig(mode="strict"),
        )
        generator = MacroTransformGenerator(config)

        result = generator.generate(realistic_fred_data, "DGS10")

        # ts_utc should be the release time, not obs_date
        first_ts = result["ts_utc"].iloc[0]
        first_release = realistic_fred_data["release_datetime_utc"].iloc[0]
        assert first_ts == first_release


# ============================================================================
# Determinism Tests
# ============================================================================


class TestDeterminism:
    """Test that feature generation is deterministic."""

    def test_global_determinism(self, realistic_underlying_data):
        """Test global features are deterministic."""
        returns_gen = ReturnsGenerator()
        rv_gen = RealizedVolGenerator()

        results = []
        for _ in range(3):
            with_returns = returns_gen.generate(realistic_underlying_data.copy())
            result = rv_gen.generate(with_returns)
            results.append(result)

        # Compare all runs
        for i in range(1, len(results)):
            for col in results[0].columns:
                if results[0][col].dtype == float:
                    np.testing.assert_allclose(
                        results[0][col].values,
                        results[i][col].values,
                        rtol=1e-10,
                        equal_nan=True,
                    )

    def test_macro_determinism(self, realistic_fred_data):
        """Test macro features are deterministic."""
        generator = MacroTransformGenerator()

        results = []
        for _ in range(3):
            result = generator.generate(realistic_fred_data.copy(), "DGS10")
            results.append(result)

        # Compare all runs
        for i in range(1, len(results)):
            for col in results[0].columns:
                if results[0][col].dtype == float:
                    np.testing.assert_allclose(
                        results[0][col].values,
                        results[i][col].values,
                        rtol=1e-10,
                        equal_nan=True,
                    )


# ============================================================================
# Combined Pipeline Tests
# ============================================================================


class TestCombinedPipeline:
    """Test combined global and macro pipeline."""

    def test_combined_features_mergeable(
        self, realistic_underlying_data, realistic_fred_data
    ):
        """Test global and macro features can be merged."""
        # Generate global features
        returns_gen = ReturnsGenerator()
        rv_gen = RealizedVolGenerator()
        with_returns = returns_gen.generate(realistic_underlying_data)
        global_features = rv_gen.generate(with_returns)

        # Generate macro features
        macro_gen = MacroTransformGenerator()
        macro_features = macro_gen.generate(realistic_fred_data, "DGS10")

        # Both should have ts_utc for merging
        assert "ts_utc" in global_features.columns
        assert "ts_utc" in macro_features.columns

        # Merge should work (outer join for different frequencies)
        merged = global_features.merge(macro_features, on="ts_utc", how="outer")
        assert len(merged) >= max(len(global_features), len(macro_features))


# ============================================================================
# Edge Cases
# ============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_short_underlying_data(self):
        """Test with short underlying data (less than longest window)."""
        df = pd.DataFrame({
            "ts_utc": pd.date_range("2024-01-01", periods=10, freq="D"),
            "close": [100 + i for i in range(10)],
        })

        returns_gen = ReturnsGenerator()
        rv_gen = RealizedVolGenerator()

        with_returns = returns_gen.generate(df)
        result = rv_gen.generate(with_returns)

        # Should not crash, but 21d features will be mostly NaN
        assert len(result) == 10
        assert result["rv_5d"].notna().any()

    def test_short_fred_data(self):
        """Test with short FRED data."""
        df = pd.DataFrame({
            "obs_date": pd.date_range("2024-01-01", periods=10, freq="W"),
            "value": [4.5 + i * 0.01 for i in range(10)],
        })

        generator = MacroTransformGenerator()
        result = generator.generate(df, "TEST")

        # Should not crash
        assert len(result) == 10

    def test_nan_in_underlying(self):
        """Test handling of NaN in underlying data."""
        df = pd.DataFrame({
            "ts_utc": pd.date_range("2024-01-01", periods=20, freq="D"),
            "close": [100 + i if i % 5 != 0 else np.nan for i in range(20)],
        })

        returns_gen = ReturnsGenerator()
        result = returns_gen.generate(df)

        # Should not crash
        assert len(result) == 20

    def test_nan_in_fred(self):
        """Test handling of NaN in FRED data."""
        df = pd.DataFrame({
            "obs_date": pd.date_range("2024-01-01", periods=20, freq="W"),
            "value": [4.5 if i % 3 != 0 else np.nan for i in range(20)],
        })

        generator = MacroTransformGenerator()
        result = generator.generate(df, "TEST")

        # Should not crash
        assert len(result) == 20
