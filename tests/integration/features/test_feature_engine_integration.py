"""Integration tests for the full feature engine pipeline.

These tests verify the end-to-end feature generation workflow:
- Surface snapshots → node features → global features → macro features → merged panel
- Database read/write operations
- Anti-leakage validation
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from src.features.engine import FeatureEngine, FeatureEngineConfig
from src.features.validators import FeatureValidator


@pytest.fixture
def sample_surface_snapshots():
    """Create realistic surface snapshot data with all required columns."""
    np.random.seed(42)

    # 60 days of data
    dates = pd.date_range("2024-01-01", periods=60, freq="D")
    tenors = [7, 30, 60]
    buckets = ["P10", "P25", "ATM", "C25", "C10"]

    rows = []
    for ts in dates:
        for tenor in tenors:
            for bucket in buckets:
                # Determine delta and right based on bucket
                bucket_params = {
                    "P10": ("P", -0.10, 0.04),
                    "P25": ("P", -0.25, 0.02),
                    "ATM": ("C", 0.50, 0.0),
                    "C25": ("C", 0.25, -0.01),
                    "C10": ("C", 0.10, -0.02),
                }
                right, delta_val, skew_adj = bucket_params[bucket]

                # Generate realistic IV surface
                base_iv = 0.20
                term_adj = 0.005 * (tenor / 30 - 1)
                noise = np.random.randn() * 0.01
                iv = max(0.05, base_iv + skew_adj + term_adj + noise)

                # Pricing data
                underlying_price = 450 + np.random.randn() * 5
                bid = 1.0 + np.random.rand() * 0.5
                ask = bid + 0.05 + np.random.rand() * 0.1
                spread = ask - bid
                mid_price = (bid + ask) / 2
                spread_pct = spread / mid_price if mid_price > 0 else 0

                # Expiry
                exp_date = (ts + timedelta(days=tenor)).date()

                # Vary strike by bucket for unique option_symbols
                strike_offset = {"P10": -20, "P25": -10, "ATM": 0, "C25": 10, "C10": 20}
                strike = 500.0 + strike_offset[bucket]

                rows.append({
                    "ts_utc": ts,
                    "option_symbol": f"SPY{exp_date.strftime('%y%m%d')}{right}0{int(strike*1000):08d}",
                    "exp_date": exp_date,
                    "strike": strike,
                    "right": right,
                    "bid": bid,
                    "ask": ask,
                    "mid_price": mid_price,
                    "spread": spread,
                    "spread_pct": spread_pct,
                    "tte_years": tenor / 365.0,
                    "tenor_days": tenor,
                    "underlying_price": underlying_price,
                    "rf_rate": 0.05,
                    "dividend_yield": 0.015,
                    "iv_mid": iv,
                    "iv_bid": iv - 0.01,
                    "iv_ask": iv + 0.01,
                    "delta": delta_val,
                    "gamma": 0.01 + np.random.rand() * 0.02,
                    "vega": 0.1 + np.random.rand() * 0.1,
                    "theta": -0.05 - np.random.rand() * 0.05,
                    "delta_bucket": bucket,
                    "flags": 0,
                    "volume": np.random.randint(100, 5000),
                    "open_interest": np.random.randint(1000, 50000),
                })

    return pd.DataFrame(rows)


@pytest.fixture
def sample_underlying_bars():
    """Create realistic underlying bar data."""
    np.random.seed(42)
    n = 90  # Extra days for lookback

    # Generate realistic price path
    returns = np.random.randn(n) * 0.01  # ~16% annualized vol
    prices = 450 * np.exp(np.cumsum(returns))

    return pd.DataFrame({
        "dataset": ["GLBX.MDP3"] * n,
        "schema": ["ohlcv-1m"] * n,
        "stype_in": ["continuous"] * n,
        "ts_utc": pd.date_range("2023-11-01", periods=n, freq="D"),
        "symbol": ["SPY"] * n,
        "timeframe": ["1m"] * n,
        "open": prices * (1 + np.random.randn(n) * 0.002),
        "high": prices * (1 + abs(np.random.randn(n) * 0.01)),
        "low": prices * (1 - abs(np.random.randn(n) * 0.01)),
        "close": prices,
        "volume": np.random.randint(50000000, 100000000, n),
    })


@pytest.fixture
def sample_fred_data():
    """Create realistic FRED series data."""
    np.random.seed(42)
    n = 180  # Extra days for lookback and z-score

    dates = pd.date_range("2023-07-01", periods=n, freq="D")

    # DGS10 - 10Y Treasury rate
    rate = 4.0
    rates = []
    for _ in range(n):
        rate = rate + 0.01 * (4.0 - rate) + np.random.randn() * 0.03
        rate = max(3.0, min(5.5, rate))
        rates.append(rate)

    return {
        "DGS10": pd.DataFrame({
            "series_id": ["DGS10"] * n,
            "obs_date": dates,
            "value": rates,
            "release_datetime_utc": dates + pd.Timedelta(days=1),
        })
    }


@pytest.fixture
def populated_db(db_engine, raw_repo, derived_repo, sample_surface_snapshots, sample_underlying_bars, sample_fred_data):
    """Populate database with test data."""
    # Write surface snapshots
    derived_repo.write_surface_snapshots(
        sample_surface_snapshots,
        build_run_id="test_run",
        version="v1.0",
    )

    # Write underlying bars
    raw_repo.write_underlying_bars(sample_underlying_bars, run_id="test_run")

    # Write FRED data
    raw_repo.write_fred_series(sample_fred_data["DGS10"])

    return db_engine


# ============================================================================
# Full Pipeline Tests
# ============================================================================


class TestFullFeaturePipeline:
    """Test complete feature generation pipeline."""

    def test_full_pipeline_builds_panel(self, populated_db, derived_repo, raw_repo):
        """Test full pipeline produces valid feature panel."""
        config = FeatureEngineConfig(
            underlying_symbol="SPY",
            fred_series=["DGS10"],
            lookback_buffer_days=30,
        )

        engine = FeatureEngine(
            config=config,
            engine=populated_db,
            raw_repo=raw_repo,
            derived_repo=derived_repo,
        )

        panel_df, result = engine.build_feature_panel(
            start=datetime(2024, 1, 15),
            end=datetime(2024, 2, 15),
            surface_version="v1.0",
            feature_version="v1.0",
            write_to_db=False,
        )

        # Should have data
        assert result.row_count > 0

        # Should have features from all generators
        assert result.node_features_count > 0
        assert result.global_features_count > 0
        assert result.macro_features_count > 0

        # Should have expected columns
        assert "iv_mid" in panel_df.columns
        assert "iv_change_1d" in panel_df.columns
        assert "returns_1d" in panel_df.columns
        assert "DGS10_level" in panel_df.columns

    def test_pipeline_writes_to_database(self, populated_db, derived_repo, raw_repo):
        """Test pipeline writes results to database."""
        config = FeatureEngineConfig(
            underlying_symbol="SPY",
            fred_series=["DGS10"],
            lookback_buffer_days=30,
        )

        engine = FeatureEngine(
            config=config,
            engine=populated_db,
            raw_repo=raw_repo,
            derived_repo=derived_repo,
        )

        panel_df, result = engine.build_feature_panel(
            start=datetime(2024, 1, 15),
            end=datetime(2024, 2, 15),
            surface_version="v1.0",
            feature_version="v1.0",
            write_to_db=True,
        )

        # Read back from database
        read_df = derived_repo.read_node_panel(
            feature_version="v1.0",
            start=datetime(2024, 1, 15),
            end=datetime(2024, 2, 15),
        )

        assert len(read_df) == len(panel_df)

    def test_pipeline_respects_date_range(self, populated_db, derived_repo, raw_repo):
        """Test pipeline filters to requested date range."""
        config = FeatureEngineConfig(
            underlying_symbol="SPY",
            fred_series=["DGS10"],
            lookback_buffer_days=30,
        )

        engine = FeatureEngine(
            config=config,
            engine=populated_db,
            raw_repo=raw_repo,
            derived_repo=derived_repo,
        )

        start = datetime(2024, 1, 20)
        end = datetime(2024, 1, 25)

        panel_df, result = engine.build_feature_panel(
            start=start,
            end=end,
            surface_version="v1.0",
            feature_version="v1.0",
            write_to_db=False,
        )

        # All timestamps should be in range
        assert panel_df["ts_utc"].min() >= start
        assert panel_df["ts_utc"].max() < end


# ============================================================================
# Feature Merge Tests
# ============================================================================


class TestFeatureMerge:
    """Test feature merging logic."""

    def test_global_features_merged_correctly(self, populated_db, derived_repo, raw_repo):
        """Test global features are merged with node features."""
        config = FeatureEngineConfig(
            underlying_symbol="SPY",
            fred_series=[],  # Skip macro
            include_macro_features=False,
        )

        engine = FeatureEngine(
            config=config,
            engine=populated_db,
            raw_repo=raw_repo,
            derived_repo=derived_repo,
        )

        panel_df, result = engine.build_feature_panel(
            start=datetime(2024, 1, 15),
            end=datetime(2024, 2, 15),
            surface_version="v1.0",
            feature_version="v1.0",
            write_to_db=False,
        )

        # Global features should be present on each node row
        assert "returns_1d" in panel_df.columns
        assert "underlying_rv_5d" in panel_df.columns

        # Check that global features are replicated across nodes for same timestamp
        for ts in panel_df["ts_utc"].unique()[:3]:
            ts_data = panel_df[panel_df["ts_utc"] == ts]
            if len(ts_data) > 1:
                # All nodes at same timestamp should have same global features
                returns = ts_data["returns_1d"].dropna()
                if len(returns) > 1:
                    assert returns.nunique() == 1

    def test_macro_features_merged_correctly(self, populated_db, derived_repo, raw_repo):
        """Test macro features are merged with node features."""
        config = FeatureEngineConfig(
            underlying_symbol="SPY",
            fred_series=["DGS10"],
            include_global_features=False,
        )

        engine = FeatureEngine(
            config=config,
            engine=populated_db,
            raw_repo=raw_repo,
            derived_repo=derived_repo,
        )

        panel_df, result = engine.build_feature_panel(
            start=datetime(2024, 1, 15),
            end=datetime(2024, 2, 15),
            surface_version="v1.0",
            feature_version="v1.0",
            write_to_db=False,
        )

        # Macro features should be present
        assert "DGS10_level" in panel_df.columns

        # Macro features should have valid values (not all NaN)
        assert panel_df["DGS10_level"].notna().sum() > 0


# ============================================================================
# Anti-Leakage Validation Tests
# ============================================================================


class TestAntiLeakageValidation:
    """Test anti-leakage validation integration."""

    def test_pipeline_runs_validation(self, populated_db, derived_repo, raw_repo):
        """Test that pipeline runs anti-leakage validation."""
        config = FeatureEngineConfig(
            underlying_symbol="SPY",
            fred_series=["DGS10"],
        )

        engine = FeatureEngine(
            config=config,
            engine=populated_db,
            raw_repo=raw_repo,
            derived_repo=derived_repo,
        )

        panel_df, result = engine.build_feature_panel(
            start=datetime(2024, 1, 15),
            end=datetime(2024, 2, 15),
            surface_version="v1.0",
            feature_version="v1.0",
            write_to_db=False,
        )

        # Result should indicate validation status
        assert hasattr(result, "validation_passed")

    def test_manual_validation_passes(self, populated_db, derived_repo, raw_repo):
        """Test manual validation on generated features."""
        config = FeatureEngineConfig(
            underlying_symbol="SPY",
            fred_series=["DGS10"],
        )

        engine = FeatureEngine(
            config=config,
            engine=populated_db,
            raw_repo=raw_repo,
            derived_repo=derived_repo,
        )

        panel_df, result = engine.build_feature_panel(
            start=datetime(2024, 1, 15),
            end=datetime(2024, 2, 15),
            surface_version="v1.0",
            feature_version="v1.0",
            write_to_db=False,
        )

        # Run manual validation
        validator = FeatureValidator()
        validation_result = validator.validate_no_future_leakage(panel_df)

        # Should pass (no error-level issues)
        assert validation_result.passed


# ============================================================================
# Node-Only Pipeline Tests
# ============================================================================


class TestNodeOnlyPipeline:
    """Test node-only feature generation."""

    def test_node_only_skips_global_and_macro(self, populated_db, derived_repo, raw_repo):
        """Test node-only config skips global and macro features."""
        config = FeatureEngineConfig(
            include_node_features=True,
            include_global_features=False,
            include_macro_features=False,
        )

        engine = FeatureEngine(
            config=config,
            engine=populated_db,
            raw_repo=raw_repo,
            derived_repo=derived_repo,
        )

        panel_df, result = engine.build_feature_panel(
            start=datetime(2024, 1, 15),
            end=datetime(2024, 2, 15),
            surface_version="v1.0",
            feature_version="v1.0",
            write_to_db=False,
        )

        # Should have node features
        assert result.node_features_count > 0
        assert "iv_change_1d" in panel_df.columns

        # Should NOT have global/macro features
        assert result.global_features_count == 0
        assert result.macro_features_count == 0
        assert "returns_1d" not in panel_df.columns


# ============================================================================
# Edge Cases
# ============================================================================


class TestEdgeCases:
    """Test edge cases."""

    def test_empty_surface_data(self, db_engine, derived_repo, raw_repo):
        """Test handling of empty surface data."""
        # Don't populate any data
        config = FeatureEngineConfig()

        engine = FeatureEngine(
            config=config,
            engine=db_engine,
            raw_repo=raw_repo,
            derived_repo=derived_repo,
        )

        panel_df, result = engine.build_feature_panel(
            start=datetime(2024, 1, 15),
            end=datetime(2024, 2, 15),
            surface_version="v1.0",
            feature_version="v1.0",
            write_to_db=False,
        )

        assert len(panel_df) == 0
        assert result.row_count == 0

    def test_missing_underlying_data(self, db_engine, derived_repo, raw_repo, sample_surface_snapshots):
        """Test handling when underlying data is missing."""
        # Only populate surface snapshots, not underlying bars
        derived_repo.write_surface_snapshots(
            sample_surface_snapshots,
            build_run_id="test",
            version="v1.0",
        )

        config = FeatureEngineConfig(
            include_global_features=True,
            include_macro_features=False,
        )

        engine = FeatureEngine(
            config=config,
            engine=db_engine,
            raw_repo=raw_repo,
            derived_repo=derived_repo,
        )

        panel_df, result = engine.build_feature_panel(
            start=datetime(2024, 1, 15),
            end=datetime(2024, 2, 15),
            surface_version="v1.0",
            feature_version="v1.0",
            write_to_db=False,
        )

        # Should still work (node features at minimum)
        assert result.node_features_count > 0
        # Global features will be empty due to missing data
        assert result.global_features_count == 0


# ============================================================================
# Determinism Tests
# ============================================================================


class TestDeterminism:
    """Test feature generation determinism."""

    def test_same_output_multiple_runs(self, populated_db, derived_repo, raw_repo):
        """Test that multiple runs produce identical output."""
        config = FeatureEngineConfig(
            underlying_symbol="SPY",
            fred_series=["DGS10"],
        )

        engine = FeatureEngine(
            config=config,
            engine=populated_db,
            raw_repo=raw_repo,
            derived_repo=derived_repo,
        )

        results = []
        for _ in range(3):
            panel_df, result = engine.build_feature_panel(
                start=datetime(2024, 1, 20),
                end=datetime(2024, 1, 25),
                surface_version="v1.0",
                feature_version="v1.0",
                write_to_db=False,
            )
            results.append(panel_df)

        # Compare outputs
        for i in range(1, len(results)):
            # Same shape
            assert results[0].shape == results[i].shape

            # Same columns
            assert list(results[0].columns) == list(results[i].columns)

            # Same values (for numeric columns)
            for col in results[0].select_dtypes(include=[np.number]).columns:
                np.testing.assert_array_almost_equal(
                    results[0][col].values,
                    results[i][col].values,
                    decimal=10,
                    err_msg=f"Column {col} differs between runs",
                )
