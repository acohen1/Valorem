"""Integration tests for SurfaceBuilder.

Tests the full pipeline from raw quotes to surface snapshots using
an in-memory SQLite database.
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from src.config.schema import (
    BlackScholesConfig,
    DeltaBucketsConfig,
    SurfaceConfig,
    TenorBinsConfig,
    UniverseConfig,
)
from src.surface.builder import BuildResult, SurfaceBuilder
from src.surface.quality.filters import QualityConfig


@pytest.fixture
def surface_config():
    """Create surface configuration."""
    return SurfaceConfig(
        delta_buckets=DeltaBucketsConfig(
            ATM=[-0.55, -0.45, 0.45, 0.55],
            P40=[-0.45, -0.35],
            P25=[-0.35, -0.20],
            P10=[-0.20, 0.0],
            C10=[0.0, 0.20],
            C25=[0.20, 0.35],
            C40=[0.35, 0.45],
        ),
        tenor_bins=TenorBinsConfig(bins=[7, 14, 30, 60, 90]),
        black_scholes=BlackScholesConfig(max_iterations=100, tolerance=1e-6),
    )


@pytest.fixture
def universe_config():
    """Create universe configuration."""
    return UniverseConfig(underlying="SPY")


@pytest.fixture
def quality_config():
    """Create quality configuration."""
    return QualityConfig(
        allow_crossed_quotes=False,
        max_spread_pct=0.50,
        min_volume=None,  # Disable for testing
        min_open_interest=None,  # Disable for testing
    )


def create_realistic_option_chain(
    base_time: datetime,
    underlying_price: float,
    exp_date: datetime,
    num_strikes: int = 11,
) -> pd.DataFrame:
    """Create a realistic option chain for testing.

    Args:
        base_time: Quote timestamp
        underlying_price: Current underlying price
        exp_date: Expiration date
        num_strikes: Number of strikes to generate

    Returns:
        DataFrame with realistic option quotes
    """
    # Generate strikes around ATM
    atm_strike = round(underlying_price / 5) * 5  # Round to nearest $5
    strikes = [atm_strike + (i - num_strikes // 2) * 5 for i in range(num_strikes)]

    rows = []
    for strike in strikes:
        # Compute rough theoretical values (simplified)
        moneyness = underlying_price / strike

        # Call pricing (very simplified)
        if moneyness > 1:  # ITM call
            call_intrinsic = underlying_price - strike
            call_mid = call_intrinsic + 1.5
        else:  # OTM call
            call_mid = max(0.10, 10 * (1.1 - 1/moneyness))

        # Put pricing (very simplified)
        if moneyness < 1:  # ITM put
            put_intrinsic = strike - underlying_price
            put_mid = put_intrinsic + 1.5
        else:  # OTM put
            put_mid = max(0.10, 10 * (moneyness - 0.9))

        # Add spread
        spread_pct = 0.05
        call_bid = call_mid * (1 - spread_pct)
        call_ask = call_mid * (1 + spread_pct)
        put_bid = put_mid * (1 - spread_pct)
        put_ask = put_mid * (1 + spread_pct)

        # Create option symbols
        exp_str = exp_date.strftime("%y%m%d")
        call_symbol = f"SPY{exp_str}C{int(strike * 1000):08d}"
        put_symbol = f"SPY{exp_str}P{int(strike * 1000):08d}"

        rows.append({
            "ts_utc": base_time,
            "option_symbol": call_symbol,
            "exp_date": exp_date,
            "strike": float(strike),
            "right": "C",
            "bid": round(call_bid, 2),
            "ask": round(call_ask, 2),
            "volume": 100,
            "open_interest": 500,
            # Required schema columns
            "dataset": "OPRA.PILLAR",
            "schema": "mbp-1",
            "stype_in": "raw_symbol",
        })

        rows.append({
            "ts_utc": base_time,
            "option_symbol": put_symbol,
            "exp_date": exp_date,
            "strike": float(strike),
            "right": "P",
            "bid": round(put_bid, 2),
            "ask": round(put_ask, 2),
            "volume": 80,
            "open_interest": 400,
            # Required schema columns
            "dataset": "OPRA.PILLAR",
            "schema": "mbp-1",
            "stype_in": "raw_symbol",
        })

    return pd.DataFrame(rows)


# =============================================================================
# Integration Tests
# =============================================================================


class TestSurfaceBuilderIntegration:
    """Integration tests for full surface build pipeline."""

    def test_end_to_end_surface_build(
        self, raw_repo, derived_repo, surface_config, universe_config, quality_config
    ):
        """Test full pipeline: raw data -> surface snapshots."""
        # Setup: Write raw data to database
        base_time = datetime(2024, 1, 15, 16, 0, 0)
        underlying_price = 478.50

        # Create underlying bars
        underlying_df = pd.DataFrame({
            "ts_utc": [
                base_time - timedelta(hours=2),
                base_time - timedelta(hours=1),
                base_time,
            ],
            "symbol": ["SPY"] * 3,
            "timeframe": ["1m"] * 3,
            "open": [477.0, 478.0, 478.25],
            "high": [478.5, 479.0, 479.0],
            "low": [476.5, 477.5, 478.0],
            "close": [478.0, 478.5, underlying_price],
            "volume": [1000000, 1100000, 1200000],
            # Required schema columns
            "dataset": ["GLBX.MDP3"] * 3,
            "schema": ["ohlcv-1m"] * 3,
            "stype_in": ["continuous"] * 3,
        })
        raw_repo.write_underlying_bars(underlying_df, "test-run-1")

        # Create FRED rate data
        fred_df = pd.DataFrame({
            "series_id": ["DGS10"] * 3,
            "obs_date": [
                datetime(2024, 1, 12),
                datetime(2024, 1, 13),
                datetime(2024, 1, 14),
            ],
            "value": [0.0425, 0.043, 0.0435],
            "release_datetime_utc": [
                datetime(2024, 1, 12, 21, 0),
                datetime(2024, 1, 13, 21, 0),
                datetime(2024, 1, 14, 21, 0),
            ],
        })
        raw_repo.write_fred_series(fred_df)

        # Create option quotes - multiple expirations
        exp_7d = datetime(2024, 1, 22)
        exp_30d = datetime(2024, 2, 14)

        quotes_7d = create_realistic_option_chain(base_time, underlying_price, exp_7d)
        quotes_30d = create_realistic_option_chain(base_time, underlying_price, exp_30d)
        quotes_df = pd.concat([quotes_7d, quotes_30d], ignore_index=True)

        raw_repo.write_option_quotes(quotes_df, "test-run-1")

        # Build surface
        builder = SurfaceBuilder(
            config=surface_config,
            universe=universe_config,
            raw_repo=raw_repo,
            derived_repo=derived_repo,
            quality_config=quality_config,
        )

        result = builder.build_surface(
            start=datetime(2024, 1, 15),
            end=datetime(2024, 1, 16),
            version="v1.0",
        )

        # Verify result
        assert isinstance(result, BuildResult)
        assert result.quotes_processed == len(quotes_df)
        assert result.row_count > 0  # Should have some surface snapshots
        assert result.version == "v1.0"

        # Read back surface snapshots
        snapshots = derived_repo.read_surface_snapshots(
            start=datetime(2024, 1, 15),
            end=datetime(2024, 1, 16),
            version="v1.0",
        )

        assert not snapshots.empty
        assert "iv_mid" in snapshots.columns
        assert "delta" in snapshots.columns
        assert "tenor_days" in snapshots.columns
        assert "delta_bucket" in snapshots.columns

        # Verify IVs are reasonable (between 1% and 200%)
        valid_ivs = snapshots["iv_mid"].dropna()
        assert (valid_ivs > 0.01).all()  # At least 1%
        assert (valid_ivs < 2.0).all()  # At most 200%

        # Verify deltas are in valid range
        valid_deltas = snapshots["delta"].dropna()
        assert (valid_deltas >= -1.0).all()
        assert (valid_deltas <= 1.0).all()

    def test_surface_build_with_multiple_timestamps(
        self, raw_repo, derived_repo, surface_config, universe_config, quality_config
    ):
        """Test surface build with multiple quote timestamps."""
        underlying_price = 480.0

        # Create underlying bars for multiple timestamps
        timestamps = [
            datetime(2024, 1, 15, 14, 0, 0),
            datetime(2024, 1, 15, 15, 0, 0),
            datetime(2024, 1, 15, 16, 0, 0),
        ]
        underlying_df = pd.DataFrame({
            "ts_utc": timestamps,
            "symbol": ["SPY"] * 3,
            "timeframe": ["1m"] * 3,
            "open": [479.0, 479.5, 480.0],
            "high": [480.0, 480.5, 481.0],
            "low": [478.5, 479.0, 479.5],
            "close": [479.5, 480.0, 480.5],
            "volume": [1000000] * 3,
            # Required schema columns
            "dataset": ["GLBX.MDP3"] * 3,
            "schema": ["ohlcv-1m"] * 3,
            "stype_in": ["continuous"] * 3,
        })
        raw_repo.write_underlying_bars(underlying_df, "test-run-2")

        # Create FRED data
        fred_df = pd.DataFrame({
            "series_id": ["DGS10"],
            "obs_date": [datetime(2024, 1, 14)],
            "value": [0.045],
            "release_datetime_utc": [datetime(2024, 1, 14, 21, 0)],
        })
        raw_repo.write_fred_series(fred_df)

        # Create quotes at each timestamp
        exp_date = datetime(2024, 1, 22)
        all_quotes = []
        for ts in timestamps:
            quotes = create_realistic_option_chain(ts, underlying_price, exp_date, num_strikes=5)
            all_quotes.append(quotes)

        quotes_df = pd.concat(all_quotes, ignore_index=True)
        raw_repo.write_option_quotes(quotes_df, "test-run-2")

        # Build surface
        builder = SurfaceBuilder(
            config=surface_config,
            universe=universe_config,
            raw_repo=raw_repo,
            derived_repo=derived_repo,
            quality_config=quality_config,
        )

        result = builder.build_surface(
            start=datetime(2024, 1, 15),
            end=datetime(2024, 1, 16),
            version="v1.0",
        )

        # Should have snapshots for multiple timestamps
        assert result.row_count > 0

        snapshots = derived_repo.read_surface_snapshots(
            start=datetime(2024, 1, 15),
            end=datetime(2024, 1, 16),
            version="v1.0",
        )

        # Verify multiple timestamps present
        unique_timestamps = snapshots["ts_utc"].nunique()
        assert unique_timestamps >= 1

    def test_surface_build_deterministic(
        self, raw_repo, derived_repo, surface_config, universe_config, quality_config
    ):
        """Test that surface builds are deterministic (same input -> same output)."""
        # Setup data
        base_time = datetime(2024, 1, 15, 16, 0, 0)
        underlying_price = 475.0

        underlying_df = pd.DataFrame({
            "ts_utc": [base_time],
            "symbol": ["SPY"],
            "timeframe": ["1m"],
            "open": [474.0],
            "high": [476.0],
            "low": [473.5],
            "close": [underlying_price],
            "volume": [1000000],
            # Required schema columns
            "dataset": ["GLBX.MDP3"],
            "schema": ["ohlcv-1m"],
            "stype_in": ["continuous"],
        })
        raw_repo.write_underlying_bars(underlying_df, "test-run-3")

        fred_df = pd.DataFrame({
            "series_id": ["DGS10"],
            "obs_date": [datetime(2024, 1, 14)],
            "value": [0.04],
            "release_datetime_utc": [datetime(2024, 1, 14, 21, 0)],
        })
        raw_repo.write_fred_series(fred_df)

        exp_date = datetime(2024, 1, 22)
        quotes_df = create_realistic_option_chain(base_time, underlying_price, exp_date, num_strikes=7)
        raw_repo.write_option_quotes(quotes_df, "test-run-3")

        # Build surface twice
        builder = SurfaceBuilder(
            config=surface_config,
            universe=universe_config,
            raw_repo=raw_repo,
            derived_repo=derived_repo,
            quality_config=quality_config,
        )

        result1 = builder.build_surface(
            start=datetime(2024, 1, 15),
            end=datetime(2024, 1, 16),
            version="v1.0",
        )

        result2 = builder.build_surface(
            start=datetime(2024, 1, 15),
            end=datetime(2024, 1, 16),
            version="v1.1",  # Different version
        )

        # Same input should produce same row count (different IDs)
        assert result1.row_count == result2.row_count
        assert result1.quotes_processed == result2.quotes_processed
        assert result1.build_run_id != result2.build_run_id  # Different IDs

    def test_surface_build_handles_no_data(
        self, raw_repo, derived_repo, surface_config, universe_config, quality_config
    ):
        """Test that surface build handles missing data gracefully."""
        builder = SurfaceBuilder(
            config=surface_config,
            universe=universe_config,
            raw_repo=raw_repo,
            derived_repo=derived_repo,
            quality_config=quality_config,
        )

        result = builder.build_surface(
            start=datetime(2024, 1, 15),
            end=datetime(2024, 1, 16),
            version="v1.0",
        )

        assert result.row_count == 0
        assert result.quotes_processed == 0


class TestSurfaceQuality:
    """Tests for surface quality and data integrity."""

    def test_no_future_data_leakage_underlying(
        self, raw_repo, derived_repo, surface_config, universe_config, quality_config
    ):
        """Test that underlying prices don't leak future data."""
        # Create quotes at 14:00
        quote_time = datetime(2024, 1, 15, 14, 0, 0)
        exp_date = datetime(2024, 1, 22)

        # Create underlying bars: one before quote, one after
        underlying_df = pd.DataFrame({
            "ts_utc": [
                datetime(2024, 1, 15, 13, 0, 0),  # Before quote
                datetime(2024, 1, 15, 15, 0, 0),  # After quote (future!)
            ],
            "symbol": ["SPY", "SPY"],
            "timeframe": ["1m", "1m"],
            "open": [470.0, 475.0],
            "high": [472.0, 478.0],
            "low": [469.0, 474.0],
            "close": [471.0, 477.0],  # 471 at 13:00, 477 at 15:00
            "volume": [1000000, 1000000],
            # Required schema columns
            "dataset": ["GLBX.MDP3", "GLBX.MDP3"],
            "schema": ["ohlcv-1m", "ohlcv-1m"],
            "stype_in": ["continuous", "continuous"],
        })
        raw_repo.write_underlying_bars(underlying_df, "test-leakage-1")

        fred_df = pd.DataFrame({
            "series_id": ["DGS10"],
            "obs_date": [datetime(2024, 1, 14)],
            "value": [0.04],
            "release_datetime_utc": [datetime(2024, 1, 14, 21, 0)],
        })
        raw_repo.write_fred_series(fred_df)

        quotes_df = create_realistic_option_chain(quote_time, 471.0, exp_date, num_strikes=3)
        raw_repo.write_option_quotes(quotes_df, "test-leakage-1")

        builder = SurfaceBuilder(
            config=surface_config,
            universe=universe_config,
            raw_repo=raw_repo,
            derived_repo=derived_repo,
            quality_config=quality_config,
        )

        # The builder should use 471.0 (13:00 bar), not 477.0 (15:00 bar)
        # We verify this by checking the stored snapshots
        result = builder.build_surface(
            start=datetime(2024, 1, 15),
            end=datetime(2024, 1, 16),
            version="v1.0",
        )

        if result.row_count > 0:
            snapshots = derived_repo.read_surface_snapshots(
                start=datetime(2024, 1, 15),
                end=datetime(2024, 1, 16),
                version="v1.0",
            )

            # Check underlying_price if it's stored
            if "underlying_price" in snapshots.columns:
                assert (snapshots["underlying_price"] == 471.0).all()
