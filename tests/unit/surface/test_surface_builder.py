"""Unit tests for the SurfaceBuilder class.

Tests cover initialization, helper methods, and the full build pipeline.
"""

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch
import uuid

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


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def surface_config():
    """Create a test surface configuration."""
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
    """Create a test universe configuration."""
    return UniverseConfig(underlying="SPY")


@pytest.fixture
def quality_config():
    """Create a test quality configuration."""
    return QualityConfig(
        allow_crossed_quotes=False,
        max_spread_pct=0.50,
        min_volume=10,
        min_open_interest=100,
    )


@pytest.fixture
def mock_raw_repo():
    """Create a mock raw repository."""
    return MagicMock()


@pytest.fixture
def mock_derived_repo():
    """Create a mock derived repository."""
    return MagicMock()


@pytest.fixture
def builder(surface_config, universe_config, quality_config, mock_raw_repo, mock_derived_repo):
    """Create a SurfaceBuilder instance with mocked repositories."""
    return SurfaceBuilder(
        config=surface_config,
        universe=universe_config,
        raw_repo=mock_raw_repo,
        derived_repo=mock_derived_repo,
        quality_config=quality_config,
    )


@pytest.fixture
def sample_quotes_df():
    """Create sample option quotes DataFrame."""
    base_time = datetime(2024, 1, 15, 16, 0, 0)
    return pd.DataFrame({
        "ts_utc": [base_time] * 6,
        "option_symbol": ["SPY240119C00470000", "SPY240119P00470000",
                         "SPY240119C00480000", "SPY240119P00480000",
                         "SPY240119C00490000", "SPY240119P00490000"],
        "exp_date": [datetime(2024, 1, 19)] * 6,
        "strike": [470.0, 470.0, 480.0, 480.0, 490.0, 490.0],
        "right": ["C", "P", "C", "P", "C", "P"],
        "bid": [8.50, 1.20, 3.50, 3.80, 0.80, 10.50],
        "ask": [8.70, 1.30, 3.70, 4.00, 0.90, 10.70],
        "volume": [100, 50, 200, 150, 80, 120],
        "open_interest": [500, 300, 800, 600, 200, 400],
    })


@pytest.fixture
def sample_underlying_df():
    """Create sample underlying bars DataFrame."""
    base_time = datetime(2024, 1, 15, 15, 0, 0)
    return pd.DataFrame({
        "ts_utc": [base_time, base_time + timedelta(hours=1)],
        "symbol": ["SPY", "SPY"],
        "close": [478.50, 479.00],
        "open": [477.00, 478.50],
        "high": [479.00, 479.50],
        "low": [476.50, 478.00],
        "volume": [1000000, 1200000],
    })


@pytest.fixture
def sample_fred_df():
    """Create sample FRED series DataFrame."""
    return pd.DataFrame({
        "series_id": ["DGS10", "DGS10", "DGS10"],
        "obs_date": [datetime(2024, 1, 12), datetime(2024, 1, 13), datetime(2024, 1, 14)],
        "value": [0.0425, 0.0430, 0.0435],
        "release_datetime_utc": [
            datetime(2024, 1, 12, 21, 0),
            datetime(2024, 1, 13, 21, 0),
            datetime(2024, 1, 14, 21, 0),
        ],
    })


# =============================================================================
# Initialization Tests
# =============================================================================


class TestSurfaceBuilderInitialization:
    """Tests for SurfaceBuilder initialization."""

    def test_initialization_with_config(
        self, surface_config, universe_config, mock_raw_repo, mock_derived_repo
    ):
        """Test that builder initializes correctly with configuration."""
        builder = SurfaceBuilder(
            config=surface_config,
            universe=universe_config,
            raw_repo=mock_raw_repo,
            derived_repo=mock_derived_repo,
        )

        assert builder._config == surface_config
        assert builder._universe == universe_config
        assert builder._iv_solver is not None
        assert builder._greeks_calculator is not None
        assert builder._bucket_assigner is not None
        assert builder._quality_filter is not None

    def test_initialization_with_custom_quality_config(
        self, surface_config, universe_config, quality_config, mock_raw_repo, mock_derived_repo
    ):
        """Test that builder accepts custom quality configuration."""
        builder = SurfaceBuilder(
            config=surface_config,
            universe=universe_config,
            raw_repo=mock_raw_repo,
            derived_repo=mock_derived_repo,
            quality_config=quality_config,
        )

        assert builder._quality_config == quality_config

    def test_initialization_with_custom_rate_series(
        self, surface_config, universe_config, mock_raw_repo, mock_derived_repo
    ):
        """Test that builder accepts custom rate series ID."""
        builder = SurfaceBuilder(
            config=surface_config,
            universe=universe_config,
            raw_repo=mock_raw_repo,
            derived_repo=mock_derived_repo,
            rate_series_id="DGS2",
        )

        assert builder._rate_series_id == "DGS2"

    def test_buckets_config_to_dict(self, builder, surface_config):
        """Test conversion of DeltaBucketsConfig to dictionary."""
        bucket_dict = builder._buckets_config_to_dict(surface_config.delta_buckets)

        assert "ATM" in bucket_dict
        assert "P10" in bucket_dict
        assert "C10" in bucket_dict
        assert bucket_dict["ATM"] == [-0.55, -0.45, 0.45, 0.55]


# =============================================================================
# Helper Method Tests
# =============================================================================


class TestJoinUnderlyingPrice:
    """Tests for _join_underlying_price method."""

    def test_join_underlying_price_basic(self, builder, sample_quotes_df, sample_underlying_df):
        """Test basic underlying price join."""
        result = builder._join_underlying_price(sample_quotes_df, sample_underlying_df)

        assert "underlying_price" in result.columns
        # Should use the most recent underlying close (479.00 from 16:00)
        assert (result["underlying_price"] == 479.00).all()

    def test_join_underlying_price_backward_direction(self, builder):
        """Test that join uses backward direction (no future data)."""
        quotes = pd.DataFrame({
            "ts_utc": [datetime(2024, 1, 15, 14, 30, 0)],  # Before underlying bar
            "option_symbol": ["TEST"],
        })
        underlying = pd.DataFrame({
            "ts_utc": [datetime(2024, 1, 15, 14, 0, 0), datetime(2024, 1, 15, 15, 0, 0)],
            "close": [100.0, 101.0],
        })

        result = builder._join_underlying_price(quotes, underlying)

        # Should use 14:00 bar (100.0), not 15:00 bar (101.0)
        assert result["underlying_price"].iloc[0] == 100.0

    def test_join_underlying_price_empty_underlying(self, builder, sample_quotes_df):
        """Test join with empty underlying data."""
        empty_underlying = pd.DataFrame()
        result = builder._join_underlying_price(sample_quotes_df, empty_underlying)

        assert "underlying_price" in result.columns
        assert result["underlying_price"].isna().all()


class TestJoinRiskFreeRate:
    """Tests for _join_risk_free_rate method."""

    def test_join_risk_free_rate_basic(self, builder, sample_quotes_df, sample_fred_df):
        """Test basic risk-free rate join."""
        result = builder._join_risk_free_rate(sample_quotes_df, sample_fred_df)

        assert "rf_rate" in result.columns
        # Should use the most recent rate (0.0435 from Jan 14)
        assert (result["rf_rate"] == 0.0435).all()

    def test_join_risk_free_rate_respects_release_time(self, builder):
        """Test that join respects release timestamps."""
        quotes = pd.DataFrame({
            "ts_utc": [datetime(2024, 1, 14, 15, 0, 0)],  # Before Jan 14 release
        })
        fred = pd.DataFrame({
            "obs_date": [datetime(2024, 1, 13), datetime(2024, 1, 14)],
            "value": [0.04, 0.05],
            "release_datetime_utc": [datetime(2024, 1, 13, 21, 0), datetime(2024, 1, 14, 21, 0)],
        })

        result = builder._join_risk_free_rate(quotes, fred)

        # Should use Jan 13 rate (0.04) since Jan 14 wasn't released yet
        assert result["rf_rate"].iloc[0] == 0.04

    def test_join_risk_free_rate_empty_fred(self, builder, sample_quotes_df):
        """Test join with empty FRED data uses default rate."""
        empty_fred = pd.DataFrame()
        result = builder._join_risk_free_rate(sample_quotes_df, empty_fred)

        assert "rf_rate" in result.columns
        assert (result["rf_rate"] == 0.05).all()  # Default fallback

    def test_join_risk_free_rate_without_release_datetime(self, builder, sample_quotes_df):
        """Test join when release_datetime_utc is not available."""
        fred = pd.DataFrame({
            "obs_date": [datetime(2024, 1, 14)],
            "value": [0.045],
        })

        result = builder._join_risk_free_rate(sample_quotes_df, fred)

        assert "rf_rate" in result.columns
        assert (result["rf_rate"] == 0.045).all()


class TestComputeTTE:
    """Tests for _compute_tte method."""

    def test_compute_tte_basic(self, builder):
        """Test basic TTE computation."""
        df = pd.DataFrame({
            "ts_utc": [datetime(2024, 1, 15, 16, 0, 0)],
            "exp_date": [datetime(2024, 1, 19)],
        })

        result = builder._compute_tte(df)

        # 4 days to expiry at close (Jan 19 21:00 UTC - Jan 15 16:00 UTC)
        # = 4 days + 5 hours = 4.208 days = 4.208/365 years
        expected_days = 4 + 5/24  # ~4.208 days
        expected_years = expected_days / 365
        assert abs(result.iloc[0] - expected_years) < 0.001

    def test_compute_tte_same_day_expiry(self, builder):
        """Test TTE for same-day expiry."""
        df = pd.DataFrame({
            "ts_utc": [datetime(2024, 1, 19, 16, 0, 0)],
            "exp_date": [datetime(2024, 1, 19)],
        })

        result = builder._compute_tte(df)

        # 5 hours to close = 5/24/365 years
        expected_years = (5/24) / 365
        assert abs(result.iloc[0] - expected_years) < 0.001

    def test_compute_tte_expired_option(self, builder):
        """Test TTE for expired option returns small positive value."""
        df = pd.DataFrame({
            "ts_utc": [datetime(2024, 1, 20, 16, 0, 0)],  # After expiry
            "exp_date": [datetime(2024, 1, 19)],
        })

        result = builder._compute_tte(df)

        # Should return minimum positive value
        assert result.iloc[0] > 0
        assert result.iloc[0] < 0.001


class TestAssignTenorBins:
    """Tests for _assign_tenor_bins method."""

    def test_assign_tenor_bins_exact_match(self, builder):
        """Test tenor assignment with exact DTE match."""
        df = pd.DataFrame({
            "ts_utc": [datetime(2024, 1, 15)],
            "exp_date": [datetime(2024, 1, 22)],  # 7 days
        })

        result = builder._assign_tenor_bins(df)

        assert result.iloc[0] == 7

    def test_assign_tenor_bins_nearest(self, builder):
        """Test tenor assignment finds nearest bin."""
        df = pd.DataFrame({
            "ts_utc": [datetime(2024, 1, 15)] * 3,
            "exp_date": [
                datetime(2024, 1, 25),  # 10 days -> nearest is 7 or 14
                datetime(2024, 1, 30),  # 15 days -> nearest is 14
                datetime(2024, 2, 20),  # 36 days -> nearest is 30
            ],
        })

        result = builder._assign_tenor_bins(df)

        # 10 days is equidistant from 7 and 14, numpy argmin picks first (7)
        assert result.iloc[0] in [7, 14]
        assert result.iloc[1] == 14
        assert result.iloc[2] == 30

    def test_assign_tenor_bins_handles_nan(self, builder):
        """Test tenor assignment handles NaN dates."""
        df = pd.DataFrame({
            "ts_utc": [datetime(2024, 1, 15)],
            "exp_date": [pd.NaT],
        })

        result = builder._assign_tenor_bins(df)

        assert pd.isna(result.iloc[0])


class TestSelectRepresentatives:
    """Tests for _select_representatives method."""

    def test_select_representatives_one_per_node(self, builder):
        """Test that only one option is selected per node."""
        # Create multiple options in same node
        df = pd.DataFrame({
            "ts_utc": [datetime(2024, 1, 15)] * 3,
            "tenor_days": [30, 30, 30],
            "delta_bucket": ["ATM", "ATM", "ATM"],
            "delta": [0.48, 0.52, 0.55],
            "iv_mid": [0.20, 0.21, 0.22],
            "flags": [0, 0, 0],
        })

        result = builder._select_representatives(df)

        # Should have only 1 representative for ATM/30
        assert len(result) == 1
        # Should select delta closest to 0.5 (which is 0.48 -> |0.48| = 0.48, distance = 0.02)
        # or 0.52 (|0.52| = 0.52, distance = 0.02)
        assert result["delta"].iloc[0] in [0.48, 0.52]

    def test_select_representatives_deprioritizes_flagged(self, builder):
        """Test that options with quality flags are deprioritized."""
        df = pd.DataFrame({
            "ts_utc": [datetime(2024, 1, 15)] * 2,
            "tenor_days": [30, 30],
            "delta_bucket": ["ATM", "ATM"],
            "delta": [0.50, 0.51],  # 0.50 is closer to center
            "iv_mid": [0.20, 0.21],
            "flags": [1, 0],  # First has crossed flag
        })

        result = builder._select_representatives(df)

        # Should select 0.51 despite 0.50 being closer (0.50 has flag)
        assert result["delta"].iloc[0] == 0.51

    def test_select_representatives_filters_invalid(self, builder):
        """Test that invalid rows are filtered out."""
        df = pd.DataFrame({
            "ts_utc": [datetime(2024, 1, 15)] * 3,
            "tenor_days": [30, np.nan, 30],
            "delta_bucket": ["ATM", "ATM", None],
            "delta": [0.50, 0.51, 0.52],
            "iv_mid": [0.20, 0.21, 0.22],
            "flags": [0, 0, 0],
        })

        result = builder._select_representatives(df)

        # Should only have 1 valid row
        assert len(result) == 1
        assert result["delta"].iloc[0] == 0.50


# =============================================================================
# Build Surface Tests
# =============================================================================


class TestBuildSurface:
    """Tests for build_surface method."""

    def test_build_surface_empty_quotes(self, builder, mock_raw_repo):
        """Test build with no quotes returns empty result."""
        mock_raw_repo.read_option_quotes.return_value = pd.DataFrame()

        result = builder.build_surface(
            start=datetime(2024, 1, 15),
            end=datetime(2024, 1, 16),
            version="v1.0",
        )

        assert isinstance(result, BuildResult)
        assert result.row_count == 0
        assert result.quotes_processed == 0

    def test_build_surface_generates_unique_id(self, builder, mock_raw_repo):
        """Test that each build generates a unique ID."""
        mock_raw_repo.read_option_quotes.return_value = pd.DataFrame()

        result1 = builder.build_surface(
            start=datetime(2024, 1, 15),
            end=datetime(2024, 1, 16),
            version="v1.0",
        )
        result2 = builder.build_surface(
            start=datetime(2024, 1, 15),
            end=datetime(2024, 1, 16),
            version="v1.0",
        )

        assert result1.build_run_id != result2.build_run_id
        assert uuid.UUID(result1.build_run_id)  # Valid UUID

    def test_build_surface_full_pipeline(
        self, builder, mock_raw_repo, mock_derived_repo,
        sample_quotes_df, sample_underlying_df, sample_fred_df
    ):
        """Test full surface build pipeline."""
        mock_raw_repo.read_option_quotes.return_value = sample_quotes_df
        mock_raw_repo.read_underlying_bars.return_value = sample_underlying_df
        mock_raw_repo.read_fred_series.return_value = sample_fred_df

        result = builder.build_surface(
            start=datetime(2024, 1, 15),
            end=datetime(2024, 1, 16),
            version="v1.0",
        )

        assert isinstance(result, BuildResult)
        assert result.quotes_processed == 6
        assert result.version == "v1.0"
        # Should have written to derived repo
        mock_derived_repo.write_surface_snapshots.assert_called_once()

    def test_build_surface_tracks_iv_failures(
        self, builder, mock_raw_repo, mock_derived_repo,
        sample_underlying_df, sample_fred_df
    ):
        """Test that IV failures are tracked."""
        # Create quotes with extreme values that will fail IV
        quotes = pd.DataFrame({
            "ts_utc": [datetime(2024, 1, 15, 16, 0, 0)] * 2,
            "option_symbol": ["TEST1", "TEST2"],
            "exp_date": [datetime(2024, 1, 19)] * 2,
            "strike": [1.0, 1000000.0],  # Extreme strikes
            "right": ["C", "P"],
            "bid": [0.01, 0.01],
            "ask": [0.02, 0.02],
        })
        mock_raw_repo.read_option_quotes.return_value = quotes
        mock_raw_repo.read_underlying_bars.return_value = sample_underlying_df
        mock_raw_repo.read_fred_series.return_value = sample_fred_df

        result = builder.build_surface(
            start=datetime(2024, 1, 15),
            end=datetime(2024, 1, 16),
            version="v1.0",
        )

        # Some IV inversions should fail
        assert result.iv_failures >= 0


class TestBuildSurfaceSnapshot:
    """Tests for build_surface_snapshot method (live trading)."""

    def test_build_surface_snapshot_basic(self, builder):
        """Test building surface from pre-loaded quotes."""
        quotes = pd.DataFrame({
            "ts_utc": [datetime(2024, 1, 15, 16, 0, 0)],
            "exp_date": [datetime(2024, 1, 22)],
            "strike": [480.0],
            "right": ["C"],
            "bid": [5.00],
            "ask": [5.20],
            "underlying_price": [479.0],
            "rf_rate": [0.045],
        })

        result = builder.build_surface_snapshot(quotes)

        assert isinstance(result, pd.DataFrame)
        if not result.empty:
            assert "iv_mid" in result.columns
            assert "delta" in result.columns
            assert "tenor_days" in result.columns
            assert "delta_bucket" in result.columns

    def test_build_surface_snapshot_missing_required_columns(self, builder):
        """Test that missing required columns raise error."""
        quotes = pd.DataFrame({
            "ts_utc": [datetime(2024, 1, 15)],
            "exp_date": [datetime(2024, 1, 22)],
            "strike": [480.0],
            "right": ["C"],
            "bid": [5.00],
            "ask": [5.20],
            # Missing underlying_price and rf_rate
        })

        with pytest.raises(ValueError, match="Missing required column"):
            builder.build_surface_snapshot(quotes)

    def test_build_surface_snapshot_empty_quotes(self, builder):
        """Test building surface from empty quotes."""
        result = builder.build_surface_snapshot(pd.DataFrame())

        assert isinstance(result, pd.DataFrame)
        assert result.empty


# =============================================================================
# BuildResult Tests
# =============================================================================


class TestBuildResult:
    """Tests for BuildResult dataclass."""

    def test_build_result_creation(self):
        """Test BuildResult creation with all fields."""
        result = BuildResult(
            build_run_id="test-id",
            version="v1.0",
            row_count=100,
            start=datetime(2024, 1, 15),
            end=datetime(2024, 1, 16),
            quotes_processed=500,
            iv_failures=10,
        )

        assert result.build_run_id == "test-id"
        assert result.version == "v1.0"
        assert result.row_count == 100
        assert result.quotes_processed == 500
        assert result.iv_failures == 10

    def test_build_result_defaults(self):
        """Test BuildResult default values."""
        result = BuildResult(
            build_run_id="test-id",
            version="v1.0",
            row_count=0,
            start=datetime(2024, 1, 15),
            end=datetime(2024, 1, 16),
        )

        assert result.quotes_processed == 0
        assert result.iv_failures == 0

    def test_iv_failure_ratio(self):
        """Test IV failure ratio computation."""
        result = BuildResult(
            build_run_id="test-id",
            version="v1.0",
            row_count=100,
            start=datetime(2024, 1, 15),
            end=datetime(2024, 1, 16),
            quotes_processed=1000,
            iv_failures=250,
        )

        assert result.iv_failure_ratio == pytest.approx(0.25)

    def test_iv_failure_ratio_zero_quotes(self):
        """Test IV failure ratio when no quotes processed."""
        result = BuildResult(
            build_run_id="test-id",
            version="v1.0",
            row_count=0,
            start=datetime(2024, 1, 15),
            end=datetime(2024, 1, 16),
            quotes_processed=0,
            iv_failures=0,
        )

        assert result.iv_failure_ratio == 0.0

    def test_iv_failure_ratio_no_failures(self):
        """Test IV failure ratio with zero failures."""
        result = BuildResult(
            build_run_id="test-id",
            version="v1.0",
            row_count=500,
            start=datetime(2024, 1, 15),
            end=datetime(2024, 1, 16),
            quotes_processed=500,
            iv_failures=0,
        )

        assert result.iv_failure_ratio == 0.0
