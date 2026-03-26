"""Unit tests for release-time alignment."""

from datetime import timedelta

import pandas as pd
import pytest

from src.features.macro.alignment import AlignmentConfig, ReleaseTimeAligner


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_fred_df():
    """Create sample FRED series data."""
    # Use daily dates for predictable testing
    obs_dates = pd.date_range("2024-01-01", periods=10, freq="D")
    return pd.DataFrame({
        "obs_date": obs_dates,
        "value": [4.5, 4.55, 4.6, 4.58, 4.62, 4.65, 4.63, 4.68, 4.7, 4.72],
        "release_datetime_utc": obs_dates + pd.Timedelta(days=7),  # Released 7 days later
    })


@pytest.fixture
def aligner():
    """Create aligner with default config."""
    return ReleaseTimeAligner()


@pytest.fixture
def strict_aligner():
    """Create aligner with strict mode."""
    config = AlignmentConfig(mode="strict")
    return ReleaseTimeAligner(config)


# ============================================================================
# Initialization Tests
# ============================================================================


class TestAlignerInit:
    """Test aligner initialization."""

    def test_default_config(self):
        """Test default configuration values."""
        aligner = ReleaseTimeAligner()
        assert aligner._config.mode == "conservative"
        assert aligner._config.conservative_delay_days == 2
        assert aligner._config.time_column == "ts_utc"

    def test_custom_config(self):
        """Test custom configuration."""
        config = AlignmentConfig(
            mode="strict",
            conservative_delay_days=2,
            time_column="effective_time",
        )
        aligner = ReleaseTimeAligner(config)
        assert aligner._config.mode == "strict"
        assert aligner._config.conservative_delay_days == 2
        assert aligner._config.time_column == "effective_time"


# ============================================================================
# Conservative Mode Tests
# ============================================================================


class TestConservativeMode:
    """Test conservative alignment mode."""

    def test_align_shifts_by_delay(self, aligner, sample_fred_df):
        """Test that align shifts dates by conservative delay."""
        result = aligner.align(sample_fred_df)

        # Default delay is 2 days
        expected_first = pd.Timestamp("2024-01-01") + timedelta(days=2)
        assert result["ts_utc"].iloc[0] == expected_first

    def test_align_preserves_values(self, aligner, sample_fred_df):
        """Test that values are preserved after alignment."""
        result = aligner.align(sample_fred_df)

        # Values should be unchanged
        assert result["value"].tolist() == sample_fred_df["value"].tolist()

    def test_align_custom_delay(self, sample_fred_df):
        """Test alignment with custom delay."""
        config = AlignmentConfig(mode="conservative", conservative_delay_days=5)
        aligner = ReleaseTimeAligner(config)
        result = aligner.align(sample_fred_df)

        expected_first = pd.Timestamp("2024-01-01") + timedelta(days=5)
        assert result["ts_utc"].iloc[0] == expected_first

    def test_align_keeps_all_rows(self, aligner, sample_fred_df):
        """Test that conservative mode keeps all rows."""
        result = aligner.align(sample_fred_df)
        assert len(result) == len(sample_fred_df)

    def test_align_empty_df(self, aligner):
        """Test alignment with empty DataFrame."""
        empty_df = pd.DataFrame(columns=["obs_date", "value"])
        result = aligner.align(empty_df)
        assert len(result) == 0


# ============================================================================
# Strict Mode Tests
# ============================================================================


class TestStrictMode:
    """Test strict alignment mode."""

    def test_align_uses_release_time(self, strict_aligner, sample_fred_df):
        """Test that strict mode uses release timestamps."""
        result = strict_aligner.align(sample_fred_df)

        # Should use release_datetime_utc
        assert result["ts_utc"].iloc[0] == sample_fred_df["release_datetime_utc"].iloc[0]

    def test_align_filters_missing_release(self, strict_aligner):
        """Test that rows without release timestamp are filtered."""
        df = pd.DataFrame({
            "obs_date": pd.date_range("2024-01-01", periods=5, freq="D"),
            "value": [1, 2, 3, 4, 5],
            "release_datetime_utc": [
                pd.Timestamp("2024-01-02"),
                None,
                pd.Timestamp("2024-01-04"),
                pd.NaT,
                pd.Timestamp("2024-01-06"),
            ],
        })
        result = strict_aligner.align(df)

        # Only 3 rows have valid release timestamps
        assert len(result) == 3

    def test_strict_mode_requires_release_column(self, strict_aligner):
        """Test that strict mode raises without release column."""
        df = pd.DataFrame({
            "obs_date": pd.date_range("2024-01-01", periods=3),
            "value": [1, 2, 3],
        })
        with pytest.raises(ValueError, match="release_datetime_utc"):
            strict_aligner.align(df)


# ============================================================================
# Validation Tests
# ============================================================================


class TestAlignValidation:
    """Test alignment validation."""

    def test_align_missing_columns_raises(self, aligner):
        """Test that missing columns raise ValueError."""
        df = pd.DataFrame({"value": [1, 2, 3]})
        with pytest.raises(ValueError, match="Missing required columns"):
            aligner.align(df)

    def test_align_sorts_by_time(self, aligner):
        """Test that result is sorted by timestamp."""
        df = pd.DataFrame({
            "obs_date": pd.to_datetime(["2024-01-10", "2024-01-01", "2024-01-05"]),
            "value": [3, 1, 2],
        })
        result = aligner.align(df)

        # Should be sorted by ts_utc
        assert result["ts_utc"].is_monotonic_increasing


# ============================================================================
# Utility Method Tests
# ============================================================================


class TestGetEffectiveTime:
    """Test get_effective_time utility method."""

    def test_conservative_mode_effective_time(self, aligner):
        """Test effective time in conservative mode."""
        obs_date = pd.Timestamp("2024-01-15")
        effective = aligner.get_effective_time(obs_date)

        expected = obs_date + timedelta(days=2)
        assert effective == expected

    def test_strict_mode_effective_time(self, strict_aligner):
        """Test effective time in strict mode."""
        obs_date = pd.Timestamp("2024-01-15")
        release = pd.Timestamp("2024-01-20")
        effective = strict_aligner.get_effective_time(obs_date, release)

        assert effective == release

    def test_strict_mode_requires_release(self, strict_aligner):
        """Test that strict mode requires release datetime."""
        obs_date = pd.Timestamp("2024-01-15")
        with pytest.raises(ValueError, match="release_datetime"):
            strict_aligner.get_effective_time(obs_date)


# ============================================================================
# Leakage Validation Tests
# ============================================================================


class TestLeakageValidation:
    """Test leakage validation."""

    def test_validate_no_leakage_empty(self, aligner):
        """Test validation with empty data."""
        df = pd.DataFrame(columns=["ts_utc", "value"])
        reference = pd.Series(dtype="datetime64[ns]")
        assert aligner.validate_no_leakage(df, reference) is True

    def test_validate_no_leakage_basic(self, aligner, sample_fred_df):
        """Test basic validation passes."""
        aligned = aligner.align(sample_fred_df)
        reference = pd.Series([pd.Timestamp("2024-01-15")])
        assert aligner.validate_no_leakage(aligned, reference) is True
