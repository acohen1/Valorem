"""Unit tests for quality filtering."""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from src.surface.quality.filters import QualityConfig, QualityFilter


class TestQualityConfig:
    """Test QualityConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = QualityConfig()
        assert config.allow_crossed_quotes is False
        assert config.max_spread_pct == 0.50
        assert config.min_volume == 10
        assert config.min_open_interest == 100
        assert config.eod_max_staleness_days == 1

    def test_custom_values(self):
        """Test custom configuration values."""
        config = QualityConfig(
            allow_crossed_quotes=True,
            max_spread_pct=0.25,
            min_volume=50,
            min_open_interest=500,
            eod_max_staleness_days=3,
        )
        assert config.allow_crossed_quotes is True
        assert config.max_spread_pct == 0.25
        assert config.min_volume == 50
        assert config.min_open_interest == 500
        assert config.eod_max_staleness_days == 3

    def test_disabled_volume_check(self):
        """Test disabling volume check."""
        config = QualityConfig(min_volume=None)
        assert config.min_volume is None

    def test_disabled_oi_check(self):
        """Test disabling open interest check."""
        config = QualityConfig(min_open_interest=None)
        assert config.min_open_interest is None


class TestFlagConstants:
    """Test quality flag constants."""

    def test_flag_values(self):
        """Test that flag values are correct powers of 2."""
        assert QualityFilter.FLAG_CROSSED == 1
        assert QualityFilter.FLAG_STALE == 2
        assert QualityFilter.FLAG_WIDE_SPREAD == 4
        assert QualityFilter.FLAG_LOW_VOLUME == 8
        assert QualityFilter.FLAG_LOW_OI == 16

    def test_flags_are_unique_bits(self):
        """Test that flags don't overlap."""
        flags = [
            QualityFilter.FLAG_CROSSED,
            QualityFilter.FLAG_STALE,
            QualityFilter.FLAG_WIDE_SPREAD,
            QualityFilter.FLAG_LOW_VOLUME,
            QualityFilter.FLAG_LOW_OI,
        ]
        # Sum of unique powers of 2 should equal bitwise OR
        assert sum(flags) == (flags[0] | flags[1] | flags[2] | flags[3] | flags[4])


class TestCrossedQuotes:
    """Test crossed quote detection."""

    @pytest.fixture
    def filter(self):
        """Create quality filter."""
        config = QualityConfig(allow_crossed_quotes=False)
        return QualityFilter(config)

    def test_detect_crossed_quote(self, filter):
        """Test detection of crossed quotes (ask < bid)."""
        df = pd.DataFrame({
            "bid": [10.0, 10.0],
            "ask": [9.0, 11.0],  # First is crossed
        })
        flags = filter.compute_flags(df)

        assert flags.iloc[0] == QualityFilter.FLAG_CROSSED
        assert flags.iloc[1] == 0  # Not crossed

    def test_allow_crossed_quotes(self):
        """Test allowing crossed quotes."""
        config = QualityConfig(allow_crossed_quotes=True)
        filter = QualityFilter(config)

        df = pd.DataFrame({
            "bid": [10.0],
            "ask": [9.0],  # Crossed
        })
        flags = filter.compute_flags(df)

        assert flags.iloc[0] == 0  # No flag set

    def test_no_crossed_check_without_columns(self, filter):
        """Test graceful handling when bid/ask columns missing."""
        df = pd.DataFrame({"price": [10.0]})
        flags = filter.compute_flags(df)
        assert flags.iloc[0] == 0


class TestWideSpread:
    """Test wide spread detection."""

    @pytest.fixture
    def filter(self):
        """Create quality filter with 20% max spread."""
        config = QualityConfig(max_spread_pct=0.20)
        return QualityFilter(config)

    def test_detect_wide_spread_from_spread_pct(self, filter):
        """Test detection using spread_pct column."""
        df = pd.DataFrame({
            "spread_pct": [0.10, 0.25, 0.50],  # 10%, 25%, 50%
        })
        flags = filter.compute_flags(df)

        assert flags.iloc[0] == 0  # 10% < 20%
        assert flags.iloc[1] == QualityFilter.FLAG_WIDE_SPREAD  # 25% > 20%
        assert flags.iloc[2] == QualityFilter.FLAG_WIDE_SPREAD  # 50% > 20%

    def test_compute_spread_from_bid_ask(self, filter):
        """Test spread computation from bid/ask."""
        df = pd.DataFrame({
            "bid": [9.0, 8.0],
            "ask": [11.0, 12.0],  # 20% and 40% spreads
        })
        flags = filter.compute_flags(df)

        # (11-9)/10 = 20% = boundary, (12-8)/10 = 40% > 20%
        assert flags.iloc[0] == 0  # At boundary
        assert flags.iloc[1] == QualityFilter.FLAG_WIDE_SPREAD


class TestStaleness:
    """Test staleness detection."""

    @pytest.fixture
    def filter(self):
        """Create quality filter with 1 day max staleness."""
        config = QualityConfig(eod_max_staleness_days=1)
        return QualityFilter(config)

    def test_detect_stale_quote(self, filter):
        """Test detection of stale quotes."""
        now = datetime(2024, 1, 15)
        df = pd.DataFrame({
            "ts_utc": [
                datetime(2024, 1, 15),  # Same day
                datetime(2024, 1, 14),  # 1 day old
                datetime(2024, 1, 13),  # 2 days old
            ],
        })
        flags = filter.compute_flags(df, reference_time=now)

        assert flags.iloc[0] == 0  # Same day
        assert flags.iloc[1] == 0  # 1 day old (at boundary)
        assert flags.iloc[2] == QualityFilter.FLAG_STALE  # 2 days old

    def test_staleness_without_reference_time(self, filter):
        """Test staleness using max timestamp as reference."""
        df = pd.DataFrame({
            "ts_utc": [
                datetime(2024, 1, 15),  # Most recent
                datetime(2024, 1, 13),  # 2 days older
            ],
        })
        flags = filter.compute_flags(df)  # No reference_time

        assert flags.iloc[0] == 0  # Most recent
        assert flags.iloc[1] == QualityFilter.FLAG_STALE  # 2 days old


class TestVolumeCheck:
    """Test volume threshold checking."""

    @pytest.fixture
    def filter(self):
        """Create quality filter with 100 min volume."""
        config = QualityConfig(min_volume=100)
        return QualityFilter(config)

    def test_detect_low_volume(self, filter):
        """Test detection of low volume."""
        df = pd.DataFrame({
            "volume": [50, 100, 200],
        })
        flags = filter.compute_flags(df)

        assert flags.iloc[0] == QualityFilter.FLAG_LOW_VOLUME  # 50 < 100
        assert flags.iloc[1] == 0  # 100 >= 100
        assert flags.iloc[2] == 0  # 200 >= 100

    def test_handle_nan_volume(self, filter):
        """Test NaN volume treated as 0."""
        df = pd.DataFrame({
            "volume": [np.nan, 100],
        })
        flags = filter.compute_flags(df)

        assert flags.iloc[0] == QualityFilter.FLAG_LOW_VOLUME  # NaN → 0 < 100
        assert flags.iloc[1] == 0

    def test_disabled_volume_check(self):
        """Test disabled volume check."""
        config = QualityConfig(min_volume=None)
        filter = QualityFilter(config)

        df = pd.DataFrame({
            "volume": [0],  # Would fail if check was enabled
        })
        flags = filter.compute_flags(df)

        assert flags.iloc[0] == 0


class TestOpenInterestCheck:
    """Test open interest threshold checking."""

    @pytest.fixture
    def filter(self):
        """Create quality filter with 500 min OI."""
        config = QualityConfig(min_open_interest=500)
        return QualityFilter(config)

    def test_detect_low_oi(self, filter):
        """Test detection of low open interest."""
        df = pd.DataFrame({
            "open_interest": [100, 500, 1000],
        })
        flags = filter.compute_flags(df)

        assert flags.iloc[0] == QualityFilter.FLAG_LOW_OI  # 100 < 500
        assert flags.iloc[1] == 0  # 500 >= 500
        assert flags.iloc[2] == 0  # 1000 >= 500

    def test_handle_nan_oi(self, filter):
        """Test NaN open interest treated as 0."""
        df = pd.DataFrame({
            "open_interest": [np.nan, 500],
        })
        flags = filter.compute_flags(df)

        assert flags.iloc[0] == QualityFilter.FLAG_LOW_OI
        assert flags.iloc[1] == 0

    def test_disabled_oi_check(self):
        """Test disabled open interest check."""
        config = QualityConfig(min_open_interest=None)
        filter = QualityFilter(config)

        df = pd.DataFrame({
            "open_interest": [0],
        })
        flags = filter.compute_flags(df)

        assert flags.iloc[0] == 0


class TestCombinedFlags:
    """Test multiple flags combined."""

    @pytest.fixture
    def filter(self):
        """Create quality filter with all checks enabled."""
        config = QualityConfig(
            allow_crossed_quotes=False,
            max_spread_pct=0.20,
            min_volume=100,
            min_open_interest=500,
        )
        return QualityFilter(config)

    def test_multiple_flags(self, filter):
        """Test option with multiple quality issues."""
        df = pd.DataFrame({
            "bid": [10.0],
            "ask": [9.0],  # Crossed
            "spread_pct": [0.50],  # Wide
            "volume": [50],  # Low volume
            "open_interest": [100],  # Low OI
        })
        flags = filter.compute_flags(df)

        expected = (
            QualityFilter.FLAG_CROSSED |
            QualityFilter.FLAG_WIDE_SPREAD |
            QualityFilter.FLAG_LOW_VOLUME |
            QualityFilter.FLAG_LOW_OI
        )
        assert flags.iloc[0] == expected

    def test_bitfield_encoding(self, filter):
        """Test that flags are properly bitfield encoded."""
        df = pd.DataFrame({
            "bid": [10.0],
            "ask": [9.0],  # Crossed (bit 0)
            "spread_pct": [0.50],  # Wide (bit 2)
        })
        flags = filter.compute_flags(df)

        # Should be 0b00101 = 5
        assert flags.iloc[0] == 5
        assert (flags.iloc[0] & QualityFilter.FLAG_CROSSED) != 0
        assert (flags.iloc[0] & QualityFilter.FLAG_WIDE_SPREAD) != 0
        assert (flags.iloc[0] & QualityFilter.FLAG_STALE) == 0


class TestHelperMethods:
    """Test helper methods."""

    @pytest.fixture
    def filter(self):
        """Create quality filter."""
        return QualityFilter(QualityConfig())

    def test_is_good_quality(self, filter):
        """Test is_good_quality method."""
        flags = pd.Series([0, 1, 4, 5])
        good = filter.is_good_quality(flags)

        assert good.iloc[0] == True  # No flags
        assert good.iloc[1] == False  # Has CROSSED
        assert good.iloc[2] == False  # Has WIDE_SPREAD
        assert good.iloc[3] == False  # Has multiple

    def test_has_flag(self, filter):
        """Test has_flag method."""
        flags = pd.Series([0, 1, 4, 5])

        crossed = filter.has_flag(flags, QualityFilter.FLAG_CROSSED)
        assert crossed.iloc[0] == False
        assert crossed.iloc[1] == True
        assert crossed.iloc[2] == False
        assert crossed.iloc[3] == True  # 5 = 1 + 4

        wide = filter.has_flag(flags, QualityFilter.FLAG_WIDE_SPREAD)
        assert wide.iloc[0] == False
        assert wide.iloc[1] == False
        assert wide.iloc[2] == True
        assert wide.iloc[3] == True

    def test_describe_flags(self, filter):
        """Test describe_flags method."""
        assert filter.describe_flags(0) == []
        assert filter.describe_flags(1) == ["CROSSED"]
        assert filter.describe_flags(4) == ["WIDE_SPREAD"]
        assert filter.describe_flags(5) == ["CROSSED", "WIDE_SPREAD"]
        assert filter.describe_flags(31) == [
            "CROSSED", "STALE", "WIDE_SPREAD", "LOW_VOLUME", "LOW_OI"
        ]


class TestEdgeCases:
    """Test edge cases."""

    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        filter = QualityFilter(QualityConfig())
        df = pd.DataFrame()
        flags = filter.compute_flags(df)
        assert len(flags) == 0

    def test_preserves_index(self):
        """Test that flags preserve DataFrame index."""
        filter = QualityFilter(QualityConfig())
        df = pd.DataFrame({"bid": [10.0, 11.0]}, index=["opt1", "opt2"])
        flags = filter.compute_flags(df)
        assert list(flags.index) == ["opt1", "opt2"]

    def test_missing_optional_columns(self):
        """Test graceful handling of missing optional columns."""
        filter = QualityFilter(QualityConfig())
        df = pd.DataFrame({"price": [10.0]})  # No quality columns
        flags = filter.compute_flags(df)
        assert flags.iloc[0] == 0  # No flags without data


class TestIntegration:
    """Integration tests for bucket assignment + quality filtering."""

    def test_full_pipeline(self):
        """Test full pipeline: deltas → buckets → quality flags."""
        from src.surface.buckets.assign import DeltaBucketAssigner

        # Create bucket assigner
        bucket_config = {
            "ATM": [-0.55, -0.45, 0.45, 0.55],
            "C25": [0.15, 0.35],
        }
        assigner = DeltaBucketAssigner(bucket_config)

        # Create quality filter
        quality_config = QualityConfig(max_spread_pct=0.30, min_volume=50)
        filter = QualityFilter(quality_config)

        # Create sample data
        df = pd.DataFrame({
            "delta": [0.50, 0.25, 0.52, 0.20],
            "bid": [5.0, 2.0, 4.9, 1.5],
            "ask": [5.5, 3.0, 5.1, 2.5],
            "volume": [100, 30, 200, 60],
        })

        # Assign buckets
        buckets = assigner.assign(df["delta"])
        df["bucket"] = buckets

        # Compute quality flags
        flags = filter.compute_flags(df)
        df["flags"] = flags

        # Verify results
        assert df.loc[0, "bucket"] == "ATM"
        assert df.loc[1, "bucket"] == "C25"
        assert df.loc[2, "bucket"] == "ATM"
        assert df.loc[3, "bucket"] == "C25"

        # Option 1 (ATM): 10% spread, 100 volume → good
        assert df.loc[0, "flags"] == 0

        # Option 2 (C25): 40% spread, 30 volume → wide spread + low volume
        assert (df.loc[1, "flags"] & QualityFilter.FLAG_WIDE_SPREAD) != 0
        assert (df.loc[1, "flags"] & QualityFilter.FLAG_LOW_VOLUME) != 0
