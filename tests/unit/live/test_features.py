"""Unit tests for feature providers."""

from datetime import date, timedelta
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from src.live.features import (
    DatabaseFeatureProvider,
    MockFeatureProvider,
    RollingFeatureProvider,
)


@pytest.fixture
def sample_surface() -> pd.DataFrame:
    """Create sample surface data with all standard tenors and buckets."""
    records = []
    tenors = [7, 14, 30, 45, 60, 90]
    buckets = ["P40", "P25", "P10", "ATM", "C10", "C25", "C40"]

    for tenor in tenors:
        for bucket in buckets:
            # Set IV based on bucket for skew testing
            if bucket.startswith("P"):
                iv = 0.22  # Higher IV for puts (skew)
            elif bucket.startswith("C"):
                iv = 0.18  # Lower IV for calls
            else:
                iv = 0.20  # ATM

            # Add term structure
            iv += 0.001 * (90 - tenor)  # Higher IV for shorter tenors

            records.append({
                "option_symbol": f"SPY_T{tenor}_{bucket}",
                "tenor_days": tenor,
                "delta_bucket": bucket,
                "strike": 450.0,
                "expiry": date.today() + timedelta(days=tenor),
                "right": "C" if bucket.startswith("C") or bucket == "ATM" else "P",
                "bid": 5.0,
                "ask": 5.10,
                "iv": iv,
                "iv_mid": iv,
                "delta": 0.5 if bucket == "ATM" else 0.25,
                "underlying_price": 450.0,
            })

    return pd.DataFrame(records)


class TestRollingFeatureProviderInit:
    """Tests for RollingFeatureProvider initialization."""

    def test_init_defaults(self) -> None:
        """Test initialization with default parameters."""
        provider = RollingFeatureProvider()

        assert provider._lookback_days == 22
        assert provider._min_history == 5
        assert provider.history_length == 0
        assert not provider.is_ready

    def test_init_custom_params(self) -> None:
        """Test initialization with custom parameters."""
        provider = RollingFeatureProvider(
            lookback_days=10,
            min_history=3,
        )

        assert provider._lookback_days == 10
        assert provider._min_history == 3


class TestRollingFeatureProviderUpdate:
    """Tests for RollingFeatureProvider.update()."""

    def test_update_adds_to_history(self, sample_surface: pd.DataFrame) -> None:
        """Test that update adds surface to history."""
        provider = RollingFeatureProvider(lookback_days=20, min_history=5)

        assert provider.history_length == 0

        provider.update(sample_surface)
        assert provider.history_length == 1

        provider.update(sample_surface)
        assert provider.history_length == 2

    def test_update_truncates_at_lookback(self, sample_surface: pd.DataFrame) -> None:
        """Test that history is truncated at lookback limit."""
        provider = RollingFeatureProvider(lookback_days=5, min_history=2)

        for _ in range(10):
            provider.update(sample_surface)

        assert provider.history_length == 5

    def test_update_empty_surface_skipped(self) -> None:
        """Test that empty surfaces are skipped."""
        provider = RollingFeatureProvider()

        provider.update(pd.DataFrame())
        assert provider.history_length == 0

    def test_is_ready_after_min_history(self, sample_surface: pd.DataFrame) -> None:
        """Test is_ready flag after reaching min_history."""
        provider = RollingFeatureProvider(lookback_days=20, min_history=3)

        provider.update(sample_surface)
        assert not provider.is_ready

        provider.update(sample_surface)
        assert not provider.is_ready

        provider.update(sample_surface)
        assert provider.is_ready


class TestRollingFeatureProviderGetFeatures:
    """Tests for RollingFeatureProvider.get_features()."""

    def test_get_features_insufficient_history(
        self, sample_surface: pd.DataFrame
    ) -> None:
        """Test returns empty DataFrame when insufficient history."""
        provider = RollingFeatureProvider(lookback_days=20, min_history=5)

        # Add only 2 surfaces (need 5)
        provider.update(sample_surface)
        provider.update(sample_surface)

        features = provider.get_features(sample_surface)
        assert features.empty

    def test_get_features_returns_dataframe(
        self, sample_surface: pd.DataFrame
    ) -> None:
        """Test returns DataFrame with features when ready."""
        provider = RollingFeatureProvider(lookback_days=20, min_history=3)

        # Add enough history
        for _ in range(5):
            provider.update(sample_surface)

        features = provider.get_features(sample_surface)

        assert not features.empty
        assert "tenor_days" in features.columns
        assert "delta_bucket" in features.columns
        assert "iv_zscore_21d" in features.columns
        assert "iv_change_5d" in features.columns
        assert "spread_pct" in features.columns

    def test_get_features_computes_zscore(
        self, sample_surface: pd.DataFrame
    ) -> None:
        """Test z-score computation."""
        provider = RollingFeatureProvider(lookback_days=20, min_history=3)

        # Add history with same surface (z-score should be ~0)
        for _ in range(5):
            provider.update(sample_surface)

        features = provider.get_features(sample_surface)

        # With identical surfaces, z-score should be 0
        zscores = features["iv_zscore_21d"]
        assert all(abs(z) < 0.01 for z in zscores if pd.notna(z))

    def test_get_features_computes_term_slope(
        self, sample_surface: pd.DataFrame
    ) -> None:
        """Test term slope computation."""
        provider = RollingFeatureProvider(lookback_days=20, min_history=3)

        for _ in range(5):
            provider.update(sample_surface)

        features = provider.get_features(sample_surface)

        # Get ATM term slope
        atm_features = features[features["delta_bucket"] == "ATM"]
        if not atm_features.empty:
            term_slope = atm_features.iloc[0]["term_slope"]
            # Our test surface has higher IV for shorter tenors, so slope is negative
            assert term_slope < 0

    def test_get_features_computes_skew(
        self, sample_surface: pd.DataFrame
    ) -> None:
        """Test skew_slope computation (linear regression across delta buckets)."""
        provider = RollingFeatureProvider(lookback_days=22, min_history=3)

        for _ in range(5):
            provider.update(sample_surface)

        features = provider.get_features(sample_surface)

        # skew_slope is the linear regression slope of IV vs delta.
        # Puts (negative delta) have higher IV (0.22), calls (positive delta)
        # have lower IV (0.18), so slope (dIV/dDelta) should be negative.
        atm_features = features[features["delta_bucket"] == "ATM"]
        if not atm_features.empty:
            skew = atm_features.iloc[0]["skew_slope"]
            assert skew < 0

    def test_get_features_spread_pct(
        self, sample_surface: pd.DataFrame
    ) -> None:
        """Test spread percentage is extracted correctly."""
        provider = RollingFeatureProvider(lookback_days=20, min_history=3)

        for _ in range(5):
            provider.update(sample_surface)

        features = provider.get_features(sample_surface)

        # Spread = (5.10 - 5.00) / 5.05 ≈ 0.0198
        spread_pcts = features["spread_pct"]
        assert all(0.01 < s < 0.03 for s in spread_pcts if pd.notna(s))


class TestRollingFeatureProviderReset:
    """Tests for RollingFeatureProvider.reset()."""

    def test_reset_clears_history(self, sample_surface: pd.DataFrame) -> None:
        """Test that reset clears all history."""
        provider = RollingFeatureProvider(lookback_days=20, min_history=3)

        for _ in range(5):
            provider.update(sample_surface)

        assert provider.history_length == 5
        assert provider.is_ready

        provider.reset()

        assert provider.history_length == 0
        assert not provider.is_ready


class TestRollingFeatureProviderZScoreComputation:
    """Tests for z-score computation with varying data."""

    def test_zscore_high_value(self) -> None:
        """Test z-score computation for elevated value."""
        provider = RollingFeatureProvider(lookback_days=20, min_history=3)

        # Create surfaces with varying IV (needed for non-zero std)
        base_ivs = [0.19, 0.20, 0.21, 0.20, 0.19]
        for iv in base_ivs:
            records = [{
                "tenor_days": 30,
                "delta_bucket": "ATM",
                "iv": iv,
                "iv_mid": iv,
                "bid": 5.0,
                "ask": 5.10,
            }]
            provider.update(pd.DataFrame(records))

        # Current surface with elevated IV (clearly above mean)
        current = pd.DataFrame([{
            "tenor_days": 30,
            "delta_bucket": "ATM",
            "iv": 0.28,  # Much higher than mean of ~0.198
            "iv_mid": 0.28,
            "bid": 5.0,
            "ask": 5.10,
        }])

        features = provider.get_features(current)

        # Z-score should be positive (current > mean)
        if not features.empty:
            zscore = features.iloc[0]["iv_zscore_21d"]
            assert zscore > 0

    def test_zscore_low_value(self) -> None:
        """Test z-score computation for depressed value."""
        provider = RollingFeatureProvider(lookback_days=20, min_history=3)

        # Create surfaces with varying IV around 0.25
        base_ivs = [0.24, 0.25, 0.26, 0.25, 0.24]
        for iv in base_ivs:
            records = [{
                "tenor_days": 30,
                "delta_bucket": "ATM",
                "iv": iv,
                "iv_mid": iv,
                "bid": 5.0,
                "ask": 5.10,
            }]
            provider.update(pd.DataFrame(records))

        # Current surface with lower IV (clearly below mean)
        current = pd.DataFrame([{
            "tenor_days": 30,
            "delta_bucket": "ATM",
            "iv": 0.18,  # Much lower than mean of ~0.248
            "iv_mid": 0.18,
            "bid": 5.0,
            "ask": 5.10,
        }])

        features = provider.get_features(current)

        # Z-score should be negative (current < mean)
        if not features.empty:
            zscore = features.iloc[0]["iv_zscore_21d"]
            assert zscore < 0

    def test_zscore_zero_std_returns_zero(self) -> None:
        """Test that identical values (zero std) returns zscore of 0."""
        provider = RollingFeatureProvider(lookback_days=20, min_history=3)

        # Create surfaces with identical IV
        for _ in range(5):
            records = [{
                "tenor_days": 30,
                "delta_bucket": "ATM",
                "iv": 0.20,
                "iv_mid": 0.20,
                "bid": 5.0,
                "ask": 5.10,
            }]
            provider.update(pd.DataFrame(records))

        # Current surface with different IV
        current = pd.DataFrame([{
            "tenor_days": 30,
            "delta_bucket": "ATM",
            "iv": 0.25,
            "iv_mid": 0.25,
            "bid": 5.0,
            "ask": 5.10,
        }])

        features = provider.get_features(current)

        # Z-score should be 0 when std is 0 (avoid division by zero)
        if not features.empty:
            zscore = features.iloc[0]["iv_zscore_21d"]
            assert zscore == 0.0


class TestDatabaseFeatureProvider:
    """Tests for DatabaseFeatureProvider."""

    def test_init(self) -> None:
        """Test initialization."""
        mock_repo = MagicMock()
        provider = DatabaseFeatureProvider(mock_repo, lookback_days=2)

        assert provider._derived_repo == mock_repo
        assert provider._lookback_days == 2

    def test_update_is_noop(self, sample_surface: pd.DataFrame) -> None:
        """Test that update is a no-op."""
        mock_repo = MagicMock()
        provider = DatabaseFeatureProvider(mock_repo)

        # Should not raise
        provider.update(sample_surface)

    def test_get_features_queries_repo(self, sample_surface: pd.DataFrame) -> None:
        """Test that get_features queries the repository."""
        from datetime import datetime, timezone

        mock_repo = MagicMock()
        mock_repo.read_node_panel.return_value = pd.DataFrame([{
            "ts_utc": datetime.now(timezone.utc),
            "tenor_days": 30,
            "delta_bucket": "ATM",
            "iv_zscore_21d": 1.5,
            "iv_change_5d": 0.02,
        }])

        provider = DatabaseFeatureProvider(mock_repo, feature_version="v1.0")
        features = provider.get_features(sample_surface)

        mock_repo.read_node_panel.assert_called_once()
        assert not features.empty
        assert "iv_zscore_21d" in features.columns  # Raw columns, no rename

    def test_get_features_empty_result(self, sample_surface: pd.DataFrame) -> None:
        """Test handling of empty database result."""
        mock_repo = MagicMock()
        mock_repo.read_node_panel.return_value = pd.DataFrame()

        provider = DatabaseFeatureProvider(mock_repo)
        features = provider.get_features(sample_surface)

        assert features.empty

    def test_get_features_handles_error(self, sample_surface: pd.DataFrame) -> None:
        """Test graceful handling of database error."""
        mock_repo = MagicMock()
        mock_repo.read_node_panel.side_effect = Exception("DB error")

        provider = DatabaseFeatureProvider(mock_repo)
        features = provider.get_features(sample_surface)

        assert features.empty

    def test_reset_is_noop(self) -> None:
        """Test that reset is a no-op."""
        mock_repo = MagicMock()
        provider = DatabaseFeatureProvider(mock_repo)

        # Should not raise
        provider.reset()


class TestMockFeatureProvider:
    """Tests for MockFeatureProvider."""

    def test_init_defaults(self) -> None:
        """Test initialization with defaults."""
        provider = MockFeatureProvider()

        assert provider._default_zscore == 0.0
        assert provider._default_change == 0.0
        assert provider._ready_after == 0

    def test_get_features_returns_mock_values(
        self, sample_surface: pd.DataFrame
    ) -> None:
        """Test that get_features returns configured values."""
        provider = MockFeatureProvider(
            default_zscore=2.5,
            default_change=0.05,
        )

        features = provider.get_features(sample_surface)

        assert not features.empty
        assert all(features["iv_zscore_5d"] == 2.5)
        assert all(features["iv_change_5d"] == 0.05)

    def test_get_features_respects_ready_after(
        self, sample_surface: pd.DataFrame
    ) -> None:
        """Test that get_features respects ready_after parameter."""
        provider = MockFeatureProvider(ready_after=3)

        # Not ready yet
        features = provider.get_features(sample_surface)
        assert features.empty

        provider.update(sample_surface)
        features = provider.get_features(sample_surface)
        assert features.empty

        provider.update(sample_surface)
        features = provider.get_features(sample_surface)
        assert features.empty

        provider.update(sample_surface)
        features = provider.get_features(sample_surface)
        assert not features.empty

    def test_reset_clears_counter(self, sample_surface: pd.DataFrame) -> None:
        """Test that reset clears the update counter."""
        provider = MockFeatureProvider(ready_after=2)

        provider.update(sample_surface)
        provider.update(sample_surface)
        assert not provider.get_features(sample_surface).empty

        provider.reset()
        assert provider.get_features(sample_surface).empty


class TestFeatureProviderIntegration:
    """Integration tests for feature providers with TradingLoop."""

    def test_rolling_provider_with_signal_generator(
        self, sample_surface: pd.DataFrame
    ) -> None:
        """Test RollingFeatureProvider produces features usable by signal generator."""
        from src.live.signal_generator import RuleBasedSignalGenerator

        provider = RollingFeatureProvider(lookback_days=5, min_history=3)
        signal_gen = RuleBasedSignalGenerator(iv_zscore_threshold=1.0)

        # Warm up feature history
        for _ in range(3):
            provider.update(sample_surface)

        # Get features
        features = provider.get_features(sample_surface)
        assert not features.empty
        assert "iv_zscore_21d" in features.columns

        # Generate signals (may or may not produce signals depending on thresholds)
        signals = signal_gen.generate_signals(sample_surface, features)
        # Just verify it doesn't crash
        assert isinstance(signals, list)
