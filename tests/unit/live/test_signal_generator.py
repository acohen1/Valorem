"""Unit tests for ModelSignalGenerator and RuleBasedSignalGenerator."""

from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import torch

from src.config.schema import SignalThresholdConfig
from src.strategy.types import SignalType


@pytest.fixture
def sample_surface() -> pd.DataFrame:
    """Create sample surface data with all standard tenors and buckets."""
    records = []
    # Include all standard tenors
    tenors = [7, 14, 30, 45, 60, 90]
    # Include all standard buckets
    buckets = ["P40", "P25", "P10", "ATM", "C10", "C25", "C40"]

    for tenor in tenors:
        for bucket in buckets:
            # Set delta based on bucket
            if bucket == "ATM":
                delta = 0.5
            elif bucket.startswith("P"):
                delta = -0.25
            else:
                delta = 0.25

            records.append({
                "option_symbol": f"SPY_T{tenor}_{bucket}",
                "tenor_days": tenor,
                "delta_bucket": bucket,
                "strike": 450.0,
                "expiry": date.today() + timedelta(days=tenor),
                "right": "C",
                "bid": 5.0,
                "ask": 5.10,
                "iv": 0.20,
                "delta": delta,
                "underlying_price": 450.0,
            })

    return pd.DataFrame(records)


@pytest.fixture
def sample_features() -> pd.DataFrame:
    """Create sample feature data."""
    records = []
    tenors = [30, 60]
    buckets = ["P25", "ATM", "C25"]

    for tenor in tenors:
        for bucket in buckets:
            records.append({
                "tenor_days": tenor,
                "delta_bucket": bucket,
                "iv_change_5d": 0.01,
                "iv_zscore_21d": 1.5,
                "spread_pct": 0.02,
            })

    return pd.DataFrame(records)


@pytest.fixture
def threshold_config() -> SignalThresholdConfig:
    """Create threshold configuration."""
    return SignalThresholdConfig(
        min_edge=0.02,
        min_confidence=0.5,
    )


class TestRuleBasedSignalGenerator:
    """Tests for RuleBasedSignalGenerator."""

    def test_init(self) -> None:
        """Test initialization."""
        from src.live.signal_generator import RuleBasedSignalGenerator

        gen = RuleBasedSignalGenerator(
            iv_zscore_threshold=2.0,
            term_slope_threshold=0.01,
            min_edge=0.02,
        )

        assert gen._iv_zscore_threshold == 2.0
        assert gen._term_slope_threshold == 0.01
        assert gen._min_edge == 0.02

    def test_generate_signals_no_features(
        self, sample_surface: pd.DataFrame
    ) -> None:
        """Test signal generation without features returns empty."""
        from src.live.signal_generator import RuleBasedSignalGenerator

        gen = RuleBasedSignalGenerator()
        signals = gen.generate_signals(sample_surface, features=None)

        assert signals == []

    def test_generate_signals_high_iv_zscore(
        self, sample_surface: pd.DataFrame
    ) -> None:
        """Test signal generation with high IV z-score."""
        from src.live.signal_generator import RuleBasedSignalGenerator

        gen = RuleBasedSignalGenerator(iv_zscore_threshold=1.0)

        # Features with high IV z-score
        features = pd.DataFrame([
            {
                "tenor_days": 30,
                "delta_bucket": "ATM",
                "iv_zscore_21d": 2.5,  # Above threshold
            }
        ])

        signals = gen.generate_signals(sample_surface, features)

        assert len(signals) == 1
        assert signals[0].signal_type == SignalType.ELEVATED_IV
        assert signals[0].edge < 0  # Sell vol

    def test_generate_signals_low_iv_zscore(
        self, sample_surface: pd.DataFrame
    ) -> None:
        """Test signal generation with low IV z-score."""
        from src.live.signal_generator import RuleBasedSignalGenerator

        gen = RuleBasedSignalGenerator(iv_zscore_threshold=1.0)

        # Features with low IV z-score
        features = pd.DataFrame([
            {
                "tenor_days": 30,
                "delta_bucket": "ATM",
                "iv_zscore_21d": -2.5,  # Below negative threshold
            }
        ])

        signals = gen.generate_signals(sample_surface, features)

        assert len(signals) == 1
        assert signals[0].signal_type == SignalType.DIRECTIONAL_VOL
        assert signals[0].edge > 0  # Buy vol

    def test_generate_signals_within_threshold(
        self, sample_surface: pd.DataFrame
    ) -> None:
        """Test no signals when within threshold."""
        from src.live.signal_generator import RuleBasedSignalGenerator

        gen = RuleBasedSignalGenerator(iv_zscore_threshold=2.0)

        # Features within threshold
        features = pd.DataFrame([
            {
                "tenor_days": 30,
                "delta_bucket": "ATM",
                "iv_zscore_21d": 1.0,  # Below threshold
            }
        ])

        signals = gen.generate_signals(sample_surface, features)

        assert len(signals) == 0


class TestModelSignalGeneratorInit:
    """Tests for ModelSignalGenerator initialization."""

    def test_init_with_model(self, threshold_config: SignalThresholdConfig) -> None:
        """Test initialization with model."""
        from src.live.signal_generator import ModelSignalGenerator

        mock_model = MagicMock()
        mock_model.to.return_value = mock_model

        gen = ModelSignalGenerator(
            model=mock_model,
            feature_columns=["iv_change_5d", "spread_pct"],
            threshold_config=threshold_config,
            device="cpu",
        )

        assert gen._feature_columns == ["iv_change_5d", "spread_pct"]
        assert gen._device == "cpu"
        mock_model.eval.assert_called_once()

    def test_from_checkpoint_not_found(
        self, threshold_config: SignalThresholdConfig
    ) -> None:
        """Test loading from non-existent checkpoint."""
        from src.live.signal_generator import ModelSignalGenerator

        with pytest.raises(FileNotFoundError):
            ModelSignalGenerator.from_checkpoint(
                checkpoint_path="/nonexistent/path.pt",
                threshold_config=threshold_config,
            )


class TestModelSignalGeneratorFeatureHistory:
    """Tests for feature history management."""

    def test_update_feature_history(
        self,
        sample_features: pd.DataFrame,
        threshold_config: SignalThresholdConfig,
    ) -> None:
        """Test feature history accumulation."""
        from src.live.signal_generator import ModelSignalGenerator

        mock_model = MagicMock()
        mock_model.to.return_value = mock_model

        gen = ModelSignalGenerator(
            model=mock_model,
            feature_columns=["iv_change_5d"],
            threshold_config=threshold_config,
            lookback_periods=5,
        )

        # Add features
        for _ in range(3):
            gen._update_feature_history(sample_features)

        assert len(gen._feature_history) == 3

    def test_feature_history_truncation(
        self,
        sample_features: pd.DataFrame,
        threshold_config: SignalThresholdConfig,
    ) -> None:
        """Test that history is truncated at lookback limit."""
        from src.live.signal_generator import ModelSignalGenerator

        mock_model = MagicMock()
        mock_model.to.return_value = mock_model

        gen = ModelSignalGenerator(
            model=mock_model,
            feature_columns=["iv_change_5d"],
            threshold_config=threshold_config,
            lookback_periods=5,
        )

        # Add more than lookback periods
        for _ in range(10):
            gen._update_feature_history(sample_features)

        assert len(gen._feature_history) == 5

    def test_reset_history(
        self,
        sample_features: pd.DataFrame,
        threshold_config: SignalThresholdConfig,
    ) -> None:
        """Test history reset."""
        from src.live.signal_generator import ModelSignalGenerator

        mock_model = MagicMock()
        mock_model.to.return_value = mock_model

        gen = ModelSignalGenerator(
            model=mock_model,
            feature_columns=["iv_change_5d"],
            threshold_config=threshold_config,
        )

        gen._update_feature_history(sample_features)
        gen._update_feature_history(sample_features)
        assert len(gen._feature_history) == 2

        gen.reset_history()
        assert len(gen._feature_history) == 0


class TestModelSignalGeneratorSignalClassification:
    """Tests for signal classification logic."""

    def test_classify_atm_short_tenor(
        self,
        sample_surface: pd.DataFrame,
        threshold_config: SignalThresholdConfig,
    ) -> None:
        """Test ATM short tenor classified as term anomaly."""
        from src.live.signal_generator import ModelSignalGenerator

        mock_model = MagicMock()
        mock_model.to.return_value = mock_model

        gen = ModelSignalGenerator(
            model=mock_model,
            feature_columns=[],
            threshold_config=threshold_config,
        )

        signal_type = gen._classify_signal(
            edge=0.05,
            tenor=7,  # Short tenor
            bucket="ATM",
            surface=sample_surface,
        )

        assert signal_type == SignalType.TERM_ANOMALY

    def test_classify_atm_long_tenor(
        self,
        sample_surface: pd.DataFrame,
        threshold_config: SignalThresholdConfig,
    ) -> None:
        """Test ATM long tenor classified as directional vol."""
        from src.live.signal_generator import ModelSignalGenerator

        mock_model = MagicMock()
        mock_model.to.return_value = mock_model

        gen = ModelSignalGenerator(
            model=mock_model,
            feature_columns=[],
            threshold_config=threshold_config,
        )

        signal_type = gen._classify_signal(
            edge=0.05,
            tenor=30,  # Long tenor
            bucket="ATM",
            surface=sample_surface,
        )

        assert signal_type == SignalType.DIRECTIONAL_VOL

    def test_classify_wing_as_skew(
        self,
        sample_surface: pd.DataFrame,
        threshold_config: SignalThresholdConfig,
    ) -> None:
        """Test wing bucket classified as skew anomaly."""
        from src.live.signal_generator import ModelSignalGenerator

        mock_model = MagicMock()
        mock_model.to.return_value = mock_model

        gen = ModelSignalGenerator(
            model=mock_model,
            feature_columns=[],
            threshold_config=threshold_config,
        )

        for bucket in ["P25", "P10", "C10", "C25"]:
            signal_type = gen._classify_signal(
                edge=0.05,
                tenor=30,
                bucket=bucket,
                surface=sample_surface,
            )
            assert signal_type == SignalType.SKEW_ANOMALY

    def test_classify_near_money_elevated_iv(
        self,
        sample_surface: pd.DataFrame,
        threshold_config: SignalThresholdConfig,
    ) -> None:
        """P40/C40 with high IV and low edge → ELEVATED_IV."""
        from src.live.signal_generator import ModelSignalGenerator

        mock_model = MagicMock()
        mock_model.to.return_value = mock_model

        gen = ModelSignalGenerator(
            model=mock_model,
            feature_columns=[],
            threshold_config=threshold_config,
        )

        # Surface fixture has iv=0.20; override to 0.35 for P40/C40 rows
        high_iv_surface = sample_surface.copy()
        mask = high_iv_surface["delta_bucket"].isin(["P40", "C40"])
        high_iv_surface.loc[mask, "iv"] = 0.35

        for bucket in ["P40", "C40"]:
            signal_type = gen._classify_signal(
                edge=0.02,  # Low edge (< 0.03)
                tenor=30,
                bucket=bucket,
                surface=high_iv_surface,
            )
            assert signal_type == SignalType.ELEVATED_IV

    def test_classify_near_money_directional_vol(
        self,
        sample_surface: pd.DataFrame,
        threshold_config: SignalThresholdConfig,
    ) -> None:
        """P40/C40 with low IV or high edge → DIRECTIONAL_VOL."""
        from src.live.signal_generator import ModelSignalGenerator

        mock_model = MagicMock()
        mock_model.to.return_value = mock_model

        gen = ModelSignalGenerator(
            model=mock_model,
            feature_columns=[],
            threshold_config=threshold_config,
        )

        # Low IV (fixture default 0.20) — doesn't meet iv > 0.30 threshold
        for bucket in ["P40", "C40"]:
            signal_type = gen._classify_signal(
                edge=0.02,
                tenor=30,
                bucket=bucket,
                surface=sample_surface,
            )
            assert signal_type == SignalType.DIRECTIONAL_VOL

        # High edge (0.05 >= 0.03) — doesn't meet abs(edge) < 0.03 threshold
        high_iv_surface = sample_surface.copy()
        mask = high_iv_surface["delta_bucket"].isin(["P40", "C40"])
        high_iv_surface.loc[mask, "iv"] = 0.35

        for bucket in ["P40", "C40"]:
            signal_type = gen._classify_signal(
                edge=0.05,
                tenor=30,
                bucket=bucket,
                surface=high_iv_surface,
            )
            assert signal_type == SignalType.DIRECTIONAL_VOL


class TestModelSignalGeneratorNodeMapping:
    """Tests for node index mapping."""

    def test_get_node_mapping(
        self, threshold_config: SignalThresholdConfig
    ) -> None:
        """Test node mapping creation."""
        from src.live.signal_generator import ModelSignalGenerator

        mock_model = MagicMock()
        mock_model.to.return_value = mock_model

        gen = ModelSignalGenerator(
            model=mock_model,
            feature_columns=[],
            threshold_config=threshold_config,
        )

        mapping = gen._get_node_mapping()

        # 6 tenors x 7 buckets = 42 nodes
        assert len(mapping) == 42

        # Check structure
        assert isinstance(mapping[0], tuple)
        assert len(mapping[0]) == 2
        assert mapping[0][0] in [7, 14, 30, 60, 90, 120]

    def test_extract_node_features_applies_checkpoint_normalization(
        self,
        threshold_config: SignalThresholdConfig,
    ) -> None:
        """Per-node checkpoint stats should be applied before inference."""
        from src.live.signal_generator import ModelSignalGenerator

        mock_model = MagicMock()
        mock_model.to.return_value = mock_model

        feature_stats = {
            "iv_change_5d": (
                np.zeros(42, dtype=np.float32),
                np.full(42, 2.0, dtype=np.float32),
            )
        }
        gen = ModelSignalGenerator(
            model=mock_model,
            feature_columns=["iv_change_5d"],
            threshold_config=threshold_config,
            feature_stats=feature_stats,
        )

        tenor, bucket = gen._get_node_mapping()[0]
        features_df = pd.DataFrame(
            [{
                "tenor_days": tenor,
                "delta_bucket": bucket,
                "iv_change_5d": 2.0,
            }]
        )

        node_features, node_mask = gen._extract_node_features(features_df)

        assert node_mask[0]
        assert node_features[0, 0] == pytest.approx(1.0)
        assert np.all(node_features[1:, 0] == 0.0)


class TestModelSignalGeneratorGenerateSignals:
    """Tests for signal generation."""

    def test_generate_signals_insufficient_history(
        self,
        sample_surface: pd.DataFrame,
        threshold_config: SignalThresholdConfig,
    ) -> None:
        """Test returns empty when insufficient history."""
        from src.live.signal_generator import ModelSignalGenerator

        mock_model = MagicMock()
        mock_model.to.return_value = mock_model

        gen = ModelSignalGenerator(
            model=mock_model,
            feature_columns=["iv_change_5d"],
            threshold_config=threshold_config,
            lookback_periods=20,
        )

        # No features added yet
        signals = gen.generate_signals(sample_surface)

        assert signals == []

    def test_generate_signals_filters_by_edge(
        self,
        sample_surface: pd.DataFrame,
        sample_features: pd.DataFrame,
    ) -> None:
        """Test that signals below min_edge are filtered."""
        from src.live.signal_generator import ModelSignalGenerator

        mock_model = MagicMock()
        mock_model.to.return_value = mock_model

        threshold_config = SignalThresholdConfig(
            min_edge=0.10,  # High threshold
            min_confidence=0.1,
        )

        gen = ModelSignalGenerator(
            model=mock_model,
            feature_columns=["iv_change_5d"],
            threshold_config=threshold_config,
            lookback_periods=2,
        )

        # Add enough history
        gen._update_feature_history(sample_features)
        gen._update_feature_history(sample_features)

        # Mock model to return low predictions
        mock_output = torch.zeros(1, 42, 1) + 0.01  # Below threshold
        mock_model.return_value = mock_output

        signals = gen.generate_signals(sample_surface, sample_features)

        # All filtered due to low edge
        assert len(signals) == 0

    def test_run_inference_uses_internal_edges_when_available(
        self,
        sample_surface: pd.DataFrame,
        threshold_config: SignalThresholdConfig,
    ) -> None:
        """When checkpoint carries internal edges, inference should not override them."""
        from src.live.signal_generator import ModelSignalGenerator

        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_model.return_value = torch.zeros(1, 42, 1)
        mock_model._edge_index = torch.zeros((2, 1), dtype=torch.long)

        gen = ModelSignalGenerator(
            model=mock_model,
            feature_columns=["iv_change_5d"],
            threshold_config=threshold_config,
            lookback_periods=2,
        )

        tenor, bucket = gen._get_node_mapping()[0]
        features = pd.DataFrame(
            [{"tenor_days": tenor, "delta_bucket": bucket, "iv_change_5d": 0.1}]
        )
        gen._update_feature_history(features)
        gen._update_feature_history(features)

        _ = gen._run_inference(sample_surface)

        assert mock_model.call_count == 1
        assert len(mock_model.call_args.args) == 1  # X tensor only
        assert "mask" in mock_model.call_args.kwargs
