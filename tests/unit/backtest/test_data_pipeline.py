"""Unit tests for BacktestDataPipeline."""

from datetime import date, datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import torch

from src.backtest.data_pipeline import (
    BacktestData,
    BacktestDataConfig,
    BacktestDataPipeline,
)
from src.config.schema import SignalThresholdConfig
from src.models.dataset import DEFAULT_FEATURE_COLS
from src.pricing import PositionPricer
from src.strategy.types import SignalType

N_FEATURES = len(DEFAULT_FEATURE_COLS)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_surface_df(
    dates: list[date],
    tenors: tuple[int, ...] = (7, 14, 30),
    buckets: tuple[str, ...] = ("ATM", "C25", "P25"),
) -> pd.DataFrame:
    """Build a minimal surface_snapshots DataFrame."""
    records = []
    for d in dates:
        ts = datetime.combine(d, datetime.min.time())
        for tenor in tenors:
            for bucket in buckets:
                records.append(
                    {
                        "ts_utc": ts,
                        "exp_date": d + pd.Timedelta(days=tenor),
                        "tenor_days": tenor,
                        "delta_bucket": bucket,
                        "strike": 450.0,
                        "right": "C",
                        "bid": 5.0,
                        "ask": 5.5,
                        "iv_mid": 0.22,
                        "delta": 0.5,
                        "gamma": 0.02,
                        "vega": 0.3,
                        "theta": -0.02,
                        "option_symbol": f"SPY{tenor}{bucket}",
                    }
                )
    return pd.DataFrame(records)


def _make_bars_df(dates: list[date], close: float = 450.0) -> pd.DataFrame:
    """Build a minimal underlying bars DataFrame."""
    records = []
    for d in dates:
        records.append(
            {
                "ts_utc": datetime.combine(d, datetime.min.time()),
                "close": close,
            }
        )
    return pd.DataFrame(records)


def _make_node_panel(
    dates: list[date],
    tenors: tuple[int, ...] = (7, 14, 30, 60, 90, 120),
    buckets: tuple[str, ...] = ("P10", "P25", "P40", "ATM", "C40", "C25", "C10"),
    num_features: int = N_FEATURES,
) -> pd.DataFrame:
    """Build a minimal node panel DataFrame."""
    records = []
    for d in dates:
        ts = datetime.combine(d, datetime.min.time())
        for tenor in tenors:
            for bucket in buckets:
                row = {
                    "ts_utc": ts,
                    "tenor_days": tenor,
                    "delta_bucket": bucket,
                }
                for i in range(num_features):
                    row[f"feat_{i}"] = np.random.randn()
                records.append(row)
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Tests: Surface loading
# ---------------------------------------------------------------------------


class TestLoadSurfaces:
    """Tests for _load_surfaces()."""

    def test_groups_by_date(self):
        """Verify surface_snapshots are grouped correctly by date."""
        dates = [date(2023, 8, 1), date(2023, 8, 2), date(2023, 8, 3)]
        surface_df = _make_surface_df(dates)

        derived_repo = MagicMock()
        derived_repo.read_surface_snapshots.return_value = surface_df

        pipeline = BacktestDataPipeline(
            raw_repo=MagicMock(),
            derived_repo=derived_repo,
            config=BacktestDataConfig(),
        )

        start_dt = datetime.combine(dates[0], datetime.min.time())
        end_dt = datetime.combine(dates[-1], datetime.min.time())
        surfaces = pipeline._load_surfaces(start_dt, end_dt)

        assert len(surfaces) == 3
        for d in dates:
            assert d in surfaces
            assert isinstance(surfaces[d], pd.DataFrame)
            assert len(surfaces[d]) > 0

    def test_renames_exp_date_to_expiry(self):
        """Verify exp_date column is renamed to expiry."""
        dates = [date(2023, 8, 1)]
        surface_df = _make_surface_df(dates)
        assert "exp_date" in surface_df.columns

        derived_repo = MagicMock()
        derived_repo.read_surface_snapshots.return_value = surface_df

        pipeline = BacktestDataPipeline(
            raw_repo=MagicMock(),
            derived_repo=derived_repo,
            config=BacktestDataConfig(),
        )

        start_dt = datetime.combine(dates[0], datetime.min.time())
        end_dt = datetime.combine(dates[0], datetime.min.time())
        surfaces = pipeline._load_surfaces(start_dt, end_dt)

        surface = surfaces[dates[0]]
        assert "expiry" in surface.columns
        assert "exp_date" not in surface.columns

    def test_adds_iv_alias(self):
        """Verify iv column is created from iv_mid if absent."""
        dates = [date(2023, 8, 1)]
        surface_df = _make_surface_df(dates)
        assert "iv_mid" in surface_df.columns
        assert "iv" not in surface_df.columns

        derived_repo = MagicMock()
        derived_repo.read_surface_snapshots.return_value = surface_df

        pipeline = BacktestDataPipeline(
            raw_repo=MagicMock(),
            derived_repo=derived_repo,
            config=BacktestDataConfig(),
        )

        start_dt = datetime.combine(dates[0], datetime.min.time())
        end_dt = datetime.combine(dates[0], datetime.min.time())
        surfaces = pipeline._load_surfaces(start_dt, end_dt)

        surface = surfaces[dates[0]]
        assert "iv" in surface.columns

    def test_empty_raises_value_error(self):
        """Verify clear error when no surface data found."""
        derived_repo = MagicMock()
        derived_repo.read_surface_snapshots.return_value = pd.DataFrame()

        pipeline = BacktestDataPipeline(
            raw_repo=MagicMock(),
            derived_repo=derived_repo,
            config=BacktestDataConfig(),
        )

        start_dt = datetime(2023, 8, 1)
        end_dt = datetime(2023, 8, 31)
        with pytest.raises(ValueError, match="No surface snapshots found"):
            pipeline._load_surfaces(start_dt, end_dt)


# ---------------------------------------------------------------------------
# Tests: Underlying price merge
# ---------------------------------------------------------------------------


class TestMergeUnderlyingPrice:
    """Tests for _merge_underlying_price()."""

    def test_adds_underlying_price_column(self):
        """Verify underlying_price is merged from bars."""
        dates = [date(2023, 8, 1), date(2023, 8, 2)]
        surface_df = _make_surface_df(dates)

        derived_repo = MagicMock()
        derived_repo.read_surface_snapshots.return_value = surface_df

        raw_repo = MagicMock()
        raw_repo.read_underlying_bars.return_value = _make_bars_df(
            dates, close=452.0
        )

        pipeline = BacktestDataPipeline(
            raw_repo=raw_repo,
            derived_repo=derived_repo,
            config=BacktestDataConfig(),
        )

        # Build surfaces first
        start_dt = datetime.combine(dates[0], datetime.min.time())
        end_dt = datetime.combine(dates[-1], datetime.min.time())
        surfaces = pipeline._load_surfaces(start_dt, end_dt)

        # Merge prices
        pipeline._merge_underlying_price(surfaces, start_dt, end_dt)

        for d in dates:
            assert "underlying_price" in surfaces[d].columns
            assert (surfaces[d]["underlying_price"] == 452.0).all()

    def test_missing_bars_warns(self):
        """Verify warning when no bars found (no crash)."""
        dates = [date(2023, 8, 1)]
        surface_df = _make_surface_df(dates)

        derived_repo = MagicMock()
        derived_repo.read_surface_snapshots.return_value = surface_df

        raw_repo = MagicMock()
        raw_repo.read_underlying_bars.return_value = pd.DataFrame()

        pipeline = BacktestDataPipeline(
            raw_repo=raw_repo,
            derived_repo=derived_repo,
            config=BacktestDataConfig(),
        )

        start_dt = datetime.combine(dates[0], datetime.min.time())
        end_dt = datetime.combine(dates[0], datetime.min.time())
        surfaces = pipeline._load_surfaces(start_dt, end_dt)

        # Should not raise
        pipeline._merge_underlying_price(surfaces, start_dt, end_dt)


# ---------------------------------------------------------------------------
# Tests: Node panel loading
# ---------------------------------------------------------------------------


class TestLoadNodePanel:
    """Tests for _load_node_panel()."""

    def test_empty_raises_value_error(self):
        """Verify clear error when no node panel data found."""
        derived_repo = MagicMock()
        derived_repo.read_node_panel.return_value = pd.DataFrame()

        pipeline = BacktestDataPipeline(
            raw_repo=MagicMock(),
            derived_repo=derived_repo,
            config=BacktestDataConfig(),
        )

        with pytest.raises(ValueError, match="No node panel data found"):
            pipeline._load_node_panel(datetime(2023, 8, 1), datetime(2023, 8, 31))


# ---------------------------------------------------------------------------
# Tests: Model loading
# ---------------------------------------------------------------------------


class TestLoadModel:
    """Tests for _load_model()."""

    def test_checkpoint_not_found_raises(self, tmp_path):
        """Verify FileNotFoundError for missing checkpoint."""
        pipeline = BacktestDataPipeline(
            raw_repo=MagicMock(),
            derived_repo=MagicMock(),
            config=BacktestDataConfig(),
        )

        with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
            pipeline._load_model(tmp_path / "nonexistent.pt")

    def test_loads_self_describing_checkpoint(self, tmp_path):
        """Verify model reconstruction from checkpoint with metadata."""
        from src.models.ensemble import PatchTST_GNN_Ensemble
        from src.models.gnn.model import GNNModelConfig
        from src.models.patchtst.model import PatchTSTModelConfig

        # Create a real model, save its state
        patchtst_cfg = PatchTSTModelConfig(d_model=64, n_layers=1)
        gnn_cfg = GNNModelConfig(hidden_dim=32, n_layers=1)
        model = PatchTST_GNN_Ensemble(
            patchtst_config=patchtst_cfg,
            gnn_config=gnn_cfg,
            input_dim=N_FEATURES,
            output_horizons=3,
        )

        from dataclasses import asdict

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "model_metadata": {
                "patchtst_config": asdict(patchtst_cfg),
                "gnn_config": asdict(gnn_cfg),
                "input_dim": N_FEATURES,
                "output_horizons": 3,
                "feature_columns": [f"feat_{i}" for i in range(N_FEATURES)],
                "feature_stats": {"feat_0": (0.0, 1.0)},
            },
        }

        ckpt_path = tmp_path / "model.pt"
        torch.save(checkpoint, ckpt_path)

        pipeline = BacktestDataPipeline(
            raw_repo=MagicMock(),
            derived_repo=MagicMock(),
            config=BacktestDataConfig(device="cpu"),
        )

        loaded_model, feature_cols, feature_stats, label_stats = pipeline._load_model(ckpt_path)

        assert isinstance(loaded_model, PatchTST_GNN_Ensemble)
        assert len(feature_cols) == N_FEATURES
        assert feature_stats is not None
        assert "feat_0" in feature_stats
        assert label_stats is None  # Not present in this checkpoint

    def test_legacy_checkpoint_without_metadata_raises(self, tmp_path):
        """Verify RuntimeError for checkpoints missing model_metadata."""
        from src.models.ensemble import PatchTST_GNN_Ensemble
        from src.models.gnn.model import GNNModelConfig
        from src.models.patchtst.model import PatchTSTModelConfig

        model = PatchTST_GNN_Ensemble(
            patchtst_config=PatchTSTModelConfig(d_model=64, n_layers=1),
            gnn_config=GNNModelConfig(hidden_dim=32, n_layers=1),
            input_dim=N_FEATURES,
            output_horizons=3,
        )

        checkpoint = {
            "model_state_dict": model.state_dict(),
            # No model_metadata — legacy checkpoint
        }

        ckpt_path = tmp_path / "legacy.pt"
        torch.save(checkpoint, ckpt_path)

        pipeline = BacktestDataPipeline(
            raw_repo=MagicMock(),
            derived_repo=MagicMock(),
            config=BacktestDataConfig(device="cpu"),
        )

        with pytest.raises(RuntimeError, match="lacks model_metadata"):
            pipeline._load_model(ckpt_path)


# ---------------------------------------------------------------------------
# Tests: Batch inference
# ---------------------------------------------------------------------------


class TestRunInference:
    """Tests for _run_inference()."""

    def test_uses_internal_edges_when_model_provides_them(self):
        """Inference should not override checkpoint-internal edge attributes."""
        dates = [date(2023, 8, 1), date(2023, 8, 2)]
        panel_df = _make_node_panel(dates)

        pipeline = BacktestDataPipeline(
            raw_repo=MagicMock(),
            derived_repo=MagicMock(),
            config=BacktestDataConfig(lookback_days=1, batch_size=1, device="cpu"),
        )

        mock_model = MagicMock()
        mock_model._edge_index = torch.zeros((2, 1), dtype=torch.long)
        mock_model.return_value = torch.zeros(1, 42, 1)

        preds = pipeline._run_inference(
            model=mock_model,
            panel_df=panel_df,
            feature_columns=["feat_0"],
            feature_stats=None,
            start_date=dates[0],
            end_date=dates[-1],
        )

        assert isinstance(preds, dict)
        assert mock_model.call_count >= 1
        assert len(mock_model.call_args.args) == 1  # X tensor only
        assert "mask" in mock_model.call_args.kwargs


# ---------------------------------------------------------------------------
# Tests: Signal conversion
# ---------------------------------------------------------------------------


class TestPredictionsToSignals:
    """Tests for _predictions_to_signals()."""

    def _make_pipeline(self, min_edge: float = 0.01, min_confidence: float = 0.5):
        return BacktestDataPipeline(
            raw_repo=MagicMock(),
            derived_repo=MagicMock(),
            config=BacktestDataConfig(
                signal_threshold=SignalThresholdConfig(
                    min_edge=min_edge,
                    min_confidence=min_confidence,
                ),
            ),
        )

    def test_filters_below_edge_threshold(self):
        """Verify predictions below min_edge are filtered out."""
        pipeline = self._make_pipeline(min_edge=0.05)

        predictions = {
            date(2023, 8, 1): {
                (30, "ATM"): 0.01,  # Below threshold → filtered
                (30, "C25"): 0.08,  # Above threshold → kept
            }
        }
        surfaces: dict[date, pd.DataFrame] = {}

        signals = pipeline._predictions_to_signals(predictions, surfaces)

        assert date(2023, 8, 1) in signals
        assert len(signals[date(2023, 8, 1)]) == 1
        assert signals[date(2023, 8, 1)][0].delta_bucket == "C25"

    def test_filters_below_confidence_threshold(self):
        """Verify predictions with low confidence are filtered out."""
        # confidence = min(1.0, abs(edge) / 0.1)
        # edge=0.03 → confidence=0.3 → below min_confidence=0.5
        pipeline = self._make_pipeline(min_edge=0.01, min_confidence=0.5)

        predictions = {
            date(2023, 8, 1): {
                (30, "ATM"): 0.03,  # confidence=0.3 → filtered
            }
        }
        surfaces: dict[date, pd.DataFrame] = {}

        signals = pipeline._predictions_to_signals(predictions, surfaces)

        # No signals above confidence threshold
        assert date(2023, 8, 1) not in signals

    def test_classifies_atm_short_tenor_as_term_anomaly(self):
        """ATM bucket with short tenor (<=14d) → TERM_ANOMALY."""
        pipeline = self._make_pipeline(min_edge=0.01, min_confidence=0.0)

        predictions = {
            date(2023, 8, 1): {
                (7, "ATM"): 0.08,
            }
        }

        signals = pipeline._predictions_to_signals(predictions, {})
        assert signals[date(2023, 8, 1)][0].signal_type == SignalType.TERM_ANOMALY

    def test_classifies_atm_long_tenor_as_directional_vol(self):
        """ATM bucket with longer tenor (>14d) → DIRECTIONAL_VOL."""
        pipeline = self._make_pipeline(min_edge=0.01, min_confidence=0.0)

        predictions = {
            date(2023, 8, 1): {
                (30, "ATM"): 0.08,
            }
        }

        signals = pipeline._predictions_to_signals(predictions, {})
        assert signals[date(2023, 8, 1)][0].signal_type == SignalType.DIRECTIONAL_VOL

    def test_classifies_wing_bucket_as_skew_anomaly(self):
        """Wing buckets (P25, P10, C10, C25) → SKEW_ANOMALY."""
        pipeline = self._make_pipeline(min_edge=0.01, min_confidence=0.0)

        predictions = {
            date(2023, 8, 1): {
                (30, "C25"): 0.08,
                (60, "P10"): 0.06,
            }
        }

        signals = pipeline._predictions_to_signals(predictions, {})
        for sig in signals[date(2023, 8, 1)]:
            assert sig.signal_type == SignalType.SKEW_ANOMALY

    def test_classifies_near_money_elevated_iv(self):
        """P40/C40 with high IV and low edge → ELEVATED_IV."""
        pipeline = self._make_pipeline(min_edge=0.01, min_confidence=0.0)

        d = date(2023, 8, 1)
        # Build surface with high IV for the P40 node
        surface = _make_surface_df([d], tenors=(30,), buckets=("P40",))
        surface["iv_mid"] = 0.35  # Above 0.30 threshold
        surfaces = {d: surface}

        predictions = {
            d: {
                (30, "P40"): 0.02,  # Low edge (< 0.03) → ELEVATED_IV
            }
        }

        signals = pipeline._predictions_to_signals(predictions, surfaces)
        assert len(signals[d]) == 1
        assert signals[d][0].signal_type == SignalType.ELEVATED_IV

    def test_classifies_near_money_directional_vol(self):
        """P40/C40 with low IV or high edge → DIRECTIONAL_VOL."""
        pipeline = self._make_pipeline(min_edge=0.01, min_confidence=0.0)

        # Case 1: Low IV (default 0.22 from helper) — no ELEVATED_IV
        predictions = {
            date(2023, 8, 1): {
                (30, "P40"): 0.02,
            }
        }
        signals = pipeline._predictions_to_signals(predictions, {})
        assert signals[date(2023, 8, 1)][0].signal_type == SignalType.DIRECTIONAL_VOL

        # Case 2: High IV but large edge (>= 0.03) — no ELEVATED_IV
        d = date(2023, 8, 2)
        surface = _make_surface_df([d], tenors=(30,), buckets=("C40",))
        surface["iv_mid"] = 0.35
        surfaces = {d: surface}

        predictions2 = {
            d: {
                (30, "C40"): 0.05,  # High edge (>= 0.03)
            }
        }
        signals2 = pipeline._predictions_to_signals(predictions2, surfaces)
        assert signals2[d][0].signal_type == SignalType.DIRECTIONAL_VOL

    def test_empty_predictions_returns_empty(self):
        """No predictions → no signals."""
        pipeline = self._make_pipeline()

        signals = pipeline._predictions_to_signals({}, {})
        assert signals == {}

    def test_all_filtered_date_omitted(self):
        """Dates where all predictions are filtered should not appear."""
        pipeline = self._make_pipeline(min_edge=0.10)

        predictions = {
            date(2023, 8, 1): {
                (30, "ATM"): 0.02,  # Below threshold
                (60, "C25"): 0.01,  # Below threshold
            }
        }

        signals = pipeline._predictions_to_signals(predictions, {})
        assert len(signals) == 0

    def test_filters_signals_without_surface_data(self):
        """Signals at nodes absent from the surface are filtered out."""
        pipeline = self._make_pipeline(min_edge=0.01, min_confidence=0.0)

        d = date(2023, 8, 1)
        # Surface only has 30d ATM — no 120d data
        surfaces = {d: _make_surface_df([d], tenors=(30,), buckets=("ATM",))}
        predictions = {
            d: {
                (30, "ATM"): 0.08,   # Present in surface → kept
                (120, "ATM"): 0.08,  # Absent from surface → filtered
            }
        }

        signals = pipeline._predictions_to_signals(predictions, surfaces)

        assert len(signals[d]) == 1
        assert signals[d][0].tenor_days == 30

    def test_keeps_all_signals_with_matching_surface(self):
        """All signals at nodes present in the surface pass through."""
        pipeline = self._make_pipeline(min_edge=0.01, min_confidence=0.0)

        d = date(2023, 8, 1)
        surfaces = {d: _make_surface_df([d], tenors=(30, 60), buckets=("ATM", "C25"))}
        predictions = {
            d: {
                (30, "ATM"): 0.08,
                (60, "C25"): 0.06,
            }
        }

        signals = pipeline._predictions_to_signals(predictions, surfaces)

        assert len(signals[d]) == 2
        tenors = {s.tenor_days for s in signals[d]}
        assert tenors == {30, 60}

    def test_no_surface_filtering_when_surface_absent(self):
        """When surface is missing for a date, no filtering is applied."""
        pipeline = self._make_pipeline(min_edge=0.01, min_confidence=0.0)

        d = date(2023, 8, 1)
        # No surface for this date
        surfaces: dict[date, pd.DataFrame] = {}
        predictions = {
            d: {
                (30, "ATM"): 0.08,
                (120, "ATM"): 0.08,
            }
        }

        signals = pipeline._predictions_to_signals(predictions, surfaces)

        # Both signals created — no surface to filter against
        assert len(signals[d]) == 2


# ---------------------------------------------------------------------------
# Tests: BacktestData dataclass
# ---------------------------------------------------------------------------


class TestBacktestData:
    """Tests for BacktestData result structure."""

    def test_structure(self):
        """Verify BacktestData holds expected types."""
        data = BacktestData(
            surfaces={date(2023, 8, 1): pd.DataFrame()},
            signals_by_date={},
            trading_dates=[date(2023, 8, 1)],
            pricer=PositionPricer(),
        )
        assert isinstance(data.surfaces, dict)
        assert isinstance(data.signals_by_date, dict)
        assert isinstance(data.trading_dates, list)
        assert isinstance(data.pricer, PositionPricer)


# ---------------------------------------------------------------------------
# Tests: BacktestDataConfig
# ---------------------------------------------------------------------------


class TestBacktestDataConfig:
    """Tests for BacktestDataConfig defaults and overrides."""

    def test_defaults(self):
        cfg = BacktestDataConfig()
        assert cfg.checkpoint_path == "artifacts/checkpoints/best_model.pt"
        assert cfg.feature_version == "v1.0"
        assert cfg.surface_version == "v1.0"
        assert cfg.underlying_symbol == "SPY"
        assert cfg.lookback_days == 22
        assert cfg.batch_size == 32
        assert cfg.device == "cpu"
        assert isinstance(cfg.signal_threshold, SignalThresholdConfig)

    def test_overrides(self):
        cfg = BacktestDataConfig(
            checkpoint_path="/tmp/model.pt",
            device="cuda",
            lookback_days=30,
        )
        assert cfg.checkpoint_path == "/tmp/model.pt"
        assert cfg.device == "cuda"
        assert cfg.lookback_days == 30
