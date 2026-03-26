"""Signal generator protocol and implementations.

This module defines the SignalGenerator protocol and provides implementations
that use trained models (PatchTST, GNN, Ensemble) or simple rules to generate
trading signals from volatility surface data.

Classes:
    SignalGenerator: Protocol defining the signal generation interface
    SignalGeneratorBase: Abstract base class for all signal generators
    ModelSignalGenerator: ML-based signal generator using trained models
    RuleBasedSignalGenerator: Simple rule-based generator for testing
"""

import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable

import numpy as np
import pandas as pd
import torch

from src.config.schema import SignalThresholdConfig
from src.models.dataset import DEFAULT_FEATURE_COLS
from src.models.ensemble import PatchTST_GNN_Ensemble
from src.models.gnn.model import GNNModelConfig
from src.models.graph import build_surface_graph
from src.models.patchtst.model import PatchTSTModelConfig
from src.strategy.types import Signal, SignalType

logger = logging.getLogger(__name__)

# Wing buckets for signal classification.
# P40/C40 are excluded — they are near-the-money and should fall through
# to the ELEVATED_IV check (high IV + low edge → Iron Condor).
WING_BUCKETS = ("P25", "P10", "C10", "C25")


@runtime_checkable
class SignalGenerator(Protocol):
    """Protocol for generating trading signals from surface data."""

    def generate_signals(
        self,
        surface: pd.DataFrame,
        features: Optional[pd.DataFrame] = None,
    ) -> list[Signal]:
        """Generate trading signals from current surface and features.

        Args:
            surface: Current volatility surface
            features: Optional historical features for model inference

        Returns:
            List of trading signals
        """
        ...


class SignalGeneratorBase(ABC):
    """Abstract base class for all signal generators.

    All signal generators must implement the generate_signals method.
    This ABC ensures consistent interface across ML-based, rule-based,
    and mock signal generators.

    Subclasses:
        - ModelSignalGenerator: Uses trained ML models
        - RuleBasedSignalGenerator: Uses simple rules for testing
    """

    @abstractmethod
    def generate_signals(
        self,
        surface: pd.DataFrame,
        features: Optional[pd.DataFrame] = None,
    ) -> list[Signal]:
        """Generate trading signals from surface and features.

        Args:
            surface: Current volatility surface DataFrame with columns:
                option_symbol, tenor_days, delta_bucket, strike, expiry,
                right, bid, ask, delta, gamma, vega, theta, iv
            features: Optional feature DataFrame with computed features
                like IV z-scores, term structure slopes, etc.

        Returns:
            List of Signal objects meeting generation criteria
        """
        ...


class ModelSignalGenerator(SignalGeneratorBase):
    """Signal generator using trained ML ensemble model.

    Uses the PatchTST + GNN ensemble to predict volatility changes,
    then converts predictions into trading signals based on
    configured thresholds.

    Example:
        generator = ModelSignalGenerator.from_checkpoint(
            checkpoint_path="artifacts/models/best_model.pt",
            threshold_config=SignalThresholdConfig(),
        )
        signals = generator.generate_signals(surface_df, features_df)
    """

    def __init__(
        self,
        model: torch.nn.Module,
        feature_columns: list[str],
        threshold_config: SignalThresholdConfig,
        device: str = "cpu",
        lookback_periods: int = 22,
        feature_stats: Optional[dict] = None,
        label_stats: Optional[dict[str, tuple[float, float]]] = None,
    ):
        """Initialize model signal generator.

        Args:
            model: Trained PyTorch model (Ensemble, PatchTST, or GNN)
            feature_columns: List of feature column names expected by model
            threshold_config: Signal threshold configuration
            device: Device for inference ("cpu", "cuda", "mps")
            lookback_periods: Number of historical periods for input
            feature_stats: Optional normalization stats from training checkpoint
            label_stats: Optional label normalization stats from training checkpoint.
                Dict mapping label column name to (mean, std). Used to denormalize
                predictions back to raw DHR scale at inference time.
        """
        self._model = model.to(device)
        self._model.eval()
        self._feature_columns = feature_columns
        self._threshold_config = threshold_config
        self._device = device
        self._lookback_periods = lookback_periods
        self._feature_stats = feature_stats or {}
        self._label_stats = label_stats or {}
        self._model_has_internal_edges = (
            getattr(self._model, "_edge_index", None) is not None
        )

        # Build static graph for GNN
        self._graph = build_surface_graph()

        # Feature history buffer (for lookback)
        self._feature_history: list[pd.DataFrame] = []

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        threshold_config: SignalThresholdConfig | None = None,
        device: str | None = None,
    ) -> "ModelSignalGenerator":
        """Load model from checkpoint file.

        Args:
            checkpoint_path: Path to model checkpoint
            threshold_config: Signal thresholds (uses defaults if None)
            device: Device for inference (auto-detect if None)

        Returns:
            Initialized ModelSignalGenerator

        Raises:
            FileNotFoundError: If checkpoint not found
            RuntimeError: If checkpoint is invalid
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        logger.info(f"Loading model from {checkpoint_path} on {device}")

        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )

        state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict"))
        if state_dict is None:
            raise RuntimeError("Checkpoint does not contain model state dict")

        # Preferred path: self-describing checkpoints from scripts/train_model.py
        model_metadata = checkpoint.get("model_metadata")
        if model_metadata is not None:
            ablation_variant = model_metadata.get("ablation_variant", "ensemble")
            if ablation_variant != "ensemble":
                raise RuntimeError(
                    "ModelSignalGenerator currently supports ensemble checkpoints "
                    f"only, got ablation_variant='{ablation_variant}'."
                )

            patchtst_cfg = model_metadata.get("patchtst_config")
            gnn_cfg = model_metadata.get("gnn_config")
            if patchtst_cfg is None or gnn_cfg is None:
                raise RuntimeError(
                    "Checkpoint model_metadata is missing patchtst_config/gnn_config"
                )

            patchtst_config = PatchTSTModelConfig(**patchtst_cfg)
            gnn_config = GNNModelConfig(**gnn_cfg)
            feature_columns = list(
                model_metadata.get(
                    "feature_columns",
                    checkpoint.get("feature_columns", list(DEFAULT_FEATURE_COLS)),
                )
            )
            input_dim = int(model_metadata.get("input_dim", len(feature_columns)))
            output_horizons = int(model_metadata.get("output_horizons", 3))
            lookback_periods = int(model_metadata.get("lookback_periods", 22))
            feature_stats = model_metadata.get(
                "feature_stats",
                checkpoint.get("feature_stats"),
            )
            label_stats = model_metadata.get("label_stats")
        else:
            # Legacy fallback path
            model_config = checkpoint.get("model_config")
            if model_config is None:
                raise RuntimeError(
                    "Checkpoint missing model_metadata and model_config; "
                    "cannot reconstruct model architecture."
                )

            model_type = model_config.get("type", "ensemble")
            if model_type != "ensemble":
                raise RuntimeError(f"Unsupported model type: {model_type}")

            patchtst_config = PatchTSTModelConfig(**model_config.get("patchtst", {}))
            gnn_config = GNNModelConfig(**model_config.get("gnn", {}))
            feature_columns = list(
                checkpoint.get("feature_columns", list(DEFAULT_FEATURE_COLS))
            )
            input_dim = int(model_config.get("input_dim", len(feature_columns)))
            output_horizons = int(model_config.get("output_horizons", 3))
            lookback_periods = int(model_config.get("lookback_periods", 22))
            feature_stats = checkpoint.get("feature_stats")
            label_stats = None  # Legacy checkpoints don't have label normalization

        volume_feature_idx = None
        if gnn_config.use_dynamic_volume_edges:
            if "log_volume" in feature_columns:
                volume_feature_idx = feature_columns.index("log_volume")
            else:
                logger.warning(
                    "Checkpoint enables dynamic volume edges but feature_columns "
                    "does not include 'log_volume'; dynamic edge augmentation "
                    "will be disabled for live inference."
                )

        has_internal_edges = "_edge_index" in state_dict or "_edge_attr" in state_dict
        model_graph = build_surface_graph() if has_internal_edges else None

        model = PatchTST_GNN_Ensemble(
            patchtst_config=patchtst_config,
            gnn_config=gnn_config,
            input_dim=input_dim,
            output_horizons=output_horizons,
            graph=model_graph,
            volume_feature_idx=volume_feature_idx,
        )

        model.load_state_dict(state_dict)

        threshold_config = threshold_config or SignalThresholdConfig()

        return cls(
            model=model,
            feature_columns=feature_columns,
            threshold_config=threshold_config,
            device=device,
            lookback_periods=lookback_periods,
            feature_stats=feature_stats,
            label_stats=label_stats,
        )

    def generate_signals(
        self,
        surface: pd.DataFrame,
        features: Optional[pd.DataFrame] = None,
    ) -> list[Signal]:
        """Generate trading signals from surface and features.

        Args:
            surface: Current volatility surface DataFrame
            features: Optional feature DataFrame. If None, uses cached history.

        Returns:
            List of Signal objects for nodes with predicted edge
        """
        if features is not None:
            self._update_feature_history(features)

        # Check if we have enough history
        if len(self._feature_history) < self._lookback_periods:
            logger.debug(
                f"Insufficient history: {len(self._feature_history)}/{self._lookback_periods}"
            )
            return []

        # Prepare model input
        try:
            predictions = self._run_inference(surface)
        except Exception as e:
            logger.error(f"Model inference failed: {e}")
            return []

        # Convert predictions to signals
        signals = self._predictions_to_signals(predictions, surface)

        logger.info(f"Generated {len(signals)} signals from model predictions")
        return signals

    def _update_feature_history(self, features: pd.DataFrame) -> None:
        """Add features to rolling history buffer.

        Args:
            features: Feature DataFrame for current period
        """
        self._feature_history.append(features.copy())

        # Keep only lookback periods
        if len(self._feature_history) > self._lookback_periods:
            self._feature_history = self._feature_history[-self._lookback_periods:]

    def _run_inference(self, surface: pd.DataFrame) -> dict[tuple[int, str], float]:
        """Run model inference to get predictions.

        Args:
            surface: Current surface DataFrame

        Returns:
            Dictionary mapping (tenor_days, delta_bucket) to predicted edge
        """
        # Stack feature history into tensor
        # Shape: (lookback, nodes, features)
        feature_tensors = []
        mask_tensors = []

        for feat_df in self._feature_history:
            # Create node features array (nodes x features)
            node_features, node_mask = self._extract_node_features(feat_df)
            feature_tensors.append(node_features)
            mask_tensors.append(node_mask)

        # Stack: (lookback, nodes, features)
        X = np.stack(feature_tensors, axis=0)
        mask = mask_tensors[-1]

        # Convert to tensor and add batch dimension: (1, lookback, nodes, features)
        X_tensor = torch.tensor(X, dtype=torch.float32, device=self._device).unsqueeze(0)
        mask_tensor = torch.tensor(mask, dtype=torch.bool, device=self._device).unsqueeze(0)

        # Get graph edge structure when model does not carry internal edges.
        edge_index = None
        edge_attr = None
        if not self._model_has_internal_edges:
            edge_index = self._graph.edge_index.to(dtype=torch.long, device=self._device)
            edge_attr = (
                self._graph.edge_attr.to(dtype=torch.float32, device=self._device)
                if self._graph.edge_attr is not None
                else None
            )

        # Run inference
        with torch.no_grad():
            # Model returns (batch, nodes, horizons) or (batch, nodes)
            if self._model_has_internal_edges:
                output = self._model(X_tensor, mask=mask_tensor)
            else:
                output = self._model(X_tensor, edge_index, edge_attr, mask_tensor)

            # Take first horizon if multi-horizon
            if output.dim() == 3:
                output = output[:, :, 0]

            predictions = output.squeeze(0).cpu().numpy()  # (nodes,)

        # Denormalize predictions if label stats are available.
        # Model outputs are in z-score space; convert back to raw DHR scale.
        if self._label_stats:
            # First horizon (index 0) is used for signal generation
            first_label = list(self._label_stats.keys())[0] if self._label_stats else None
            if first_label and first_label in self._label_stats:
                mean, std = self._label_stats[first_label]
                predictions = predictions * std + mean

        # Map predictions to (tenor, bucket) tuples
        node_mapping = self._get_node_mapping()
        result = {}

        for node_idx, pred in enumerate(predictions):
            if node_idx < len(node_mapping) and mask[node_idx]:
                if not np.isfinite(pred):
                    continue
                tenor, bucket = node_mapping[node_idx]
                result[(tenor, bucket)] = float(pred)

        return result

    def _extract_node_features(
        self,
        features_df: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract node features from DataFrame.

        Args:
            features_df: Feature DataFrame with node data

        Returns:
            Tuple of:
                - features array, shape (nodes, features)
                - node validity mask, shape (nodes,)
        """
        # Use graph node ordering so feature tensor indexing matches training.
        node_mapping = self._get_node_mapping()
        n_nodes = len(node_mapping)
        n_features = len(self._feature_columns)
        result = np.full((n_nodes, n_features), np.nan, dtype=np.float32)
        node_mask = np.zeros(n_nodes, dtype=bool)

        for node_idx, (tenor, bucket) in enumerate(node_mapping):
            # Find matching row in features
            mask = (
                (features_df.get("tenor_days", pd.Series()) == tenor) &
                (features_df.get("delta_bucket", pd.Series()) == bucket)
            )

            if mask.any():
                row = features_df[mask].iloc[0]
                row_is_masked = (
                    bool(row["is_masked"])
                    if "is_masked" in row and pd.notna(row["is_masked"])
                    else False
                )
                if row_is_masked:
                    continue

                node_mask[node_idx] = True
                for feat_idx, col in enumerate(self._feature_columns):
                    if col in row:
                        val = row[col]
                        if pd.notna(val) and np.isfinite(val):
                            result[node_idx, feat_idx] = np.float32(val)

        if self._feature_stats:
            result = self._normalize_features(result)
        np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0, copy=False)

        return result, node_mask

    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Apply checkpoint feature normalization stats to node feature matrix."""
        n_nodes = features.shape[0]
        for feat_idx, col in enumerate(self._feature_columns):
            stats = self._feature_stats.get(col)
            if not isinstance(stats, (list, tuple)) or len(stats) != 2:
                continue
            mean_raw, std_raw = stats

            mean_arr = np.asarray(mean_raw, dtype=np.float32)
            std_arr = np.asarray(std_raw, dtype=np.float32)

            if mean_arr.ndim == 0:
                mean_scalar = float(mean_arr)
                std_scalar = float(std_arr.reshape(-1)[0]) if std_arr.size else 1.0
                if not np.isfinite(std_scalar) or abs(std_scalar) < 1e-10:
                    std_scalar = 1.0
                features[:, feat_idx] = (features[:, feat_idx] - mean_scalar) / std_scalar
                continue

            if mean_arr.size != n_nodes:
                continue

            if std_arr.ndim == 0:
                std_arr = np.full(n_nodes, float(std_arr), dtype=np.float32)
            elif std_arr.size != n_nodes:
                continue

            std_safe = np.where(
                np.isfinite(std_arr) & (np.abs(std_arr) >= 1e-10),
                std_arr,
                1.0,
            )
            features[:, feat_idx] = (features[:, feat_idx] - mean_arr) / std_safe

        return features

    def _get_node_mapping(self) -> list[tuple[int, str]]:
        """Get ordered list of (tenor, bucket) for each node index.

        Returns:
            List of (tenor_days, delta_bucket) tuples
        """
        return [
            self._graph.index_mapping[i]
            for i in range(self._graph.num_nodes)
        ]

    def _predictions_to_signals(
        self,
        predictions: dict[tuple[int, str], float],
        surface: pd.DataFrame,
    ) -> list[Signal]:
        """Convert model predictions to trading signals.

        Args:
            predictions: Dictionary of (tenor, bucket) -> predicted edge
            surface: Current surface for context

        Returns:
            List of Signal objects meeting threshold criteria
        """
        signals = []
        now = datetime.now(timezone.utc)
        if surface.empty:
            return signals

        available_nodes = set(zip(surface["tenor_days"], surface["delta_bucket"]))

        for (tenor, bucket), edge in predictions.items():
            if not np.isfinite(edge):
                continue

            if (tenor, bucket) not in available_nodes:
                continue

            # Apply thresholds
            if abs(edge) < self._threshold_config.min_edge:
                continue

            # Determine signal type based on prediction characteristics
            signal_type = self._classify_signal(edge, tenor, bucket, surface)

            # Compute confidence from prediction magnitude
            # Higher magnitude -> higher confidence (capped at 1.0)
            confidence = min(1.0, abs(edge) / 0.1)

            if confidence < self._threshold_config.min_confidence:
                continue

            signals.append(
                Signal(
                    signal_type=signal_type,
                    edge=edge,
                    confidence=confidence,
                    tenor_days=tenor,
                    delta_bucket=bucket,
                    timestamp=now,
                )
            )

        return signals

    def _classify_signal(
        self,
        edge: float,
        tenor: int,
        bucket: str,
        surface: pd.DataFrame,
    ) -> SignalType:
        """Classify signal type based on prediction and context.

        Args:
            edge: Predicted edge value
            tenor: Tenor in days
            bucket: Delta bucket
            surface: Current surface for context

        Returns:
            Appropriate SignalType
        """
        # Get surface context for this node
        node_data = surface[
            (surface["tenor_days"] == tenor) &
            (surface["delta_bucket"] == bucket)
        ]

        if node_data.empty:
            # Default to directional vol
            return SignalType.DIRECTIONAL_VOL

        if "iv_mid" in node_data.columns:
            iv = node_data["iv_mid"].iloc[0]
        elif "iv" in node_data.columns:
            iv = node_data["iv"].iloc[0]
        else:
            iv = 0.20

        # Classify based on edge direction and magnitude
        if bucket == "ATM":
            # ATM predictions -> directional vol or term structure
            if tenor <= 14:
                return SignalType.TERM_ANOMALY
            else:
                return SignalType.DIRECTIONAL_VOL
        elif bucket in WING_BUCKETS:
            # Wing predictions -> skew anomaly
            return SignalType.SKEW_ANOMALY
        else:
            # High IV with low confidence -> iron condor
            if iv > 0.30 and abs(edge) < 0.03:
                return SignalType.ELEVATED_IV
            else:
                return SignalType.DIRECTIONAL_VOL

    def reset_history(self) -> None:
        """Clear feature history buffer."""
        self._feature_history = []


class RuleBasedSignalGenerator(SignalGeneratorBase):
    """Simple rule-based signal generator for testing without ML model.

    Generates signals based on IV percentile ranks and term structure.
    Useful for testing the pipeline before training a model.
    """

    def __init__(
        self,
        iv_zscore_threshold: float = 2.0,
        term_slope_threshold: float = 0.01,
        min_edge: float = 0.02,
    ):
        """Initialize rule-based generator.

        Args:
            iv_zscore_threshold: Z-score for IV mean reversion signal
            term_slope_threshold: Term slope for calendar signal
            min_edge: Minimum edge for signal generation
        """
        self._iv_zscore_threshold = iv_zscore_threshold
        self._term_slope_threshold = term_slope_threshold
        self._min_edge = min_edge

    def generate_signals(
        self,
        surface: pd.DataFrame,
        features: Optional[pd.DataFrame] = None,
    ) -> list[Signal]:
        """Generate signals using simple rules.

        Args:
            surface: Current volatility surface
            features: Optional feature DataFrame with IV z-scores, etc.

        Returns:
            List of Signal objects
        """
        signals = []
        now = datetime.now(timezone.utc)

        if features is None:
            return signals

        # Look for IV mean reversion opportunities
        for _, row in features.iterrows():
            iv_zscore = row.get("iv_zscore_21d", row.get("iv_zscore_20d", 0))
            tenor = row.get("tenor_days", 30)
            bucket = row.get("delta_bucket", "ATM")

            # High IV z-score -> sell vol (negative edge)
            if iv_zscore > self._iv_zscore_threshold:
                signals.append(
                    Signal(
                        signal_type=SignalType.ELEVATED_IV,
                        edge=-0.05,  # Sell vol
                        confidence=min(0.9, iv_zscore / 3.0),
                        tenor_days=int(tenor),
                        delta_bucket=str(bucket),
                        timestamp=now,
                    )
                )

            # Low IV z-score -> buy vol (positive edge)
            elif iv_zscore < -self._iv_zscore_threshold:
                signals.append(
                    Signal(
                        signal_type=SignalType.DIRECTIONAL_VOL,
                        edge=0.05,  # Buy vol
                        confidence=min(0.9, abs(iv_zscore) / 3.0),
                        tenor_days=int(tenor),
                        delta_bucket=str(bucket),
                        timestamp=now,
                    )
                )

        return signals
