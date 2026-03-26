"""Training loop for PatchTST+GNN ensemble model.

This module provides the Trainer class for training volatility surface models
with early stopping, checkpointing, learning rate scheduling, and metrics tracking.
"""

import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from loguru import logger
from torch.utils.data import DataLoader
from torch_geometric.data import Data

from src.models.eval.metrics import MetricsCalculator
from src.models.results import TrainingResults
from src.models.train.loss import MaskedLoss, VolumeWeightedMaskedLoss, build_loss


@dataclass
class TrainerConfig:
    """Configuration for model training.

    Attributes:
        device: Device to train on ('cuda', 'mps', 'cpu', or 'auto')
        batch_size: Training batch size
        max_epochs: Maximum number of training epochs
        learning_rate: Initial learning rate for optimizer
        weight_decay: L2 regularization strength
        early_stopping_patience: Epochs to wait before stopping if no improvement
        early_stopping_metric: Metric to monitor for early stopping. Recommended:
            'temporal_ic' (default) — per-node Spearman IC across timestamps,
            measures whether the model predicts *when* DHR is large vs small.
            'xs_demeaned_ic' — cross-sectionally demeaned IC, measures relative
            node ranking within each timestamp.
            Avoid pooled metrics like 'ic' for DHR targets — these are dominated
            by the trivial gamma·ΔS² cross-sectional relationship and do not
            reflect genuine predictive skill.
        early_stopping_mode: 'max' for IC metrics, 'min' for loss
        gradient_clip_val: Maximum gradient norm for clipping (0 to disable)
        checkpoint_dir: Directory to save checkpoints
        loss_type: Loss function type ('mse', 'huber', 'quantile', 'mae')
        huber_delta: Delta parameter for Huber loss
        quantile: Quantile parameter for quantile loss
        scheduler_type: LR scheduler type ('cosine', 'step', 'plateau', 'none')
        scheduler_warmup_epochs: Number of warmup epochs for cosine scheduler
        scheduler_step_size: Step size for step scheduler
        scheduler_gamma: Decay factor for step scheduler
        plateau_factor: Factor by which LR is reduced on plateau (default 0.5)
        plateau_patience: Epochs with no improvement before reducing LR (default 5)
        log_every_n_steps: Log training progress every N batches
        use_amp: Enable automatic mixed precision (CUDA only, no-op on MPS/CPU)
    """

    device: str = "auto"
    batch_size: int = 32
    max_epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    early_stopping_patience: int = 10
    early_stopping_metric: str = "temporal_ic"
    early_stopping_mode: Literal["min", "max"] = "max"
    gradient_clip_val: float = 1.0
    checkpoint_dir: str = "checkpoints"
    loss_type: str = "huber"
    huber_delta: float = 1.0
    quantile: float = 0.5
    scheduler_type: str = "cosine"
    scheduler_warmup_epochs: int = 5
    scheduler_step_size: int = 10
    scheduler_gamma: float = 0.1
    plateau_factor: float = 0.5
    plateau_patience: int = 5
    log_every_n_steps: int = 10
    use_amp: bool = True
    volume_weight_enabled: bool = False
    volume_weight_feature_idx: int | None = None


@dataclass
class TrainResult:
    """Results from training run.

    Attributes:
        best_val_metric: Best validation metric achieved
        best_epoch: Epoch that achieved best metric
        epochs_trained: Total number of epochs trained
        train_losses: Training loss per epoch
        val_losses: Validation loss per epoch
        val_metrics: Validation metrics per epoch
        best_checkpoint_path: Path to best checkpoint file
    """

    best_val_metric: float
    best_epoch: int
    epochs_trained: int
    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    val_metrics: list[dict[str, float]] = field(default_factory=list)
    best_checkpoint_path: str = ""


class Trainer:
    """Training loop with early stopping, checkpointing, and metrics.

    Handles the complete training workflow including:
    - Training and validation epochs
    - Learning rate scheduling with warmup
    - Early stopping based on validation metrics
    - Model checkpointing
    - Gradient clipping
    - Metrics computation (Temporal IC, XS-Demeaned IC, RMSE, MAE)

    Example:
        >>> config = TrainerConfig(max_epochs=100, learning_rate=1e-3)
        >>> trainer = Trainer(model, config, graph)
        >>> result = trainer.train(train_loader, val_loader)
        >>> print(f"Best metric: {result.best_val_metric:.4f}")
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainerConfig,
        graph: Data,
    ) -> None:
        """Initialize trainer.

        Args:
            model: PyTorch model to train. If the model was initialized with a graph,
                edge structure is managed internally by the model. Otherwise, graph
                edge_index/edge_attr are passed to model.forward() during training.
            config: Training configuration
            graph: Static graph for the volatility surface (may be used for backward
                compatibility with models that don't store edge structure internally)
        """
        self._config = config
        self._model = model
        self._graph = graph
        self._metrics_calculator = MetricsCalculator()
        self._model_metadata: dict | None = None

        # Determine device
        self._device = self._get_device(config.device)
        logger.info(f"Using device: {self._device}")

        # Move model to device
        self._model = self._model.to(self._device)

        # Check if model manages edge structure internally
        self._model_manages_edges = (
            hasattr(model, '_edge_index') and
            model._edge_index is not None
        )

        if not self._model_manages_edges:
            # Backward compatibility: extract from graph and pass explicitly
            self._edge_index = graph.edge_index.to(self._device)
            self._edge_attr = (
                graph.edge_attr.to(self._device)
                if graph.edge_attr is not None
                else None
            )
        else:
            # Model owns the edge structure (already on device after to())
            self._edge_index = None
            self._edge_attr = None

        # Build optimizer, scheduler, criterion
        self._optimizer = self._build_optimizer()
        self._scheduler = self._build_scheduler(self._optimizer)
        self._base_criterion = build_loss(
            config.loss_type,
            huber_delta=config.huber_delta,
            quantile=config.quantile,
        )
        if config.volume_weight_enabled:
            self._criterion = VolumeWeightedMaskedLoss(
                self._base_criterion,
                base_loss_type=config.loss_type,
                huber_delta=config.huber_delta,
                quantile=config.quantile,
            )
        else:
            self._criterion = MaskedLoss(self._base_criterion)

        # Automatic mixed precision (CUDA only)
        self._use_amp = config.use_amp and self._device.type == "cuda"
        self._scaler = torch.amp.GradScaler("cuda", enabled=self._use_amp)
        if self._use_amp:
            logger.info("Automatic mixed precision (AMP) enabled")

        # Enable cuDNN auto-tuner for fixed-size inputs (batch shape is constant)
        if self._device.type == "cuda":
            torch.backends.cudnn.benchmark = True

        # Training state
        self._best_val_metric = float("-inf") if config.early_stopping_mode == "max" else float("inf")
        self._best_epoch = 0
        self._patience_counter = 0

        # Create checkpoint directory
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    def _get_device(self, device: str) -> torch.device:
        """Determine the device to use for training."""
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        return torch.device(device)

    def _build_optimizer(self) -> torch.optim.Optimizer:
        """Build AdamW optimizer."""
        return torch.optim.AdamW(
            self._model.parameters(),
            lr=self._config.learning_rate,
            weight_decay=self._config.weight_decay,
        )

    def _build_scheduler(
        self,
        optimizer: torch.optim.Optimizer,
    ) -> torch.optim.lr_scheduler.LRScheduler | None:
        """Build learning rate scheduler."""
        if self._config.scheduler_type == "none":
            return None

        if self._config.scheduler_type == "step":
            return torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self._config.scheduler_step_size,
                gamma=self._config.scheduler_gamma,
            )

        if self._config.scheduler_type == "cosine":
            # Cosine annealing with linear warmup
            warmup_epochs = self._config.scheduler_warmup_epochs
            max_epochs = self._config.max_epochs

            if warmup_epochs >= max_epochs:
                # Just use linear warmup if warmup covers all epochs
                return torch.optim.lr_scheduler.LinearLR(
                    optimizer,
                    start_factor=0.1,
                    end_factor=1.0,
                    total_iters=max_epochs,
                )

            # Linear warmup followed by cosine decay
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=warmup_epochs,
            )
            cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max_epochs - warmup_epochs,
            )
            return torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_epochs],
            )

        if self._config.scheduler_type == "plateau":
            # ReduceLROnPlateau — decays LR when monitored metric stalls.
            # Unlike epoch-count schedulers, this ties LR decay directly to
            # the early stopping metric, so the model refines around plateaus.
            mode = self._config.early_stopping_mode
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=mode,
                factor=self._config.plateau_factor,
                patience=self._config.plateau_patience,
            )

        raise ValueError(f"Unknown scheduler type: {self._config.scheduler_type}")

    def train(
        self,
        train_loader: DataLoader[dict[str, torch.Tensor]],
        val_loader: DataLoader[dict[str, torch.Tensor]],
        *,
        run_id: str | None = None,
        config_dict: dict | None = None,
    ) -> TrainingResults:
        """Run full training loop.

        Args:
            train_loader: DataLoader for training data.
            val_loader: DataLoader for validation data.
            run_id: Optional unique identifier for this run (generated if not provided).
            config_dict: Optional full configuration dict for reproducibility.

        Returns:
            TrainingResults with structured training history and metrics.
        """
        train_losses: list[float] = []
        val_losses: list[float] = []
        val_metrics_history: list[dict[str, float]] = []
        lr_history: list[float] = []
        best_checkpoint_path = ""

        # Generate run_id if not provided
        if run_id is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_id = f"training_run_{timestamp}"

        # Build config dict for TrainingResults
        if config_dict is None:
            config_dict = {
                "batch_size": self._config.batch_size,
                "max_epochs": self._config.max_epochs,
                "learning_rate": self._config.learning_rate,
                "weight_decay": self._config.weight_decay,
                "early_stopping_patience": self._config.early_stopping_patience,
                "early_stopping_metric": self._config.early_stopping_metric,
                "loss_type": self._config.loss_type,
                "scheduler_type": self._config.scheduler_type,
            }

        train_start_time = datetime.now()
        logger.info(f"Starting training for up to {self._config.max_epochs} epochs")

        for epoch in range(self._config.max_epochs):
            # Record LR at start of epoch (before scheduler step)
            lr_history.append(self._optimizer.param_groups[0]["lr"])

            # Training epoch
            train_loss = self._train_epoch(train_loader, epoch)
            train_losses.append(train_loss)

            # Validation epoch
            val_loss, val_metrics = self._validate_epoch(val_loader)
            val_losses.append(val_loss)
            val_metrics_history.append(val_metrics)

            # Get monitoring metric
            if self._config.early_stopping_metric == "val_loss":
                current_metric = val_loss
            else:
                current_metric = val_metrics.get(self._config.early_stopping_metric, 0.0)

            # Check for improvement
            is_better = self._is_better(current_metric)
            if is_better:
                self._best_val_metric = current_metric
                self._best_epoch = epoch
                self._patience_counter = 0

                # Save best checkpoint — include run_id to prevent clobbering
                best_checkpoint_path = os.path.join(
                    self._config.checkpoint_dir, f"best_model_{run_id}.pt"
                )
                self.save_checkpoint(
                    best_checkpoint_path,
                    epoch,
                    {"train_loss": train_loss, "val_loss": val_loss, **val_metrics},
                )
            else:
                self._patience_counter += 1

            # Log progress — lead with decomposed metrics (temporal IC, XS IC)
            # which reflect genuine predictive skill for DHR targets
            lr = self._optimizer.param_groups[0]["lr"]
            temporal_ic = val_metrics.get('temporal_ic', 0)
            xs_ic = val_metrics.get('xs_demeaned_ic', 0)
            logger.info(
                f"Epoch {epoch + 1}/{self._config.max_epochs} - "
                f"Train: {train_loss:.4f}, Val: {val_loss:.4f}, "
                f"Temporal IC: {temporal_ic:.4f}, "
                f"XS IC: {xs_ic:.4f}, "
                f"LR: {lr:.2e}"
            )

            # Step scheduler
            if self._scheduler is not None:
                if isinstance(
                    self._scheduler,
                    torch.optim.lr_scheduler.ReduceLROnPlateau,
                ):
                    self._scheduler.step(current_metric)
                else:
                    self._scheduler.step()

            # Early stopping check
            if self._patience_counter >= self._config.early_stopping_patience:
                logger.info(
                    f"Early stopping triggered after {epoch + 1} epochs "
                    f"(no improvement for {self._config.early_stopping_patience} epochs)"
                )
                break

        train_duration = (datetime.now() - train_start_time).total_seconds()

        # Build training history DataFrame
        history_records = []
        for epoch_idx, (train_loss, val_loss, val_metrics) in enumerate(
            zip(train_losses, val_losses, val_metrics_history), start=1
        ):
            record = {
                "epoch": epoch_idx,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "temporal_ic": val_metrics.get("temporal_ic", 0.0),
                "xs_demeaned_ic": val_metrics.get("xs_demeaned_ic", 0.0),
                "rmse": val_metrics.get("rmse", 0.0),
                "mae": val_metrics.get("mae", 0.0),
                "lr": lr_history[epoch_idx - 1],
            }
            history_records.append(record)

        training_history = pd.DataFrame(history_records)

        # Get metrics from the best epoch (matching the saved checkpoint)
        best_epoch_metrics = (
            val_metrics_history[self._best_epoch]
            if val_metrics_history
            else {}
        )

        # Build metadata
        metadata = {
            "timestamp": train_start_time.isoformat(),
            "duration_seconds": train_duration,
            "device": str(self._device),
            "checkpoint_path": best_checkpoint_path,
        }

        # Reload best-epoch weights so artifact contains optimal model,
        # not the (potentially degraded) last-epoch weights.
        if best_checkpoint_path and os.path.exists(best_checkpoint_path):
            best_ckpt = torch.load(
                best_checkpoint_path, map_location=self._device, weights_only=False
            )
            self._model.load_state_dict(best_ckpt["model_state_dict"])

        model_state_dict = self._model.state_dict()

        return TrainingResults(
            run_id=run_id,
            config=config_dict,
            training_history=training_history,
            final_metrics=best_epoch_metrics,
            best_epoch=self._best_epoch + 1,  # Convert to 1-indexed
            best_metric=self._best_val_metric,
            epochs_trained=len(train_losses),
            model_state_dict=model_state_dict,
            metadata=metadata,
        )

    def _train_epoch(
        self,
        loader: DataLoader[dict[str, torch.Tensor]],
        epoch: int,
    ) -> float:
        """Train one epoch.

        Args:
            loader: Training data loader
            epoch: Current epoch number (for logging)

        Returns:
            Average training loss for the epoch
        """
        self._model.train()
        total_loss = torch.tensor(0.0, device=self._device)
        num_batches = 0

        for batch_idx, batch in enumerate(loader):
            # Move batch to device (non_blocking overlaps transfer with compute)
            X = batch["X"].to(self._device, non_blocking=True)
            y = batch["y"].to(self._device, non_blocking=True)
            mask = batch["mask"].to(self._device, non_blocking=True)
            if "label_mask" in batch:
                label_mask = batch["label_mask"].to(self._device, non_blocking=True)
            else:
                # Backward compatibility for datasets that only provide node masks.
                label_mask = mask.unsqueeze(-1).expand_as(y)

            # Forward pass with optional AMP autocast
            self._optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=self._use_amp):
                if self._model_manages_edges:
                    preds = self._model(X, mask=mask)
                else:
                    preds = self._model(X, self._edge_index, self._edge_attr, mask)
                if self._config.volume_weight_enabled:
                    vol_weights = self._compute_volume_weights(X, mask)
                    loss = self._criterion(preds, y, label_mask, weights=vol_weights)
                else:
                    loss = self._criterion(preds, y, label_mask)

            # Backward pass (scaled for AMP)
            self._scaler.scale(loss).backward()

            # Gradient clipping (must unscale first for correct magnitudes)
            if self._config.gradient_clip_val > 0:
                self._scaler.unscale_(self._optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self._model.parameters(),
                    self._config.gradient_clip_val,
                )

            # Optimizer step (scaler handles skip if gradients contain inf/NaN)
            self._scaler.step(self._optimizer)
            self._scaler.update()

            # Accumulate on GPU to avoid per-batch CUDA sync from .item()
            total_loss += loss.detach()
            num_batches += 1

            # Log progress
            if (batch_idx + 1) % self._config.log_every_n_steps == 0:
                logger.debug(
                    f"Epoch {epoch + 1}, Batch {batch_idx + 1}/{len(loader)}, "
                    f"Loss: {loss.item():.4f}"
                )

        # Single GPU->CPU sync at epoch end
        return (total_loss / max(num_batches, 1)).item()

    def _validate_epoch(
        self,
        loader: DataLoader[dict[str, torch.Tensor]],
    ) -> tuple[float, dict[str, float]]:
        """Validate one epoch.

        Accumulates predictions on the device (GPU) and performs a single
        GPU->CPU transfer at the end to avoid per-batch synchronization stalls.

        Args:
            loader: Validation data loader

        Returns:
            Tuple of (average loss, metrics dict)
        """
        self._model.eval()
        total_loss = torch.tensor(0.0, device=self._device)
        num_batches = 0

        # Accumulate on device to avoid per-batch GPU->CPU sync stalls
        all_preds: list[torch.Tensor] = []
        all_targets: list[torch.Tensor] = []
        all_masks: list[torch.Tensor] = []
        all_label_masks: list[torch.Tensor] = []

        with torch.no_grad():
            for batch in loader:
                # Move batch to device (non_blocking overlaps transfer with compute)
                X = batch["X"].to(self._device, non_blocking=True)
                y = batch["y"].to(self._device, non_blocking=True)
                mask = batch["mask"].to(self._device, non_blocking=True)
                if "label_mask" in batch:
                    label_mask = batch["label_mask"].to(self._device, non_blocking=True)
                else:
                    label_mask = mask.unsqueeze(-1).expand_as(y)

                # Forward pass with optional AMP autocast
                with torch.amp.autocast("cuda", enabled=self._use_amp):
                    if self._model_manages_edges:
                        preds = self._model(X, mask=mask)
                    else:
                        preds = self._model(X, self._edge_index, self._edge_attr, mask)
                    if self._config.volume_weight_enabled:
                        vol_weights = self._compute_volume_weights(X, mask)
                        loss = self._criterion(preds, y, label_mask, weights=vol_weights)
                    else:
                        loss = self._criterion(preds, y, label_mask)

                total_loss += loss.detach()
                num_batches += 1

                # Accumulate on device (single transfer at end)
                all_preds.append(preds)
                all_targets.append(y)
                all_masks.append(mask)
                all_label_masks.append(label_mask)

        # Single device->CPU transfer for all predictions
        preds_arr = torch.cat(all_preds, dim=0).cpu().numpy()
        targets_arr = torch.cat(all_targets, dim=0).cpu().numpy()
        masks_arr = torch.cat(all_masks, dim=0).cpu().numpy()
        label_masks_arr = torch.cat(all_label_masks, dim=0).cpu().numpy()

        # Remove invalid labels at horizon-level for both training metrics and
        # decomposed IC computation.
        preds_arr = np.where(label_masks_arr, preds_arr, np.nan)
        targets_arr = np.where(label_masks_arr, targets_arr, np.nan)

        # Keep 3D arrays for decomposed metrics (temporal IC, XS-demeaned IC)
        # preds_arr: (num_samples, num_nodes, num_horizons)
        # masks_arr: (num_samples, num_nodes)
        num_nodes = preds_arr.shape[1] if preds_arr.ndim == 3 else 0

        # Flatten for pooled error metrics. Invalid label positions already set
        # to NaN above and are ignored by metric implementations.
        preds_flat = preds_arr.reshape(-1, preds_arr.shape[-1])
        targets_flat = targets_arr.reshape(-1, targets_arr.shape[-1])

        # Compute all metrics (pooled + decomposed)
        metrics = self._metrics_calculator.compute_all(
            preds_flat,
            targets_flat,
            predictions_3d=preds_arr,
            targets_3d=targets_arr,
            masks_2d=masks_arr,
            num_nodes=num_nodes,
        )

        # Single GPU->CPU sync at epoch end
        return (total_loss / max(num_batches, 1)).item(), metrics

    def _compute_volume_weights(
        self,
        X: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-node volume weights from input features.

        Extracts log_volume from the last timestep of X, applies softplus
        for a smooth positive floor, and normalizes so mean weight across
        valid nodes = 1.0 per sample.

        Args:
            X: Input features (batch, time, nodes, features)
            mask: Valid node mask (batch, nodes)

        Returns:
            Weights tensor (batch, nodes), mean ~1.0 across valid nodes.
        """
        vol_idx = self._config.volume_weight_feature_idx
        if vol_idx is None:
            return torch.ones(X.size(0), X.size(2), device=X.device)

        # Extract log_volume at last timestep: (batch, nodes)
        raw_vol = X[:, -1, :, vol_idx]

        # Replace NaN/inf with 0
        raw_vol = torch.nan_to_num(raw_vol, nan=0.0, posinf=0.0, neginf=0.0)

        # Softplus for smooth positive floor (z-scored values can be negative)
        weights = torch.nn.functional.softplus(raw_vol)

        # Normalize: mean weight across valid nodes = 1.0 per sample
        valid_mask = mask.float()
        valid_count = valid_mask.sum(dim=1, keepdim=True).clamp(min=1)
        weight_sum = (weights * valid_mask).sum(dim=1, keepdim=True)
        weights = weights * valid_count / weight_sum.clamp(min=1e-8)

        return weights

    def _is_better(self, current: float) -> bool:
        """Check if current metric is better than best."""
        if self._config.early_stopping_mode == "max":
            return current > self._best_val_metric
        else:
            return current < self._best_val_metric

    @property
    def model_metadata(self) -> dict | None:
        """Model metadata included in checkpoints."""
        return self._model_metadata

    @model_metadata.setter
    def model_metadata(self, metadata: dict | None) -> None:
        """Set model metadata to include in checkpoints.

        Args:
            metadata: Dict with model architecture config, feature columns,
                and normalization stats. Saved alongside the state dict
                so checkpoints are self-describing.
        """
        self._model_metadata = metadata

    def save_checkpoint(
        self,
        path: str,
        epoch: int,
        metrics: dict[str, float],
    ) -> None:
        """Save model checkpoint.

        Args:
            path: Path to save checkpoint
            epoch: Current epoch number
            metrics: Current metrics dict
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self._model.state_dict(),
            "optimizer_state_dict": self._optimizer.state_dict(),
            "scheduler_state_dict": (
                self._scheduler.state_dict() if self._scheduler is not None else None
            ),
            "config": self._config,
            "metrics": metrics,
            "best_val_metric": self._best_val_metric,
            "model_metadata": self._model_metadata,
        }
        torch.save(checkpoint, path)
        logger.debug(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: str) -> dict[str, float]:
        """Load model checkpoint.

        Args:
            path: Path to checkpoint file

        Returns:
            Metrics dict from checkpoint
        """
        checkpoint = torch.load(path, map_location=self._device, weights_only=False)

        self._model.load_state_dict(checkpoint["model_state_dict"])
        self._optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self._scheduler is not None and checkpoint["scheduler_state_dict"] is not None:
            self._scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self._best_val_metric = checkpoint.get("best_val_metric", self._best_val_metric)

        logger.info(f"Loaded checkpoint from {path} (epoch {checkpoint['epoch']})")
        metrics: dict[str, float] = checkpoint.get("metrics", {})
        return metrics
