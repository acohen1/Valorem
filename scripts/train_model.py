#!/usr/bin/env python3
"""Train PatchTST+GNN ensemble model on volatility surface data.

This script provides a CLI for training the model. By default it loads
real data from the database (node panel + underlying bars). Use --synthetic
for pipeline smoke tests with random data.

Example:
    # Train on real data from DB (default)
    python scripts/train_model.py --env dev --epochs 50

    # Override split dates for limited data
    python scripts/train_model.py --train-start 2023-04-01 --val-start 2023-07-01 \
        --test-start 2023-08-01 --test-end 2023-08-31

    # Synthetic data for CI / smoke tests
    python scripts/train_model.py --synthetic --epochs 5

    # Dry run (synthetic, 1 epoch, minimal data)
    python scripts/train_model.py --dry-run

    # Use specific device
    python scripts/train_model.py --device cuda
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from loguru import logger
from torch.utils.data import DataLoader, Dataset

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.logging import add_file_handler, setup_logging

from src.models import (
    GNNModelConfig,
    LinearBaseline,
    LinearBaselineConfig,
    PatchTST_GNN_Ensemble,
    PatchTSTModel,
    PatchTSTModelConfig,
    SurfaceGraphConfig,
    build_surface_graph,
)
from src.models.dataset import (
    DEFAULT_FEATURE_COLS,
    GNN_ABLATION_FEATURE_COLS,
    LabelsConfig,
    SplitsConfig,
)
from src.models.repository import ResultsRepository
from src.models.train import Trainer, TrainerConfig, surface_collate_fn
from src.models.train.data_pipeline import (
    TrainingDataConfig,
    TrainingDataPipeline,
    build_splits_from_yaml,
)


class SyntheticSurfaceDataset(Dataset[dict[str, torch.Tensor]]):
    """Synthetic dataset for testing training pipeline.

    Generates random data matching the expected input format for
    the PatchTST+GNN ensemble model.
    """

    def __init__(
        self,
        num_samples: int,
        time_steps: int = 22,
        num_nodes: int = 42,
        num_features: int = len(DEFAULT_FEATURE_COLS),
        num_horizons: int = 3,
    ) -> None:
        self._num_samples = num_samples
        self._time_steps = time_steps
        self._num_nodes = num_nodes
        self._num_features = num_features
        self._num_horizons = num_horizons

        # Generate synthetic data — y is independent of X to avoid
        # identity-mapping leakage during pipeline validation
        self._X = torch.randn(num_samples, time_steps, num_nodes, num_features)
        self._y = torch.randn(num_samples, num_nodes, num_horizons)
        self._mask = torch.ones(num_samples, num_nodes, dtype=torch.bool)
        self._label_mask = torch.ones(
            num_samples, num_nodes, num_horizons, dtype=torch.bool
        )

    def __len__(self) -> int:
        return self._num_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "X": self._X[idx],
            "y": self._y[idx],
            "mask": self._mask[idx],
            "label_mask": self._label_mask[idx],
        }


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train PatchTST+GNN ensemble model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data source
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic random data (for CI/smoke tests). Default is real data from DB.",
    )

    # Config and environment
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--env",
        type=str,
        default="dev",
        help="Environment overlay name (loads config/environments/{env}.yaml)",
    )

    # Feature version (for real data mode)
    parser.add_argument(
        "--feature-version",
        type=str,
        default="v1.0",
        help="Feature version in node_panel table (real data mode)",
    )

    # Split overrides (for real data mode)
    parser.add_argument(
        "--train-start",
        type=str,
        default=None,
        help="Override train start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--val-start",
        type=str,
        default=None,
        help="Override val start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--test-start",
        type=str,
        default=None,
        help="Override test start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--test-end",
        type=str,
        default=None,
        help="Override test end date (YYYY-MM-DD)",
    )

    # Training parameters (defaults from YAML config; CLI overrides take precedence)
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Maximum training epochs (default: from config)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Training batch size (default: from config)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate (default: from config)",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=None,
        help="Weight decay / L2 regularization (default: from config)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=None,
        help="Early stopping patience in epochs (default: from config)",
    )

    # Model parameters (defaults from YAML config; CLI overrides take precedence)
    parser.add_argument(
        "--patchtst-d-model",
        type=int,
        default=None,
        help="PatchTST model dimension (default: from config)",
    )
    parser.add_argument(
        "--patchtst-layers",
        type=int,
        default=None,
        help="Number of PatchTST layers (default: from config)",
    )
    parser.add_argument(
        "--gnn-hidden",
        type=int,
        default=None,
        help="GNN hidden dimension (default: from config)",
    )
    parser.add_argument(
        "--gnn-layers",
        type=int,
        default=None,
        help="Number of GNN layers (default: from config)",
    )
    parser.add_argument(
        "--gnn-type",
        type=str,
        default=None,
        help="GNN layer type: GAT or GCN (default: from config)",
    )

    # Ablation study variant selection
    parser.add_argument(
        "--ablation",
        type=str,
        default="ensemble",
        choices=["patchtst", "gnn", "ensemble", "linear"],
        help="Ablation variant: linear (feature baseline), patchtst (temporal only), gnn (cross-sectional only), or ensemble (full model)",
    )

    # M3: Learnable edge weights
    parser.add_argument(
        "--learnable-edges",
        action="store_true",
        help="Enable learnable edge attribute weights (M3 experiment)",
    )

    # Volume integration features
    parser.add_argument(
        "--volume-weight",
        action="store_true",
        help="Enable volume-weighted loss (upweight liquid contracts)",
    )
    parser.add_argument(
        "--dynamic-volume-edges",
        action="store_true",
        help="Enable dynamic volume-based GNN edge attributes (liquidity gradients)",
    )

    # Early stopping metric
    parser.add_argument(
        "--early-stopping-metric",
        type=str,
        default="temporal_ic",
        choices=["temporal_ic", "xs_demeaned_ic", "val_loss"],
        help=(
            "Metric for early stopping. 'temporal_ic' (default) measures per-node "
            "prediction quality across timestamps. 'xs_demeaned_ic' measures "
            "cross-sectional ranking within each timestamp. 'val_loss' uses raw "
            "validation loss (minimize)."
        ),
    )

    # Loss and scheduler
    parser.add_argument(
        "--loss",
        type=str,
        default="huber",
        choices=["mse", "huber", "quantile", "mae"],
        help="Loss function type (default: huber for robustness with z-scored DHR labels)",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="cosine",
        choices=["cosine", "step", "plateau", "none"],
        help="Learning rate scheduler type",
    )
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=None,
        help="Linear warmup epochs for cosine scheduler (default: 5)",
    )
    parser.add_argument(
        "--scheduler-step-size",
        type=int,
        default=None,
        help="Step size for step scheduler (default: 10)",
    )
    parser.add_argument(
        "--scheduler-gamma",
        type=float,
        default=None,
        help="Decay factor for step scheduler (default: 0.1)",
    )
    parser.add_argument(
        "--plateau-factor",
        type=float,
        default=None,
        help="Factor by which LR is reduced on plateau (default: 0.5)",
    )
    parser.add_argument(
        "--plateau-patience",
        type=int,
        default=None,
        help="Epochs with no improvement before reducing LR (default: 5)",
    )
    parser.add_argument(
        "--huber-delta",
        type=float,
        default=None,
        help="Delta parameter for Huber loss (default: 1.0)",
    )

    # Device and output (defaults from YAML config; CLI overrides take precedence)
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to train on: auto, cuda, mps, cpu (default: from config)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Directory to save checkpoints (default: artifacts/checkpoints)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    # Synthetic data parameters
    parser.add_argument(
        "--train-samples",
        type=int,
        default=1000,
        help="Number of training samples (--synthetic mode only)",
    )
    parser.add_argument(
        "--val-samples",
        type=int,
        default=200,
        help="Number of validation samples (--synthetic mode only)",
    )

    # Utility
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run 1 epoch with synthetic data for pipeline testing",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    return parser.parse_args()


def _build_splits_with_overrides(args: argparse.Namespace, config) -> SplitsConfig:
    """Build SplitsConfig from YAML config with optional CLI overrides.

    Args:
        args: Parsed CLI arguments
        config: Loaded ConfigSchema

    Returns:
        SplitsConfig with any CLI overrides applied
    """
    splits = build_splits_from_yaml(config.dataset.splits)

    # Apply CLI overrides
    has_overrides = any([args.train_start, args.val_start, args.test_start, args.test_end])
    if has_overrides:
        splits = SplitsConfig(
            train_start=(
                datetime.strptime(args.train_start, "%Y-%m-%d")
                if args.train_start
                else splits.train_start
            ),
            val_start=(
                datetime.strptime(args.val_start, "%Y-%m-%d")
                if args.val_start
                else splits.val_start
            ),
            test_start=(
                datetime.strptime(args.test_start, "%Y-%m-%d")
                if args.test_start
                else splits.test_start
            ),
            test_end=(
                datetime.strptime(args.test_end, "%Y-%m-%d")
                if args.test_end
                else splits.test_end
            ),
        )

    return splits


def _resolve_param(cli_value, config_value):
    """Resolve a parameter: CLI override takes precedence over config.

    Args:
        cli_value: Value from argparse (None if not explicitly provided)
        config_value: Value from YAML config

    Returns:
        CLI value if provided, otherwise config value
    """
    return cli_value if cli_value is not None else config_value


def _load_real_data(
    args: argparse.Namespace,
    config,
    batch_size: int,
    feature_cols: list[str] | None = None,
):
    """Load real data from database via TrainingDataPipeline.

    Args:
        args: Parsed CLI arguments
        config: Loaded ConfigSchema
        batch_size: Resolved batch size
        feature_cols: Override feature columns (None = DEFAULT_FEATURE_COLS)

    Returns:
        TrainingData with loaders, datasets, graph, and metadata
    """
    from src.data.storage.engine import create_engine
    from src.data.storage.repository import DerivedRepository, RawRepository

    # Build splits from config with CLI overrides
    splits = _build_splits_with_overrides(args, config)

    # Create DB engine and repositories
    db_engine = create_engine(config.paths.db_path)
    raw_repo = RawRepository(db_engine.engine)
    derived_repo = DerivedRepository(db_engine.engine)

    # Build pipeline config
    pipeline_config = TrainingDataConfig(
        feature_version=args.feature_version,
        underlying_symbol=config.universe.underlying,
        splits=splits,
        labels=LabelsConfig(horizons_days=config.labels.horizons),
        batch_size=batch_size,
        feature_cols=feature_cols,
        lookback_days=63,
    )

    # Load data
    pipeline = TrainingDataPipeline(raw_repo, derived_repo, pipeline_config)
    return pipeline.load()


def main() -> int:
    """Main training function."""
    args = parse_args()

    # Configure logging
    setup_logging(verbose=args.verbose)

    # Set random seeds for reproducibility
    import random
    import numpy as np

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Random seed set to {args.seed}")

    # Load YAML config (base + env overlay) — used for parameter defaults and infra
    from src.config.loader import ConfigLoader

    try:
        config = ConfigLoader.load(Path(args.config), env=args.env)
        add_file_handler(
            workflow="train",
            logs_dir=config.paths.logs_dir,
            level=config.logging.level,
            fmt=config.logging.format,
            enabled=config.logging.file_enabled,
        )
    except Exception:
        logger.debug("Config unavailable, using schema defaults")
        from src.config.schema import ConfigSchema

        config = ConfigSchema()

    # Resolve parameters: CLI arg → config → schema default
    device = _resolve_param(args.device, config.training.device)
    batch_size = _resolve_param(args.batch_size, config.training.batch_size)
    max_epochs = _resolve_param(args.epochs, config.training.max_epochs)
    learning_rate = _resolve_param(args.lr, config.training.learning_rate)
    weight_decay = _resolve_param(args.weight_decay, config.training.weight_decay)
    patience = _resolve_param(args.patience, config.training.early_stopping_patience)
    checkpoint_dir = _resolve_param(args.checkpoint_dir, "artifacts/checkpoints")
    num_workers = config.training.num_workers
    use_amp = config.training.use_amp

    patchtst_d_model = _resolve_param(args.patchtst_d_model, config.model.patchtst.d_model)
    patchtst_layers = _resolve_param(args.patchtst_layers, config.model.patchtst.n_layers)
    patchtst_patch_len = config.model.patchtst.patch_len
    patchtst_stride = config.model.patchtst.stride
    gnn_hidden = _resolve_param(args.gnn_hidden, config.model.gnn.hidden_dim)
    gnn_layers = _resolve_param(args.gnn_layers, config.model.gnn.n_layers)
    gnn_type = _resolve_param(args.gnn_type, config.model.gnn.model_type)

    # Resolve scheduler parameters (CLI override or TrainerConfig defaults)
    warmup_epochs = args.warmup_epochs if args.warmup_epochs is not None else 5
    scheduler_step_size = args.scheduler_step_size if args.scheduler_step_size is not None else 10
    scheduler_gamma = args.scheduler_gamma if args.scheduler_gamma is not None else 0.1
    plateau_factor = args.plateau_factor if args.plateau_factor is not None else 0.5
    plateau_patience = args.plateau_patience if args.plateau_patience is not None else 5
    huber_delta = args.huber_delta if args.huber_delta is not None else 1.0

    # Resolve splits early so we can log them
    splits = _build_splits_with_overrides(args, config)

    logger.info(f"Starting training (env={args.env}):")
    logger.info(f"  Max epochs: {max_epochs}")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Learning rate: {learning_rate}")
    logger.info(f"  Weight decay: {weight_decay}")
    logger.info(f"  Patience: {patience}")
    logger.info(f"  Early stop metric: {args.early_stopping_metric}")
    logger.info(f"  Scheduler: {args.scheduler}")
    if args.scheduler == "cosine":
        logger.info(f"    Warmup epochs: {warmup_epochs}")
    elif args.scheduler == "step":
        logger.info(f"    Step size: {scheduler_step_size}, gamma: {scheduler_gamma}")
    elif args.scheduler == "plateau":
        logger.info(f"    Factor: {plateau_factor}, patience: {plateau_patience}")
    logger.info(f"  Device: {device}")
    logger.info(f"  Ablation: {args.ablation}")
    logger.info(f"  Train: {splits.train_start:%Y-%m-%d} to {splits.val_start:%Y-%m-%d}")
    logger.info(f"  Val:   {splits.val_start:%Y-%m-%d} to {splits.test_start:%Y-%m-%d}")
    logger.info(f"  Test:  {splits.test_start:%Y-%m-%d} to {splits.test_end:%Y-%m-%d}")
    if args.ablation == "linear":
        logger.info(f"  Linear baseline: input_dim={len(DEFAULT_FEATURE_COLS)}")
    if args.ablation in ("patchtst", "ensemble"):
        logger.info(f"  PatchTST: d_model={patchtst_d_model}, n_layers={patchtst_layers}, patch_len={patchtst_patch_len}, stride={patchtst_stride}")
    if args.ablation in ("gnn", "ensemble"):
        logger.info(f"  GNN: type={gnn_type}, hidden={gnn_hidden}, layers={gnn_layers}, learnable_edges={args.learnable_edges}")
    if args.volume_weight:
        logger.info(f"  Volume weighting: enabled")
    if args.dynamic_volume_edges:
        logger.info(f"  Dynamic volume edges: enabled (edge_dim=3)")
    # Build surface graph
    logger.info("Building surface graph...")
    graph_config = SurfaceGraphConfig()
    graph = build_surface_graph(graph_config)
    num_nodes = graph.num_nodes
    logger.info(f"Graph has {num_nodes} nodes and {graph.edge_index.shape[1]} edges")

    # Wrapper for GNN-only variant: handles temporal aggregation
    class GNNStandaloneWrapper(nn.Module):
        """Wrapper that selects last timestep before GNN forward pass.

        For ablation study: allows GNN-only variant to work with (B,T,N,F) input
        by taking the last timestep to produce (B,N,F). This ensures the GNN
        sees only the current cross-sectional snapshot with no temporal information.
        """

        def __init__(self, gnn_model: nn.Module):
            super().__init__()
            self.gnn = gnn_model

        def forward(
            self,
            x: torch.Tensor,
            edge_index: torch.Tensor,
            edge_attr: torch.Tensor | None = None,
            mask: torch.Tensor | None = None,
        ) -> torch.Tensor:
            # x: (batch, time, nodes, features)
            # Take last timestep only — true cross-sectional baseline
            x_last = x[:, -1, :, :]  # (batch, nodes, features)
            return self.gnn(x_last, edge_index, edge_attr, mask)

    # Create model
    logger.info(f"Creating model (ablation={args.ablation})...")
    patchtst_config = PatchTSTModelConfig(
        d_model=patchtst_d_model,
        n_layers=patchtst_layers,
        patch_len=patchtst_patch_len,
        stride=patchtst_stride,
    )
    gnn_config = GNNModelConfig(
        model_type=gnn_type,  # type: ignore[arg-type]
        hidden_dim=gnn_hidden,
        n_layers=gnn_layers,
        use_learnable_edge_attr=args.learnable_edges,
        use_dynamic_volume_edges=args.dynamic_volume_edges,
        edge_dim=3 if args.dynamic_volume_edges else 2,
    )
    # Resolve feature columns for this ablation variant.
    # GNN-only uses node-specific features only (22) to prevent trivial
    # denoising of global features through message passing. PatchTST and
    # ensemble use all 31 features.
    active_feature_cols = (
        GNN_ABLATION_FEATURE_COLS if args.ablation == "gnn" else DEFAULT_FEATURE_COLS
    )
    input_dim = len(active_feature_cols)

    if args.ablation == "patchtst":
        # PatchTST-only: temporal encoder with prediction head
        # Wrap to match trainer's signature (accepts edge_index/edge_attr but ignores them)
        class PatchTSTStandaloneWrapper(nn.Module):
            """Wrapper that makes PatchTST compatible with trainer signature."""

            def __init__(self, patchtst_model: nn.Module):
                super().__init__()
                self.patchtst = patchtst_model

            def forward(
                self,
                x: torch.Tensor,
                edge_index: torch.Tensor,
                edge_attr: torch.Tensor | None = None,
                mask: torch.Tensor | None = None,
            ) -> torch.Tensor:
                # Ignore edge_index and edge_attr (not needed for PatchTST)
                return self.patchtst(x, mask)

        patchtst_model = PatchTSTModel(
            config=patchtst_config,
            input_dim=input_dim,
            output_horizons=3,
        )
        model = PatchTSTStandaloneWrapper(patchtst_model)
    elif args.ablation == "gnn":
        # GNN-only: raw features → GNN with prediction head
        from src.models.gnn import SurfaceGNN

        gnn_model = SurfaceGNN(
            config=gnn_config,
            input_dim=input_dim,
            output_horizons=3,
        )
        model = GNNStandaloneWrapper(gnn_model)
    elif args.ablation == "linear":
        # Linear baseline: shared linear layer on last timestep, no temporal/graph structure

        class LinearStandaloneWrapper(nn.Module):
            """Wrapper that makes LinearBaseline compatible with trainer signature."""

            def __init__(self, linear_model: nn.Module):
                super().__init__()
                self.linear = linear_model

            def forward(
                self,
                x: torch.Tensor,
                edge_index: torch.Tensor,
                edge_attr: torch.Tensor | None = None,
                mask: torch.Tensor | None = None,
            ) -> torch.Tensor:
                # Ignore edge_index and edge_attr (not needed for linear model)
                return self.linear(x, mask)

        linear_config = LinearBaselineConfig()
        linear_model = LinearBaseline(
            config=linear_config,
            input_dim=input_dim,
            output_horizons=3,
        )
        model = LinearStandaloneWrapper(linear_model)
    else:  # "ensemble"
        # Full ensemble: PatchTST → GNN → head
        vol_idx = (
            active_feature_cols.index("log_volume")
            if args.dynamic_volume_edges
            else None
        )
        model = PatchTST_GNN_Ensemble(
            patchtst_config=patchtst_config,
            gnn_config=gnn_config,
            input_dim=input_dim,
            output_horizons=3,
            graph=graph,
            volume_feature_idx=vol_idx,
        )

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model has {num_params:,} parameters")

    # Load data
    use_synthetic = args.synthetic or args.dry_run

    if use_synthetic:
        # Synthetic data path (for CI/smoke tests)
        if args.dry_run:
            logger.info("DRY RUN: Using minimal synthetic data")
            train_samples = 32
            val_samples = 16
            epochs = 1
        else:
            logger.info("Using synthetic data (--synthetic)")
            train_samples = args.train_samples
            val_samples = args.val_samples
            epochs = max_epochs

        train_dataset = SyntheticSurfaceDataset(
            num_samples=train_samples,
            num_nodes=num_nodes,
            num_features=input_dim,
        )
        val_dataset = SyntheticSurfaceDataset(
            num_samples=val_samples,
            num_nodes=num_nodes,
            num_features=input_dim,
        )
        use_cuda = torch.cuda.is_available()
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=surface_collate_fn,
            num_workers=num_workers,
            pin_memory=use_cuda,
            persistent_workers=num_workers > 0,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=surface_collate_fn,
            num_workers=num_workers,
            pin_memory=use_cuda,
            persistent_workers=num_workers > 0,
        )
        feature_stats = None
        label_stats = None
        lookback_periods = int(train_dataset[0]["X"].shape[0]) if len(train_dataset) > 0 else 22
    else:
        # Real data path (default)
        logger.info("Loading real data from database...")
        logger.info(f"  Config: {args.config} (env={args.env})")
        logger.info(f"  Feature version: {args.feature_version}")
        feature_cols_override = (
            list(active_feature_cols) if args.ablation == "gnn" else None
        )
        training_data = _load_real_data(
            args, config, batch_size, feature_cols=feature_cols_override,
        )
        train_loader = training_data.train_loader
        val_loader = training_data.val_loader
        graph = training_data.graph
        num_nodes = graph.num_nodes
        epochs = max_epochs
        feature_stats = training_data.train_dataset.get_feature_stats()
        label_stats = training_data.train_dataset.get_label_stats()
        lookback_periods = (
            int(training_data.train_dataset[0]["X"].shape[0])
            if len(training_data.train_dataset) > 0
            else 22
        )

    # Create trainer
    # Resolve volume weight feature index
    vol_weight_idx = (
        active_feature_cols.index("log_volume")
        if args.volume_weight
        else None
    )

    # Resolve early stopping mode from metric name
    es_metric = args.early_stopping_metric
    es_mode = "min" if es_metric == "val_loss" else "max"

    trainer_config = TrainerConfig(
        device=device,
        batch_size=batch_size,
        max_epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        early_stopping_patience=patience,
        early_stopping_metric=es_metric,
        early_stopping_mode=es_mode,
        checkpoint_dir=checkpoint_dir,
        loss_type=args.loss,
        huber_delta=huber_delta,
        scheduler_type=args.scheduler,
        scheduler_warmup_epochs=warmup_epochs,
        scheduler_step_size=scheduler_step_size,
        scheduler_gamma=scheduler_gamma,
        plateau_factor=plateau_factor,
        plateau_patience=plateau_patience,
        use_amp=use_amp,
        volume_weight_enabled=args.volume_weight,
        volume_weight_feature_idx=vol_weight_idx,
    )

    trainer = Trainer(model, trainer_config, graph)

    # Attach model metadata for self-describing checkpoints
    from dataclasses import asdict

    trainer.model_metadata = {
        "ablation_variant": args.ablation,
        "patchtst_config": asdict(patchtst_config),
        "gnn_config": asdict(gnn_config),
        "input_dim": input_dim,
        "output_horizons": 3,
        "lookback_periods": lookback_periods,
        "feature_columns": list(active_feature_cols),
        "feature_stats": feature_stats if not use_synthetic else None,
        "label_stats": label_stats if not use_synthetic else None,
    }

    # Build comprehensive config dict for TrainingResults
    config_dict = {
        "ablation": args.ablation,
        "learnable_edges": args.learnable_edges,
        "batch_size": batch_size,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "patience": patience,
        "early_stopping_metric": es_metric,
        "loss_type": args.loss,
        "huber_delta": huber_delta,
        "scheduler_type": args.scheduler,
        "scheduler_warmup_epochs": warmup_epochs,
        "scheduler_step_size": scheduler_step_size,
        "scheduler_gamma": scheduler_gamma,
        "plateau_factor": plateau_factor,
        "plateau_patience": plateau_patience,
        "patchtst_d_model": patchtst_d_model,
        "patchtst_n_layers": patchtst_layers,
        "gnn_type": gnn_type,
        "gnn_hidden": gnn_hidden,
        "gnn_layers": gnn_layers,
        "volume_weight": args.volume_weight,
        "dynamic_volume_edges": args.dynamic_volume_edges,
        "use_synthetic": use_synthetic,
        "environment": args.env,
        "feature_version": args.feature_version,
        "train_start": splits.train_start.strftime("%Y-%m-%d"),
        "val_start": splits.val_start.strftime("%Y-%m-%d"),
        "test_start": splits.test_start.strftime("%Y-%m-%d"),
        "test_end": splits.test_end.strftime("%Y-%m-%d"),
    }

    # Generate meaningful run_id
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    edge_suffix = "_learnable" if args.learnable_edges else "_fixed"
    run_id = f"{args.ablation}{edge_suffix}_{timestamp}"

    # Train
    logger.info("Starting training...")
    results = trainer.train(
        train_loader,
        val_loader,
        run_id=run_id,
        config_dict=config_dict,
    )

    # Save to repository
    repo = ResultsRepository()
    tags = [args.ablation]
    if args.learnable_edges:
        tags.append("learnable-edges")
    else:
        tags.append("fixed-edges")
    if args.volume_weight:
        tags.append("volume-weight")
    if args.dynamic_volume_edges:
        tags.append("dynamic-volume-edges")

    saved_name = repo.save(
        results,
        name=run_id,
        tags=tags,
        include_weights=True,
    )

    # Report results
    logger.info("=" * 60)
    logger.info("Training complete!")
    edge_label = f" (learnable edges)" if args.learnable_edges else ""
    logger.info(f"  Ablation variant:        {args.ablation}{edge_label}")
    logger.info(f"  Feature version:         {args.feature_version}")
    logger.info(f"  Data period:             {splits.train_start:%Y-%m-%d} to {splits.test_end:%Y-%m-%d}")
    logger.info(f"    Train:                 {splits.train_start:%Y-%m-%d} to {splits.val_start:%Y-%m-%d}")
    logger.info(f"    Val:                   {splits.val_start:%Y-%m-%d} to {splits.test_start:%Y-%m-%d}")
    logger.info(f"    Test:                  {splits.test_start:%Y-%m-%d} to {splits.test_end:%Y-%m-%d}")
    logger.info(f"  Epochs trained:          {results.epochs_trained}")
    logger.info(f"  Best epoch:              {results.best_epoch}")
    logger.info(f"  Best validation metric:  {results.best_metric:.4f}")
    best_row = results.training_history.iloc[results.best_epoch - 1]
    logger.info(f"  Best epoch train loss:   {best_row['train_loss']:.4f}")
    logger.info(f"  Best epoch val loss:     {best_row['val_loss']:.4f}")
    logger.info("  Best epoch metrics:")
    logger.info(f"    Temporal IC:       {results.final_metrics.get('temporal_ic', 0):>7.4f}")
    logger.info(f"    XS-Demeaned IC:    {results.final_metrics.get('xs_demeaned_ic', 0):>7.4f}")
    logger.info(f"    RMSE:              {results.final_metrics.get('rmse', 0):>7.4f}")
    logger.info(f"    MAE:               {results.final_metrics.get('mae', 0):>7.4f}")
    logger.info(f"  Best checkpoint:         {results.metadata['checkpoint_path']}")
    logger.info(f"  Results saved as:        {saved_name}")
    logger.info("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
