# Valorem

**Production-Grade Volatility Arbitrage Trading System**

A modular, scalable system for predicting and exploiting volatility mispricing in SPY options using advanced machine learning (PatchTST + Graph Neural Networks).

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [CLI Reference](#cli-reference)
- [Trading Modes](#trading-modes)
- [Module Reference](#module-reference)
- [Configuration](#configuration)
- [Development](#development)
- [Testing](#testing)
- [Project Structure](#project-structure)
- [Documentation](#documentation)
- [References](#references)

---

## Overview

Valorem forecasts volatility mispricing at the (timestamp, tenor, delta_bucket) level and translates predictions into bounded-risk option trades with strict risk controls and realistic execution modeling.

### Key Features

- **Modular Architecture**: Independently testable, replaceable components via Protocol interfaces
- **Provider-Agnostic**: Clean abstractions for data providers (Databento, FRED, database, mock)
- **Type-Safe**: Pydantic-based configuration with strict validation
- **ML Pipeline**: PatchTST temporal model + Graph Neural Network for volatility surface, with volume-aware loss weighting and dynamic liquidity-gradient edge attributes
- **Risk Management**: Pre-trade checks, stress testing, kill switches
- **Multiple Trading Modes**: Mock (synthetic), paper (live data), and live (real execution)
- **State Persistence**: Crash recovery with JSON state snapshots
- **Memory-Safe Pipelines**: Chunk-write-flush pattern throughout ingestion, surface building, and feature generation; memmap-backed training datasets enable multi-year data without OOM

---

## Architecture

### Model Architecture

The ML pipeline uses a two-component ensemble to capture both temporal dynamics and cross-sectional structure:

```
Historical Features ──→ PatchTST ──→ Per-node temporal embeddings
                                              │
                                              ▼
Surface Graph ────────→ GNN ──────→ Cross-sectionally informed embeddings
                                              │
                                              ▼
                                    Multi-horizon predictions
```

**PatchTST (Temporal Encoder)**: Processes each surface node's historical time series using a patched transformer architecture. Captures regime changes, mean reversion patterns, and temporal dependencies.

**Graph Neural Network (Cross-Sectional Encoder)**: Propagates information across the surface graph based on known adjacency (delta neighbors, tenor neighbors), allowing nodes to inform each other. Supports learnable edge weights and dynamic volume-based edge attributes that encode per-sample liquidity gradients between adjacent nodes.

### Ablation Variants

The training script supports three ablation variants via `--ablation`:

| Variant | Architecture | Input Features | Purpose |
|---------|-------------|----------------|---------|
| `ensemble` (default) | PatchTST -> GNN -> prediction head | All 29 (`DEFAULT_FEATURE_COLS`) | Target architecture. PatchTST encodes temporal dynamics per node; GNN propagates cross-sectionally. |
| `patchtst` | PatchTST -> prediction head | All 29 (`DEFAULT_FEATURE_COLS`) | Temporal-only baseline. Each node processed independently -- no cross-sectional information. |
| `gnn` | Last-timestep features -> GNN -> prediction head | 20 node-specific (`GNN_ABLATION_FEATURE_COLS`) | Cross-sectional-only baseline. Sees only the current snapshot -- no temporal history. |

The ensemble is the production architecture. The two baselines quantify what each component contributes: if the ensemble outperforms both, the temporal and cross-sectional pathways capture complementary information.

**GNN ablation uses node-specific features only.** The 9 omitted features -- 3 realized vol (`underlying_rv_*`) and 6 macro (`VIXCLS_*`, `DGS10_*`, `DGS2_*`) -- are identical across all 42 surface nodes at each timestamp. Through message passing, the GNN trivially denoises these by averaging 42 identical copies, an advantage PatchTST (which processes nodes independently) does not have. Stripping global features makes the ablation comparison valid.

### Feature Philosophy

Input features fall into two categories:

**Node-specific (20 features)** -- vary across nodes at a given timestamp:
- Greeks and surface structure: delta, gamma, vega, theta, spread_pct, skew_slope, term_slope, curvature
- IV dynamics: iv_change_1d, iv_change_5d
- Vol-of-vol: iv_vol_5d, iv_vol_10d, iv_vol_21d
- IV richness (z-score): iv_zscore_5d, iv_zscore_10d, iv_zscore_21d
- Volume/liquidity: log_volume, volume_ratio_5d, log_oi, oi_change_5d

**Global (9 features)** -- identical across all nodes at a given timestamp:
- Realized vol: underlying_rv_5d, underlying_rv_10d, underlying_rv_21d
- Macro regime: VIXCLS_level, VIXCLS_change_1w, DGS10_level, DGS10_change_1w, DGS2_level, DGS2_change_1w

Global features provide regime context (is the market calm or stressed? are rates rising?). In the ensemble and PatchTST variants, each node independently processes these alongside its node-specific features. iv_mid is deliberately excluded from all feature sets to prevent label-feature leakage (DHR labels are driven by IV level via gamma).

### DHR Labels and Evaluation

Labels are **delta-hedged returns** (DHR): the gamma P&L of a delta-hedged option position. For each node at horizon H:

```
DHR(H) = 0.5 * gamma * (S(t+H) - S(t))^2 * (252/H)
```

DHR has genuine cross-sectional variation because gamma differs across strike/tenor nodes. This is what makes cross-sectional modeling meaningful -- unlike IV level or returns, which are largely homogeneous across the surface. See [docs/MODEL_ROADMAP.md](docs/MODEL_ROADMAP.md) for design rationale and [References](#references) for academic citations.

**Label normalization**: Raw DHR values are O(10-100) due to the 252/H annualization factor and gamma magnitude. Labels are z-score normalized (global mean/std computed on training data, applied to val/test) so all horizons are O(1) scale. The normalization statistics are stored in the checkpoint metadata (`label_stats`) for denormalization at inference time.

**Loss function**: Huber loss (default) provides robustness against outlier DHR values from high-gamma short-dated OTM nodes. The delta parameter (default 1.0) acts as the transition point between MSE (small errors) and MAE (large errors) on the normalized labels. Use `--loss mse` to revert to pure MSE if needed.

**Evaluation metrics** (used for early stopping and ablation comparison):
- **Temporal IC** (primary, used for early stopping): Per-node Spearman correlation across time -- does the model predict *when* DHR is large for a given node?
- **XS-Demeaned IC**: Cross-sectionally demeaned Spearman correlation per timestamp -- does the model predict *which* nodes have higher DHR?
- **RMSE / MAE**: Scale-dependent error metrics for monitoring convergence

Pooled IC (Pearson/Spearman across all sample-node pairs) is deliberately excluded from training metrics. For DHR targets, it is dominated by the trivial gamma-ΔS^2 cross-sectional relationship and does not reflect genuine predictive skill.

### Trading Loop Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        TradingLoop                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────┐    ┌──────────────────┐                   │
│  │ SurfaceProvider  │───→│ SignalGenerator  │                   │
│  │ (live/mock/db)   │    │ (model/rules)    │                   │
│  └──────────────────┘    └────────┬─────────┘                   │
│                                   │                             │
│                                   ▼                             │
│  ┌──────────────────┐    ┌──────────────────┐                   │
│  │ FeatureProvider  │───→│ StructureSelector│                   │
│  │ (rolling/db)     │    │ + PositionSizer  │                   │
│  └──────────────────┘    └────────┬─────────┘                   │
│                                   │                             │
│                                   ▼                             │
│  ┌──────────────────┐    ┌──────────────────┐                   │
│  │   RiskChecker    │───→│  OrderGenerator  │                   │
│  │   + KillSwitch   │    │                  │                   │
│  └──────────────────┘    └────────┬─────────┘                   │
│                                   │                             │
│                                   ▼                             │
│  ┌──────────────────┐    ┌──────────────────┐                   │
│  │   OrderRouter    │───→│  StateManager    │                   │
│  │ (paper/live)     │    │  + Portfolio     │                   │
│  └──────────────────┘    └──────────────────┘                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Installation

### Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) package manager

### Setup

```bash
# Clone repository
git clone <repository-url>
cd Valorem

# Copy environment template
cp .env.example .env
# Edit .env with your API keys (see Environment Variables below)

# Install dependencies
make install-dev
```

### Environment Variables

| Variable | Required For | Description |
|----------|--------------|-------------|
| `DATABENTO_API_KEY` | Data ingestion, `paper_live`, `live` | Databento market data API key |
| `FRED_API_KEY` | Feature building | Federal Reserve Economic Data API key |
| `VALOREM_DATABASE_URL` | `paper_db` | Database connection URL |
| `VALOREM_ENV` | Optional | Environment selection (`dev`, `backtest`, `paper`, `live`; default: `dev`) |
| `VALOREM_MODE` | Optional | Trading mode override (`mock`, `paper_live`, `paper_db`, `live`) |
| `IBKR_HOST` | `live` | Interactive Brokers TWS/Gateway host (default: 127.0.0.1) |
| `IBKR_PORT` | `live` | Interactive Brokers port (default: 7497) |
| `IBKR_CLIENT_ID` | `live` | Interactive Brokers client ID (default: 1) |
| `VALOREM_LOG_LEVEL` | Optional | Logging level (DEBUG, INFO, WARNING, ERROR; default: INFO) |

---

## Quick Start

### Mock Mode (No API Keys Required)

```bash
# Train with synthetic data (dry run)
python scripts/train_model.py --dry-run

# Run paper trading with synthetic data
python scripts/run_paper_trading.py --mode mock --max-iterations 10
```

### Full Pipeline (Requires API Keys)

```bash
# 1. Preview ingestion cost (no data fetched, no credits spent)
python scripts/ingest_raw.py --start-date 2018-06-01 --end-date 2021-12-31 --preview-only

# 2. Ingest raw data (equity bars + option quotes + option bars + FRED)
python scripts/ingest_raw.py --start-date 2018-06-01 --end-date 2021-12-31 --yes

# 2b. Add a new FRED series without re-running Databento (free, instant)
python scripts/ingest_raw.py --start-date 2018-06-01 --end-date 2021-12-31 --fred-only

# 3. Check what's in the database
python scripts/manage_data.py list

# 4. Build surfaces + features (includes volume from option bars)
python scripts/build_features.py --start-date 2018-06-01 --end-date 2021-12-31 \
    --surface-version v1.0 --feature-version v1.0

# 4a. (Optional) Clear and rebuild derived tables after schema changes
python scripts/manage_data.py clear --derived
python scripts/build_features.py --start-date 2018-06-01 --end-date 2021-12-31 \
    --surface-version v1.0 --feature-version v1.0

# 5. Train model on real data from DB (uses dev.yaml splits)
python scripts/train_model.py --env dev --epochs 10

# 5a. Train with volume features enabled
python scripts/train_model.py --env dev --epochs 50 --volume-weight --dynamic-volume-edges

# 5b. (Alternative) Train on synthetic data for CI/smoke testing
python scripts/train_model.py --synthetic --epochs 5

# 5c. (Alternative) Quick dry run (synthetic, 1 epoch)
python scripts/train_model.py --dry-run

# 5d. (Alternative) Run ablation variant
python scripts/train_model.py --ablation patchtst --epochs 10

# 6. Run backtest
python scripts/run_backtest.py --env dev

# 7. Paper trade (mock, validate loop)
python scripts/run_paper_trading.py --mode mock --max-iterations 10 --interval 0
```

Steps 4-6 pull dates/config from `dev.yaml` automatically (no CLI date args needed).

### Environment Configuration

The base config is `config/config.yaml`. Environment-specific overrides live in `config/environments/`:

```bash
# Dev environment is the default (no flag needed)
python scripts/ingest_raw.py --start-date 2023-04-01 --end-date 2023-08-31

# Explicitly select an environment
python scripts/ingest_raw.py --env dev --start-date 2023-04-01 --end-date 2023-08-31

# Or set via environment variable
export VALOREM_ENV=dev
```

Available environments:
- `dev` (default): MPS training, batch size 4, $300 Databento cost cap
- `dev-cuda`: CUDA/GPU training with production-tuned hyperparameters (batch 64, AMP, lower LR)

---

## CLI Reference

### `scripts/train_model.py` - Model Training

Train a PatchTST+GNN ensemble model on volatility surface data.

```bash
python scripts/train_model.py [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--env` | dev | Environment overlay (loads `config/environments/{env}.yaml`) |
| `--config` | config/config.yaml | Path to config file |
| `--synthetic` | - | Use synthetic random data (for CI/smoke tests) |
| `--feature-version` | v1.0 | Feature version in node_panel table (real data mode) |
| `--train-start` | - | Override train start date (YYYY-MM-DD) |
| `--val-start` | - | Override val start date (YYYY-MM-DD) |
| `--test-start` | - | Override test start date (YYYY-MM-DD) |
| `--test-end` | - | Override test end date (YYYY-MM-DD) |
| `--epochs` | from config | Maximum training epochs |
| `--batch-size` | from config | Training batch size |
| `--lr` | from config | Learning rate |
| `--weight-decay` | from config | L2 regularization |
| `--patience` | from config | Early stopping patience |
| `--patchtst-d-model` | from config | PatchTST model dimension |
| `--patchtst-layers` | from config | Number of PatchTST layers |
| `--gnn-hidden` | from config | GNN hidden dimension |
| `--gnn-layers` | from config | Number of GNN layers |
| `--gnn-type` | from config | GNN architecture (GAT/GCN) |
| `--ablation` | ensemble | Ablation variant: `patchtst`, `gnn`, or `ensemble` |
| `--learnable-edges` | - | Enable learnable edge attribute weights |
| `--volume-weight` | - | Enable volume-weighted loss (upweight liquid contracts) |
| `--dynamic-volume-edges` | - | Enable dynamic volume-based GNN edge attributes |
| `--loss` | huber | Loss function (mse/huber/quantile/mae) |
| `--scheduler` | cosine | LR scheduler (cosine/step/none) |
| `--device` | from config | Device (cuda/mps/cpu/auto) |
| `--checkpoint-dir` | artifacts/checkpoints | Checkpoint directory |
| `--train-samples` | 1000 | Training samples (--synthetic mode only) |
| `--val-samples` | 200 | Validation samples (--synthetic mode only) |
| `--dry-run` | - | Run 1 epoch with synthetic data |
| `--verbose` | - | Enable verbose logging |

**Examples:**

```bash
# Full training on real data from DB
python scripts/train_model.py --env dev --epochs 50 --lr 1e-4

# Synthetic smoke test
python scripts/train_model.py --synthetic --epochs 5

# Quick dry run
python scripts/train_model.py --dry-run

# Ablation: temporal-only baseline
python scripts/train_model.py --ablation patchtst --env dev

# Enable learnable edge weights
python scripts/train_model.py --learnable-edges --env dev-cuda

# Volume-weighted loss (upweight liquid contracts)
python scripts/train_model.py --volume-weight --env dev-cuda

# Dynamic volume edges + volume-weighted loss
python scripts/train_model.py --volume-weight --dynamic-volume-edges --env dev-cuda
```

---

### `scripts/run_paper_trading.py` - Paper/Live Trading

Run the trading loop with configurable data sources and execution modes.

```bash
python scripts/run_paper_trading.py [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--mode` | mock | Trading mode (mock/paper_live/paper_db/live) |
| `--config` | config/config.yaml | Path to config file |
| `--checkpoint` | - | Path to trained model checkpoint |
| `--symbols-file` | - | Path to JSON manifest with option symbols |
| `--max-iterations` | 0 | Maximum loop iterations (0 = unlimited) |
| `--interval` | 5 | Seconds between loop iterations |
| `--validate` | - | Validate configuration and exit |
| `--discover-symbols` | - | Discover symbols from Databento |
| `--underlying` | SPY | Underlying for symbol discovery |
| `--output` | - | Output path for symbol manifest |
| `--min-dte` | 7 | Minimum DTE for discovery |
| `--max-dte` | 90 | Maximum DTE for discovery |
| `--verbose` | - | Enable verbose logging |

**Examples:**

```bash
# Mock mode (synthetic data)
python scripts/run_paper_trading.py --mode mock --max-iterations 10

# Paper trading with live data
python scripts/run_paper_trading.py --mode paper_live \
    --symbols-file data/manifest/spy_options.json \
    --checkpoint artifacts/checkpoints/best.pt

# Discover and save symbols
python scripts/run_paper_trading.py --discover-symbols \
    --underlying SPY \
    --output data/manifest/spy_options.json

# Validate configuration
python scripts/run_paper_trading.py --validate --mode paper_live

# Live trading (REAL MONEY - use with caution!)
python scripts/run_paper_trading.py --mode live \
    --symbols-file data/manifest/spy_options.json
```

---

### `scripts/ingest_raw.py` - Data Ingestion

Ingest market data from Databento into the database. Fetches equity bars (XNAS.ITCH), option quotes (OPRA.PILLAR CBBO-1m), and option daily bars (OPRA.PILLAR OHLCV-1d for volume data). Writes chunks to DB immediately to prevent OOM on large date ranges.

```bash
python scripts/ingest_raw.py [OPTIONS]
```

| Option | Required | Description |
|--------|----------|-------------|
| `--start-date` | Yes | Start date (YYYY-MM-DD) |
| `--end-date` | Yes | End date (YYYY-MM-DD) |
| `--env` | - | Environment overlay (default: dev) |
| `--config` | - | Config file path |
| `--preview-only` | - | Estimate costs without fetching |
| `--skip-validation` | - | Skip data quality validation |
| `--manifest` | - | Path to existing manifest file |
| `--mock` | - | Use mock providers for testing |
| `--force` | - | Re-fetch all data even if already ingested |
| `--fred-only` | - | Only fetch FRED macro series (skip all Databento market data) |
| `--yes` / `-y` | - | Skip confirmation prompts (for automated/non-interactive use) |
| `--verbose` | - | Enable verbose logging |

---

### `scripts/build_features.py` - Surface + Feature Engineering

Build volatility surfaces and feature panels. Surfaces are built in daily chunks and features in monthly chunks to bound memory usage.

```bash
python scripts/build_features.py [OPTIONS]
```

| Option | Required | Description |
|--------|----------|-------------|
| `--start-date` | Yes | Start date (YYYY-MM-DD) |
| `--end-date` | Yes | End date (YYYY-MM-DD) |
| `--surface-version` | Yes | Version of surface (e.g., v1.0) |
| `--feature-version` | Yes | Version for output features |
| `--env` | - | Environment overlay (default: dev) |
| `--config` | - | Config file path |
| `--underlying` | - | Underlying symbol (default: from config) |
| `--fred-series` | - | FRED series to include (default: from config) |
| `--lookback-buffer` | - | Days of lookback buffer for rolling calculations (default: from config) |
| `--dry-run` | - | Generate but don't write |
| `--skip-validation` | - | Skip anti-leakage validation |
| `--skip-surfaces` | - | Skip surface building (assumes surfaces exist in DB) |
| `--node-only` | - | Only generate node features |
| `--global-only` | - | Only generate global features |
| `--macro-only` | - | Only generate macro features |
| `--output-csv` | - | Path to save features as CSV |
| `--verbose` | - | Enable verbose logging |

---

### `scripts/run_backtest.py` - Backtesting

Run backtest over historical data.

```bash
python scripts/run_backtest.py [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--env` | dev | Environment overlay |
| `--config` | config/config.yaml | Config file path |
| `--start-date` | from config | Backtest start date (YYYY-MM-DD) |
| `--end-date` | from config | Backtest end date (YYYY-MM-DD) |
| `--initial-capital` | from config | Starting capital (USD) |
| `--checkpoint` | artifacts/checkpoints/best_model.pt | Path to trained model checkpoint |
| `--feature-version` | v1.0 | Feature version in node_panel table |
| `--surface-version` | v1.0 | Surface snapshot version |
| `--device` | from config | Device for model inference (auto/cpu/cuda/mps) |
| `--mock` | - | Use mock/synthetic data |
| `--dry-run` | - | Validate config without running |
| `--output-dir` | artifacts/reports | Output directory |
| `--verbose` | - | Enable verbose logging |

---

### `scripts/manage_data.py` - Database Management

Inspect and manage database tables. List row counts or selectively clear tables.

```bash
python scripts/manage_data.py COMMAND [OPTIONS]
```

**Subcommands:**

| Command | Description |
|---------|-------------|
| `list` | Show all tables with row counts and disk usage |
| `clear` | Delete rows from specific tables or table groups |

**Clear options:**

| Option | Description |
|--------|-------------|
| `<table_names>` | Clear specific tables by name |
| `--derived` | Clear derived tables (surface_snapshots, node_panel) |
| `--raw` | Clear raw ingested tables (warns about re-ingestion cost) |
| `--all` | Clear all tables |
| `--yes` / `-y` | Skip confirmation prompt |

**Examples:**

```bash
# Show tables and row counts
python scripts/manage_data.py list

# Clear surfaces + features (keeps raw ingested data)
python scripts/manage_data.py clear --derived

# Clear a specific table
python scripts/manage_data.py clear surface_snapshots

# Clear without confirmation
python scripts/manage_data.py clear --derived --yes
```

---

## Trading Modes

| Mode | Data Source | Execution | Use Case |
|------|-------------|-----------|----------|
| `mock` | Synthetic | Simulated | Development, testing |
| `paper_live` | Databento (live) | Simulated | Strategy validation |
| `paper_db` | Database (replay) | Simulated | Historical analysis |
| `live` | Databento (live) | Real (IBKR) | Production trading |

### Mode Selection

```bash
# Mock - No external dependencies
python scripts/run_paper_trading.py --mode mock

# Paper with live data - Requires DATABENTO_API_KEY
python scripts/run_paper_trading.py --mode paper_live

# Paper with database replay - Requires VALOREM_DATABASE_URL
python scripts/run_paper_trading.py --mode paper_db

# Live trading - Requires DATABENTO_API_KEY + IBKR connection
python scripts/run_paper_trading.py --mode live
```

---

## Module Reference

### `src.live` - Trading Infrastructure

```python
from src.live import (
    # Core Loop
    TradingLoop,           # Main trading loop orchestrator
    LoopState,             # Current loop state (iteration, signals, errors)
    LoopMetrics,           # Aggregate metrics (total signals, fills, P&L)

    # Protocols (interfaces for dependency injection)
    SurfaceProvider,       # Protocol: get_latest_surface() -> DataFrame
    SignalGenerator,       # Protocol: generate_signals(surface, features) -> list[Signal]
    FeatureProvider,       # Protocol: get_features(surface) -> DataFrame
    IngestionService,      # Protocol: ingest data for a time range

    # Surface Providers
    DatabaseSurfaceProvider,  # Historical data from database

    # Ingestion Services
    BaseIngestionService,     # Abstract base for ingestion
    DatabentoIngestionService,  # Live data ingestion from Databento
    MockIngestionService,     # Synthetic data for testing

    # Signal Generators
    SignalGeneratorBase,      # Abstract base for signal generators
    ModelSignalGenerator,     # ML model-based signals
    RuleBasedSignalGenerator, # Rule-based signals (for testing)

    # Feature Providers
    RollingFeatureProvider,   # In-memory rolling buffer
    DatabaseFeatureProvider,  # Query from node_panel table
    MockFeatureProvider,      # Synthetic features for testing

    # Order Routing
    OrderRouter,           # Protocol: route_order(order, surface) -> Fill
    PaperOrderRouter,      # Simulated execution with slippage/fees
    Fill,                  # Fill result (fill_id, price, qty, fees)

    # State Management
    StateManager,          # Persist/recover trading state
    TradingState,          # Complete state snapshot

    # Position Tracking
    PositionTracker,       # Track open positions
    PositionSnapshot,      # Position state at point in time

    # Monitoring
    TradingMonitor,        # Metrics collection and alerting
    TradingMetrics,        # Collected metrics
    Alert,                 # Alert notification
    AlertLevel,            # Enum: INFO, WARNING, CRITICAL

    # Symbol Discovery
    SymbolProvider,        # Protocol for symbol discovery
    DatabentoSymbolProvider,  # Fetch symbols from Databento API
    ManifestSymbolProvider,   # Load symbols from JSON file
    MockSymbolProvider,       # Generate synthetic symbols
    save_symbols_manifest,    # Save symbols to JSON file
)
```

### `src.strategy` - Trade Structures

```python
from src.strategy import (
    # Core Types
    Signal,                # Trading signal (type, edge, confidence, tenor, delta)
    SignalType,            # Enum: TERM_ANOMALY, DIRECTIONAL_VOL, SKEW_ANOMALY, ELEVATED_IV
    Greeks,                # Option Greeks (delta, gamma, vega, theta)
    OptionLeg,             # Single leg (symbol, qty, price, strike, expiry, greeks)
    OptionRight,           # Enum: CALL, PUT

    # Exit Signals
    ExitSignal,            # Exit signal for position management
    ExitSignalType,        # Enum: STOP_LOSS, TAKE_PROFIT, EDGE_DECAY, TIME_DECAY

    # Trade Structures
    TradeStructure,        # Abstract base for bounded-risk structures
    CalendarSpread,        # Sell near-term, buy far-term (term structure)
    VerticalSpread,        # Buy/sell different strikes (directional)
    CONTRACT_MULTIPLIER,   # Standard multiplier (100)

    # Structure Selection
    StructureSelector,     # Map signals to appropriate structures

    # Position Sizing
    PositionSizer,         # Calculate position size
    SizingResult,          # Sizing result with contracts and rationale

    # Order Generation
    Order,                 # Complete order (legs, max_loss, structure_type)
    OrderGenerator,        # Generate orders from signals
    OrderGenerationResult, # Result with orders and rejections
)
```

### `src.risk` - Risk Management

```python
from src.risk import (
    # Portfolio
    Portfolio,             # Portfolio state and Greek aggregation
    PortfolioState,        # Serializable portfolio snapshot
    Position,              # Individual position
    PositionState,         # Enum: OPEN, CLOSED, EXPIRED
    CONTRACT_MULTIPLIER,   # Standard multiplier (100)

    # Pre-Trade Checks
    RiskChecker,           # Validate trades against limits
    RiskCheckResult,       # Check result with approval/rejection
    RiskCheckStatus,       # Enum: APPROVED, REJECTED, WARNING

    # Stress Testing
    StressEngine,          # Run stress scenarios
    StressResult,          # Aggregated stress results
    StressScenario,        # Single scenario result

    # Kill Switch
    KillSwitch,            # Automated trading halt
    KillSwitchResult,      # Check result
    KillSwitchTrigger,     # Enum: DAILY_LOSS, STRESS_BREACH, LIQUIDITY
)
```

### `src.models` - Machine Learning

```python
from src.models import (
    # Graph Construction
    SurfaceGraphConfig,    # Graph configuration (tenors, deltas)
    build_surface_graph,   # Build PyTorch Geometric graph

    # Dataset
    DatasetConfig,         # Dataset configuration
    LabelsConfig,          # Label configuration (DHR horizons)
    SplitsConfig,          # Train/val/test date split configuration
    SurfaceDataset,        # PyTorch Dataset for surfaces
    DatasetBuilder,        # Build train/val/test splits

    # PatchTST
    PatchTSTModel,         # PatchTST temporal encoder
    PatchTSTModelConfig,   # Model configuration
    PatchEmbedding,        # Patch embedding layer

    # GNN
    SurfaceGNN,            # Graph neural network
    GNNModelConfig,        # GNN configuration

    # Ensemble
    PatchTST_GNN_Ensemble, # Combined model

    # Training
    Trainer,               # Training loop with early stopping
    TrainerConfig,         # Training configuration
    TrainResult,           # Training result (losses, metrics, checkpoint)
    TrainingDataPipeline,  # Load real data from DB into DataLoaders
    TrainingDataConfig,    # Pipeline configuration
    TrainingData,          # Container for loaders + graph
    build_splits_from_yaml,  # Build SplitsConfig from YAML environment

    # Loss Functions
    HuberLoss,             # Robust L1-L2 hybrid
    QuantileLoss,          # Quantile regression
    MaskedLoss,            # Apply node masks
    VolumeWeightedMaskedLoss,  # Volume-weighted loss (upweight liquid nodes)
    build_loss,            # Loss factory

    # Collation
    surface_collate_fn,    # Custom collate for surface batches

    # Evaluation
    compute_ic,            # Information coefficient
    compute_rank_ic,       # Rank IC (Spearman)
    compute_temporal_ic,   # Per-node IC across time
    compute_xs_demeaned_ic,  # Cross-sectionally demeaned IC
    compute_rmse,          # Root mean squared error
    compute_mae,           # Mean absolute error
    MetricsCalculator,     # Compute all metrics
)
```

### `src.backtest` - Backtesting

```python
from src.backtest import (
    BacktestEngine,        # Main backtest orchestrator
    create_backtest_engine,  # Factory function

    # Data Pipeline
    BacktestDataPipeline,  # Load and prepare backtest data
    BacktestDataConfig,    # Data pipeline configuration
    BacktestData,          # Container for prepared data

    # Execution
    ExecutionSimulator,    # Simulate trade execution
    FillResult,            # Simulated fill

    # Results
    BacktestResult,        # Complete result container
    BacktestMetrics,       # Performance metrics
    BacktestReporter,      # Generate reports
    ReportConfig,          # Report configuration
    generate_comparison_report,  # Compare multiple backtest runs

    TradeRecord,           # Individual trade record
    PortfolioSnapshot,     # Portfolio state over time
    PositionSnapshot,      # Position state over time
    calculate_metrics,     # Calculate performance metrics
)
```

### `src.features` - Feature Engineering

```python
from src.features import (
    # Feature Engine
    FeatureEngine,         # Full orchestrator
    create_feature_engine, # Factory function
    FeatureEngineConfig,   # Engine configuration
    FeatureEngineResult,   # Complete result

    # Node Features
    NodeFeatureGenerator,  # Generate per-node features
    NodeFeatureConfig,     # Node feature configuration
    FeatureResult,         # Feature generation result
    IVFeatureGenerator,    # IV-based features
    IVFeatureConfig,       # IV feature configuration
    MicrostructureFeatureGenerator,  # Bid-ask spread, volume
    MicrostructureConfig,  # Microstructure configuration
    SurfaceFeatureGenerator,  # Cross-sectional features
    SurfaceFeatureConfig,  # Surface feature configuration

    # Global Features
    GlobalFeatureConfig,   # Global feature configuration
    ReturnsGenerator,      # Underlying returns
    ReturnsConfig,         # Returns configuration
    RealizedVolGenerator,  # Realized volatility
    RealizedVolConfig,     # Realized vol configuration

    # Macro Features
    MacroTransformGenerator,  # FRED series transforms
    MacroTransformConfig,  # Macro transform configuration
    ReleaseTimeAligner,    # Align to release times
    AlignmentConfig,       # Alignment configuration

    # Validation
    FeatureValidator,      # Validate feature quality
    ValidationResult,      # Validation result
    ValidationIssue,       # Individual issue
    IssueSeverity,         # Enum: INFO, WARNING, ERROR
)
```

### `src.config` - Configuration

```python
from src.config import (
    ConfigSchema,          # Root configuration schema
    ConfigLoader,          # Load and validate config
    PathResolver,          # Resolve file paths
    setup_logging,         # Centralized loguru-based logging

    # Constants
    SurfaceConstants,      # Tenor and delta bucket definitions
    TradingConstants,      # Contract multiplier and execution constants
    MarketConstants,       # Market hours and default data sources

    # Environment
    EnvironmentConfig,     # Trading environment config
    TradingMode,           # Enum: MOCK, PAPER_LIVE, PAPER_DB, LIVE
    validate_cli_config,   # Validate CLI arguments
    print_validation_results,  # Print validation results
)
```

---

## Configuration

### YAML Configuration (`config/config.yaml`)

The main configuration file uses Pydantic models for type-safe validation:

```yaml
version: v1

project:
  name: valorem
  version: "2.0.0"

universe:
  underlying: SPY

surface:
  delta_buckets:
    P10: [-1.0, -0.15]
    P25: [-0.15, -0.35]
    P40: [-0.35, -0.45]
    ATM: [-0.45, -0.55, 0.45, 0.55]
    C40: [0.35, 0.45]
    C25: [0.15, 0.35]
    C10: [0.0, 0.15]
  tenor_bins:
    bins: [7, 14, 30, 60, 90, 120]
  black_scholes:
    method: newton-raphson
    max_iterations: 100
    tolerance: 1.0e-6

model:
  patchtst:
    d_model: 128
    n_layers: 3
    n_heads: 8
  gnn:
    model_type: GAT
    hidden_dim: 64
    n_layers: 2

training:
  device: cuda
  batch_size: 32
  max_epochs: 100
  learning_rate: 0.001
  early_stopping_patience: 10

execution:
  pricing:
    buy_at: ask
    sell_at: bid
  slippage:
    model: fixed_bps
    fixed_bps: 5.0

risk:
  per_trade:
    max_loss: 500.0
    max_contracts: 10
  caps:
    max_abs_delta: 100.0
    max_abs_vega: 1000.0
    max_daily_loss: 2000.0
  kill_switch:
    halt_on_daily_loss: true
    max_daily_loss: 2000.0

logging:
  level: INFO
  format: text
  console_enabled: true
```

Environment-specific overrides live in `config/environments/`. The dev environment (`config/environments/dev.yaml`) overrides training to use MPS, batch size 4, and shorter date ranges. The dev-cuda environment (`config/environments/dev-cuda.yaml`) targets NVIDIA GPUs with production-tuned hyperparameters and AMP.

---

## Development

### Workflow Commands

```bash
# Install dependencies
make install-dev

# Run all tests
make test

# Run specific test suites
make test-unit
make test-integration

# Code quality
make lint       # Run ruff linter
make format     # Format with black + ruff
make typecheck  # Run mypy

# Coverage report
make coverage
```

### Data Pipeline

```bash
# 1. Ingest raw data
make ingest

# 2. Build surfaces + features (surfaces built automatically unless --skip-surfaces)
make features

# 3. Train model
make train

# 4. Run backtest
make backtest

# 5. Run paper trading
make paper
```

---

## Testing

### Test Structure

```
tests/
├── unit/              # Unit tests (fast, isolated)
│   ├── backtest/      # Backtesting tests
│   ├── config/        # Configuration tests
│   ├── data/          # Data layer tests
│   ├── features/      # Feature engineering tests
│   ├── live/          # Trading infrastructure tests
│   ├── models/        # ML model tests
│   ├── pricing/       # Options pricing tests
│   ├── risk/          # Risk management tests
│   ├── strategy/      # Strategy tests
│   ├── surface/       # Surface engine tests
│   └── utils/         # Utility tests
└── integration/       # Integration tests (component interaction)
    ├── backtest/      # Backtest integration
    ├── config/        # Config integration
    ├── data/          # Data pipeline integration
    ├── features/      # Feature pipeline integration
    ├── live/          # Trading loop integration
    ├── models/        # Training integration
    ├── risk/          # Risk integration
    ├── strategy/      # Strategy integration
    └── surface/       # Surface integration
```

### Running Tests

```bash
# All tests
pytest tests/ -v

# Unit tests only
pytest tests/unit/ -v

# Integration tests only
pytest tests/integration/ -v

# Specific module
pytest tests/unit/live/ -v

# With coverage
pytest tests/ --cov=src --cov-report=html

# Parallel execution
pytest tests/ -n auto
```

### Test Coverage

- Unit tests: >85% coverage
- Integration tests: Key workflows covered
- Current: 1971 tests (104 test files)

---

## Project Structure

```
Valorem/
├── config/                     # YAML configuration files
│   ├── config.yaml             # Main configuration
│   └── environments/
│       ├── dev.yaml            # Dev environment overrides (MPS)
│       └── dev-cuda.yaml       # CUDA dev overrides (NVIDIA GPU)
├── src/                        # Source code
│   ├── config/                 # Configuration management
│   │   ├── schema.py           # Pydantic models
│   │   ├── loader.py           # Config loading
│   │   ├── paths.py            # Path resolution
│   │   ├── env.py              # Environment config
│   │   ├── environments.py     # Environment overlay loading
│   │   ├── constants.py        # Surface, trading, market constants
│   │   └── logging.py          # Centralized loguru logging
│   ├── data/                   # Data abstraction + ingestion
│   │   ├── providers/          # Data providers (Databento, FRED, mock)
│   │   ├── storage/            # Storage backends (SQLite)
│   │   ├── ingest/             # Ingestion orchestration + manifests
│   │   └── quality/            # Data quality validators
│   ├── surface/                # Surface engine
│   │   ├── builder.py          # Surface builder orchestrator (daily chunks)
│   │   ├── iv/                 # IV solver (Newton-Raphson Black-Scholes)
│   │   ├── greeks/             # Greeks calculator (analytical)
│   │   ├── buckets/            # Delta bucket assignment
│   │   └── quality/            # Surface quality filters
│   ├── features/               # Feature engineering
│   │   ├── engine.py           # Feature engine orchestrator
│   │   ├── validators.py       # Feature quality validation
│   │   ├── node/               # Per-node features (IV, microstructure, surface)
│   │   ├── global_/            # Surface-level features (returns, realized vol)
│   │   └── macro/              # Macro features (FRED transforms, alignment)
│   ├── pricing/                # Options pricing
│   │   ├── pricer.py           # Option pricer
│   │   ├── historical.py       # Historical pricing
│   │   └── protocol.py         # Pricing protocol
│   ├── models/                 # ML pipeline
│   │   ├── graph.py            # Surface graph construction
│   │   ├── dataset.py          # Dataset + splits configuration
│   │   ├── ensemble.py         # PatchTST+GNN ensemble model
│   │   ├── patchtst/           # PatchTST temporal encoder
│   │   ├── gnn/                # Graph neural network
│   │   ├── train/              # Training (trainer, loss, data pipeline, collate)
│   │   └── eval/               # Evaluation metrics
│   ├── strategy/               # Strategy + execution
│   │   ├── types.py            # Core types (Signal, Greeks, OptionLeg)
│   │   ├── selector.py         # Structure selection
│   │   ├── sizing.py           # Position sizing
│   │   ├── orders.py           # Order generation
│   │   ├── structures/         # Trade structures (calendar, vertical, iron condor, skew)
│   │   └── positions/          # Position management (lifecycle, exits, rebalancing)
│   ├── risk/                   # Risk management
│   │   ├── portfolio.py        # Portfolio tracking
│   │   ├── checker.py          # Pre-trade checks
│   │   ├── stress.py           # Stress testing
│   │   └── kill_switch.py      # Automated halt
│   ├── backtest/               # Backtesting engine
│   │   ├── engine.py           # Main orchestrator
│   │   ├── execution.py        # Execution simulation
│   │   ├── results.py          # Result containers
│   │   └── reporting.py        # Report generation
│   ├── live/                   # Live trading infrastructure
│   │   ├── loop.py             # Trading loop
│   │   ├── router.py           # Order routing
│   │   ├── state.py            # State persistence
│   │   ├── features.py         # Feature providers
│   │   ├── ingestion.py        # Data ingestion services
│   │   ├── signal_generator.py # Signal generation
│   │   ├── surface_provider.py # Surface providers
│   │   ├── symbols.py          # Symbol discovery
│   │   ├── positions.py        # Position tracking
│   │   └── monitoring.py       # Metrics + alerting
│   └── utils/                  # Shared utilities
│       ├── calculations.py     # Common calculations
│       └── validation.py       # Shared validation helpers
├── scripts/                    # CLI scripts
│   ├── train_model.py          # Model training
│   ├── run_paper_trading.py    # Paper/live trading
│   ├── run_backtest.py         # Backtesting
│   ├── ingest_raw.py           # Data ingestion
│   ├── build_features.py       # Surface + feature building
│   └── manage_data.py          # Database inspection + table management
├── notebooks/                  # Jupyter notebooks
│   ├── 00_data_providers_demo.ipynb
│   ├── 01_surface_engine_demo.ipynb
│   ├── 02_ingestion_pipeline.ipynb
│   ├── 03_feature_engineering_demo.ipynb
│   ├── 04_model_exploration.ipynb
│   ├── 05_training_demo.ipynb
│   ├── 06_strategy_risk_demo.ipynb
│   ├── 07_backtest_demo.ipynb
│   ├── 08_paper_trading_demo.ipynb
│   └── 09_ablation_analysis.ipynb
├── tests/                      # Test suite (1964 tests)
│   ├── unit/                   # Unit tests
│   └── integration/            # Integration tests
├── docs/                       # Documentation
│   ├── SPEC.md                 # System specification
│   ├── SPEC_REFACTOR.md        # Refactoring specification
│   ├── PRODUCTION_SEQUENCE.md  # Pipeline execution guide
│   ├── MODEL_ROADMAP.md        # Model research roadmap
│   └── REFACTOR_MILESTONES.md  # Refactoring milestones
├── data/                       # Data artifacts (not committed)
│   ├── db.sqlite               # SQLite database
│   └── manifest/               # Per-month data manifests
├── artifacts/                  # Build artifacts
│   └── logs/                   # Per-workflow log files
├── checkpoints/                # Model checkpoints
├── Makefile                    # Development commands
├── pyproject.toml              # Project configuration
└── .env.example                # Environment template
```

---

## Documentation

- [SPEC.md](docs/SPEC.md) - Comprehensive system specification
- [SPEC_REFACTOR.md](docs/SPEC_REFACTOR.md) - Refactoring specification
- [PRODUCTION_SEQUENCE.md](docs/PRODUCTION_SEQUENCE.md) - Pipeline execution guide
- [MODEL_ROADMAP.md](docs/MODEL_ROADMAP.md) - Model research roadmap
- [REFACTOR_MILESTONES.md](docs/REFACTOR_MILESTONES.md) - Refactoring milestones

---

## References

Key academic references informing the model design, label construction, and evaluation methodology:

### Variance Risk Premium
- Carr, P. & Wu, L. (2009). "Variance Risk Premiums." *Review of Financial Studies*, 22(3), 1311-1341. — Canonical VRP definition and log-difference specification.
- Bollerslev, T., Tauchen, G. & Zhou, H. (2009). "Expected Stock Returns and Variance Risk Premia." *Review of Financial Studies*, 22(11), 4463-4492. — VRP as equity return predictor; VIX² vs realized variance.

### Delta-Hedged Returns (DHR label design)
- Bakshi, G. & Kapadia, N. (2003). "Delta-Hedged Gains and the Negative Market Volatility Risk Premium." *Review of Financial Studies*, 16(2), 527-566. — Per-option VRP via delta-hedged gains; theoretical foundation for node-varying labels.
- Bali, T., Beckmeyer, H., Moerke, M. & Weigert, F. (2023). "Option Return Predictability with Machine Learning and Big Data." *Review of Financial Studies*, 36(9), 3548-3602. — Delta-hedged returns as ML labels; OOS cross-sectional R² ~2.5%; long-short Sharpe evaluation.
- Cao, J. & Han, B. (2013). "Cross Section of Option Returns and Idiosyncratic Stock Volatility." *Journal of Financial Economics*, 108(1), 231-249. — Option-specific realized variance (ORV) formulation.

### Cross-Sectional Evaluation
- Goyal, A. & Saretto, A. (2009). "Cross-Section of Option Returns and Volatility." *Journal of Financial Economics*, 94(2), 310-326. — HV-IV spread; cross-sectional sorts; implicitly market-neutral long-short portfolios.
- Kelly, B., Kuznetsov, B., Malamud, S. & Xu, T. (2023). "Deep Learning from Implied Volatility Surfaces." SSRN Working Paper. — CNN on IV surfaces; Fama-MacBeth cross-sectional regression evaluation.

### Label-Feature Leakage
- Elejalde, H. et al. (2025). "Examining Challenges in Implied Volatility Forecasting." *Computational Economics*. — Identifies IV-in-features + IV-in-label as a leakage pattern; chronological splitting requirements.
- Wang, J. et al. (2024). "Considering Momentum Spillover Effects via GNN in Option Pricing." *Journal of Futures Markets*. — GNN for options; uses features orthogonal to IV levels.

### Architecture
- Nie, Y. et al. (2023). "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers." *ICLR 2023*. — PatchTST architecture for time series patching and channel-independent encoding.

---

## License

Private project - All rights reserved
