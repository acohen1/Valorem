# Claude Code Context

Environment-specific notes for Claude Code when working on Valorem. See [README.md](README.md) for project structure and general documentation.

## Development Environment

- **Python**: 3.12+ with venv at `.venv/`
- **Platform**: Dual development (macOS + Windows)

### Windows-Specific

When on Windows, **prefer WSL2** for running Python commands, tests, and scripts:

```bash
wsl bash -c "cd /mnt/c/Users/alexc/Projects/Valorem && . .venv/bin/activate && <command>"
```

Avoid Git Bash/PowerShell for Python operations (venv symlink issues).

### macOS-Specific

Standard terminal and venv activation work fine. No special considerations.

## Running Tests

Activate venv first, then use pytest:

```bash
# Specific test
pytest tests/unit/models/test_gnn.py::TestClass::test_name -v

# Test directory
pytest tests/integration/models/ -v
```

## Key Scripts

### Data Ingestion (`scripts/ingest_raw.py`)

Ingests market data from Databento and FRED into the database. **Costs real money** via Databento API — always preview first.

```bash
# Preview costs (no data fetched)
python scripts/ingest_raw.py --preview-only --start-date 2024-01-01 --end-date 2024-02-01

# Full ingestion (prompts for cost confirmation)
python scripts/ingest_raw.py --start-date 2024-01-01 --end-date 2024-02-01

# FRED macro data only (free, no Databento cost)
python scripts/ingest_raw.py --start-date 2024-01-01 --end-date 2024-02-01 --fred-only

# Quotes only (skip bars/stats)
python scripts/ingest_raw.py --start-date 2024-01-01 --end-date 2024-02-01 --quotes-only

# Skip confirmation for automation
python scripts/ingest_raw.py --start-date 2024-01-01 --end-date 2024-02-01 --yes
```

### Feature Building (`scripts/build_features.py`)

Builds node_panel features from surface snapshots. Requires `--start-date`, `--end-date`, `--surface-version`, `--feature-version`.

```bash
# Full rebuild (surfaces + features)
python scripts/build_features.py --start-date 2018-05-01 --end-date 2021-12-31 \
  --surface-version v1.0 --feature-version v1.0

# Features only (skip surface rebuild if surfaces already exist)
python scripts/build_features.py --start-date 2018-05-01 --end-date 2021-12-31 \
  --surface-version v1.0 --feature-version v1.0 --skip-surfaces

# Dry run (compute but don't write to DB)
python scripts/build_features.py --start-date 2018-05-01 --end-date 2021-12-31 \
  --surface-version v1.0 --feature-version v1.0 --dry-run
```

### Training (`scripts/train_model.py`)

Trains PatchTST+GNN ensemble or ablation variants. Reads splits from `config/config.yaml` by default.

```bash
# Train ensemble (default)
python scripts/train_model.py

# Ablation variants
python scripts/train_model.py --ablation patchtst
python scripts/train_model.py --ablation gnn

# Override splits from CLI
python scripts/train_model.py --train-start 2018-05-01 --val-start 2020-07-01 \
  --test-start 2021-04-01 --test-end 2021-12-31

# Quick pipeline smoke test with synthetic data
python scripts/train_model.py --synthetic
python scripts/train_model.py --dry-run

# Key hyperparameter overrides
python scripts/train_model.py --lr 0.001 --batch-size 32 --epochs 50 --patience 10
python scripts/train_model.py --loss huber --huber-delta 1.0
python scripts/train_model.py --ablation ensemble --learnable-edges --dynamic-volume-edges
```

### Backtest (`scripts/run_backtest.py`)

```bash
python scripts/run_backtest.py --start-date 2021-04-01 --end-date 2021-12-31 \
  --checkpoint checkpoints/best_model.pt --feature-version v1.0 --surface-version v1.0
```

### Database Management (`scripts/manage_data.py`)

```bash
# List all tables with row counts
python scripts/manage_data.py list

# Clear derived tables (surfaces + node_panel), keeps raw data
python scripts/manage_data.py clear --derived --yes

# Clear specific table
python scripts/manage_data.py clear node_panel --yes

# Clear everything (destructive!)
python scripts/manage_data.py clear --all --yes
```

### Typical Rebuild Workflow

When features or labels change, the standard workflow is:

```bash
python scripts/manage_data.py clear node_panel --yes
python scripts/build_features.py --start-date 2018-05-01 --end-date 2021-12-31 \
  --surface-version v1.0 --feature-version v1.0 --skip-surfaces
python scripts/train_model.py
```

Surface snapshots rarely need rebuilding — only clear `--derived` if the surface builder itself changed.

## Model Ablation Variants

Training script supports three variants via `--ablation` flag:
- `patchtst`: Temporal-only baseline (PatchTST → prediction head, no GNN)
- `gnn`: Cross-sectional-only baseline (last timestep → GNN, no temporal context)
- `ensemble`: Full combined model (PatchTST → GNN → prediction head) — **default, production architecture**

All variants use the same feature set (`DEFAULT_FEATURE_COLS`, 29 features). The GNN ablation uses a reduced set (`GNN_ABLATION_FEATURE_COLS`, 20 features) that strips 9 global features identical across all nodes.

## Config

- Main config: `config/config.yaml`
- Schema with defaults: `src/config/schema.py`
- Train/val/test splits are in the YAML under `dataset.splits`
- Date range: train 2018-05-01→2020-06-30, val 2020-07-01→2021-03-31, test 2021-04-01→2021-12-31
