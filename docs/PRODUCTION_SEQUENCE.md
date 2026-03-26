# Valorem - Production Sequence

Complete pipeline from data ingestion through model training, backtesting, and paper trading.

---

## Data Availability

**Historical Range:**
- **Underlying (XNAS.ITCH)**: May 1, 2018 → Present (~8 years)
- **Options (OPRA.PILLAR, cbbo-1m)**: April 1, 2013 → Present (~13 years)
- **Limiting Factor**: Underlying data (2018+)

**Key Regimes Covered (2018-2026):**
- 2018: Vol spike (VIX >50), trade war
- 2019: Recovery, rate cuts
- 2020: COVID crash + rally
- 2021: Meme stocks, low vol grind
- 2022: Bear market, rate hikes
- 2023: Banking crisis, recovery
- 2024: Election, Aug vol spike
- 2025-2026: Recent period

---

## Prerequisites

**API Keys:**
- [x] `DATABENTO_API_KEY` in `.env` (for market data)
- [x] `FRED_API_KEY` in `.env` (for macro data)

**Configuration:**
- Base config: `config/config.yaml` (XNAS.ITCH + OPRA.PILLAR)
- Dev overlay: `config/environments/dev.yaml` (MPS, small batches, Apr-Aug 2023)
- Dev-CUDA overlay: `config/environments/dev-cuda.yaml` (CUDA, larger batches)

---

## Pipeline Overview

```
1. Data Ingestion    → Raw market data (XNAS.ITCH + OPRA.PILLAR + FRED)
2. Surface Building  → IV calculation, Greeks, delta bucketing
3. Feature Engineering → Node/Global/Macro features
4. Model Training    → PatchTST+GNN ensemble (with ablation variants)
5. Backtesting       → Historical strategy validation
6. Paper Trading     → Live loop validation (mock/paper/live modes)
```

---

## 1. Data Ingestion

### Quick Start (Single Period)

```bash
# Preview cost (underlying only - requires cached manifests for options estimate)
python scripts/ingest_raw.py --start-date 2023-04-01 --end-date 2023-08-31 --preview-only

# Ingest data (shows cost confirmation prompt, then fetches)
python scripts/ingest_raw.py --start-date 2023-04-01 --end-date 2023-08-31

# Skip confirmation prompt (for automation / non-interactive use)
python scripts/ingest_raw.py --start-date 2023-04-01 --end-date 2023-08-31 --yes
```

**Cost Reference:** ~$2/month for SPY (2000+ option symbols + underlying at cbbo-1m)

**Cost Confirmation:** In interactive mode, ingestion runs a cost preview first and prompts for confirmation before spending money. Use `--yes` / `-y` to skip the prompt.

### Multi-Year Ingestion (for Ablation Studies)

For robust cross-regime model evaluation, ingest 2018-2026 data:

```bash
# Option A: Ingest by year (limits financial risk per run)
python scripts/ingest_raw.py --start-date 2018-05-01 --end-date 2019-12-31
python scripts/ingest_raw.py --start-date 2020-01-01 --end-date 2021-12-31
python scripts/ingest_raw.py --start-date 2022-01-01 --end-date 2023-12-31
python scripts/ingest_raw.py --start-date 2024-01-01 --end-date 2026-02-28

# Option B: Ingest full range in one command
python scripts/ingest_raw.py --start-date 2018-05-01 --end-date 2026-02-28
```

**Skip/Resume:** Ingestion automatically skips already-ingested data. If a run is interrupted, re-running the same command resumes where it left off — underlying bars and completed option quote chunks are detected and skipped (no duplicate API charges). Use `--force` to re-fetch everything.

**Per-Chunk Writes:** Option quotes are written to the DB immediately after each monthly chunk is fetched, so data is preserved even if the process crashes mid-run.

### Preview Workflow

`--preview-only` generates manifests and estimates costs before ingestion:

```bash
# Preview cost for a date range
python scripts/ingest_raw.py --start-date 2018-05-01 --end-date 2018-05-31 --preview-only

# Shows confirmation prompt:
# ============================================================
# PREVIEW MODE WARNING
# ============================================================
# Generating manifests requires API calls and will cost money.
#
# Date range: 2018-05-01 to 2018-05-31 (30 days)
# Estimated manifest generation cost: ~$0.13
#
# Note: This is ONLY the cost to generate manifests.
#       Actual data ingestion will cost additional.
# ============================================================
#
# Proceed with preview? [y/N]:
```

**Manifest Generation Cost:** ~$0.0044/day (~$0.13/month, ~$1.59/year). This is separate from actual data ingestion costs.

**Note:** If cached manifests exist for the date range, they will be reused (no additional cost).

---

## 2. Surface Building + Feature Engineering

Build surfaces and features for the full ingested range:

```bash
# Full range (2018-2026)
python scripts/build_features.py \
    --start-date 2018-05-01 \
    --end-date 2026-02-28 \
    --surface-version v2.0 \
    --feature-version v2.0

# Or by year for memory management
python scripts/build_features.py \
    --start-date 2018-05-01 --end-date 2019-12-31 \
    --surface-version v2.0 --feature-version v2.0

python scripts/build_features.py \
    --start-date 2020-01-01 --end-date 2021-12-31 \
    --surface-version v2.0 --feature-version v2.0

# ... continue for remaining years
```

**What this does:**
- Computes IV and Greeks for all options
- Assigns options to (tenor, delta_bucket) grid nodes
- Generates node features (IV dynamics, microstructure)
- Generates global features (returns, realized vol)
- Generates macro features (FRED series, release-time aligned)
- Writes to `surface_snapshots` and `node_panel` tables

**Versioning & Rebuilds:**

Both `surface_snapshots` and `node_panel` use upsert-replace keyed on `(timestamp, node, version)`. This means:

- **Same version is safe to rebuild.** If you build 2018-2019 with `v1.0`, then later rebuild 2018-2026 with `v1.0`, the overlapping rows are overwritten (identical values) and new dates are inserted.
- **Different versions coexist.** `v1.0` and `v2.0` live side-by-side in the same table — reads filter by version.
- **Not incremental.** You can't add just 2020-2026 to an existing `v1.0` build — the feature engine needs a lookback buffer of historical surfaces. Rebuild the full range; the DB handles deduplication.

---

## 3. Model Training

### Quick Start (Dev Environment)

```bash
# Train on dev data (Apr-Aug 2023, defined in dev.yaml)
python scripts/train_model.py --env dev --epochs 50

# Synthetic data for CI/smoke testing
python scripts/train_model.py --synthetic --epochs 5

# Quick dry run (synthetic, 1 epoch)
python scripts/train_model.py --dry-run
```

### Ablation Studies (M1 + M3)

Train all variants for cross-regime comparison:

```bash
# M1: Component ablations (temporal vs spatial vs ensemble)
python scripts/train_model.py --env dev --ablation patchtst --epochs 50
python scripts/train_model.py --env dev --ablation gnn --epochs 50
python scripts/train_model.py --env dev --ablation ensemble --epochs 50

# M3: Edge weight variants (fixed vs learnable)
python scripts/train_model.py --env dev --ablation ensemble --epochs 50
python scripts/train_model.py --env dev --ablation ensemble --learnable-edges --epochs 50
```

**Results tracking:** All runs save to `ResultsRepository` for analysis in `notebooks/09_ablation_analysis.ipynb`

### Multi-Regime Training

To train across multiple regimes, override split dates:

```bash
# 2018 H2 regime (Vol spike)
python scripts/train_model.py \
    --train-start 2018-07-01 --val-start 2018-10-01 \
    --test-start 2018-11-01 --test-end 2018-12-31 \
    --ablation ensemble --epochs 50

# 2020 H1 regime (COVID)
python scripts/train_model.py \
    --train-start 2020-01-01 --val-start 2020-03-01 \
    --test-start 2020-04-01 --test-end 2020-06-30 \
    --ablation ensemble --epochs 50

# ... repeat for other regimes
```

**Suggested Regimes for Ablation Analysis:**
- 2018 H2: Vol spike (VIX >50)
- 2020 H1: COVID crash
- 2021 H1: Recovery rally, low vol
- 2022 H1: Bear market start, rate hikes
- 2023 H1: Banking crisis recovery
- 2024 H2: Election + Aug vol spike (VIX >65)
- 2025 H2: Recent period

---

## 4. Backtesting

Run backtest on test period using trained model:

```bash
# Backtest on dev test period (Aug 2023, defined in dev.yaml)
python scripts/run_backtest.py --env dev

# Override dates or checkpoint
python scripts/run_backtest.py --env dev \
    --start-date 2023-08-01 --end-date 2023-08-31 \
    --checkpoint artifacts/checkpoints/best_model.pt

# Backtest with mock/synthetic data (no DB needed)
python scripts/run_backtest.py --mock

# Backtest on a specific regime
python scripts/run_backtest.py \
    --start-date 2020-03-01 --end-date 2020-06-30 \
    --checkpoint artifacts/checkpoints/ensemble_2020H1.pt \
    --feature-version v2.0 --surface-version v2.0
```

### Backtest CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--env` | `dev` | Environment overlay (`dev`, `backtest`, `paper`, `live`) |
| `--checkpoint` | `artifacts/checkpoints/best_model.pt` | Path to trained model checkpoint |
| `--feature-version` | `v1.0` | Node panel feature version in DB |
| `--surface-version` | `v1.0` | Surface snapshot version in DB |
| `--device` | `cpu` | Inference device (`cpu`, `cuda`, `mps`) |
| `--mock` | off | Use synthetic data instead of DB + model |
| `--start-date` | from config | Override backtest start date (YYYY-MM-DD) |
| `--end-date` | from config | Override backtest end date (YYYY-MM-DD) |
| `--initial-capital` | from config | Override starting capital |
| `--output-dir` | `artifacts/reports` | Directory for result exports |

---

## 5. Paper Trading

Validate trading loop with mock, paper, or live data:

```bash
# Mock mode (synthetic data, no API keys needed)
python scripts/run_paper_trading.py --mode mock --max-iterations 10 --interval 0

# Paper trading with live data (requires DATABENTO_API_KEY)
python scripts/run_paper_trading.py --mode paper_live \
    --symbols-file data/manifests/spy_options.json \
    --checkpoint artifacts/checkpoints/best_model.pt

# Discover and save symbols for paper trading
python scripts/run_paper_trading.py --discover-symbols \
    --underlying SPY \
    --output data/manifests/spy_options.json

# Validate configuration before running
python scripts/run_paper_trading.py --validate --mode paper_live

# Live trading (REAL MONEY - use with extreme caution!)
python scripts/run_paper_trading.py --mode live \
    --symbols-file data/manifests/spy_options.json \
    --checkpoint artifacts/checkpoints/best_model.pt
```

---

## GPU Training (CUDA)

**Recommended Setup for Long Training Runs:**

MPS (Apple Silicon) works but is slow for this workload. For multi-regime training across 2018-2026 data, use CUDA:

### WSL2 + NVIDIA GPU (Windows)

1. **Install WSL2**
   ```bash
   wsl --install
   ```

2. **Install NVIDIA CUDA Toolkit inside WSL2**
   ```bash
   wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
   sudo dpkg -i cuda-keyring_1.1-1_all.deb
   sudo apt-get update
   sudo apt-get -y install cuda-toolkit-12-6
   ```

3. **Clone repo and install dependencies inside WSL2**
   ```bash
   cd /mnt/c/Users/<username>/Projects/Valorem
   python -m venv .venv
   source .venv/bin/activate
   pip install -e ".[dev]"
   ```

4. **Run training with CUDA**
   ```bash
   python scripts/train_model.py --env dev-cuda --device cuda --epochs 50
   ```

### Native Linux

```bash
# Install CUDA toolkit
sudo apt-get install nvidia-cuda-toolkit

# Verify CUDA
nvidia-smi
nvcc --version

# Train
python scripts/train_model.py --device cuda --epochs 50
```

**Performance:** CUDA is ~5-10x faster than MPS for this workload (PatchTST + GNN ensemble).

---

## Full Workflow Example (2018-2026)

Complete pipeline for multi-regime ablation studies:

```bash
# 1. Ingest data by year (2018-2026)
for year in 2018 2019 2020 2021 2022 2023 2024 2025; do
    if [ $year -eq 2018 ]; then
        start="${year}-05-01"
    else
        start="${year}-01-01"
    fi

    if [ $year -eq 2025 ]; then
        end="2026-02-28"
    else
        end="${year}-12-31"
    fi

    echo "Ingesting $year..."
    python scripts/ingest_raw.py --start-date $start --end-date $end --yes
done

# 2. Build surfaces + features for full range
python scripts/build_features.py \
    --start-date 2018-05-01 \
    --end-date 2026-02-28 \
    --surface-version v2.0 \
    --feature-version v2.0

# 3. Train ablation variants across key regimes
# Define regimes in notebooks/09_ablation_analysis.ipynb, then:

# Example: 2020 H1 (COVID crash)
for ablation in patchtst gnn ensemble; do
    python scripts/train_model.py \
        --train-start 2020-01-01 --val-start 2020-03-01 \
        --test-start 2020-04-01 --test-end 2020-06-30 \
        --ablation $ablation --epochs 50 --device cuda
done

# Example: 2022 H1 (Bear market)
for ablation in patchtst gnn ensemble; do
    python scripts/train_model.py \
        --train-start 2022-01-01 --val-start 2022-03-01 \
        --test-start 2022-04-01 --test-end 2022-06-30 \
        --ablation $ablation --epochs 50 --device cuda
done

# M3: Learnable edges
python scripts/train_model.py \
    --train-start 2020-01-01 --val-start 2020-03-01 \
    --test-start 2020-04-01 --test-end 2020-06-30 \
    --ablation ensemble --learnable-edges --epochs 50 --device cuda

# 4. Analyze results in Jupyter
jupyter notebook notebooks/09_ablation_analysis.ipynb

# 5. Backtest best variant on out-of-sample period
python scripts/run_backtest.py \
    --start-date 2025-07-01 --end-date 2026-02-28 \
    --checkpoint artifacts/checkpoints/ensemble_best.pt \
    --feature-version v2.0 --surface-version v2.0

# 6. Paper trade (mock validation)
python scripts/run_paper_trading.py --mode mock --max-iterations 100
```

---

## Cost Summary

**Calibrated from actual ingestion runs:**
- 2024 Jan-Feb (2 months): $3.95 actual → ~$2/month
- 2018-2019 (20 months): ~$50 actual (from Databento billing)

**Estimated Total (2018-2026, ~93 months):**
- Earlier years (2018-2020) have fewer option symbols → lower cost per month
- Later years (2023-2026) have more option symbols → higher cost per month
- **Rough estimate: ~$150-200 total**

Use `--preview-only` for accurate cost estimates before committing to a date range. The interactive cost confirmation prompt will show the exact estimate before any money is spent.

---

## Troubleshooting

### Out of memory during feature building

Ingestion is safe from OOM (per-chunk writes keep memory bounded). Feature building loads surfaces into memory, which can be large for multi-year ranges.

**Solution:** Process data in smaller chunks

```bash
# Instead of full range
python scripts/build_features.py --start-date 2018-05-01 --end-date 2026-02-28 ...

# Do it yearly
for year in 2018 2019 2020 2021 2022 2023 2024 2025; do
    if [ $year -eq 2018 ]; then
        start="${year}-05-01"
    else
        start="${year}-01-01"
    fi

    if [ $year -eq 2025 ]; then
        end="2026-02-28"
    else
        end="${year}-12-31"
    fi

    python scripts/build_features.py \
        --start-date $start --end-date $end \
        --surface-version v2.0 --feature-version v2.0
done
```

### Training too slow on CPU

**Solution:** Use CUDA (see GPU Training section above) or reduce model size

```bash
# Smaller model for faster iteration
python scripts/train_model.py --env dev \
    --patchtst-d-model 64 --patchtst-layers 1 \
    --gnn-hidden 32 --gnn-layers 1 \
    --batch-size 16
```

---

## Next Steps

1. **Ingest Historical Data:** Start with 2018-2019 to validate pipeline
2. **Generate Features:** Build surfaces and features for ingested data
3. **Run Ablation Studies:** Train all variants across regimes
4. **Analyze Results:** Use `notebooks/09_ablation_analysis.ipynb` for cross-regime comparison
5. **Select Best Model:** Based on mean performance + robustness (low variance)
6. **Backtest:** Validate on out-of-sample test periods
7. **Paper Trade:** Live loop validation before production
