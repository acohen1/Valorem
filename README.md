# Valorem

**Valorem** is a deep-learning-driven forecasting system for SPY options, combining macroeconomic, credit, and microstructure features into a unified training and inference stack. Built around the PatchTST architecture in PyTorch, it leverages a robust local SQLite feature store for efficient time-series ingestion, preprocessing, and model execution.

---

## Quick Start

```bash
# 1. clone & enter
$ git clone <your‑fork>
$ cd valorem

# 2. pull the upstream PatchTST repo (subfolder)
$ git clone https://github.com/yuqinie98/PatchTST external/PatchTST

# 3. create env + install
$ python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
$ pip install -e .

# 4. scaffold .env with API keys (Polygon, FRED)
$ python scripts/setup_env.py
```
---

## Core Data Inflows

### 1. Macroeconomic Metrics — FRED API

| Theme                        | Representative Series (ticker)                                                                                  |
| ---------------------------- | --------------------------------------------------------------------------------------------------------------- |
| **Interest Rates & Curves**  | Fed Funds Rate `FEDFUNDS`, 3M T-Bill `DTB3`, 2Y `DGS2`, 10Y `DGS10`, 10Y–2Y Spread `T10Y2Y`                     |
| **Inflation & Expectations** | CPI `CPIAUCSL`, Core PCE `PCEPILFE`, 5Y/5Y Breakeven `T5YIFR`, UMich 1Y Inflation Exp. `MICH`                   |
| **Labor Market**             | Nonfarm Payrolls `PAYEMS`, Unemployment `UNRATE`, Initial Claims `ICSA`, Job Openings `JTSJOL`                  |
| **Growth & Sentiment**       | Industrial Production `INDPRO`, ISM PMI `ADPWINDMANNERSA`, Conf. Board LEI `USSLIND`, UMich Sentiment `UMCSENT` |
| **Credit & Liquidity**       | Baa Spread `BAA10Y`, BBB OAS `BAMLC0A4CBBB`, St. Louis FCI `STLFSI4`, Fed Balance Sheet `WALCL`                 |
| **Benchmarks & Volatility**  | S\&P 500 `SP500`, DJIA `DJIA`, Nasdaq `NASDAQCOM`, VIX Index `VIXCLS`                                           |

### 2. Market Structure — Polygon API

* **SPY Aggregates** — 1-second and 1-minute OHLCV bars
* **Trades** — Tick-level prints with price/size/condition codes
* **Quotes** — Full NBBO quote feed (pending access tier)
* **Options Chain** — Greeks, IV, OI, delta/theta/gamma
* **L2 Snapshots** — Order book depth, spread, imbalance
* **Secondary Features** — Realized volatility, Garman-Klass, Bipower Variance

---

## Project Structure

```
valorem/
├── data/
|   ├── patchtst_loader.py             # PatchTST DataLoader
│   └── ingestion/
│       ├── fred.py                    # Macroeconomic data fetcher from FRED
│       └── polygon.py                 # Market microstructure fetcher from Polygon.io
├── evaluation/                        # Evaluation scripts for model performance
│   └── eval_patchtst.py               # Evaluates PatchTST model on hold-out data
├── features/
│   └── store.py                       # Upsert, load, schema‑migration, WAL SQLite store
│   └── compute/                       # Secondary feature computation modules
│       └── realised_vol.py            # Garman-Klass/Bipower Var
├── models/
│   └── patchtst_model.py              # Wrapper around upstream PatchTST model
├── pipelines/
│   ├── update_macro.py                # FRED pipeline for macro_daily
│   ├── update_spy_market.py           # Polygon pipeline for SPY data
│   └── update_secondary.py            # Secondary features pipeline (realized volatility, etc.)
├── preprocessing/
│   └── align.py                       # Aligns DataFrames to a shared timeline
├── training/
│   └── train_patchtst.py              # Full training loop for PatchTST
|
checkpoints/                           # Checkpoints for PatchTST model
│   ├── patchtst_epoch001.pth
|   └── ...
external/
|   └── PatchTST/                     # PatchTST source code (submodule)
notebooks/
│   └── evaluate_patchtst.ipynb       # Notebook for evaluating trained PatchTST model
scripts/
│   └── setup_env.py                   # One-time setup for .env scaffolding
tests/                                 
│   └── test_align.py                  # Unit tests for time alignment
.env                                   # Local API keys (not tracked)
.env.example                           # Template for .env (tracked)
.gitignore
pyproject.toml
README.md                              # You are here!
valorem.db                             # SQLite feature store (auto-created)
```

---
## End-to-end Workflow
```bash
# ingest macro & market data, creates valorem.db
python -m valorem.pipelines.update_macro        --start 2020-01-01 --end 2025-06-30
python -m valorem.pipelines.update_spy_market   --start 2024-06-01 --end 2025-06-30

# compute secondary features (realized volatility, etc.)
python -m valorem.pipelines.update_secondary    --missing

# train PatchTST (≈30s on GPU)
python -m valorem.training.train_patchtst \
       --batch 32 --past 60 --horizon 5 --epochs 50

# hold‑out evaluation -> metrics + CSV
python -m valorem.evaluation.eval_patchtst \
       --ckpt checkpoints/patchtst_epoch050.pt \
       --out preds_vs_actuals.csv

# visualize results
jupyter notebook notebooks/evaluate_patchtst.ipynb
```


---

## Roadmap

* [x] SQLite feature store with automatic schema migration
* [x] Modular ingestion via `fred.py` and `polygon.py`
* [x] PatchTST-compatible time alignment and flattening
* [x] Realized volatility and secondary feature pipelines
* [x] Full model training loop w/ DataLoader from SQLite
* [x] Backtesting framework for model evaluation

---

## License

MIT License. See `LICENSE` file for details.

---

## References

The model architecture is adapted from the official PatchTST implementation
[@inproceedings{Yuqietal-2023-PatchTST, title={A Time Series is Worth 64 Words: Long-term Forecasting with Transformers}, author={Nie, Yuqi and Nguyen, Nam H. and Sinthong, Phanwadee and Kalagnanam, Jayant}, booktitle={International Conference on Learning Representations}, year={2023}}].
