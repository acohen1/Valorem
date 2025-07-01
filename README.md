# Valorem

**Valorem** is a deep-learning-driven forecasting system for SPY options, combining macroeconomic, credit, and microstructure features into a unified training and inference stack. Built around the PatchTST architecture in PyTorch, it leverages a robust local SQLite feature store for efficient time-series ingestion, preprocessing, and model execution.

---

## Quick Start

```bash
# 1. Clone and set up environment
$ git clone <your-fork-url>
$ cd valorem
$ python -m venv .venv && source .venv/bin/activate

# 2. Install dependencies
$ pip install -r requirements.txt

# 3. Set up environment variables
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
* **L2 Snapshots** — Order book depth, spread, imbalance (future)

---

## Project Structure

```
valorem/
├── data/
│   └── ingestion/
│       ├── fred.py                    # Macroeconomic data fetcher from FRED
│       └── polygon.py                 # Market microstructure fetcher from Polygon.io
├── preprocessing/
│   └── align.py                       # Aligns DataFrames to a shared timeline
├── features/
│   └── store.py                       # Upsert, load, schema‑migration, WAL SQLite store
├── models/
│   └── patchtst_model.py              # PatchTST forecasting model definition  (WIP)
├── pipelines/
│   ├── update_macro.py                # FRED pipeline for macro_daily
│   ├── update_spy_market.py           # Polygon pipeline for SPY data

.notebooks/                            # Jupyter notebooks for exploration and testing
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

## Roadmap

* [x] SQLite feature store with automatic schema migration
* [x] Modular ingestion via `fred.py` and `polygon.py`
* [x] PatchTST-compatible time alignment and flattening
* [ ] Realized volatility and secondary feature pipelines
* [ ] Full model training loop w/ DataLoader from SQLite
* [ ] Backtesting framework for model evaluation

---

## License

MIT License. See `LICENSE` file for details.

---
