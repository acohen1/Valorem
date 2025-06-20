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
│       ├── fred.py              # FRED API fetchers
│       └── polygon.py           # Polygon fetchers (bars, trades, options)
│
├── features/
│   └── store.py                # SQLite persistence and schema management
│
├── pipelines/
│   ├── update_macro.py         # Macro table loader (REPLACE mode)
│   └── update_spy_market.py    # SPY bars/trades/chain loader (APPEND mode)
│
├── preprocessing/
│   └── align.py                # Timestamp alignment and forward-filling
│
├── models/
│   └── patchtst/               # PatchTST model components (WIP)
│
├── scripts/
│   └── setup_env.py            # Environment scaffolding
│
├── notebooks/                  # Exploratory notebooks & visualizations
├── tests/                      # Unit + integration tests
├── .env.template               # Example env file for API keys
└── README.md                   # You are here
```

---

## Roadmap

* [x] SQLite feature store with automatic schema migration
* [x] Modular ingestion via `fred.py` and `polygon.py`
* [x] PatchTST-compatible time alignment and flattening
* [ ] Realized volatility and secondary feature pipelines
* [ ] Full model training loop w/ DataLoader from SQLite
* [ ] Optional cloud sync or DuckDB scale-out for >10M samples

---

## License

MIT License. See `LICENSE` file for details.

---
