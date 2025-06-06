# Valorem

**Valorem** is a deep‑learning‑driven trading system that forecasts short‑term SPY returns and optimizes options positioning by combining macroeconomic, credit, and market‑structure signals. It is built on **PyTorch** (PatchTST architecture), with a local SQLite feature store for high‑frequency inference and disciplined signal generation.

---

## Quick Start

```bash
# 1. Clone & create an isolated environment
$ git clone <your‑fork‑url>
$ cd valorem
$ python -m venv .venv && source .venv/bin/activate

# 2. Install dependencies
$ pip install -r requirements.txt   # or: poetry install

# 3. Create the `.env` from template (runs once)
$ python scripts/setup_env.py
```

---

## Core Data Inflows

Each subsection lists concrete series or endpoints that will be scripted under `valorem/data/ingestion`. This checklist is **non‑exhaustive**—expand it as the project evolves.

### 1. Macroeconomic Metrics — FRED API

| Theme                        | Representative Series (ticker)                                                                                           |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| **Interest Rates & Curves**  | Fed Funds Effective `FEDFUNDS`, 3‑M T‑Bill `DTB3`, 2‑Y Note `DGS2`, 10‑Y Note `DGS10`, 10Y–2Y Spread `T10Y2Y`            |
| **Inflation & Expectations** | CPI `CPIAUCSL`, Core PCE `PCEPILFE`, 5‑Y/5‑Y Breakeven `T5YIFR`, UMich Inflation Expectations `MICH`                     |
| **Labor Market**             | Non‑Farm Payrolls `PAYEMS`, Unemployment Rate `UNRATE`, Initial Claims `ICSA`, Job Openings `JTSJOL`                     |
| **Growth & Sentiment**       | Industrial Production `INDPRO`, ISM Manufacturing PMI `NAPM`, Conference Board LEI `USSLIND`, UMich Sentiment `UMCSENT`  |
| **Credit & Liquidity**       | Moody’s Baa Spread `BAA10Y`, BBB OAS `BAMLC0A4CBBB`, St. Louis Financial Conditions `STLFSI4`, Fed Balance Sheet `WALCL` |
| **Volatility & Risk**        | ICE BofAML MOVE Index `MOVE`, VIX (FRED‐sourced or Polygon derivative)                                                   |

### 2. Market Structure / Micro‑Structure — Polygon API

* **SPY Bar / Quote / Trade** — 1‑second bars (Aggregates v3), OHLCV, full NBBO quotes & condition codes.
* **SPY Order‑Book (L2)** — Book v2 snapshots for depth, imbalance, spread.
* **Options Chain** — Real‑time chain snapshots (implied vol, delta, theta, vega, gamma), historical open interest & volume.
* **Options Trades & Quotes** — Tick‑level option prints; derive flow metrics & trade aggressor side.
* **Aggregated Greeks / GEX** — Pre‑aggregated gamma exposure, vanna & charm by strike/expiry.

### 3. Cross‑Asset Indicators (Polygon or Alt Feeds)

* **Volatility Surfaces** — VIX term structure, VVIX, SKEW.
* **Credit ETFs** — Intraday bars for **HYG**, **LQD**.
* **Rates & Futures** — 2‑Y `ZT`, 10‑Y `ZN`, 30‑Y `ZB` treasuries; Eurodollar / SOFR implied curve.
* **Commodities & FX** — Crude `CL`, Gold `GC` futures; U.S. Dollar Index `DXY`.

### 4. Derived / Secondary Features (Pre‑processing)

* **Realized Volatility** — Rolling Garman‑Klass, Parkinson, bipower.
* **Macro Surprise Index** — CESI or DIY z‑score (actual vs. consensus releases).
* **Liquidity Metrics** — Bid‑ask spread, order‑book depth percentile, VPIN.
* **Regime Labels** — Risk‑on/off (e.g., PCA of cross‑asset returns), macro‑regime clustering.
* **Term‑Structure Factors** — Slope, curvature, level for both yields & vols.

### 5. Reference Calendars / Exogenous Schedules

* **Economic Release Calendar** — Auto‑pull BLS, BEA, Census schedules & consensus; store surprise sign.
* **FOMC Calendar & Dot Plot** — Meeting dates, probability‑weighted rate path (CME FedWatch scraping).
* **Options Expiry Calendar** — Weekly & monthly SPY expiries, quarterly PM‑settled futures.
* **U.S. Holidays / Early Closes** — NYSE holiday calendar for trading‑hour alignment.

---

## Project Structure (Initial Sketch)

```
valorem/
├── data/
│   └── ingestion/            # Data‑feed scripts (FRED, Polygon, …)
├── models/
│   ├── patchtst/             # Forecasting architecture
│   └── …
├── notebooks/                # Exploratory analysis & feature research
├── scripts/
│   └── setup_env.py          # `.env` scaffolding helper
├── tests/                    # Unit & integration tests
├── .env.template             # Sample environment variables (API keys, etc.)
└── README.md                 # You are here 🚀
```

---

## Contributing

1. Fork the repo & create a feature branch (`git checkout -b feature/<name>`).
2. Run `pre‑commit install` to enable linting hooks.
3. Add or update **unit tests** for new features.
4. Ensure `pytest` passes and `ruff`/`black` show no issues.
5. Submit a pull request—detailing motivation and implementation.

---

## License

This project is licensed under the **MIT License**. See `LICENSE` for details.
