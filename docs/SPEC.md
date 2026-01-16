# Rhubarb System Specification
**Production-Grade Volatility Arbitrage Trading System**

Version: 2.0
Last Updated: 2026-01-10

---

## Table of Contents

1. [Vision & Objectives](#1-vision--objectives)
2. [System Architecture](#2-system-architecture)
3. [Core Modules](#3-core-modules)
4. [Data Flow](#4-data-flow)
5. [Technology Stack](#5-technology-stack)
6. [Implementation Guidelines](#6-implementation-guidelines)
7. [Migration Path](#7-migration-path)
8. [Appendices](#8-appendices)

---

## 1. Vision & Objectives

### 1.1 Mission Statement

Build a production-grade, modular trading system that predicts and exploits volatility mispricing in SPY options using advanced machine learning (PatchTST + Graph Neural Networks) with strict risk controls and realistic execution modeling.

### 1.2 Core Values

1. **Modularity**: Every component is independently testable, replaceable, and composable
2. **Organization**: Clear separation of concerns with intuitive package hierarchy
3. **Scalability**: Architecture supports evolution (multi-underlying, new models, live brokers)

### 1.3 Design Philosophy

- **Systems design first**: Plan the entire architecture before writing implementation code
- **Production-ready from day one**: No "research code" that needs rewriting for production
- **Provider-agnostic abstractions**: Clean interfaces with concrete implementations
- **Type safety everywhere**: Leverage Python type hints and Pydantic for contracts
- **Immutable data**: Raw data is append-only; derived data is versioned and reproducible
- **Fail loudly**: Validation errors should crash early, not propagate silently

### 1.4 Primary Objective

Forecast implied vs realized variance mispricing over horizons H ∈ {5, 10, 21} trading days at the (timestamp, tenor, delta_bucket) level, and translate predictions into bounded-risk option trades executed with realistic constraints.

### 1.5 Explicit Non-Goals (v1)

- Multi-underlying portfolio (SPY only)
- Exotic options (vanilla calls/puts only)
- Real-time streaming quotes (1-minute bars sufficient)
- Dynamic delta hedging (static structures only)
- Full broker integration (paper trading → manual execution bridge)

---

## 2. System Architecture

### 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     EXTERNAL SOURCES                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  Databento   │  │     FRED     │  │   Broker     │      │
│  │  (Market)    │  │    (Macro)   │  │   (Future)   │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
         │                    │                    │
         ▼                    ▼                    ▼
┌─────────────────────────────────────────────────────────────┐
│                   DATA ABSTRACTION LAYER                     │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Provider Interface (Protocol)                        │   │
│  │    - DatabentoProvider  - FREDProvider               │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    STORAGE & PERSISTENCE                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │  Raw Tables │  │   Derived   │  │   Parquet   │        │
│  │   (SQLite)  │  │   Tables    │  │   Archive   │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      SURFACE ENGINE                          │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Raw → IV/Greeks → Bucket Assignment → Snapshots     │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   FEATURE ENGINEERING                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │    Node     │  │   Global    │  │    Macro    │        │
│  │  Features   │  │  Features   │  │  Features   │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                       ML PIPELINE                            │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Dataset Builder → Model Training → Inference        │   │
│  │  PatchTST + GNN                                      │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  STRATEGY & EXECUTION                        │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Signal → Structure Selection → Order Generation     │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     RISK MANAGEMENT                          │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Pre-Trade Checks → Portfolio Limits → Stress Tests  │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  LIVE TRADING INFRASTRUCTURE                 │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  State Manager → Order Router → Position Tracker     │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Design Principles by Layer

#### Data Abstraction Layer
- **Purpose**: Decouple system from specific data vendors
- **Pattern**: Protocol-based interfaces with concrete implementations
- **Key Abstraction**: `MarketDataProvider`, `MacroDataProvider`

#### Storage Layer
- **Purpose**: Immutable raw storage + versioned derived artifacts
- **Pattern**: Repository pattern with SQL + Parquet hybrid
- **Key Abstraction**: `RawRepository`, `DerivedRepository`

#### Surface Engine
- **Purpose**: Transform raw quotes into structured IV surface
- **Pattern**: Pipeline with explicit stages (IV → Greeks → Buckets)
- **Key Abstraction**: `SurfaceBuilder`, `BucketAssigner`

#### Feature Engineering
- **Purpose**: Generate model inputs with strict time integrity
- **Pattern**: Composable feature generators with validation
- **Key Abstraction**: `FeatureEngine`, `FeatureValidator`

#### ML Pipeline
- **Purpose**: Train, evaluate, and deploy models
- **Pattern**: Scikit-learn-style fit/predict with graph support
- **Key Abstraction**: `TimeSeriesModel`, `GraphModel`, `Ensemble`

#### Strategy & Execution
- **Purpose**: Map predictions to trades with realistic constraints
- **Pattern**: Strategy pattern with pluggable trade structures
- **Key Abstraction**: `TradingStrategy`, `StructureSelector`, `OrderGenerator`

#### Risk Management
- **Purpose**: Enforce hard constraints on positions and portfolio
- **Pattern**: Chain of responsibility for risk checks
- **Key Abstraction**: `RiskChecker`, `StressEngine`, `KillSwitch`

#### Live Trading Infrastructure
- **Purpose**: Manage state, route orders, track positions
- **Pattern**: Event-driven state machine
- **Key Abstraction**: `TradingLoop`, `StateManager`, `PositionTracker`

---

## 3. Core Modules

### 3.1 Configuration Management

#### 3.1.1 Design Goals
- Type-safe configuration with Pydantic models
- Environment-specific overrides (dev, backtest, paper, live)
- Validation at load time (fail fast)
- Centralized path resolution
- Schema versioning

#### 3.1.2 Package Structure
```
src/config/
├── __init__.py
├── schema.py          # Pydantic models for entire config tree
├── loader.py          # Load and validate YAML → Pydantic
├── validator.py       # Cross-field validation rules
├── paths.py           # Path resolution utilities
└── environments.py    # Environment-specific overlays
```

#### 3.1.3 Key Classes

**ConfigSchema** (Pydantic BaseModel)
```python
class ConfigSchema(BaseModel):
    version: Literal["v1"]
    project: ProjectConfig
    data: DataConfig
    calendar: CalendarConfig
    universe: UniverseConfig
    surface: SurfaceConfig
    features: FeaturesConfig
    labels: LabelsConfig
    dataset: DatasetConfig
    model: ModelConfig
    training: TrainingConfig
    backtest: BacktestConfig
    execution: ExecutionConfig
    risk: RiskConfig
    paper: PaperConfig
    harness: HarnessConfig
    logging: LoggingConfig
    paths: PathsConfig
```

**ConfigLoader**
```python
class ConfigLoader:
    @staticmethod
    def load(path: Path, env: str = "dev") -> ConfigSchema:
        """Load, validate, and merge environment overrides."""

    @staticmethod
    def validate(config: ConfigSchema) -> None:
        """Run cross-field validation."""
```

**PathResolver**
```python
class PathResolver:
    def __init__(self, root: Path, paths_config: PathsConfig):
        """Initialize with repo root and paths config."""

    def resolve(self, key: str) -> Path:
        """Resolve logical path key to absolute path."""
```

#### 3.1.4 Configuration Validation Rules
- `data.providers.databento.dataset_*` must not be empty
- `surface.delta_buckets.ATM` must be 4-element list
- `execution.pricing.buy_at` must be "ask" (no mid in v1)
- `risk.caps.*` must be positive
- `dataset.splits.*_start` must be chronologically ordered
- All file paths must be relative to repo root

#### 3.1.5 Environment Overrides
```yaml
# config/config.yaml (base)
training:
  device: "cuda"
  batch_size: 32

# config/environments/dev.yaml (override)
training:
  device: "cpu"
  batch_size: 4
```

---

### 3.2 Data Abstraction Layer

#### 3.2.1 Design Goals
- Vendor-agnostic interfaces
- Concrete implementations for Databento and FRED
- Standardized data contracts (schemas)
- Retry logic and error handling
- Cost estimation and rate limiting

#### 3.2.2 Package Structure
```
src/data/
├── __init__.py
├── providers/
│   ├── __init__.py
│   ├── protocol.py           # Abstract interfaces
│   ├── databento.py          # Concrete Databento implementation
│   ├── fred.py               # Concrete FRED implementation
│   └── mock.py               # Mock provider for testing
├── ingest/
│   ├── __init__.py
│   ├── orchestrator.py       # Coordinates ingestion runs
│   ├── manifest.py           # Options symbol manifest generator
│   └── cost_estimator.py     # Pre-ingestion cost checks
├── storage/
│   ├── __init__.py
│   ├── schema.py             # SQLAlchemy table definitions
│   ├── repository.py         # Data access layer (raw + derived)
│   ├── engine.py             # Database connection management
│   └── parquet_writer.py     # Parquet archival
└── quality/
    ├── __init__.py
    ├── validators.py         # Data quality checks
    └── diagnostics.py        # Quality reporting
```

#### 3.2.3 Provider Protocols

**MarketDataProvider** (Abstract Interface)
```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class MarketDataProvider(Protocol):
    """Abstract interface for market data vendors."""

    def fetch_underlying_bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        interval: str = "1m",
    ) -> pd.DataFrame:
        """Fetch OHLCV bars for underlying."""
        ...

    def fetch_option_quotes(
        self,
        symbols: list[str],
        start: datetime,
        end: datetime,
        schema: str = "cbbo-1m",
    ) -> pd.DataFrame:
        """Fetch option quote data."""
        ...

    def estimate_cost(
        self,
        dataset: str,
        schema: str,
        symbols: list[str],
        start: datetime,
        end: datetime,
    ) -> float:
        """Estimate USD cost before fetching."""
        ...

    def resolve_option_symbols(
        self,
        parent: str,
        as_of: datetime,
        dte_min: int,
        dte_max: int,
        moneyness_min: float,
        moneyness_max: float,
    ) -> list[str]:
        """Resolve available option symbols for given criteria."""
        ...
```

**MacroDataProvider** (Abstract Interface)
```python
@runtime_checkable
class MacroDataProvider(Protocol):
    """Abstract interface for macro/fundamental data vendors."""

    def fetch_series(
        self,
        series_id: str,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        """Fetch macro time series with release timestamps."""
        ...

    def get_latest_value(
        self,
        series_id: str,
        as_of: datetime,
    ) -> tuple[datetime, float]:
        """Get latest released value as of timestamp."""
        ...
```

#### 3.2.4 Concrete Implementations

**DatabentoProvider**
```python
class DatabentoProvider:
    """Databento implementation of MarketDataProvider protocol."""

    def __init__(self, config: DatabentoConfig):
        self._client = db.Historical(key=os.getenv(config.api_key_env))
        self._config = config
        self._logger = logging.getLogger(__name__)

    def fetch_underlying_bars(self, ...) -> pd.DataFrame:
        """Implementation using databento client."""
        data = self._client.timeseries.get_range(
            dataset=self._config.dataset_equities,
            schema="ohlcv-1m",
            symbols=symbol,
            stype_in=self._config.stype_in,
            start=start.isoformat(),
            end=end.isoformat(),
        )
        return self._normalize_bars(data.to_df())

    def _normalize_bars(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names and types."""
        return df.rename(columns={
            "ts_event": "ts_utc",
            # ... other mappings
        })
```

**FREDProvider**
```python
class FREDProvider:
    """FRED implementation of MacroDataProvider protocol."""

    def __init__(self, config: FREDConfig):
        self._api_key = os.getenv(config.api_key_env)
        self._base_url = config.base_url
        self._session = requests.Session()

    def fetch_series(self, series_id: str, start: datetime, end: datetime) -> pd.DataFrame:
        """Fetch series observations with release metadata."""
        # Implementation using FRED API
        ...
```

#### 3.2.5 Ingestion Orchestrator

**IngestionOrchestrator**
```python
class IngestionOrchestrator:
    """Coordinates multi-source ingestion runs."""

    def __init__(
        self,
        market_provider: MarketDataProvider,
        macro_provider: MacroDataProvider,
        repository: RawRepository,
        config: DataConfig,
    ):
        self._market = market_provider
        self._macro = macro_provider
        self._repo = repository
        self._config = config

    def run_ingestion(
        self,
        start: datetime,
        end: datetime,
        preview_only: bool = False,
    ) -> IngestionResult:
        """Execute full ingestion workflow."""

        # 1. Generate run ID
        run_id = self._generate_run_id()

        # 2. Estimate costs
        cost = self._estimate_total_cost(start, end)
        if cost > self._config.ingestion.databento.cost.max_usd:
            raise CostExceededException(f"Estimated cost ${cost:.2f} exceeds limit")

        if preview_only:
            return IngestionResult(run_id=run_id, cost=cost, preview=True)

        # 3. Load or generate options manifest
        manifest = self._load_or_generate_manifest(start, end)

        # 4. Fetch underlying bars
        underlying_df = self._market.fetch_underlying_bars(
            symbol=self._config.universe.underlying,
            start=start,
            end=end,
        )

        # 5. Fetch option quotes
        options_df = self._market.fetch_option_quotes(
            symbols=manifest.symbols,
            start=start,
            end=end,
            schema=self._config.ingestion.databento.options.schema,
        )

        # 6. Fetch macro series
        macro_dfs = {}
        for series_id in self._config.features.macro.series:
            macro_dfs[series_id] = self._macro.fetch_series(
                series_id=series_id,
                start=start,
                end=end,
            )

        # 7. Run data quality checks
        self._validate_data_quality(underlying_df, options_df, macro_dfs)

        # 8. Write to storage
        self._repo.write_underlying_bars(underlying_df, run_id)
        self._repo.write_option_quotes(options_df, run_id)
        for series_id, df in macro_dfs.items():
            self._repo.write_fred_series(series_id, df)

        # 9. Log ingestion metadata
        self._repo.write_ingestion_log(run_id, cost, start, end, ...)

        return IngestionResult(run_id=run_id, cost=cost, rows=len(underlying_df) + len(options_df))
```

#### 3.2.6 Data Quality Checks

**DataQualityValidator**
```python
class DataQualityValidator:
    """Validate ingested data meets quality standards."""

    @staticmethod
    def check_underlying_bars(df: pd.DataFrame) -> ValidationResult:
        """Check bars for issues."""
        issues = []

        # Check for nulls
        if df.isnull().any().any():
            issues.append("Null values found in bars")

        # Check timestamp ordering
        if not df["ts_utc"].is_monotonic_increasing:
            issues.append("Timestamps not monotonic")

        # Check for duplicates
        if df.duplicated(subset=["ts_utc", "symbol"]).any():
            issues.append("Duplicate bars found")

        # Check OHLC relationships
        invalid_ohlc = (
            (df["high"] < df["low"]) |
            (df["high"] < df["open"]) |
            (df["high"] < df["close"]) |
            (df["low"] > df["open"]) |
            (df["low"] > df["close"])
        )
        if invalid_ohlc.any():
            issues.append(f"Invalid OHLC relationships in {invalid_ohlc.sum()} rows")

        return ValidationResult(passed=len(issues) == 0, issues=issues)
```

---

### 3.3 Storage & Persistence

#### 3.3.1 Design Goals
- Immutable raw data (append-only)
- Versioned derived data (reproducible)
- Hybrid SQL + Parquet (SQL for queries, Parquet for bulk)
- Clear separation: raw vs derived
- Audit trail (ingestion logs, build versions)

#### 3.3.2 Raw Tables (SQLite)

**raw_underlying_bars**
```sql
CREATE TABLE raw_underlying_bars (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset TEXT NOT NULL,
    schema TEXT NOT NULL,
    stype_in TEXT NOT NULL,
    instrument_id INTEGER,
    publisher_id INTEGER,
    ts_utc TIMESTAMP NOT NULL,
    ts_recv_utc TIMESTAMP,
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    open REAL NOT NULL,
    high REAL NOT NULL,
    low REAL NOT NULL,
    close REAL NOT NULL,
    volume INTEGER,
    source_ingested_at TIMESTAMP NOT NULL,
    ingest_run_id TEXT NOT NULL,
    UNIQUE(symbol, ts_utc, ingest_run_id)
);
CREATE INDEX idx_bars_ts ON raw_underlying_bars(ts_utc);
CREATE INDEX idx_bars_symbol ON raw_underlying_bars(symbol);
```

**raw_option_quotes**
```sql
CREATE TABLE raw_option_quotes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset TEXT NOT NULL,
    schema TEXT NOT NULL,
    stype_in TEXT NOT NULL,
    instrument_id INTEGER,
    publisher_id INTEGER,
    ts_utc TIMESTAMP NOT NULL,
    ts_recv_utc TIMESTAMP,
    option_symbol TEXT NOT NULL,
    exp_date DATE NOT NULL,
    strike REAL NOT NULL,
    right TEXT NOT NULL,  -- 'C' or 'P'
    bid REAL,
    ask REAL,
    bid_size INTEGER,
    ask_size INTEGER,
    volume INTEGER,
    open_interest INTEGER,
    source_ingested_at TIMESTAMP NOT NULL,
    ingest_run_id TEXT NOT NULL,
    UNIQUE(option_symbol, ts_utc, ingest_run_id)
);
CREATE INDEX idx_quotes_ts ON raw_option_quotes(ts_utc);
CREATE INDEX idx_quotes_exp ON raw_option_quotes(exp_date);
```

**raw_fred_series**
```sql
CREATE TABLE raw_fred_series (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    series_id TEXT NOT NULL,
    obs_date DATE NOT NULL,
    value REAL NOT NULL,
    release_datetime_utc TIMESTAMP,  -- NULL if unknown
    source_ingested_at TIMESTAMP NOT NULL,
    UNIQUE(series_id, obs_date)
);
CREATE INDEX idx_fred_series ON raw_fred_series(series_id, obs_date);
```

**raw_ingestion_log**
```sql
CREATE TABLE raw_ingestion_log (
    ingest_run_id TEXT PRIMARY KEY,
    dataset TEXT NOT NULL,
    schema TEXT NOT NULL,
    stype_in TEXT NOT NULL,
    symbols TEXT NOT NULL,  -- JSON array
    start_date DATE NOT NULL,
    end_date DATE NOT NULL,
    row_count INTEGER NOT NULL,
    min_ts_event_utc TIMESTAMP,
    max_ts_event_utc TIMESTAMP,
    min_ts_recv_utc TIMESTAMP,
    max_ts_recv_utc TIMESTAMP,
    cost_usd REAL,
    git_sha TEXT,
    config_snapshot TEXT,  -- JSON
    source_ingested_at TIMESTAMP NOT NULL
);
```

#### 3.3.3 Derived Tables (SQLite)

**surface_snapshots**
```sql
CREATE TABLE surface_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_utc TIMESTAMP NOT NULL,
    option_symbol TEXT NOT NULL,
    exp_date DATE NOT NULL,
    strike REAL NOT NULL,
    right TEXT NOT NULL,

    -- Pricing
    bid REAL NOT NULL,
    ask REAL NOT NULL,
    mid_price REAL NOT NULL,
    spread REAL NOT NULL,
    spread_pct REAL NOT NULL,

    -- Time
    tte_years REAL NOT NULL,
    tenor_days INTEGER NOT NULL,

    -- Market data
    underlying_price REAL NOT NULL,
    rf_rate REAL NOT NULL,
    dividend_yield REAL NOT NULL,

    -- IV
    iv_mid REAL NOT NULL,
    iv_bid REAL,
    iv_ask REAL,

    -- Greeks
    delta REAL NOT NULL,
    gamma REAL NOT NULL,
    vega REAL NOT NULL,
    theta REAL NOT NULL,

    -- Bucket assignment
    delta_bucket TEXT,  -- 'P10', 'P25', etc. (NULL if not assigned)

    -- Quality
    flags INTEGER NOT NULL,  -- Bitfield: 1=crossed, 2=stale, 4=wide_spread, etc.
    volume INTEGER,
    open_interest INTEGER,

    -- Metadata
    snapshot_version TEXT NOT NULL,
    build_run_id TEXT NOT NULL,
    source_created_at TIMESTAMP NOT NULL,

    UNIQUE(ts_utc, option_symbol, snapshot_version)
);
CREATE INDEX idx_snapshots_ts_bucket ON surface_snapshots(ts_utc, delta_bucket, tenor_days);
CREATE INDEX idx_snapshots_version ON surface_snapshots(snapshot_version);
```

**node_panel**
```sql
CREATE TABLE node_panel (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_utc TIMESTAMP NOT NULL,
    tenor_days INTEGER NOT NULL,
    delta_bucket TEXT NOT NULL,

    -- Representative option (may be NULL if node masked)
    option_symbol TEXT,

    -- Node features
    iv_mid REAL,
    iv_bid REAL,
    iv_ask REAL,
    spread_pct REAL,
    delta REAL,
    gamma REAL,
    vega REAL,
    theta REAL,

    -- Derived features (examples)
    iv_change_1d REAL,
    iv_change_5d REAL,
    skew_slope REAL,
    term_slope REAL,
    curvature REAL,
    oi_change_5d REAL,
    volume_ratio REAL,

    -- Global features (denormalized for convenience)
    underlying_rv_5d REAL,
    underlying_rv_10d REAL,
    underlying_rv_21d REAL,

    -- Macro features (denormalized)
    sofr_level REAL,
    sofr_change_1w REAL,
    -- ... other macro features

    -- Metadata
    feature_version TEXT NOT NULL,
    is_masked BOOLEAN NOT NULL,
    mask_reason TEXT,

    UNIQUE(ts_utc, tenor_days, delta_bucket, feature_version)
);
CREATE INDEX idx_panel_ts ON node_panel(ts_utc);
CREATE INDEX idx_panel_node ON node_panel(tenor_days, delta_bucket);
```

#### 3.3.4 Repository Pattern

**RawRepository**
```python
class RawRepository:
    """Data access layer for raw tables."""

    def __init__(self, engine: Engine):
        self._engine = engine

    def write_underlying_bars(self, df: pd.DataFrame, run_id: str) -> None:
        """Write underlying bars to raw table."""
        df["ingest_run_id"] = run_id
        df["source_ingested_at"] = datetime.now(timezone.utc)
        df.to_sql("raw_underlying_bars", self._engine, if_exists="append", index=False)

    def read_underlying_bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        """Read underlying bars for date range."""
        query = """
            SELECT * FROM raw_underlying_bars
            WHERE symbol = :symbol
              AND ts_utc >= :start
              AND ts_utc < :end
            ORDER BY ts_utc
        """
        return pd.read_sql(query, self._engine, params={"symbol": symbol, "start": start, "end": end})

    # Similar methods for option_quotes, fred_series, ingestion_log
```

**DerivedRepository**
```python
class DerivedRepository:
    """Data access layer for derived tables."""

    def __init__(self, engine: Engine):
        self._engine = engine

    def write_surface_snapshots(self, df: pd.DataFrame, build_run_id: str, version: str) -> None:
        """Write surface snapshots with version."""
        df["build_run_id"] = build_run_id
        df["snapshot_version"] = version
        df["source_created_at"] = datetime.now(timezone.utc)
        df.to_sql("surface_snapshots", self._engine, if_exists="append", index=False)

    def read_surface_snapshots(
        self,
        start: datetime,
        end: datetime,
        version: str,
        delta_buckets: list[str] | None = None,
    ) -> pd.DataFrame:
        """Read surface snapshots for analysis."""
        query = """
            SELECT * FROM surface_snapshots
            WHERE ts_utc >= :start
              AND ts_utc < :end
              AND snapshot_version = :version
        """
        if delta_buckets:
            query += f" AND delta_bucket IN ({','.join('?' * len(delta_buckets))})"

        return pd.read_sql(query, self._engine, params={...})

    # Similar methods for node_panel
```

#### 3.3.5 Parquet Archival

For bulk export and long-term storage, periodically export derived tables to Parquet:

```python
class ParquetWriter:
    """Export derived data to Parquet for archival."""

    def __init__(self, root: Path):
        self._root = root

    def export_surface_snapshots(
        self,
        repo: DerivedRepository,
        start: datetime,
        end: datetime,
        version: str,
    ) -> Path:
        """Export surface snapshots to Parquet."""
        df = repo.read_surface_snapshots(start, end, version)

        # Partition by year-month
        output_path = self._root / "surface_snapshots" / version / f"{start.year}-{start.month:02d}.parquet"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        df.to_parquet(output_path, engine="pyarrow", compression="snappy")
        return output_path
```

---

### 3.4 Surface Engine

#### 3.4.1 Design Goals
- Deterministic surface construction (same inputs → same outputs)
- Explicit stages: IV inversion → Greeks → Bucket assignment → Quality filtering
- Vectorized operations (avoid row-by-row iteration)
- Comprehensive quality flags (crossed, stale, wide spread, low volume)
- Auditable (log which contracts are selected for each node)

#### 3.4.2 Package Structure
```
src/surface/
├── __init__.py
├── builder.py            # Orchestrates surface construction pipeline
├── iv/
│   ├── __init__.py
│   ├── black_scholes.py  # BS IV inversion (Newton-Raphson)
│   └── solver.py         # Numerical solver utilities
├── greeks/
│   ├── __init__.py
│   └── analytical.py     # Analytical Greeks (BS formulas)
├── buckets/
│   ├── __init__.py
│   └── assign.py         # Delta bucket assignment logic
├── quality/
│   ├── __init__.py
│   └── filters.py        # Quality checks and flagging
└── snapshots.py          # Snapshot data structures
```

#### 3.4.3 Surface Builder

**SurfaceBuilder**
```python
class SurfaceBuilder:
    """Orchestrates surface construction from raw quotes."""

    def __init__(
        self,
        config: SurfaceConfig,
        raw_repo: RawRepository,
        derived_repo: DerivedRepository,
    ):
        self._config = config
        self._raw_repo = raw_repo
        self._derived_repo = derived_repo

        self._iv_solver = BlackScholesIVSolver()
        self._greeks_calculator = AnalyticalGreeks()
        self._bucket_assigner = DeltaBucketAssigner(config.delta_buckets)
        self._quality_filter = QualityFilter(config.quality)

    def build_surface(
        self,
        start: datetime,
        end: datetime,
        version: str,
    ) -> BuildResult:
        """Build surface snapshots for date range."""

        build_run_id = self._generate_build_id()

        # 1. Load raw data
        quotes_df = self._raw_repo.read_option_quotes(start, end)
        underlying_df = self._raw_repo.read_underlying_bars(
            symbol=self._config.universe.underlying,
            start=start,
            end=end,
        )
        fred_df = self._raw_repo.read_fred_series(
            series_id=self._config.surface.pricing.rate_proxy.series,
            start=start - timedelta(days=365),  # Buffer for lookback
            end=end,
        )

        # 2. Join underlying price (time-aware)
        quotes_df = self._join_underlying_price(quotes_df, underlying_df)

        # 3. Join risk-free rate (time-aware, release-aligned)
        quotes_df = self._join_risk_free_rate(quotes_df, fred_df)

        # 4. Compute time-to-expiry
        quotes_df["tte_years"] = self._compute_tte(quotes_df)

        # 5. Compute mid price and spread
        quotes_df["mid_price"] = (quotes_df["bid"] + quotes_df["ask"]) / 2
        quotes_df["spread"] = quotes_df["ask"] - quotes_df["bid"]
        quotes_df["spread_pct"] = quotes_df["spread"] / quotes_df["mid_price"]

        # 6. Invert IV (vectorized)
        quotes_df["iv_mid"] = self._iv_solver.solve_iv_vectorized(
            prices=quotes_df["mid_price"],
            S=quotes_df["underlying_price"],
            K=quotes_df["strike"],
            T=quotes_df["tte_years"],
            r=quotes_df["rf_rate"],
            q=quotes_df.get("dividend_yield", 0.0),
            right=quotes_df["right"],
        )

        # Similarly for iv_bid, iv_ask

        # 7. Compute Greeks (vectorized)
        greeks = self._greeks_calculator.compute_greeks_vectorized(
            S=quotes_df["underlying_price"],
            K=quotes_df["strike"],
            T=quotes_df["tte_years"],
            r=quotes_df["rf_rate"],
            q=quotes_df.get("dividend_yield", 0.0),
            sigma=quotes_df["iv_mid"],
            right=quotes_df["right"],
        )
        quotes_df = pd.concat([quotes_df, greeks], axis=1)

        # 8. Assign tenor bins
        quotes_df["tenor_days"] = self._assign_tenor_bins(quotes_df)

        # 9. Assign delta buckets
        quotes_df["delta_bucket"] = self._bucket_assigner.assign(quotes_df["delta"])

        # 10. Apply quality filters and flag
        quotes_df["flags"] = self._quality_filter.compute_flags(quotes_df)

        # 11. Select representative option per node
        snapshots_df = self._select_representatives(quotes_df)

        # 12. Write to derived storage
        self._derived_repo.write_surface_snapshots(snapshots_df, build_run_id, version)

        return BuildResult(
            build_run_id=build_run_id,
            version=version,
            row_count=len(snapshots_df),
            start=start,
            end=end,
        )

    def _join_underlying_price(self, quotes_df: pd.DataFrame, underlying_df: pd.DataFrame) -> pd.DataFrame:
        """Join most recent underlying close for each quote timestamp."""
        # Implementation: merge_asof with direction='backward'
        ...

    def _join_risk_free_rate(self, quotes_df: pd.DataFrame, fred_df: pd.DataFrame) -> pd.DataFrame:
        """Join risk-free rate respecting release timestamps."""
        # Implementation: merge_asof with release_datetime_utc <= ts_utc
        ...

    def _compute_tte(self, df: pd.DataFrame) -> pd.Series:
        """Compute time-to-expiry in years using ACT/365."""
        # expiry_ts = exp_date at close time in project timezone → UTC
        # tte_years = (expiry_ts - ts_utc) / 365 days
        ...

    def _assign_tenor_bins(self, df: pd.DataFrame) -> pd.Series:
        """Assign each option to nearest tenor bin."""
        dte_days = (df["exp_date"] - df["ts_utc"].dt.date).dt.days
        tenor_bins = self._config.surface.tenors_days

        # Find nearest tenor within max_days_to_tenor threshold
        ...

    def _select_representatives(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select one option per (ts_utc, tenor_days, delta_bucket)."""
        # Group by (ts_utc, tenor_days, delta_bucket)
        # Select option with delta closest to bucket center
        # Exclude options with disqualifying flags (if configured)
        ...
```

#### 3.4.4 IV Solver

**BlackScholesIVSolver**
```python
class BlackScholesIVSolver:
    """Black-Scholes implied volatility solver using Newton-Raphson."""

    def __init__(self, max_iters: int = 100, tolerance: float = 1e-6):
        self._max_iters = max_iters
        self._tolerance = tolerance

    def solve_iv_vectorized(
        self,
        prices: pd.Series,
        S: pd.Series,
        K: pd.Series,
        T: pd.Series,
        r: pd.Series,
        q: pd.Series | float,
        right: pd.Series,
    ) -> pd.Series:
        """Solve for IV using vectorized Newton-Raphson."""

        # Initial guess (Brenner-Subrahmanyam approximation)
        iv_guess = np.sqrt(2 * np.pi / T) * (prices / S)

        for _ in range(self._max_iters):
            bs_price = self._black_scholes_price(S, K, T, r, q, iv_guess, right)
            vega = self._vega(S, K, T, r, q, iv_guess)

            diff = bs_price - prices
            iv_guess = iv_guess - diff / (vega + 1e-10)  # Avoid division by zero

            if np.abs(diff).max() < self._tolerance:
                break

        # Mask failed convergence
        iv_guess = np.where(np.isfinite(iv_guess) & (iv_guess > 0), iv_guess, np.nan)

        return pd.Series(iv_guess, index=prices.index)

    def _black_scholes_price(self, S, K, T, r, q, sigma, right):
        """Vectorized BS pricing."""
        ...

    def _vega(self, S, K, T, r, q, sigma):
        """Vectorized vega."""
        ...
```

#### 3.4.5 Greeks Calculator

**AnalyticalGreeks**
```python
class AnalyticalGreeks:
    """Analytical Greeks using Black-Scholes formulas."""

    def compute_greeks_vectorized(
        self,
        S: pd.Series,
        K: pd.Series,
        T: pd.Series,
        r: pd.Series,
        q: pd.Series | float,
        sigma: pd.Series,
        right: pd.Series,
    ) -> pd.DataFrame:
        """Compute all Greeks in one pass."""

        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T) + 1e-10)
        d2 = d1 - sigma * np.sqrt(T)

        is_call = (right == "C")

        # Delta
        delta = np.where(
            is_call,
            np.exp(-q * T) * norm.cdf(d1),
            -np.exp(-q * T) * norm.cdf(-d1),
        )

        # Gamma (same for calls and puts)
        gamma = np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T) + 1e-10)

        # Vega (same for calls and puts, per 1.0 vol change)
        vega = S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)

        # Theta (per year)
        theta_call = (
            -S * norm.pdf(d1) * sigma * np.exp(-q * T) / (2 * np.sqrt(T))
            - r * K * np.exp(-r * T) * norm.cdf(d2)
            + q * S * np.exp(-q * T) * norm.cdf(d1)
        )
        theta_put = (
            -S * norm.pdf(d1) * sigma * np.exp(-q * T) / (2 * np.sqrt(T))
            + r * K * np.exp(-r * T) * norm.cdf(-d2)
            - q * S * np.exp(-q * T) * norm.cdf(-d1)
        )
        theta = np.where(is_call, theta_call, theta_put)

        return pd.DataFrame({
            "delta": delta,
            "gamma": gamma,
            "vega": vega,
            "theta": theta,
        }, index=S.index)
```

#### 3.4.6 Bucket Assignment

**DeltaBucketAssigner**
```python
class DeltaBucketAssigner:
    """Assign options to delta buckets."""

    def __init__(self, bucket_config: dict[str, list[float]]):
        self._buckets = self._parse_bucket_config(bucket_config)

    def _parse_bucket_config(self, config: dict[str, list[float]]) -> list[tuple[str, float, float]]:
        """Parse bucket config into (name, min, max) tuples."""
        buckets = []
        for name, bounds in config.items():
            if name == "ATM":
                # ATM encoded as 4 numbers: [-0.55, -0.45, 0.45, 0.55]
                # Means |delta| in [0.45, 0.55]
                buckets.append((name, bounds[2], bounds[3]))  # Positive side
            else:
                buckets.append((name, bounds[0], bounds[1]))
        return buckets

    def assign(self, deltas: pd.Series) -> pd.Series:
        """Assign each delta to a bucket (or None)."""

        bucket_names = pd.Series([None] * len(deltas), index=deltas.index, dtype=object)

        for name, min_delta, max_delta in self._buckets:
            if name == "ATM":
                # ATM: |delta| in range
                mask = (np.abs(deltas) >= min_delta) & (np.abs(deltas) <= max_delta)
            else:
                # Directional buckets
                mask = (deltas >= min_delta) & (deltas <= max_delta)

            bucket_names[mask] = name

        return bucket_names
```

#### 3.4.7 Quality Filtering

**QualityFilter**
```python
class QualityFilter:
    """Compute quality flags for options."""

    # Flag constants
    FLAG_CROSSED = 1 << 0
    FLAG_STALE = 1 << 1
    FLAG_WIDE_SPREAD = 1 << 2
    FLAG_LOW_VOLUME = 1 << 3
    FLAG_LOW_OI = 1 << 4

    def __init__(self, quality_config: QualityConfig):
        self._config = quality_config

    def compute_flags(self, df: pd.DataFrame) -> pd.Series:
        """Compute quality flags for each option."""

        flags = pd.Series(0, index=df.index, dtype=int)

        # Crossed quotes
        if not self._config.allow_crossed_quotes:
            crossed = df["ask"] < df["bid"]
            flags[crossed] |= self.FLAG_CROSSED

        # Stale quotes (using ts_utc, not ts_recv_utc)
        staleness_days = (df["ts_utc"].dt.date - df["ts_utc"].dt.date).dt.days  # Simplified
        stale = staleness_days > self._config.eod_max_staleness_days
        flags[stale] |= self.FLAG_STALE

        # Wide spread
        wide = df["spread_pct"] > self._config.max_spread_pct
        flags[wide] |= self.FLAG_WIDE_SPREAD

        # Low volume
        if self._config.min_volume:
            low_vol = df["volume"].fillna(0) < self._config.min_volume
            flags[low_vol] |= self.FLAG_LOW_VOLUME

        # Low open interest
        if self._config.min_open_interest:
            low_oi = df["open_interest"].fillna(0) < self._config.min_open_interest
            flags[low_oi] |= self.FLAG_LOW_OI

        return flags
```

---

### 3.5 Feature Engineering

#### 3.5.1 Design Goals
- Strict time integrity (no future data leakage)
- Composable feature generators (mix and match)
- Vectorized operations (avoid loops)
- Macro features respect release timestamps
- Feature versioning (reproducibility)

#### 3.5.2 Package Structure
```
src/features/
├── __init__.py
├── engine.py             # Orchestrates feature generation
├── node/
│   ├── __init__.py
│   ├── iv_features.py    # IV dynamics (changes, volatility)
│   ├── microstructure.py # Spread, quote stability
│   └── surface.py        # Skew slope, term slope, curvature
├── global_/
│   ├── __init__.py
│   ├── returns.py        # Underlying returns
│   └── realized_vol.py   # Realized variance windows
├── macro/
│   ├── __init__.py
│   ├── transforms.py     # Level, change, z-score
│   └── alignment.py      # Release-time alignment
└── validators.py         # Anti-leakage checks
```

#### 3.5.3 Feature Engine

**FeatureEngine**
```python
class FeatureEngine:
    """Orchestrates feature generation across all feature families."""

    def __init__(
        self,
        config: FeaturesConfig,
        raw_repo: RawRepository,
        derived_repo: DerivedRepository,
    ):
        self._config = config
        self._raw_repo = raw_repo
        self._derived_repo = derived_repo

        # Initialize feature generators
        self._node_generator = NodeFeatureGenerator(config.node)
        self._global_generator = GlobalFeatureGenerator(config.global_underlying)
        self._macro_generator = MacroFeatureGenerator(config.macro, raw_repo)

    def build_feature_panel(
        self,
        start: datetime,
        end: datetime,
        surface_version: str,
        feature_version: str,
    ) -> FeatureResult:
        """Build node_panel with all features."""

        # 1. Load surface snapshots
        surface_df = self._derived_repo.read_surface_snapshots(
            start=start - timedelta(days=self._config.lookback_buffer),
            end=end,
            version=surface_version,
        )

        # 2. Load underlying bars for global features
        underlying_df = self._raw_repo.read_underlying_bars(
            symbol=self._config.universe.underlying,
            start=start - timedelta(days=self._config.lookback_buffer),
            end=end,
        )

        # 3. Generate node-level features
        node_features_df = self._node_generator.generate(surface_df)

        # 4. Generate global features
        global_features_df = self._global_generator.generate(underlying_df)

        # 5. Generate macro features
        macro_features_df = self._macro_generator.generate(start, end)

        # 6. Merge all features into node panel
        panel_df = self._merge_features(
            surface_df,
            node_features_df,
            global_features_df,
            macro_features_df,
        )

        # 7. Run anti-leakage validation
        self._validate_no_leakage(panel_df)

        # 8. Write to storage
        self._derived_repo.write_node_panel(panel_df, feature_version)

        return FeatureResult(feature_version=feature_version, row_count=len(panel_df))
```

#### 3.5.4 Node Features

**NodeFeatureGenerator**
```python
class NodeFeatureGenerator:
    """Generate node-level features from surface snapshots."""

    def __init__(self, config: NodeFeatureConfig):
        self._config = config

    def generate(self, surface_df: pd.DataFrame) -> pd.DataFrame:
        """Generate all node features."""

        # Sort by (tenor, delta_bucket, ts_utc) for rolling windows
        surface_df = surface_df.sort_values(["tenor_days", "delta_bucket", "ts_utc"])

        features = []

        # Group by node (tenor, delta_bucket)
        for (tenor, bucket), group in surface_df.groupby(["tenor_days", "delta_bucket"]):
            node_features = self._compute_node_features(group)
            features.append(node_features)

        return pd.concat(features, ignore_index=True)

    def _compute_node_features(self, node_df: pd.DataFrame) -> pd.DataFrame:
        """Compute features for a single node time series."""

        # IV changes
        node_df["iv_change_1d"] = node_df["iv_mid"].diff(1)
        node_df["iv_change_5d"] = node_df["iv_mid"].diff(5)

        # IV volatility (rolling std)
        for window in self._config.rolling_windows_days:
            node_df[f"iv_vol_{window}d"] = (
                node_df["iv_mid"].rolling(window=window, min_periods=1).std()
            )

        # Spread dynamics
        node_df["spread_pct_ma_5d"] = node_df["spread_pct"].rolling(window=5).mean()

        # OI/volume changes
        node_df["oi_change_5d"] = node_df["open_interest"].pct_change(5)
        node_df["volume_ratio"] = node_df["volume"] / node_df["volume"].rolling(window=5).mean()

        # Skew slope (requires adjacent delta bucket data - defer to cross-sectional pass)
        # Term slope (requires adjacent tenor data - defer to cross-sectional pass)

        return node_df
```

**Surface Features (Cross-Sectional)**
```python
class SurfaceFeatureGenerator:
    """Generate features requiring cross-sectional surface structure."""

    def compute_skew_slope(self, surface_df: pd.DataFrame) -> pd.Series:
        """Compute skew slope (delta-adjacent IV gradient)."""

        # For each (ts_utc, tenor), compute slope across delta buckets
        slopes = []

        for (ts, tenor), group in surface_df.groupby(["ts_utc", "tenor_days"]):
            group = group.sort_values("delta")  # Sort by delta

            # Simple linear regression: IV ~ delta
            if len(group) >= 3:
                slope = np.polyfit(group["delta"], group["iv_mid"], deg=1)[0]
            else:
                slope = np.nan

            slopes.append(pd.DataFrame({
                "ts_utc": [ts] * len(group),
                "tenor_days": [tenor] * len(group),
                "delta_bucket": group["delta_bucket"],
                "skew_slope": [slope] * len(group),
            }))

        return pd.concat(slopes, ignore_index=True)

    def compute_term_slope(self, surface_df: pd.DataFrame) -> pd.Series:
        """Compute term slope (tenor-adjacent IV gradient)."""

        # For each (ts_utc, delta_bucket), compute slope across tenors
        ...
```

#### 3.5.5 Global Features

**GlobalFeatureGenerator**
```python
class GlobalFeatureGenerator:
    """Generate global underlying features."""

    def __init__(self, config: GlobalUnderlyingConfig):
        self._config = config

    def generate(self, underlying_df: pd.DataFrame) -> pd.DataFrame:
        """Generate global features from underlying bars."""

        underlying_df = underlying_df.sort_values("ts_utc")

        # Returns
        underlying_df["returns_1d"] = underlying_df["close"].pct_change(1)
        underlying_df["returns_5d"] = underlying_df["close"].pct_change(5)

        # Realized variance (close-to-close)
        for window in self._config.realized_var_windows_days:
            underlying_df[f"rv_{window}d"] = (
                underlying_df["returns_1d"]
                .rolling(window=window)
                .var()
                * 252  # Annualize
            )

        # Vol-of-vol
        underlying_df["vol_of_vol_21d"] = (
            underlying_df["returns_1d"]
            .rolling(window=21)
            .std()
            .rolling(window=21)
            .std()
        )

        # Drawdown
        underlying_df["drawdown"] = (
            underlying_df["close"] / underlying_df["close"].rolling(window=252).max() - 1
        )

        return underlying_df[["ts_utc", "returns_1d", "rv_5d", "rv_10d", "rv_21d", ...]]
```

#### 3.5.6 Macro Features

**MacroFeatureGenerator**
```python
class MacroFeatureGenerator:
    """Generate macro features with release-time alignment."""

    def __init__(self, config: MacroConfig, raw_repo: RawRepository):
        self._config = config
        self._raw_repo = raw_repo

    def generate(self, start: datetime, end: datetime) -> pd.DataFrame:
        """Generate macro features respecting release times."""

        macro_features = []

        for series_id in self._config.series:
            # Load FRED series
            fred_df = self._raw_repo.read_fred_series(
                series_id=series_id,
                start=start - timedelta(days=365),  # Buffer for lookback
                end=end,
            )

            # Convert percent to decimal
            fred_df["value"] = fred_df["value"] / 100.0

            # Generate transforms
            if self._config.transforms.include_level:
                macro_features.append(self._generate_level_feature(fred_df, series_id))

            if self._config.transforms.include_change_1w:
                macro_features.append(self._generate_change_feature(fred_df, series_id, "1w"))

            if self._config.transforms.include_change_1m:
                macro_features.append(self._generate_change_feature(fred_df, series_id, "1m"))

            if self._config.transforms.include_zscore:
                macro_features.append(self._generate_zscore_feature(fred_df, series_id))

        # Merge all macro features
        result = macro_features[0]
        for df in macro_features[1:]:
            result = result.merge(df, on="ts_utc", how="outer")

        return result.sort_values("ts_utc")

    def _generate_level_feature(self, fred_df: pd.DataFrame, series_id: str) -> pd.DataFrame:
        """Generate level feature with release-time alignment."""

        if self._config.release_alignment.mode == "strict":
            # Use release_datetime_utc if available
            fred_df = fred_df[fred_df["release_datetime_utc"].notna()]
            time_col = "release_datetime_utc"
        else:
            # Conservative: shift by delay days
            fred_df["effective_time"] = (
                fred_df["obs_date"] + timedelta(days=self._config.release_alignment.conservative_delay_days)
            )
            time_col = "effective_time"

        return fred_df[[time_col, "value"]].rename(columns={
            time_col: "ts_utc",
            "value": f"{series_id}_level",
        })

    def _generate_change_feature(self, fred_df: pd.DataFrame, series_id: str, period: str) -> pd.DataFrame:
        """Generate change feature."""

        if period == "1w":
            lag = 7
        elif period == "1m":
            lag = 30
        else:
            raise ValueError(f"Unknown period: {period}")

        fred_df[f"{series_id}_change_{period}"] = fred_df["value"].diff(lag)

        return fred_df[["ts_utc", f"{series_id}_change_{period}"]]

    def _generate_zscore_feature(self, fred_df: pd.DataFrame, series_id: str) -> pd.DataFrame:
        """Generate rolling z-score feature."""

        window = 252  # 1 year rolling
        fred_df[f"{series_id}_zscore"] = (
            (fred_df["value"] - fred_df["value"].rolling(window).mean()) /
            fred_df["value"].rolling(window).std()
        )

        return fred_df[["ts_utc", f"{series_id}_zscore"]]
```

#### 3.5.7 Anti-Leakage Validation

**FeatureValidator**
```python
class FeatureValidator:
    """Validate features for data leakage."""

    @staticmethod
    def validate_no_future_leakage(panel_df: pd.DataFrame) -> ValidationResult:
        """Check that features only use past data."""

        issues = []

        # Check that all feature values at time t are computable from data <= t
        # (This is a conceptual check - actual implementation would inspect feature logic)

        # Example: Check that rolling windows don't exceed available history
        for col in panel_df.columns:
            if "change" in col or "ma" in col or "vol" in col:
                # Verify first N rows are NaN where N = window size
                # (Simplified check)
                pass

        # Check that macro features respect release times
        # (Would require cross-checking with raw_fred_series.release_datetime_utc)

        return ValidationResult(passed=len(issues) == 0, issues=issues)
```

---

### 3.6 ML Pipeline

#### 3.6.1 Design Goals
- Scikit-learn-style API (fit/predict)
- Support for temporal models (PatchTST) and graph models (GNN)
- Modular architecture (combine PatchTST + GNN)
- Multi-horizon prediction (5d, 10d, 21d)
- Uncertainty quantification (quantile regression)
- Reproducible training (seed control)

#### 3.6.2 Package Structure
```
src/models/
├── __init__.py
├── base.py               # Abstract base classes
├── patchtst/
│   ├── __init__.py
│   ├── model.py          # PatchTST implementation
│   └── encoder.py        # Patch embedding logic
├── gnn/
│   ├── __init__.py
│   ├── model.py          # GNN wrapper (GAT/GCN)
│   ├── graph.py          # Graph construction from surface spec
│   └── layers.py         # Custom GNN layers
├── ensemble.py           # PatchTST + GNN ensemble
├── train/
│   ├── __init__.py
│   ├── trainer.py        # Training loop
│   ├── optimizer.py      # Optimizer/scheduler setup
│   └── loss.py           # Custom loss functions
└── eval/
    ├── __init__.py
    ├── metrics.py        # IC, Rank IC, etc.
    └── inference.py      # Batch inference utilities
```

#### 3.6.3 Base Classes

**TimeSeriesModel** (Abstract)
```python
from abc import ABC, abstractmethod

class TimeSeriesModel(ABC):
    """Abstract base for time series models."""

    @abstractmethod
    def fit(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        X_val: torch.Tensor | None = None,
        y_val: torch.Tensor | None = None,
    ) -> TrainResult:
        """Train the model."""
        ...

    @abstractmethod
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Generate predictions."""
        ...

    @abstractmethod
    def save(self, path: Path) -> None:
        """Save model checkpoint."""
        ...

    @abstractmethod
    def load(self, path: Path) -> None:
        """Load model checkpoint."""
        ...
```

#### 3.6.4 Dataset Builder

**SurfaceDataset** (PyTorch Dataset)
```python
class SurfaceDataset(Dataset):
    """PyTorch dataset for surface node panel."""

    def __init__(
        self,
        panel_df: pd.DataFrame,
        feature_cols: list[str],
        label_cols: list[str],
        lookback_days: int,
        graph: torch_geometric.data.Data | None = None,
    ):
        self._panel = panel_df.sort_values(["ts_utc", "tenor_days", "delta_bucket"])
        self._feature_cols = feature_cols
        self._label_cols = label_cols
        self._lookback_days = lookback_days
        self._graph = graph

        # Build index: (timestamp, lookback_window)
        self._samples = self._build_sample_index()

    def _build_sample_index(self) -> list[tuple[datetime, datetime]]:
        """Build list of (start_ts, end_ts) for each sample."""

        unique_timestamps = sorted(self._panel["ts_utc"].unique())
        samples = []

        for i in range(self._lookback_days, len(unique_timestamps)):
            end_ts = unique_timestamps[i]
            start_ts = unique_timestamps[i - self._lookback_days]
            samples.append((start_ts, end_ts))

        return samples

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Get a single sample."""

        start_ts, end_ts = self._samples[idx]

        # Extract window
        window_df = self._panel[
            (self._panel["ts_utc"] >= start_ts) & (self._panel["ts_utc"] <= end_ts)
        ]

        # Features: shape (lookback_days, num_nodes, num_features)
        X = self._extract_features(window_df)

        # Labels: shape (num_nodes, num_horizons)
        y = self._extract_labels(window_df.query(f"ts_utc == '{end_ts}'"))

        # Mask: shape (num_nodes,) - indicates which nodes are valid
        mask = self._extract_mask(window_df.query(f"ts_utc == '{end_ts}'"))

        return {
            "X": torch.tensor(X, dtype=torch.float32),
            "y": torch.tensor(y, dtype=torch.float32),
            "mask": torch.tensor(mask, dtype=torch.bool),
            "graph": self._graph,  # Static graph (same for all samples)
        }

    def _extract_features(self, window_df: pd.DataFrame) -> np.ndarray:
        """Extract feature tensor from window."""

        # Reshape to (time, nodes, features)
        # Handle missing nodes by padding/masking
        ...

    def _extract_labels(self, snapshot_df: pd.DataFrame) -> np.ndarray:
        """Extract label tensor."""
        ...

    def _extract_mask(self, snapshot_df: pd.DataFrame) -> np.ndarray:
        """Extract mask tensor."""
        ...
```

**DatasetBuilder**
```python
class DatasetBuilder:
    """Build train/val/test datasets with walk-forward splits."""

    def __init__(
        self,
        config: DatasetConfig,
        derived_repo: DerivedRepository,
    ):
        self._config = config
        self._repo = derived_repo

    def build_datasets(
        self,
        feature_version: str,
        label_config: LabelsConfig,
    ) -> tuple[SurfaceDataset, SurfaceDataset, SurfaceDataset]:
        """Build train, val, test datasets."""

        # Load node panel
        panel_df = self._repo.read_node_panel(feature_version=feature_version)

        # Build labels (if not already in panel)
        panel_df = self._build_labels(panel_df, label_config)

        # Split by date
        train_df = panel_df[panel_df["ts_utc"] < self._config.splits.val_start]
        val_df = panel_df[
            (panel_df["ts_utc"] >= self._config.splits.val_start) &
            (panel_df["ts_utc"] < self._config.splits.test_start)
        ]
        test_df = panel_df[panel_df["ts_utc"] >= self._config.splits.test_start]

        # Build graph (static)
        graph = self._build_graph()

        # Create datasets
        train_ds = SurfaceDataset(train_df, self._feature_cols, self._label_cols, self._config.lookback_days, graph)
        val_ds = SurfaceDataset(val_df, self._feature_cols, self._label_cols, self._config.lookback_days, graph)
        test_ds = SurfaceDataset(test_df, self._feature_cols, self._label_cols, self._config.lookback_days, graph)

        return train_ds, val_ds, test_ds

    def _build_labels(self, panel_df: pd.DataFrame, label_config: LabelsConfig) -> pd.DataFrame:
        """Compute forward-looking labels."""

        # For each horizon H, compute future realized variance
        for H in label_config.horizons_days:
            panel_df[f"rv_{H}d"] = ...  # Compute from underlying returns
            panel_df[f"y_gap_{H}d"] = np.log(panel_df[f"rv_{H}d"]) - np.log(panel_df["iv_mid"] ** 2)

        return panel_df

    def _build_graph(self) -> torch_geometric.data.Data:
        """Build static surface graph from adjacency spec."""

        # Define nodes: (tenor, delta_bucket) pairs
        tenors = self._config.surface.tenors_days
        buckets = list(self._config.surface.delta_buckets.keys())
        nodes = [(t, b) for t in tenors for b in buckets]

        node_to_idx = {node: idx for idx, node in enumerate(nodes)}

        # Build edge list
        edge_index = []
        edge_attr = []

        # Delta adjacency (within tenor)
        for tenor in tenors:
            for i in range(len(buckets) - 1):
                src = node_to_idx[(tenor, buckets[i])]
                dst = node_to_idx[(tenor, buckets[i + 1])]
                edge_index.append([src, dst])
                edge_index.append([dst, src])  # Undirected

                # Edge attribute: delta distance
                edge_attr.append([1.0, 0.0])  # [delta_distance=1, tenor_distance=0]
                edge_attr.append([1.0, 0.0])

        # Tenor adjacency (within delta bucket)
        for bucket in buckets:
            for i in range(len(tenors) - 1):
                src = node_to_idx[(tenors[i], bucket)]
                dst = node_to_idx[(tenors[i + 1], bucket)]
                edge_index.append([src, dst])
                edge_index.append([dst, src])

                # Edge attribute: tenor distance
                edge_attr.append([0.0, float(tenors[i + 1] - tenors[i])])
                edge_attr.append([0.0, float(tenors[i + 1] - tenors[i])])

        edge_index = torch.tensor(edge_index, dtype=torch.long).T
        edge_attr = torch.tensor(edge_attr, dtype=torch.float32)

        return torch_geometric.data.Data(
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_nodes=len(nodes),
        )
```

#### 3.6.5 PatchTST Model

**PatchTSTModel**
```python
class PatchTSTModel(nn.Module, TimeSeriesModel):
    """PatchTST model for per-node temporal encoding."""

    def __init__(
        self,
        config: PatchTSTConfig,
        input_dim: int,
        output_horizons: int,
    ):
        super().__init__()
        self._config = config

        # Patch embedding
        self._patch_embedding = PatchEmbedding(
            patch_len=config.patch_len,
            stride=config.stride,
            input_dim=input_dim,
            d_model=config.d_model,
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self._encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.n_layers)

        # Prediction head (multi-horizon)
        self._head = nn.Linear(config.d_model, output_horizons)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x: (batch, time, nodes, features)
            mask: (batch, nodes) - indicates valid nodes

        Returns:
            predictions: (batch, nodes, horizons)
        """

        batch_size, time_steps, num_nodes, input_dim = x.shape

        # Reshape to (batch * nodes, time, features)
        x = x.permute(0, 2, 1, 3).reshape(batch_size * num_nodes, time_steps, input_dim)

        # Patch embedding
        x = self._patch_embedding(x)  # (batch * nodes, num_patches, d_model)

        # Transformer encoding
        x = self._encoder(x)  # (batch * nodes, num_patches, d_model)

        # Aggregate patches (mean pooling)
        x = x.mean(dim=1)  # (batch * nodes, d_model)

        # Prediction head
        preds = self._head(x)  # (batch * nodes, horizons)

        # Reshape to (batch, nodes, horizons)
        preds = preds.reshape(batch_size, num_nodes, -1)

        # Apply mask
        if mask is not None:
            preds = preds * mask.unsqueeze(-1)

        return preds

    def fit(self, X, y, X_val=None, y_val=None) -> TrainResult:
        """Training implementation."""
        # Handled by Trainer class
        ...
```

#### 3.6.6 GNN Model

**SurfaceGNN**
```python
class SurfaceGNN(nn.Module):
    """Graph Neural Network over options surface."""

    def __init__(
        self,
        config: GNNConfig,
        input_dim: int,
    ):
        super().__init__()
        self._config = config

        if config.type == "gat":
            self._conv_layers = nn.ModuleList([
                torch_geometric.nn.GATConv(
                    in_channels=input_dim if i == 0 else config.hidden_dim,
                    out_channels=config.hidden_dim,
                    edge_dim=2 if config.use_edge_attributes else None,
                    dropout=config.dropout,
                )
                for i in range(config.n_layers)
            ])
        elif config.type == "gcn":
            self._conv_layers = nn.ModuleList([
                torch_geometric.nn.GCNConv(
                    in_channels=input_dim if i == 0 else config.hidden_dim,
                    out_channels=config.hidden_dim,
                )
                for i in range(config.n_layers)
            ])
        else:
            raise ValueError(f"Unknown GNN type: {config.type}")

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, nodes, features)
            edge_index: (2, num_edges)
            edge_attr: (num_edges, edge_features)

        Returns:
            node_embeddings: (batch, nodes, hidden_dim)
        """

        batch_size, num_nodes, input_dim = x.shape

        # Process each sample in batch independently
        batch_outputs = []

        for i in range(batch_size):
            node_features = x[i]  # (nodes, features)

            # Apply GNN layers
            for conv in self._conv_layers:
                if self._config.use_edge_attributes and edge_attr is not None:
                    node_features = conv(node_features, edge_index, edge_attr)
                else:
                    node_features = conv(node_features, edge_index)

                node_features = F.relu(node_features)
                node_features = F.dropout(node_features, p=self._config.dropout, training=self.training)

            batch_outputs.append(node_features)

        return torch.stack(batch_outputs)  # (batch, nodes, hidden_dim)
```

#### 3.6.7 Ensemble Model

**PatchTST_GNN_Ensemble**
```python
class PatchTST_GNN_Ensemble(nn.Module, TimeSeriesModel):
    """Combined PatchTST + GNN model."""

    def __init__(
        self,
        patchtst_config: PatchTSTConfig,
        gnn_config: GNNConfig,
        input_dim: int,
        output_horizons: int,
    ):
        super().__init__()

        # PatchTST for temporal encoding
        self._patchtst = PatchTSTModel(patchtst_config, input_dim, output_horizons=0)  # No head

        # GNN for cross-sectional encoding
        self._gnn = SurfaceGNN(gnn_config, input_dim=patchtst_config.d_model)

        # Combined prediction head
        self._head = nn.Linear(gnn_config.hidden_dim, output_horizons)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, time, nodes, features)
            edge_index: (2, num_edges)
            edge_attr: (num_edges, edge_features)
            mask: (batch, nodes)

        Returns:
            predictions: (batch, nodes, horizons)
        """

        # 1. PatchTST temporal encoding (per node)
        temporal_embeddings = self._patchtst.encode(x)  # (batch, nodes, d_model)

        # 2. GNN cross-sectional encoding
        graph_embeddings = self._gnn(temporal_embeddings, edge_index, edge_attr)  # (batch, nodes, hidden_dim)

        # 3. Prediction head
        preds = self._head(graph_embeddings)  # (batch, nodes, horizons)

        # 4. Apply mask
        if mask is not None:
            preds = preds * mask.unsqueeze(-1)

        return preds
```

#### 3.6.8 Training Loop

**Trainer**
```python
class Trainer:
    """Training loop with early stopping, checkpointing, and logging."""

    def __init__(
        self,
        model: TimeSeriesModel,
        config: TrainingConfig,
        device: torch.device,
    ):
        self._model = model.to(device)
        self._config = config
        self._device = device

        # Optimizer
        self._optimizer = self._build_optimizer()

        # Scheduler
        self._scheduler = self._build_scheduler()

        # Loss function
        self._criterion = self._build_loss()

        # Early stopping
        self._best_val_metric = -np.inf
        self._patience_counter = 0

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> TrainResult:
        """Run training loop."""

        for epoch in range(self._config.epochs):
            # Train epoch
            train_loss = self._train_epoch(train_loader)

            # Validation epoch
            val_loss, val_metrics = self._validate_epoch(val_loader)

            # Logging
            logger.info(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_rank_ic={val_metrics['rank_ic']:.4f}")

            # Scheduler step
            if self._scheduler:
                self._scheduler.step()

            # Early stopping check
            if self._config.early_stopping.enabled:
                metric = val_metrics[self._config.early_stopping.metric]

                if metric > self._best_val_metric:
                    self._best_val_metric = metric
                    self._patience_counter = 0
                    self._save_checkpoint("best")
                else:
                    self._patience_counter += 1

                    if self._patience_counter >= self._config.early_stopping.patience:
                        logger.info(f"Early stopping at epoch {epoch}")
                        break

        return TrainResult(
            best_val_metric=self._best_val_metric,
            epochs_trained=epoch + 1,
        )

    def _train_epoch(self, loader: DataLoader) -> float:
        """Train for one epoch."""

        self._model.train()
        total_loss = 0.0

        for batch in loader:
            X = batch["X"].to(self._device)
            y = batch["y"].to(self._device)
            mask = batch["mask"].to(self._device)
            graph = batch["graph"]

            # Forward
            preds = self._model(X, graph.edge_index, graph.edge_attr, mask)

            # Compute loss (masked)
            loss = self._criterion(preds[mask], y[mask])

            # Backward
            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()

            total_loss += loss.item()

        return total_loss / len(loader)

    def _validate_epoch(self, loader: DataLoader) -> tuple[float, dict]:
        """Validate for one epoch."""

        self._model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in loader:
                X = batch["X"].to(self._device)
                y = batch["y"].to(self._device)
                mask = batch["mask"].to(self._device)
                graph = batch["graph"]

                preds = self._model(X, graph.edge_index, graph.edge_attr, mask)
                loss = self._criterion(preds[mask], y[mask])

                total_loss += loss.item()
                all_preds.append(preds[mask].cpu().numpy())
                all_targets.append(y[mask].cpu().numpy())

        # Compute metrics
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)

        metrics = {
            "ic": self._compute_ic(all_preds, all_targets),
            "rank_ic": self._compute_rank_ic(all_preds, all_targets),
        }

        return total_loss / len(loader), metrics
```

---

### 3.7 Strategy & Execution

#### 3.7.1 Design Goals
- Map model predictions to bounded-risk option structures
- Realistic execution modeling (bid/ask fills, slippage, fees)
- Liquidity-aware structure selection
- Deterministic signal-to-trade logic (no human discretion)

#### 3.7.2 Package Structure
```
src/strategy/
├── __init__.py
├── signal.py             # Signal generation from model outputs
├── structures/
│   ├── __init__.py
│   ├── base.py           # Abstract trade structure
│   ├── calendar.py       # Calendar spreads
│   ├── vertical.py       # Vertical spreads
│   ├── skew.py           # Skew trades
│   └── iron_condor.py    # Iron condors
├── selector.py           # Structure selection logic
├── sizing.py             # Position sizing
└── orders.py             # Order generation
```

#### 3.7.3 Trade Structures

**TradeStructure** (Abstract)
```python
class TradeStructure(ABC):
    """Abstract base for option trade structures."""

    @abstractmethod
    def create_legs(
        self,
        signal: Signal,
        surface: pd.DataFrame,
    ) -> list[OptionLeg]:
        """Create option legs for this structure."""
        ...

    @abstractmethod
    def compute_max_loss(self, legs: list[OptionLeg]) -> float:
        """Compute max loss for this structure."""
        ...

    @abstractmethod
    def compute_greeks(self, legs: list[OptionLeg]) -> Greeks:
        """Compute aggregate Greeks."""
        ...
```

**CalendarSpread**
```python
class CalendarSpread(TradeStructure):
    """Calendar spread (sell near, buy far)."""

    def create_legs(self, signal: Signal, surface: pd.DataFrame) -> list[OptionLeg]:
        """
        Signal indicates term structure mispricing:
        - signal.node = (tenor_near, delta_bucket)
        - signal.edge = positive → buy calendar (sell near, buy far)
        """

        # Find near-tenor option
        near_option = surface[
            (surface["tenor_days"] == signal.node.tenor) &
            (surface["delta_bucket"] == signal.node.bucket)
        ].iloc[0]

        # Find far-tenor option (same delta bucket, next tenor)
        far_tenor = self._next_tenor(signal.node.tenor)
        far_option = surface[
            (surface["tenor_days"] == far_tenor) &
            (surface["delta_bucket"] == signal.node.bucket)
        ].iloc[0]

        # Determine direction
        if signal.edge > 0:
            # Buy calendar: sell near, buy far
            near_qty = -1
            far_qty = 1
        else:
            # Sell calendar: buy near, sell far
            near_qty = 1
            far_qty = -1

        return [
            OptionLeg(
                symbol=near_option["option_symbol"],
                qty=near_qty,
                entry_price=near_option["ask"] if near_qty > 0 else near_option["bid"],
            ),
            OptionLeg(
                symbol=far_option["option_symbol"],
                qty=far_qty,
                entry_price=far_option["ask"] if far_qty > 0 else far_option["bid"],
            ),
        ]

    def compute_max_loss(self, legs: list[OptionLeg]) -> float:
        """Calendar spreads: max loss = net debit (if debit spread)."""

        net_premium = sum(leg.qty * leg.entry_price for leg in legs) * 100

        if net_premium < 0:
            # Debit spread
            return abs(net_premium)
        else:
            # Credit spread (rare for calendars)
            # Max loss = credit received (if near expires worthless and far is full loss)
            # Conservative: assume max loss = far leg cost
            return max(abs(leg.qty * leg.entry_price * 100) for leg in legs)
```

#### 3.7.4 Structure Selector

**StructureSelector**
```python
class StructureSelector:
    """Select appropriate trade structure based on signal characteristics."""

    def __init__(self, config: ExecutionConfig):
        self._config = config

        self._structures = {
            "calendar": CalendarSpread(),
            "vertical": VerticalSpread(),
            "skew": SkewTrade(),
            "iron_condor": IronCondor(),
        }

    def select_structure(self, signal: Signal, surface: pd.DataFrame) -> TradeStructure:
        """Select structure based on signal type and confidence."""

        # Term structure signal → Calendar
        if signal.type == SignalType.TERM_ANOMALY:
            return self._structures["calendar"]

        # Strong directional vol signal → Vertical
        elif signal.type == SignalType.DIRECTIONAL_VOL and signal.confidence > 0.7:
            return self._structures["vertical"]

        # Skew anomaly → Skew trade
        elif signal.type == SignalType.SKEW_ANOMALY:
            return self._structures["skew"]

        # Elevated IV, low confidence → Iron condor
        elif signal.type == SignalType.ELEVATED_IV and signal.confidence < 0.5:
            return self._structures["iron_condor"]

        else:
            # Default: no trade
            return None
```

#### 3.7.5 Order Generator

**OrderGenerator**
```python
class OrderGenerator:
    """Generate executable orders from trade structures."""

    def __init__(
        self,
        config: ExecutionConfig,
        risk_checker: RiskChecker,
    ):
        self._config = config
        self._risk_checker = risk_checker

    def generate_orders(
        self,
        signals: list[Signal],
        surface: pd.DataFrame,
        portfolio: Portfolio,
    ) -> list[Order]:
        """Generate orders from signals."""

        orders = []

        for signal in signals:
            # Skip if signal below threshold
            if abs(signal.edge) < self._config.signal_threshold.min_edge:
                continue

            if signal.uncertainty > self._config.signal_threshold.max_uncertainty:
                continue

            # Select structure
            structure = self._structure_selector.select_structure(signal, surface)
            if structure is None:
                continue

            # Create legs
            legs = structure.create_legs(signal, surface)

            # Compute risk metrics
            max_loss = structure.compute_max_loss(legs)
            greeks = structure.compute_greeks(legs)

            # Position sizing
            qty_multiplier = self._position_sizer.compute_size(
                signal=signal,
                max_loss=max_loss,
                liquidity=self._get_liquidity(legs, surface),
            )

            # Scale legs
            legs = [leg._replace(qty=leg.qty * qty_multiplier) for leg in legs]

            # Pre-trade risk checks
            risk_result = self._risk_checker.check_trade(
                legs=legs,
                portfolio=portfolio,
                surface=surface,
            )

            if not risk_result.approved:
                logger.warning(f"Trade rejected: {risk_result.reason}")
                continue

            # Create order
            order = Order(
                order_id=self._generate_order_id(),
                legs=legs,
                structure_type=structure.__class__.__name__,
                signal=signal,
                max_loss=max_loss,
                greeks=greeks,
                timestamp=datetime.now(timezone.utc),
            )

            orders.append(order)

        return orders
```

---

### 3.8 Risk Management

#### 3.8.1 Design Goals
- Hard constraints enforced pre-trade and continuously
- Stress testing before execution
- Portfolio-level caps (vega, gamma, delta, loss, drawdown)
- Kill switches (automated halt triggers)
- Deterministic risk math (no approximations)

#### 3.8.2 Package Structure
```
src/risk/
├── __init__.py
├── checker.py            # Pre-trade risk checks
├── portfolio.py          # Portfolio state and aggregation
├── stress.py             # Stress testing engine
├── caps.py               # Risk limit definitions
├── greeks.py             # Greek aggregation
└── kill_switch.py        # Automated halt logic
```

#### 3.8.3 Risk Checker

**RiskChecker**
```python
class RiskChecker:
    """Pre-trade risk validation."""

    def __init__(self, config: RiskConfig):
        self._config = config
        self._stress_engine = StressEngine(config.stress)

    def check_trade(
        self,
        legs: list[OptionLeg],
        portfolio: Portfolio,
        surface: pd.DataFrame,
    ) -> RiskCheckResult:
        """Run all pre-trade checks."""

        issues = []

        # 1. Per-trade max loss
        max_loss = self._compute_max_loss(legs)
        if max_loss > self._config.per_trade.max_loss:
            issues.append(f"Max loss ${max_loss:.2f} exceeds limit ${self._config.per_trade.max_loss}")

        # 2. Per-trade contract limit
        total_contracts = sum(abs(leg.qty) for leg in legs)
        if total_contracts > self._config.per_trade.max_contracts:
            issues.append(f"Total contracts {total_contracts} exceeds limit {self._config.per_trade.max_contracts}")

        # 3. Portfolio-level caps (after adding trade)
        hypothetical_portfolio = portfolio.add_trade(legs)

        if abs(hypothetical_portfolio.net_vega) > self._config.caps.max_abs_vega:
            issues.append(f"Net vega {hypothetical_portfolio.net_vega:.0f} exceeds cap {self._config.caps.max_abs_vega}")

        if abs(hypothetical_portfolio.net_gamma) > self._config.caps.max_abs_gamma:
            issues.append(f"Net gamma {hypothetical_portfolio.net_gamma:.0f} exceeds cap {self._config.caps.max_abs_gamma}")

        if abs(hypothetical_portfolio.net_delta) > self._config.caps.max_abs_delta:
            issues.append(f"Net delta {hypothetical_portfolio.net_delta:.0f} exceeds cap {self._config.caps.max_abs_delta}")

        # 4. Stress testing
        if self._config.stress.enabled:
            stress_result = self._stress_engine.run_stress_test(
                portfolio=hypothetical_portfolio,
                surface=surface,
            )

            if stress_result.worst_case_loss > self._config.caps.max_daily_loss:
                issues.append(f"Stress test worst-case loss ${stress_result.worst_case_loss:.2f} exceeds limit")

        return RiskCheckResult(
            approved=len(issues) == 0,
            issues=issues,
        )

    def _compute_max_loss(self, legs: list[OptionLeg]) -> float:
        """Compute max loss for trade structure."""

        # Determine if debit or credit spread
        net_premium = sum(leg.qty * leg.entry_price for leg in legs) * 100

        if net_premium < 0:
            # Debit spread: max loss = net debit
            return abs(net_premium)
        else:
            # Credit spread: max loss = spread width - net credit
            # (Simplified - actual logic depends on structure)
            return self._compute_credit_spread_max_loss(legs)
```

#### 3.8.4 Stress Engine

**StressEngine**
```python
class StressEngine:
    """Portfolio stress testing."""

    def __init__(self, config: StressConfig):
        self._config = config

    def run_stress_test(
        self,
        portfolio: Portfolio,
        surface: pd.DataFrame,
    ) -> StressResult:
        """Run stress scenarios on portfolio."""

        scenarios = []

        # Underlying price shocks
        for shock_pct in self._config.underlying_shocks_pct:
            scenario = self._stress_underlying(portfolio, surface, shock_pct)
            scenarios.append(scenario)

        # IV shocks (parallel)
        for shock_pts in self._config.iv_shocks_points:
            scenario = self._stress_iv(portfolio, surface, shock_pts)
            scenarios.append(scenario)

        # Combined shocks (worst case: down move + IV spike)
        worst_underlying_shock = min(self._config.underlying_shocks_pct)
        worst_iv_shock = max(self._config.iv_shocks_points)
        scenario = self._stress_combined(portfolio, surface, worst_underlying_shock, worst_iv_shock)
        scenarios.append(scenario)

        # Find worst-case loss
        worst_case_loss = min(s.pnl for s in scenarios)

        return StressResult(
            scenarios=scenarios,
            worst_case_loss=abs(worst_case_loss),
        )

    def _stress_underlying(
        self,
        portfolio: Portfolio,
        surface: pd.DataFrame,
        shock_pct: float,
    ) -> StressScenario:
        """Stress test underlying price shock."""

        # Shock underlying price
        shocked_S = surface["underlying_price"] * (1 + shock_pct)

        # Reprice all options (first-order approximation using Greeks)
        pnl = 0.0

        for position in portfolio.positions:
            delta_pnl = position.delta * (shocked_S - surface["underlying_price"]) * 100 * position.qty
            gamma_pnl = 0.5 * position.gamma * ((shocked_S - surface["underlying_price"]) ** 2) * 100 * position.qty

            pnl += delta_pnl + gamma_pnl

        return StressScenario(
            description=f"Underlying {shock_pct:+.1%}",
            pnl=pnl,
        )

    def _stress_iv(
        self,
        portfolio: Portfolio,
        surface: pd.DataFrame,
        shock_pts: float,
    ) -> StressScenario:
        """Stress test IV shock."""

        # Shock IV (in volatility points, e.g., +10 pts = +0.10)
        shock_vol = shock_pts / 100.0

        # Reprice using vega
        pnl = 0.0

        for position in portfolio.positions:
            vega_pnl = position.vega * shock_vol * 100 * position.qty
            pnl += vega_pnl

        return StressScenario(
            description=f"IV +{shock_pts} pts",
            pnl=pnl,
        )
```

#### 3.8.5 Kill Switch

**KillSwitch**
```python
class KillSwitch:
    """Automated trading halt logic."""

    def __init__(self, config: KillSwitchConfig):
        self._config = config
        self._triggered = False
        self._trigger_reason = None

    def check(
        self,
        portfolio: Portfolio,
        surface: pd.DataFrame,
        stress_result: StressResult,
    ) -> KillSwitchResult:
        """Check if kill switch should trigger."""

        if self._triggered:
            return KillSwitchResult(triggered=True, reason=self._trigger_reason)

        # Daily loss check
        if self._config.halt_on_daily_loss:
            if portfolio.daily_pnl < -self._config.max_daily_loss:
                self._trigger(f"Daily loss ${abs(portfolio.daily_pnl):.2f} exceeds limit")

        # Stress breach check
        if self._config.halt_on_stress_breach:
            if stress_result.worst_case_loss > portfolio.max_acceptable_loss:
                self._trigger(f"Stress test worst-case ${stress_result.worst_case_loss:.2f} exceeds limit")

        # Liquidity collapse check
        if self._config.halt_on_liquidity_collapse:
            max_spread = surface["spread_pct"].max()
            if max_spread > self._config.max_spread_pct:
                self._trigger(f"Spread blowout: {max_spread:.1%} exceeds {self._config.max_spread_pct:.1%}")

        return KillSwitchResult(triggered=self._triggered, reason=self._trigger_reason)

    def _trigger(self, reason: str) -> None:
        """Trigger kill switch."""
        self._triggered = True
        self._trigger_reason = reason
        logger.critical(f"KILL SWITCH TRIGGERED: {reason}")

    def reset(self) -> None:
        """Reset kill switch (manual intervention required)."""
        self._triggered = False
        self._trigger_reason = None
```

---

### 3.9 Live Trading Infrastructure

#### 3.9.1 Design Goals
- Event-driven architecture (react to market data, orders, fills)
- State persistence (survive crashes)
- Order routing (paper trading → broker API later)
- Position tracking (real-time P&L, Greeks)
- Observability (logging, metrics, alerts)

#### 3.9.2 Package Structure
```
src/live/
├── __init__.py
├── loop.py               # Main trading loop
├── state.py              # State manager (positions, orders, risk)
├── router.py             # Order routing (paper/live)
├── positions.py          # Position tracker
└── monitoring.py         # Metrics and alerts
```

#### 3.9.3 Trading Loop

**TradingLoop**
```python
class TradingLoop:
    """Main event loop for live/paper trading."""

    def __init__(
        self,
        config: PaperConfig,
        market_provider: MarketDataProvider,
        model: TimeSeriesModel,
        order_generator: OrderGenerator,
        risk_checker: RiskChecker,
        state_manager: StateManager,
        order_router: OrderRouter,
    ):
        self._config = config
        self._market = market_provider
        self._model = model
        self._order_gen = order_generator
        self._risk = risk_checker
        self._state = state_manager
        self._router = order_router

        self._kill_switch = KillSwitch(config.risk.kill_switch)
        self._running = False

    def start(self) -> None:
        """Start the trading loop."""

        logger.info("Starting trading loop")
        self._running = True

        while self._running:
            try:
                # 1. Fetch latest market data
                surface = self._build_latest_surface()

                # 2. Run model inference
                signals = self._generate_signals(surface)

                # 3. Generate order intents
                orders = self._order_gen.generate_orders(
                    signals=signals,
                    surface=surface,
                    portfolio=self._state.portfolio,
                )

                # 4. Check kill switch
                stress_result = self._risk.stress_test(self._state.portfolio, surface)
                kill_switch_result = self._kill_switch.check(
                    self._state.portfolio,
                    surface,
                    stress_result,
                )

                if kill_switch_result.triggered:
                    logger.critical(f"Kill switch triggered: {kill_switch_result.reason}")
                    self._halt_trading()
                    break

                # 5. Route orders
                for order in orders:
                    fill = self._router.route_order(order)

                    if fill:
                        self._state.record_fill(fill)
                        logger.info(f"Order filled: {order.order_id}")

                # 6. Update positions and risk
                self._state.update_positions(surface)

                # 7. Snapshot state
                self._state.save_snapshot()

                # 8. Sleep until next cycle
                time.sleep(self._config.loop_interval_seconds)

            except Exception as e:
                logger.error(f"Error in trading loop: {e}", exc_info=True)

                # Optionally halt on errors
                if self._config.halt_on_error:
                    self._halt_trading()
                    break

    def stop(self) -> None:
        """Stop the trading loop."""
        logger.info("Stopping trading loop")
        self._running = False

    def _build_latest_surface(self) -> pd.DataFrame:
        """Build surface from latest market data."""

        # Fetch latest quotes
        now = datetime.now(timezone.utc)
        quotes = self._market.fetch_option_quotes(
            symbols=self._state.watched_symbols,
            start=now - timedelta(minutes=5),
            end=now,
        )

        # Run surface builder
        surface_builder = SurfaceBuilder(self._config.surface, self._raw_repo, self._derived_repo)
        surface = surface_builder.build_surface_snapshot(quotes)

        return surface

    def _generate_signals(self, surface: pd.DataFrame) -> list[Signal]:
        """Run model inference to generate signals."""

        # Load recent feature history
        features = self._load_recent_features()

        # Run model
        predictions = self._model.predict(features)

        # Convert predictions to signals
        signals = self._predictions_to_signals(predictions, surface)

        return signals

    def _halt_trading(self) -> None:
        """Emergency halt."""
        logger.critical("HALTING TRADING - manual intervention required")
        self._running = False

        # Send alert
        self._send_alert("Trading halted")
```

#### 3.9.4 State Manager

**StateManager**
```python
class StateManager:
    """Manage trading state (positions, orders, risk)."""

    def __init__(self, config: PaperConfig):
        self._config = config
        self._portfolio = Portfolio()
        self._order_history = []
        self._fill_history = []

        # Load from disk if exists
        self._load_state()

    @property
    def portfolio(self) -> Portfolio:
        return self._portfolio

    def record_fill(self, fill: Fill) -> None:
        """Record order fill and update portfolio."""

        self._fill_history.append(fill)

        # Update positions
        for leg in fill.legs:
            self._portfolio.add_position(
                symbol=leg.symbol,
                qty=leg.qty,
                entry_price=leg.fill_price,
                greeks=leg.greeks,
            )

    def update_positions(self, surface: pd.DataFrame) -> None:
        """Update position valuations and Greeks from latest surface."""

        for position in self._portfolio.positions:
            # Find option in surface
            option = surface[surface["option_symbol"] == position.symbol].iloc[0]

            # Update mark price and Greeks
            position.mark_price = option["mid_price"]
            position.delta = option["delta"]
            position.gamma = option["gamma"]
            position.vega = option["vega"]
            position.theta = option["theta"]

        # Recompute portfolio-level Greeks
        self._portfolio.recompute_greeks()

    def save_snapshot(self) -> None:
        """Save state to disk."""

        snapshot_path = self._config.state_snapshot_path / f"state_{datetime.now().isoformat()}.json"

        state = {
            "portfolio": self._portfolio.to_dict(),
            "order_history": [o.to_dict() for o in self._order_history],
            "fill_history": [f.to_dict() for f in self._fill_history],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        with open(snapshot_path, "w") as f:
            json.dump(state, f, indent=2)

    def _load_state(self) -> None:
        """Load most recent state snapshot."""

        # Find most recent snapshot
        snapshot_files = sorted(self._config.state_snapshot_path.glob("state_*.json"))

        if snapshot_files:
            latest = snapshot_files[-1]
            logger.info(f"Loading state from {latest}")

            with open(latest) as f:
                state = json.load(f)

            self._portfolio = Portfolio.from_dict(state["portfolio"])
            # ... load other state
```

---

## 4. Data Flow

### 4.1 Ingestion Flow

```
Databento API → DatabentoProvider.fetch_*() → RawRepository.write_*() → raw_* tables
FRED API → FREDProvider.fetch_series() → RawRepository.write_fred_series() → raw_fred_series
```

### 4.2 Surface Construction Flow

```
raw_underlying_bars + raw_option_quotes + raw_fred_series
  → SurfaceBuilder.build_surface()
    → IV inversion (BlackScholesIVSolver)
    → Greeks computation (AnalyticalGreeks)
    → Bucket assignment (DeltaBucketAssigner)
    → Quality filtering (QualityFilter)
    → Representative selection
  → surface_snapshots table
```

### 4.3 Feature Engineering Flow

```
surface_snapshots
  → NodeFeatureGenerator → node features
  → SurfaceFeatureGenerator → skew/term slopes

raw_underlying_bars
  → GlobalFeatureGenerator → realized variance, returns

raw_fred_series
  → MacroFeatureGenerator → macro features (release-aligned)

→ Merge → node_panel table
```

### 4.4 Training Flow

```
node_panel
  → DatasetBuilder → SurfaceDataset (train/val/test splits)
  → Trainer.train()
    → PatchTSTModel (temporal encoding)
    → SurfaceGNN (cross-sectional encoding)
    → Loss computation + backprop
  → Model checkpoint
```

### 4.5 Inference → Execution Flow

```
Latest surface_snapshots + node_panel
  → Model.predict() → predictions
  → Signal generation
  → StructureSelector.select_structure()
  → OrderGenerator.generate_orders()
  → RiskChecker.check_trade()
  → StressEngine.run_stress_test()
  → OrderRouter.route_order()
  → StateManager.record_fill()
```

---

## 5. Technology Stack

### 5.1 Core Languages & Frameworks

- **Python 3.11+**: Primary language
- **PyTorch 2.0+**: Deep learning framework
- **PyTorch Geometric / DGL**: Graph neural networks
- **Pydantic 2.0+**: Configuration validation and data contracts
- **Polars / Pandas**: Data manipulation
- **NumPy / SciPy**: Numerical operations

### 5.2 Data Storage

- **SQLite**: Development and backtesting (lightweight, single-file)
- **PostgreSQL**: Production (optional, for multi-user/scaling)
- **Parquet (PyArrow)**: Long-term archival and bulk exports

### 5.3 Data Providers

- **Databento Python SDK**: Market data ingestion
- **FRED API**: Macro data (via requests)

### 5.4 Utilities

- **python-dotenv**: Environment variable management
- **PyYAML**: Configuration file parsing
- **pytest**: Unit and integration testing
- **black / ruff**: Code formatting and linting
- **mypy**: Static type checking

### 5.5 Optional (Future)

- **Streamlit**: Research console / harness
- **MLflow / Weights & Biases**: Experiment tracking
- **Redis**: Caching (for live trading)
- **Docker**: Containerization
- **Kubernetes**: Orchestration (production deployment)

---

## 6. Implementation Guidelines

### 6.1 Code Organization

#### Directory Structure
```
Rhubarb/
├── config/
│   ├── config.yaml               # Base configuration
│   └── environments/
│       ├── dev.yaml
│       ├── backtest.yaml
│       ├── paper.yaml
│       └── live.yaml
├── src/
│   ├── config/                   # Configuration management
│   ├── data/                     # Data abstraction + ingestion
│   ├── surface/                  # Surface engine
│   ├── features/                 # Feature engineering
│   ├── models/                   # ML pipeline
│   ├── strategy/                 # Strategy + execution
│   ├── risk/                     # Risk management
│   └── live/                     # Live trading infrastructure
├── scripts/
│   ├── ingest_raw.py
│   ├── build_surface.py
│   ├── build_features.py
│   ├── train_model.py
│   ├── run_backtest.py
│   └── run_paper_trading.py
├── tests/
│   ├── unit/
│   ├── integration/
│   └── system/
├── docs/
│   ├── SPEC.md                   # System specification (this document)
│   ├── ROADMAP.md                # Implementation milestones
│   └── *.md                      # Legacy specs (kept for reference)
├── data/                         # Data artifacts (not committed)
│   ├── db.sqlite
│   ├── parquet/
│   └── manifest/
├── artifacts/                    # Model checkpoints, reports
│   ├── checkpoints/
│   └── reports/
├── .env.example
├── .gitignore
├── pyproject.toml
├── Makefile
└── README.md
```

### 6.2 Coding Standards

#### Type Hints
- Use type hints everywhere (functions, methods, class attributes)
- Leverage `typing` module (`Protocol`, `TypeVar`, `Generic`)
- Run `mypy` in strict mode

#### Naming Conventions
- Classes: `PascalCase` (e.g., `SurfaceBuilder`)
- Functions/methods: `snake_case` (e.g., `build_surface`)
- Constants: `UPPER_SNAKE_CASE` (e.g., `MAX_CONTRACTS`)
- Private members: prefix with `_` (e.g., `self._config`)

#### Docstrings
- Use Google-style docstrings
- Document all public classes, functions, and methods
- Include Args, Returns, Raises sections

Example:
```python
def compute_greeks(
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
) -> dict[str, float]:
    """Compute analytical Black-Scholes Greeks.

    Args:
        S: Spot price
        K: Strike price
        T: Time-to-expiry (years)
        r: Risk-free rate (decimal)
        q: Dividend yield (decimal)
        sigma: Implied volatility (decimal)

    Returns:
        Dictionary with keys: delta, gamma, vega, theta

    Raises:
        ValueError: If T <= 0 or sigma <= 0
    """
```

#### Error Handling
- Use specific exceptions (not bare `Exception`)
- Define custom exceptions for domain-specific errors
- Log errors with context
- Fail fast (don't silently catch and continue)

#### Testing
- Aim for >80% code coverage
- Unit tests: test individual functions/classes in isolation
- Integration tests: test interactions between modules
- System tests: test end-to-end workflows

### 6.3 Configuration Management

- All configurable parameters live in `config/config.yaml`
- No hardcoded magic numbers in code
- Use Pydantic models for validation
- Environment-specific overrides in `config/environments/*.yaml`

### 6.4 Logging

- Use Python `logging` module (not print statements)
- Log levels:
  - DEBUG: Verbose diagnostic info
  - INFO: General informational messages
  - WARNING: Unexpected but recoverable events
  - ERROR: Errors that don't crash the system
  - CRITICAL: Fatal errors (kill switch, data corruption)

- Structured logging (JSON format) for production
- Include context: timestamps, module names, function names

### 6.5 Version Control

- Use Git with meaningful commit messages
- Branch strategy:
  - `main`: stable, production-ready code
  - `develop`: integration branch for features
  - Feature branches: `feature/description`
  - Hotfix branches: `hotfix/description`

- Commit messages format:
  ```
  feat(surface): add IV solver with Newton-Raphson
  fix(risk): correct max loss calculation for credit spreads
  docs(spec): update data flow diagram
  test(features): add unit tests for macro alignment
  ```

### 6.6 Dependency Management

- Use `uv` or `pip` with `pyproject.toml`
- Pin versions for reproducibility
- Separate dev dependencies (`pytest`, `mypy`, etc.)

### 6.7 Deployment

#### Development
- Run locally with SQLite
- Use `dev.yaml` environment

#### Backtesting
- Run on workstation or cloud VM
- Use `backtest.yaml` environment
- Export results to `artifacts/reports/`

#### Paper Trading
- Run on always-on machine (VPS, cloud)
- Use `paper.yaml` environment
- State persisted to `data/paper/state/`

#### Live Trading (Future)
- Containerize with Docker
- Deploy to cloud (AWS, GCP, Azure)
- Use `live.yaml` environment
- Add monitoring (Prometheus, Grafana)

---

## 7. Migration Path

### 7.1 Milestone-Based Implementation

#### Phase 1: Foundation (Weeks 1-2)
- **M0: Configuration Management**
  - Implement Pydantic schema
  - Build loader with validation
  - Add environment overlay support
  - Unit tests for config validation

- **M1: Data Abstraction Layer**
  - Define provider protocols
  - Implement DatabentoProvider
  - Implement FREDProvider
  - Build RawRepository
  - Unit tests for providers

#### Phase 2: Core Pipeline (Weeks 3-4)
- **M2: Storage & Persistence**
  - Define raw table schemas (SQLAlchemy)
  - Implement DerivedRepository
  - Add Parquet writer
  - Integration tests for storage

- **M3: Ingestion**
  - Build IngestionOrchestrator
  - Add cost estimation
  - Implement manifest generator
  - Add data quality validators
  - End-to-end ingestion test

#### Phase 3: Surface Engine (Weeks 5-6)
- **M4: Surface Construction**
  - Implement IV solver (Black-Scholes)
  - Implement Greeks calculator
  - Build bucket assigner
  - Add quality filters
  - Build SurfaceBuilder orchestrator
  - Unit tests for each component
  - Integration test: raw → surface

#### Phase 4: Features & ML (Weeks 7-10)
- **M5: Feature Engineering**
  - Implement node feature generators
  - Implement global feature generators
  - Implement macro feature generators
  - Build FeatureEngine orchestrator
  - Anti-leakage validation
  - Integration test: surface → features

- **M6: Dataset Builder**
  - Implement SurfaceDataset (PyTorch)
  - Build DatasetBuilder with splits
  - Implement graph construction
  - Unit tests for dataset logic

- **M7: PatchTST Model**
  - Implement PatchTST architecture
  - Build training loop (Trainer)
  - Add evaluation metrics (IC, Rank IC)
  - Training test on synthetic data

- **M8: GNN Model**
  - Implement SurfaceGNN (GAT/GCN)
  - Build PatchTST+GNN ensemble
  - Add smoothness regularization
  - Training test on synthetic data

#### Phase 5: Strategy & Risk (Weeks 11-12)
- **M9: Trade Structures**
  - Implement TradeStructure base class
  - Implement CalendarSpread
  - Implement VerticalSpread
  - Unit tests for max loss calculation

- **M10: Strategy & Execution**
  - Build StructureSelector
  - Implement OrderGenerator
  - Add position sizing
  - Integration test: signal → orders

- **M11: Risk Management**
  - Implement RiskChecker
  - Build StressEngine
  - Add KillSwitch logic
  - Unit tests for risk checks

#### Phase 6: Backtesting (Weeks 13-14)
- **M12: Backtest Engine**
  - Build backtest orchestrator
  - Add realistic execution (bid/ask, slippage)
  - Implement portfolio tracking
  - Generate backtest reports
  - End-to-end backtest test

#### Phase 7: Live Infrastructure (Weeks 15-16)
- **M13: Paper Trading**
  - Implement TradingLoop
  - Build StateManager
  - Add OrderRouter (paper mode)
  - Add monitoring and alerts
  - Paper trading dry run (24h)

- **M14: Production Hardening**
  - Add comprehensive logging
  - Error handling and recovery
  - State persistence and recovery
  - Deployment automation (Docker)

### 7.2 Acceptance Criteria (Per Milestone)

Each milestone is complete when:
1. All code is implemented and reviewed
2. Unit tests pass with >80% coverage
3. Integration tests pass (if applicable)
4. Documentation is updated
5. No blockers or critical bugs remain

### 7.3 Testing Strategy

#### Unit Tests
- Test individual functions and classes in isolation
- Mock external dependencies (providers, databases)
- Focus on edge cases and error conditions

#### Integration Tests
- Test interactions between modules
- Use test database (in-memory SQLite)
- Validate data transformations end-to-end

#### System Tests
- Test complete workflows (ingestion → surface → features → model → backtest)
- Use realistic data samples
- Validate against known results

#### Regression Tests
- Ensure new changes don't break existing functionality
- Run full test suite on every commit (CI/CD)

---

## 8. Appendices

### 8.1 Glossary

- **ATM**: At-the-money (option with delta near ±0.50)
- **Delta Bucket**: Range of delta values representing a surface node
- **DTE**: Days-to-expiry
- **GNN**: Graph Neural Network
- **IC**: Information Coefficient (correlation between predictions and actuals)
- **IV**: Implied Volatility
- **OI**: Open Interest
- **PatchTST**: Patching Time Series Transformer
- **Rank IC**: Rank-order correlation (Spearman)
- **RV**: Realized Variance
- **TTE**: Time-to-expiry (in years)

### 8.2 References

- Legacy specs (in `docs/`):
  - VOL_ARB_SYSTEM_SPEC.md
  - RISK_MATH_AND_CONSTRAINTS.md
  - CONFIG_SCHEMA.md
  - SURFACE_GRAPH_ADJACENCY.md
  - TRADE_STRUCTURES_APPENDIX.md
  - MILESTONES.md
  - DATABENTO_INGESTION_GUIDE.md

- External references:
  - Black-Scholes model: [Wikipedia](https://en.wikipedia.org/wiki/Black%E2%80%93Scholes_model)
  - PatchTST paper: [arXiv:2211.14730](https://arxiv.org/abs/2211.14730)
  - PyTorch Geometric docs: [pytorch-geometric.readthedocs.io](https://pytorch-geometric.readthedocs.io)

### 8.3 Change Log

- 2026-01-10: Initial NEW_SPEC.md (v2.0) - Complete rewrite for modularity, organization, scalability

---

## End of Specification

This document represents the authoritative design for Rhubarb v2.0. Any implementation must adhere to the architecture, patterns, and guidelines defined herein.

For questions or clarifications, refer to the legacy specs in `docs/` or consult the project maintainer.
