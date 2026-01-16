# Rhubarb v2.0 Implementation Roadmap

## Overview

This roadmap breaks down the [SPEC.md](docs/SPEC.md) into 26 atomic, auditable milestones organized into 8 phases. Each milestone represents 1-2 hours of focused implementation and can be independently tested and reviewed.

## How to Use This Roadmap

1. **Sequential Execution**: Complete milestones in order (dependencies are explicit)
2. **Implementation → Audit Cycle**: After each milestone, pause for code review/audit
3. **Testing Required**: Every milestone includes unit tests + integration tests (where applicable)
4. **Acceptance Gates**: Don't proceed until all acceptance criteria are met

## Progress Tracking

- ✅ Complete
- 🔄 In Progress
- ⏸️ Blocked
- ⬜ Not Started

---

## Phase 1: Foundation

### Milestone 0: Project Bootstrap
**Status**: ✅
**Goal**: Set up project structure and tooling
**Reference**: [§6.1 Code Organization](docs/SPEC.md#61-code-organization)
**Estimate**: 30 minutes
**Dependencies**: None

#### Deliverables
- [x] Create directory structure (src/, tests/, config/, scripts/, docs/, data/, artifacts/)
- [x] Create pyproject.toml with core dependencies (pydantic, pyyaml, pandas, pytest)
- [x] Create Makefile with common commands (install, test, clean, lint)
- [x] Create .gitignore for Python, data files, and artifacts
- [x] Create .env.example template
- [x] Create minimal README.md with setup instructions

#### Acceptance Criteria
- ✅ `make install` creates virtual environment using uv
- ✅ Directory structure matches SPEC.md §6.1
- ✅ All paths are relative to repo root
- ✅ Git repository initialized

#### Testing
- N/A (infrastructure setup)

---

### Milestone 1: Configuration System
**Status**: ✅
**Goal**: Type-safe, validated configuration management
**Reference**: [§3.1 Configuration Management](docs/SPEC.md#31-configuration-management)
**Estimate**: 2 hours
**Dependencies**: M0

#### Deliverables
- [x] Implement `src/config/schema.py` with all Pydantic models (ConfigSchema, ProjectConfig, DataConfig, etc.)
- [x] Implement `src/config/loader.py` (ConfigLoader.load, ConfigLoader.validate)
- [x] Implement `src/config/paths.py` (PathResolver)
- [x] Implement `src/config/environments.py` (environment overlay support)
- [x] Create `config/config.yaml` (base configuration)
- [x] Create `config/environments/dev.yaml` (dev overrides)
- [x] Unit tests for all validation rules (invalid configs, missing fields, type errors)
- [x] Integration test for environment overlay merging

#### Acceptance Criteria
- ✅ Load config.yaml successfully and return validated ConfigSchema
- ✅ Environment-specific overrides work (dev/backtest/paper/live)
- ✅ Invalid configs raise ValidationError with clear messages
- ✅ PathResolver converts logical paths to absolute paths
- ✅ Test coverage 92% (exceeds 80% target)

#### Testing
```bash
pytest tests/config/ -v --cov=src/config
```

---

### Milestone 2: Database Schema & Engine
**Status**: ✅
**Goal**: Define database schema and connection management
**Reference**: [§3.3 Storage & Persistence](docs/SPEC.md#33-storage--persistence)
**Estimate**: 1.5 hours
**Dependencies**: M1

#### Deliverables
- [x] Implement `src/data/storage/schema.py` with SQLAlchemy table definitions (raw_underlying_bars, raw_option_quotes, raw_fred_series, raw_ingestion_log)
- [x] Implement `src/data/storage/schema.py` derived tables (surface_snapshots, node_panel)
- [x] Implement `src/data/storage/engine.py` (database connection management, create_engine utility)
- [x] Unit tests for schema validation
- [x] Integration test: create database, verify schema matches SQLAlchemy models

#### Acceptance Criteria
- ✅ SQLAlchemy models match SPEC.md table definitions exactly
- ✅ Database engine connects to SQLite successfully
- ✅ All indexes are created correctly
- ✅ Test coverage: schema.py 100%, engine.py 92% (exceeds 80% target)
- ✅ 31 tests pass (24 unit + 7 integration)

#### Testing
```bash
pytest tests/unit/data/storage/ tests/integration/data/storage/ -v --cov=src/data/storage
```

---

## Phase 2: Data Abstraction Layer

### Milestone 3: Data Provider Protocols
**Status**: ⬜
**Goal**: Define abstract interfaces for data providers
**Reference**: [§3.2.3 Provider Protocols](docs/SPEC.md#323-provider-protocols)
**Estimate**: 1 hour
**Dependencies**: M1

#### Deliverables
- [ ] Implement `src/data/providers/protocol.py` (MarketDataProvider protocol)
- [ ] Implement `src/data/providers/protocol.py` (MacroDataProvider protocol)
- [ ] Implement `src/data/providers/mock.py` (MockMarketDataProvider, MockMacroDataProvider for testing)
- [ ] Unit tests for mock providers
- [ ] Documentation: docstrings for all protocol methods

#### Acceptance Criteria
- Protocols use @runtime_checkable decorator
- All protocol methods have type hints and docstrings
- Mock providers pass isinstance(provider, MarketDataProvider) checks
- Mock providers return realistic test data
- Test coverage >90%

#### Testing
```bash
pytest tests/data/providers/test_protocol.py -v
pytest tests/data/providers/test_mock.py -v
```

---

### Milestone 4: Databento Provider Implementation
**Status**: ⬜
**Goal**: Concrete Databento implementation of MarketDataProvider
**Reference**: [§3.2.4 Concrete Implementations](docs/SPEC.md#324-concrete-implementations)
**Estimate**: 2 hours
**Dependencies**: M3

#### Deliverables
- [ ] Implement `src/data/providers/databento.py` (DatabentoProvider class)
- [ ] Implement fetch_underlying_bars method with normalization
- [ ] Implement fetch_option_quotes method with normalization
- [ ] Implement estimate_cost method using Databento API
- [ ] Implement resolve_option_symbols method (stubbed for now, full implementation in M11)
- [ ] Unit tests with mocked Databento client
- [ ] Integration test with real Databento API (optional, requires API key)

#### Acceptance Criteria
- DatabentoProvider passes isinstance(provider, MarketDataProvider) check
- Column names standardized (ts_event → ts_utc, etc.)
- API key loaded from environment variable
- Error handling for API failures (rate limits, network errors)
- Test coverage >80%

#### Testing
```bash
pytest tests/data/providers/test_databento.py -v --cov=src/data/providers/databento
```

---

### Milestone 5: FRED Provider Implementation
**Status**: ⬜
**Goal**: Concrete FRED implementation of MacroDataProvider
**Reference**: [§3.2.4 Concrete Implementations](docs/SPEC.md#324-concrete-implementations)
**Estimate**: 1.5 hours
**Dependencies**: M3

#### Deliverables
- [ ] Implement `src/data/providers/fred.py` (FREDProvider class)
- [ ] Implement fetch_series method with release timestamp metadata
- [ ] Implement get_latest_value method (as-of point-in-time query)
- [ ] Unit tests with mocked FRED API responses
- [ ] Integration test with real FRED API (optional, requires API key)

#### Acceptance Criteria
- FREDProvider passes isinstance(provider, MacroDataProvider) check
- Release timestamps extracted from API metadata
- Percent values converted to decimal (5.25% → 0.0525)
- Error handling for missing series, invalid dates
- Test coverage >80%

#### Testing
```bash
pytest tests/data/providers/test_fred.py -v --cov=src/data/providers/fred
```

---

### Milestone 6: Storage Repositories
**Status**: ⬜
**Goal**: Data access layer for raw and derived tables
**Reference**: [§3.3.4 Repository Pattern](docs/SPEC.md#334-repository-pattern)
**Estimate**: 2 hours
**Dependencies**: M2, M4, M5

#### Deliverables
- [ ] Implement `src/data/storage/repository.py` (RawRepository class)
  - write_underlying_bars, read_underlying_bars
  - write_option_quotes, read_option_quotes
  - write_fred_series, read_fred_series
  - write_ingestion_log, read_ingestion_log
- [ ] Implement `src/data/storage/repository.py` (DerivedRepository class)
  - write_surface_snapshots, read_surface_snapshots
  - write_node_panel, read_node_panel
- [ ] Unit tests for each repository method
- [ ] Integration test: write → read round-trip

#### Acceptance Criteria
- All CRUD operations work correctly
- Timestamps stored in UTC
- Unique constraints enforced (duplicates rejected)
- Queries use parameterized SQL (no SQL injection risk)
- Test coverage >85%

#### Testing
```bash
pytest tests/data/storage/test_repository.py -v --cov=src/data/storage/repository
```

---

## Phase 3: Surface Engine

### Milestone 7: IV Solver (Black-Scholes)
**Status**: ⬜
**Goal**: Implied volatility inversion using Newton-Raphson
**Reference**: [§3.4.4 IV Solver](docs/SPEC.md#344-iv-solver)
**Estimate**: 2 hours
**Dependencies**: M1

#### Deliverables
- [ ] Implement `src/surface/iv/black_scholes.py` (BlackScholesIVSolver class)
- [ ] Implement solve_iv_vectorized method (Newton-Raphson)
- [ ] Implement _black_scholes_price helper (vectorized)
- [ ] Implement _vega helper (vectorized)
- [ ] Unit tests: known IV values, edge cases (ITM/OTM/ATM), convergence failures
- [ ] Performance test: vectorized vs. loop (should be >10x faster)

#### Acceptance Criteria
- Solver converges for 99%+ of realistic inputs within 100 iterations
- Failed convergence returns NaN (not crash)
- Vectorized implementation works on pandas Series
- Accuracy: reconstructed prices match input prices within 1e-6
- Test coverage >90%

#### Testing
```bash
pytest tests/surface/test_black_scholes.py -v --cov=src/surface/iv
```

---

### Milestone 8: Greeks Calculator
**Status**: ⬜
**Goal**: Analytical Greeks using Black-Scholes formulas
**Reference**: [§3.4.5 Greeks Calculator](docs/SPEC.md#345-greeks-calculator)
**Estimate**: 1.5 hours
**Dependencies**: M7

#### Deliverables
- [ ] Implement `src/surface/greeks/analytical.py` (AnalyticalGreeks class)
- [ ] Implement compute_greeks_vectorized method (delta, gamma, vega, theta)
- [ ] Unit tests: known Greeks values, put-call parity, boundary conditions
- [ ] Integration test with IV solver (IV → Greeks → reprice)

#### Acceptance Criteria
- Greeks match known analytical solutions (e.g., ATM call delta ≈ 0.5)
- Put-call parity holds for delta/gamma/vega
- Theta sign correct (negative for long positions)
- Vectorized over pandas DataFrame
- Test coverage >90%

#### Testing
```bash
pytest tests/surface/test_greeks.py -v --cov=src/surface/greeks
```

---

### Milestone 9: Bucket Assignment & Quality Filters
**Status**: ⬜
**Goal**: Assign options to delta buckets and compute quality flags
**Reference**: [§3.4.6 Bucket Assignment](docs/SPEC.md#346-bucket-assignment), [§3.4.7 Quality Filtering](docs/SPEC.md#347-quality-filtering)
**Estimate**: 1.5 hours
**Dependencies**: M1

#### Deliverables
- [ ] Implement `src/surface/buckets/assign.py` (DeltaBucketAssigner class)
- [ ] Handle ATM bucket special case (4-element list → |delta| range)
- [ ] Implement `src/surface/quality/filters.py` (QualityFilter class)
- [ ] Implement compute_flags method (crossed, stale, wide_spread, low_volume, low_oi)
- [ ] Unit tests: bucket boundaries, flag combinations
- [ ] Integration test: full pipeline (deltas → buckets → flags)

#### Acceptance Criteria
- Bucket assignment matches SPEC.md config exactly
- Options on bucket boundaries assigned correctly
- Quality flags use bitfield encoding (FLAGS = 0b00101 = crossed | wide_spread)
- No option assigned to multiple buckets
- Test coverage >85%

#### Testing
```bash
pytest tests/surface/test_bucket_assignment.py -v
pytest tests/surface/test_quality_flags.py -v
```

---

### Milestone 10: Surface Builder Orchestrator
**Status**: ⬜
**Goal**: Orchestrate full surface construction pipeline
**Reference**: [§3.4.3 Surface Builder](docs/SPEC.md#343-surface-builder)
**Estimate**: 2 hours
**Dependencies**: M6, M7, M8, M9

#### Deliverables
- [ ] Implement `src/surface/builder.py` (SurfaceBuilder class)
- [ ] Implement build_surface method (full pipeline)
- [ ] Implement _join_underlying_price (time-aware merge_asof)
- [ ] Implement _join_risk_free_rate (release-time-aware merge)
- [ ] Implement _compute_tte (ACT/365 convention)
- [ ] Implement _assign_tenor_bins (nearest tenor within threshold)
- [ ] Implement _select_representatives (one option per node)
- [ ] Unit tests for each helper method
- [ ] Integration test: raw quotes → surface snapshots (end-to-end)

#### Acceptance Criteria
- Deterministic: same inputs → same outputs across runs
- Time-aware joins: no future data leakage
- TTE calculation uses correct close time in project timezone
- Representative selection: closest delta to bucket center
- Surface snapshots written to database correctly
- Test coverage >80%

#### Testing
```bash
pytest tests/surface/test_surface_selection.py -v
pytest tests/surface/integration/test_surface_builder.py -v
```

---

## Phase 4: Ingestion Pipeline

### Milestone 11: Options Manifest Generator
**Status**: ⬜
**Goal**: Generate deterministic options symbol manifest
**Reference**: [§3.2.5 Ingestion Orchestrator](docs/SPEC.md#325-ingestion-orchestrator)
**Estimate**: 1.5 hours
**Dependencies**: M4

#### Deliverables
- [ ] Implement `src/data/ingest/manifest.py` (ManifestGenerator class)
- [ ] Implement resolve_symbols method (DTE filter, moneyness filter, per-expiry cap)
- [ ] Use spot reference = last underlying close at or before as_of_ts_utc
- [ ] Write manifest to JSON file (symbols, selection params, timestamp)
- [ ] Unit tests: manifest generation logic
- [ ] Integration test: generate manifest, verify deterministic output

#### Acceptance Criteria
- Manifest is deterministic for given config and as_of_ts_utc
- Spot reference uses correct underlying close
- DTE computed as (expiry_date - as_of_session_date).days
- Per-expiry per-side caps applied correctly (nearest-to-ATM ordering)
- Manifest includes metadata (config snapshot, timestamp)
- Test coverage >85%

#### Testing
```bash
pytest tests/data/ingest/test_manifest.py -v --cov=src/data/ingest/manifest
```

---

### Milestone 12: Ingestion Orchestrator
**Status**: ⬜
**Goal**: Coordinate multi-source ingestion runs
**Reference**: [§3.2.5 Ingestion Orchestrator](docs/SPEC.md#325-ingestion-orchestrator), [§3.2.6 Data Quality Checks](docs/SPEC.md#326-data-quality-checks)
**Estimate**: 2 hours
**Dependencies**: M6, M11

#### Deliverables
- [ ] Implement `src/data/ingest/orchestrator.py` (IngestionOrchestrator class)
- [ ] Implement run_ingestion method (full workflow)
- [ ] Implement cost estimation and preview mode
- [ ] Implement `src/data/quality/validators.py` (DataQualityValidator class)
- [ ] Quality checks: nulls, timestamp ordering, duplicates, OHLC relationships
- [ ] Write ingestion log to database
- [ ] Create `scripts/ingest_raw.py` CLI script
- [ ] Unit tests for orchestration logic
- [ ] Integration test: mock providers → database (end-to-end)

#### Acceptance Criteria
- Cost estimation rejects ingestion if exceeds limit
- Preview mode shows cost without fetching data
- Data quality checks catch common issues (nulls, duplicates)
- Ingestion log records all metadata (run_id, cost, row counts, git SHA)
- CLI script accepts --start-date, --end-date, --preview-only flags
- Test coverage >80%

#### Testing
```bash
pytest tests/data/ingest/test_orchestrator.py -v
pytest tests/data/quality/test_validators.py -v
python scripts/ingest_raw.py --preview-only --start-date 2024-01-01 --end-date 2024-01-02
```

---

## Phase 5: Feature Engineering

### Milestone 13: Node Features
**Status**: ⬜
**Goal**: Generate node-level features from surface snapshots
**Reference**: [§3.5.4 Node Features](docs/SPEC.md#354-node-features)
**Estimate**: 2 hours
**Dependencies**: M10

#### Deliverables
- [ ] Implement `src/features/node/iv_features.py` (IV changes, IV volatility)
- [ ] Implement `src/features/node/microstructure.py` (spread dynamics, quote stability)
- [ ] Implement `src/features/node/surface.py` (skew slope, term slope, curvature)
- [ ] Implement `src/features/engine.py` (NodeFeatureGenerator orchestrator)
- [ ] Unit tests for each feature generator
- [ ] Integration test: surface snapshots → node features

#### Acceptance Criteria
- Features computed per node (tenor, delta_bucket)
- Rolling windows use min_periods to avoid NaN explosion
- Skew slope computed via cross-sectional linear regression
- No future data leakage (features at time t only use data <= t)
- Test coverage >85%

#### Testing
```bash
pytest tests/features/test_node_features.py -v --cov=src/features/node
```

---

### Milestone 14: Global & Macro Features
**Status**: ⬜
**Goal**: Generate global underlying and macro features
**Reference**: [§3.5.5 Global Features](docs/SPEC.md#355-global-features), [§3.5.6 Macro Features](docs/SPEC.md#356-macro-features)
**Estimate**: 2 hours
**Dependencies**: M6

#### Deliverables
- [ ] Implement `src/features/global_/returns.py` (underlying returns)
- [ ] Implement `src/features/global_/realized_vol.py` (realized variance windows, vol-of-vol, drawdown)
- [ ] Implement `src/features/macro/transforms.py` (level, change, z-score)
- [ ] Implement `src/features/macro/alignment.py` (release-time alignment logic)
- [ ] Unit tests for each feature generator
- [ ] Integration test: raw data → global/macro features

#### Acceptance Criteria
- Realized variance computed correctly (close-to-close, annualized)
- Macro features respect release timestamps (strict or conservative mode)
- Percent values converted to decimal (5.25% → 0.0525)
- No future data leakage
- Test coverage >85%

#### Testing
```bash
pytest tests/features/test_global_features.py -v
pytest tests/features/test_macro_features.py -v
```

---

### Milestone 15: Feature Engine Orchestrator
**Status**: ⬜
**Goal**: Orchestrate feature generation and merge into node panel
**Reference**: [§3.5.3 Feature Engine](docs/SPEC.md#353-feature-engine), [§3.5.7 Anti-Leakage Validation](docs/SPEC.md#357-anti-leakage-validation)
**Estimate**: 1.5 hours
**Dependencies**: M13, M14

#### Deliverables
- [ ] Implement `src/features/engine.py` (FeatureEngine class)
- [ ] Implement build_feature_panel method (orchestrate all generators)
- [ ] Implement _merge_features method (join node/global/macro)
- [ ] Implement `src/features/validators.py` (FeatureValidator, leakage checks)
- [ ] Create `scripts/build_features.py` CLI script
- [ ] Unit tests for merge logic
- [ ] Integration test: surface → node panel (end-to-end)

#### Acceptance Criteria
- All features merged correctly (time-aware joins)
- Macro features available at correct timestamps
- Anti-leakage validator passes
- Node panel written to database with feature_version
- CLI script accepts --start-date, --end-date, --surface-version, --feature-version
- Test coverage >80%

#### Testing
```bash
pytest tests/features/test_feature_engine.py -v
pytest tests/features/integration/test_feature_pipeline.py -v
python scripts/build_features.py --start-date 2024-01-01 --end-date 2024-01-31
```

---

## Phase 6: ML Pipeline

### Milestone 16: Dataset Builder & Graph Construction
**Status**: ⬜
**Goal**: Build PyTorch datasets with static surface graph
**Reference**: [§3.6.4 Dataset Builder](docs/SPEC.md#364-dataset-builder)
**Estimate**: 2 hours
**Dependencies**: M15

#### Deliverables
- [ ] Implement `src/models/dataset.py` (SurfaceDataset class)
- [ ] Implement _build_sample_index (sliding window logic)
- [ ] Implement _extract_features, _extract_labels, _extract_mask
- [ ] Implement `src/models/graph.py` (build_graph function)
- [ ] Build static graph: delta adjacency + tenor adjacency (no diagonal edges)
- [ ] Implement DatasetBuilder (train/val/test splits)
- [ ] Unit tests for dataset indexing and graph construction
- [ ] Integration test: node panel → PyTorch datasets

#### Acceptance Criteria
- Dataset __getitem__ returns (X, y, mask, graph) tensors
- Graph topology matches SPEC.md adjacency spec exactly
- Train/val/test splits respect date boundaries (no leakage)
- Lookback windows constructed correctly
- Test coverage >85%

#### Testing
```bash
pytest tests/models/test_dataset.py -v --cov=src/models/dataset
pytest tests/models/test_graph.py -v --cov=src/models/graph
```

---

### Milestone 17: PatchTST Model
**Status**: ⬜
**Goal**: Implement PatchTST temporal encoder
**Reference**: [§3.6.5 PatchTST Model](docs/SPEC.md#365-patchtst-model)
**Estimate**: 2 hours
**Dependencies**: M16

#### Deliverables
- [ ] Implement `src/models/patchtst/model.py` (PatchTSTModel class)
- [ ] Implement `src/models/patchtst/encoder.py` (PatchEmbedding layer)
- [ ] Forward pass: patch → transformer → aggregation → prediction
- [ ] Multi-horizon prediction head
- [ ] Unit tests: forward pass shapes, gradient flow
- [ ] Integration test: synthetic data → training loop (1 epoch)

#### Acceptance Criteria
- Model output shape: (batch, nodes, horizons)
- Masking works correctly (invalid nodes zeroed)
- Gradients flow through all layers
- Model can overfit on small synthetic dataset (sanity check)
- Test coverage >80%

#### Testing
```bash
pytest tests/models/test_patchtst.py -v --cov=src/models/patchtst
```

---

### Milestone 18: GNN Model
**Status**: ⬜
**Goal**: Implement GNN for cross-sectional surface learning
**Reference**: [§3.6.6 GNN Model](docs/SPEC.md#366-gnn-model)
**Estimate**: 2 hours
**Dependencies**: M16

#### Deliverables
- [ ] Implement `src/models/gnn/model.py` (SurfaceGNN class)
- [ ] Support GAT and GCN layer types
- [ ] Edge attributes (delta_distance, tenor_distance)
- [ ] Implement `src/models/ensemble.py` (PatchTST_GNN_Ensemble class)
- [ ] Unit tests: forward pass shapes, edge attribute handling
- [ ] Integration test: synthetic data → GNN → predictions

#### Acceptance Criteria
- GNN respects graph topology (only adjacent nodes exchange messages)
- Edge attributes incorporated correctly (if enabled)
- Ensemble combines PatchTST + GNN correctly
- Model can overfit on small synthetic dataset
- Test coverage >80%

#### Testing
```bash
pytest tests/models/test_gnn.py -v --cov=src/models/gnn
pytest tests/models/test_ensemble.py -v --cov=src/models/ensemble
```

---

### Milestone 19: Training Infrastructure
**Status**: ⬜
**Goal**: Training loop, metrics, and checkpointing
**Reference**: [§3.6.8 Training Loop](docs/SPEC.md#368-training-loop)
**Estimate**: 2 hours
**Dependencies**: M17, M18

#### Deliverables
- [ ] Implement `src/models/train/trainer.py` (Trainer class)
- [ ] Implement _train_epoch, _validate_epoch methods
- [ ] Implement `src/models/train/loss.py` (custom loss functions: Huber, quantile)
- [ ] Implement `src/models/eval/metrics.py` (IC, Rank IC)
- [ ] Implement early stopping logic
- [ ] Implement checkpointing (save/load model state)
- [ ] Create `scripts/train_model.py` CLI script
- [ ] Unit tests for training logic
- [ ] Integration test: train for 3 epochs, verify loss decreases

#### Acceptance Criteria
- Training loop runs without errors
- Loss decreases over epochs (on synthetic data)
- Early stopping triggers when val metric plateaus
- Checkpoints saved and loadable
- CLI script accepts config path, loads datasets, runs training
- Test coverage >75%

#### Testing
```bash
pytest tests/models/test_trainer.py -v --cov=src/models/train
python scripts/train_model.py --config config/config.yaml --epochs 3
```

---

## Phase 7: Strategy & Risk

### Milestone 20: Trade Structures
**Status**: ⬜
**Goal**: Implement bounded-risk option structures
**Reference**: [Trade Structures Appendix](docs/SPEC.md#appendices), [§3.7.3 Trade Structures](docs/SPEC.md#373-trade-structures)
**Estimate**: 2 hours
**Dependencies**: M1

#### Deliverables
- [ ] Implement `src/strategy/structures/base.py` (TradeStructure abstract class)
- [ ] Implement `src/strategy/structures/calendar.py` (CalendarSpread)
- [ ] Implement `src/strategy/structures/vertical.py` (VerticalSpread)
- [ ] Implement compute_max_loss methods (debit vs credit)
- [ ] Implement compute_greeks methods (aggregate portfolio Greeks)
- [ ] Unit tests: max loss calculations, Greek aggregation
- [ ] Integration test: signal → structure → legs

#### Acceptance Criteria
- Max loss calculation matches SPEC.md risk math
- Greek aggregation uses contract multiplier (100)
- Structures always have bounded risk (no naked positions)
- Test coverage >85%

#### Testing
```bash
pytest tests/strategy/test_trade_structures.py -v --cov=src/strategy/structures
```

---

### Milestone 21: Risk Management System
**Status**: ⬜
**Goal**: Pre-trade risk checks and stress testing
**Reference**: [§3.8 Risk Management](docs/SPEC.md#38-risk-management)
**Estimate**: 2 hours
**Dependencies**: M20

#### Deliverables
- [ ] Implement `src/risk/checker.py` (RiskChecker class)
- [ ] Implement check_trade method (per-trade and portfolio-level checks)
- [ ] Implement `src/risk/stress.py` (StressEngine class)
- [ ] Implement run_stress_test method (underlying shocks, IV shocks, combined)
- [ ] Implement `src/risk/kill_switch.py` (KillSwitch class)
- [ ] Implement `src/risk/portfolio.py` (Portfolio state and Greek aggregation)
- [ ] Unit tests for each risk check
- [ ] Integration test: trade → risk check → approve/reject

#### Acceptance Criteria
- Risk checks enforce all caps (vega, gamma, delta, loss, drawdown)
- Stress tests compute worst-case loss correctly
- Kill switch triggers on breach conditions
- Portfolio Greeks aggregate correctly
- Test coverage >85%

#### Testing
```bash
pytest tests/risk/test_risk_checker.py -v
pytest tests/risk/test_stress_engine.py -v
pytest tests/risk/test_kill_switch.py -v
```

---

### Milestone 22: Order Generator & Structure Selector
**Status**: ⬜
**Goal**: Map signals to trades with structure selection
**Reference**: [§3.7.4 Structure Selector](docs/SPEC.md#374-structure-selector), [§3.7.5 Order Generator](docs/SPEC.md#375-order-generator)
**Estimate**: 1.5 hours
**Dependencies**: M20, M21

#### Deliverables
- [ ] Implement `src/strategy/selector.py` (StructureSelector class)
- [ ] Implement select_structure method (signal type → structure mapping)
- [ ] Implement `src/strategy/orders.py` (OrderGenerator class)
- [ ] Implement generate_orders method (signals → risk-checked orders)
- [ ] Implement `src/strategy/sizing.py` (position sizing logic)
- [ ] Unit tests for selection and sizing logic
- [ ] Integration test: signals → orders (end-to-end)

#### Acceptance Criteria
- Structure selection follows SPEC.md logic (term anomaly → calendar, etc.)
- Position sizing respects risk limits
- Orders rejected if fail risk checks
- All orders have bounded max loss
- Test coverage >80%

#### Testing
```bash
pytest tests/strategy/test_selector.py -v
pytest tests/strategy/test_order_generator.py -v
```

---

## Phase 8: Execution

### Milestone 23: Backtest Engine
**Status**: ⬜
**Goal**: Realistic backtesting with execution simulation
**Reference**: [§3.7 Strategy & Execution](docs/SPEC.md#37-strategy--execution)
**Estimate**: 2 hours
**Dependencies**: M22

#### Deliverables
- [ ] Implement `src/backtest/engine.py` (BacktestEngine class)
- [ ] Implement run_backtest method (simulate trading over historical data)
- [ ] Implement bid/ask execution (buy at ask, sell at bid)
- [ ] Implement slippage model (spread_fraction or fixed)
- [ ] Implement fees (per-contract)
- [ ] Implement position tracking and P&L attribution
- [ ] Create `scripts/run_backtest.py` CLI script
- [ ] Unit tests for execution logic
- [ ] Integration test: backtest over 1 month of synthetic data

#### Acceptance Criteria
- No mid-price fills (always bid/ask)
- Slippage and fees applied correctly
- P&L matches hand-calculated expectations
- All risk constraints enforced during backtest
- CLI script outputs summary metrics
- Test coverage >80%

#### Testing
```bash
pytest tests/backtest/test_engine.py -v --cov=src/backtest
python scripts/run_backtest.py --start-date 2024-01-01 --end-date 2024-01-31
```

---

### Milestone 24: Backtest Reporting
**Status**: ⬜
**Goal**: Generate backtest reports and metrics
**Reference**: [§3.7 Strategy & Execution](docs/SPEC.md#37-strategy--execution)
**Estimate**: 1.5 hours
**Dependencies**: M23

#### Deliverables
- [ ] Implement `src/backtest/metrics.py` (compute IC, Rank IC, Sharpe, drawdowns)
- [ ] Implement `src/backtest/reporting.py` (generate summary report)
- [ ] Export trades to CSV
- [ ] Export Greeks timeseries to CSV
- [ ] Generate performance plots (cumulative P&L, drawdown)
- [ ] Unit tests for metric calculations
- [ ] Integration test: backtest → report generation

#### Acceptance Criteria
- Metrics computed correctly (IC, Sharpe, max drawdown)
- Reports include trade count, win rate, avg P&L per trade
- Greeks timeseries shows portfolio exposure over time
- Plots generated and saved to artifacts/reports/
- Test coverage >75%

#### Testing
```bash
pytest tests/backtest/test_metrics.py -v
pytest tests/backtest/test_reporting.py -v
```

---

### Milestone 25: Paper Trading Loop
**Status**: ⬜
**Goal**: Live trading loop for paper trading
**Reference**: [§3.9 Live Trading Infrastructure](docs/SPEC.md#39-live-trading-infrastructure)
**Estimate**: 2 hours
**Dependencies**: M22, M23

#### Deliverables
- [ ] Implement `src/live/loop.py` (TradingLoop class)
- [ ] Implement start, stop methods
- [ ] Implement _build_latest_surface method (fetch and process latest data)
- [ ] Implement _generate_signals method (model inference)
- [ ] Implement `src/live/router.py` (OrderRouter for paper trading)
- [ ] Implement kill switch integration
- [ ] Create `scripts/run_paper_trading.py` CLI script
- [ ] Unit tests for loop logic
- [ ] Integration test: run loop for 5 cycles (mocked data)

#### Acceptance Criteria
- Loop runs continuously without crash
- Latest surface built every loop_interval_seconds
- Signals generated and orders routed
- Kill switch halts loop when triggered
- CLI script runs in foreground (Ctrl+C to stop)
- Test coverage >75%

#### Testing
```bash
pytest tests/live/test_trading_loop.py -v --cov=src/live
python scripts/run_paper_trading.py --config config/config.yaml  # Run for 30 seconds, then Ctrl+C
```

---

### Milestone 26: State Management & Monitoring
**Status**: ⬜
**Goal**: Persist state and monitor live trading
**Reference**: [§3.9.4 State Manager](docs/SPEC.md#394-state-manager)
**Estimate**: 1.5 hours
**Dependencies**: M25

#### Deliverables
- [ ] Implement `src/live/state.py` (StateManager class)
- [ ] Implement save_snapshot, load_state methods (JSON persistence)
- [ ] Implement `src/live/positions.py` (PositionTracker class)
- [ ] Implement update_positions method (mark-to-market)
- [ ] Implement `src/live/monitoring.py` (metrics logging, alerts)
- [ ] Unit tests for state persistence
- [ ] Integration test: save → crash → reload → continue

#### Acceptance Criteria
- State persisted to data/paper/state/ after every loop cycle
- Portfolio positions updated with latest surface prices
- Greeks recomputed on each update
- State recovery works after simulated crash
- Monitoring logs sent to configured destination (stdout/file)
- Test coverage >80%

#### Testing
```bash
pytest tests/live/test_state_manager.py -v
pytest tests/live/test_position_tracker.py -v
```

---

## Completion Checklist

Once all 26 milestones are complete:

- [ ] All unit tests pass (`make test`)
- [ ] All integration tests pass
- [ ] Code coverage >80% overall
- [ ] Type checking passes (`mypy src/`)
- [ ] Linting passes (`ruff check src/`)
- [ ] Documentation complete (all public APIs documented)
- [ ] End-to-end system test: ingest → surface → features → train → backtest → paper trade
- [ ] Performance acceptable (surface build <5min for 1 day of data)
- [ ] No critical TODOs remaining in code

---

## Notes

- **Estimates are rough**: Actual time may vary based on complexity and debugging
- **Test coverage**: Aim for >80% per milestone, >85% for critical paths
- **Review before merge**: Every milestone should be audited before proceeding
- **Documentation**: Update docstrings as you implement (not as afterthought)
- **Refactoring**: If a milestone reveals design issues, pause and refactor before proceeding

---

**Version**: 1.0
**Last Updated**: 2026-01-10
**Maintainer**: alex
