# Changelog

All notable changes to Valorem are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.12.0] - 2026-03-25

### Changed
- **DHR labels use actual option P&L instead of Taylor approximation.** The old formula (`0.5 × gamma × ΔS²`) was always non-negative, meaning the model could not distinguish cheap from expensive options. The new formula (`option P&L − delta hedge P&L`) includes theta decay, producing labels with genuine sign variation — the variance risk premium signal.
- **XS-demeaned IC metric now measures time-varying skill.** Previously it was raw cross-sectional Spearman correlation, which a model outputting constant per-node predictions could trivially score high on. Now subtracts per-node temporal means before ranking.
- Increased PatchTST lookback from 21 to 63 timesteps (6 → 20 transformer patches).
- Increased macro data publication delay from 1 to 2 days to reduce look-ahead risk.
- Widened feature validator ranges to realistic values: theta (-500, 0), vega (0, 2.0).

### Added
- `mid_price` column on `node_panel` for DHR label construction (not a model feature).
- Macro leakage validation (`validate_no_leakage`) — previously a stub.
- Covariance estimator module for portfolio-level risk analysis.
- Quality masking enabled in config (`mask_crossed`, `mask_wide_spread`).

### Fixed
- Realized volatility features now apply `.shift(1)` before rolling variance, consistent with other feature generators.
- SQLite datetime adapter normalizes to UTC on write, preventing `merge_asof` failures from timezone mismatches.
- Lookback diagnostic in `TrainingDataPipeline` counted total rows per day instead of unique timestamps, producing a spurious warning about sub-day lookback coverage.

## [0.11.1] - 2026-03-12

### Fixed
- Volume and open interest carried through quote normalization (were silently dropped).

### Changed
- Demoted noisy internal logs to debug level.

## [0.11.0] - 2026-03-11

### Fixed
- P&L sign convention hardened: fees always increase cost, capital checks gate debit trades correctly.
- IV solver uses per-element convergence tracking.
- Dataset normalization uses `ddof=1` to match live feature computation.

### Added
- `--quotes-only` and `--fred-only` ingestion flags.
- DGS2 (2-year Treasury rate) added to default FRED series.
- Bounded monitoring history and state snapshot retention limits.

## [0.10.1] - 2026-03-06

### Fixed
- `min_periods` floor capped to not exceed window size.
- 7 test failures from changed runtime behavior (shift, lookback, checkpoint naming).

## [0.10.0] - 2026-03-04

### Changed
- **Ablation pipeline overhauled.** Per-node feature normalization replaces global normalization. GNN ablation uses a dedicated 20-feature subset (strips 9 global features identical across nodes). Split-boundary labels nullified to prevent forward price leakage.
- Early stopping uses Temporal IC by default instead of pooled Rank IC.

### Added
- Linear baseline ablation variant.
- Naive gamma baseline evaluation script.
- Scheduler and loss CLI controls (plateau, warmup, Huber delta).
- `--early-stopping-metric` flag.

### Fixed
- Synthetic/dry-run batch contract emits `label_mask` correctly.
- Backtest keeps latest intraday surface snapshot per date.
- Live/mock feature tenor grid aligned with model graph defaults.

## [0.9.0] - 2026-03-02

### Changed
- **Labels migrated from RV-gap to delta-hedged returns (DHR).** Feature set expanded from 16 to 29 features.
- Pooled IC removed from training metrics, replaced with Temporal IC and XS-demeaned IC.

### Added
- Vol-of-vol features (iv_vol), IV z-score features (iv_zscore), log open interest, and DGS2 macro features.
- `scripts/manage_data.py` for database table inspection and management.
- Volume-weighted loss function and dynamic volume-based GNN edge attributes.
- Option OHLCV bar ingestion from Databento.
- Quality masking framework for excluding unreliable surface nodes.

### Fixed
- Training pipeline OOM eliminated by progressive DataFrame freeing (~4 GB saved).
- Numerical robustness: epsilon guards replaced with `np.maximum` to prevent sign-flip bugs.
- Dead config fields and 9 unused microstructure features removed.

## [0.8.0] - 2026-02-07

### Changed
- Surface builder and feature generation chunked to prevent OOM on large date ranges.
- SQLite datetime adapter fixed to match pandas `to_sql()` format.

### Added
- Memmap-backed datasets for training data larger than RAM.
- Results repository for tracking experiments.
- Manifest generation optimized with dynamic query windows.
- Automatic data availability detection and graceful date range truncation.

### Fixed
- OOM crash during multi-month ingestion — chunks now written to DB immediately.
- Interactive cost confirmation prompt added before expensive Databento API calls.

## [0.7.0] - 2026-02-06

### Added
- **Leave-one-regime-out cross-validation** for ablation analysis (8 folds, 32 training runs).
- **Learnable edge weights:** GNN edge attributes can be made trainable via `--learnable-edges`.
- Ablation analysis notebook with regime-robust methodology.

### Changed
- Manifest cost estimates updated from actual production data.

## [0.6.0] - 2026-02-03

### Added
- **M1 ablation studies:** Three model variants via `--ablation` flag — `patchtst`, `gnn`, `ensemble`. Initial results: ensemble Rank IC 0.513, PatchTST 0.503, GNN 0.490.
- Ablation analysis notebook with publication-quality visualizations.
- GNN standalone prediction mode.

## [0.5.0] - 2026-02-02

### Changed
- **GPU optimization pass.** Batched GNN forward (5–10x), AMP mixed precision (1.5–2.5x), DataLoader tuning (2–5x).
- Training pipeline bottlenecks fixed: vectorized masked loss, GPU-side loss accumulation, numpy-based rank correlation.

### Fixed
- Backtest zero-predictions bug (dataset indices treated as timestamps).
- Expired option pricing bug (expiration check ran after pricing attempt).
- PatchTST config tuned: patch 8/4, dropout 0.2, LR 5e-4.

### Added
- CUDA dev environment overlay.
- Config cascade: CLI arg → YAML config → schema default.

## [0.4.0] - 2026-02-01

### Added
- Pricing layer (`src/pricing/`): centralized option leg pricing with surface → quotes → entry fallback cascade.
- Per-workflow file logging with rotation and retention.

### Fixed
- Risk checker max_loss bug inflating credit spread max loss from ~$500 to ~$88k, causing false trade rejections.
- P10 delta bucket config fixed to capture actual 10-delta OTM puts.
- Calendar spread nearest-strike matching fixed.

## [0.3.0] - 2026-01-30

### Added
- **Real-data training pipeline.** TrainingDataPipeline loads node panel and underlying bars from DB into DataLoaders. `--synthetic` preserves old behavior.
- **Real-data backtest pipeline** with self-describing checkpoints embedding model metadata.
- Idempotent ingestion: skip already-ingested data, `--force` to re-fetch.
- Per-chunk manifests with retry and exponential backoff.
- Unified logging across all scripts via loguru.

### Fixed
- 15 runtime bugs across the full workflow stack (OCC encoding, lazy imports, crash recovery, schema compatibility).
- Dead code removed, stale test references fixed, protocols relocated to implementation modules.

## [0.2.0] - 2026-01-26

### Changed
- **Architecture refactored for testability.** Dependency injection for BacktestEngine and FeatureEngine. Unified surface pipeline (all data flows through DB). Position state unified under PortfolioState.

### Added
- Domain exception hierarchy (`ValoremError` base with specific subtypes).
- Transactional database writes with automatic rollback.
- Runtime protocol validation.
- Centralized Greeks/DTE calculation utilities.
- Centralized constants module (eliminates 40+ magic values).
- Phase 2 paper trading: feature provider, symbol discovery, environment config, E2E tests.

## [0.1.0] - 2026-01-25

### Added
- **Core system.** PatchTST temporal encoder, GAT-based GNN for cross-sectional propagation, and combined ensemble prediction head.
- **Data pipeline.** Databento options quote ingestion, FRED macro data, SQLite storage with repository pattern.
- **Surface construction.** Newton-Raphson Black-Scholes IV solver, delta bucketing, tenor binning, Greeks computation, quality filtering.
- **Feature engine.** Node-level (Greeks, IV dynamics, microstructure), global (realized vol, drawdown), and macro (VIX, Treasury rates) feature generators.
- **Training infrastructure.** Huber/MSE loss, cosine/plateau schedulers, early stopping, checkpoint management.
- **Backtest engine.** Execution simulator, strategy structures (verticals, calendars, iron condors, skew trades), position lifecycle, exit signal generation, rebalancing, reporting.
- **Live trading loop.** Signal generation, order routing, risk checks (portfolio limits, stress tests, kill switches), state persistence with crash recovery, monitoring with alerts.
- **Test suite.** ~1700 unit and integration tests.
