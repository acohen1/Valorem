# Audit Fix Registry

Tracks all fixes applied across audit passes. Check this before modifying any listed file
to avoid re-introducing a previously fixed issue.

---

## Audit Pass 1 — 2026-03-02 (commit 24d87dd)

"Fix 16 audit findings: remove dead features, harden configs and validation"

| # | Severity | File | Fix |
|---|----------|------|-----|
| 1 | High | `src/models/dataset.py` | Replace hardcoded `input_dim=13` with dynamic `N_FEATURES` |
| 2 | High | `src/features/node/microstructure.py` | Remove 9 dead computed-but-dropped features (spread rolling/std/change, volume_ma, oi_ma) |
| 3 | High | `src/config/schema.py` | Consolidate `MaskingConfig` into single source of truth |
| 4 | Medium | `src/features/node/iv_features.py` | Fix `min_periods=1` default to `None` (pandas window default) |
| 5 | Medium | `src/features/validators.py` | Wire up feature range validation |
| 6 | Medium | `src/models/dataset.py` | Fix synthetic dataset y leakage |
| 7 | Low | `src/data/storage/repository.py` | Narrow exception handling in count methods |
| 8 | Low | `src/models/dataset.py` | Add memmap cleanup on init failure |
| 9 | Low | `src/config/schema.py` | Remove unused `HarnessConfig` |

---

## Audit Pass 2 — 2026-03-02 (commit 86b5488)

"Audit fixes, harden numerics, and replace pooled IC with decomposed metrics"

| # | Severity | File | Fix |
|---|----------|------|-----|
| 1 | High | `src/models/eval/metrics.py` | Remove pooled IC/Rank IC — dominated by trivial gamma cross-section |
| 2 | High | `src/models/train/trainer.py` | Change early stopping default from `rank_ic` to `temporal_ic` |
| 3 | Medium | `src/models/dataset.py` | Guard NaN propagation in feature stats normalization |
| 4 | Medium | `src/surface/iv/black_scholes.py` | Fix div-by-zero in gamma, BS, spread_pct — `np.maximum` not `+1e-10` |
| 5 | Medium | `src/features/global_/returns.py` | Fix div-by-zero in returns (close=0) |
| 6 | Low | `src/data/ingest/orchestrator.py` | Add logging for silently swallowed exceptions |
| 7 | Low | `src/data/storage/engine.py` | Wrap cursor in try/finally for reliable cleanup |

---

## Audit Pass 3 — 2026-03-02 (commit cfbe652)

"Harden ablation pipeline and add linear baseline"

| # | Severity | File | Fix |
|---|----------|------|-----|
| 1 | High | `src/models/dataset.py` | Per-node normalization (shape `(num_nodes,)`) instead of global |
| 2 | High | `src/models/dataset.py` | Label boundary nullification — train labels can't cross into val |
| 3 | High | `src/models/eval/metrics.py` | Integer ranks (`np.int64`) in Spearman computations |
| 4 | Medium | `scripts/train_model.py` | Add linear baseline ablation variant |

---

## Audit Pass 4 — 2026-03-04 (commit 09588b0)

"Harden training/backtest/live data flow before ablations"

| # | Severity | File | Fix |
|---|----------|------|-----|
| 1 | High | `src/models/train/collate.py` | Emit `label_mask` in synthetic/dry-run batches |
| 2 | High | `src/backtest/data_pipeline.py` | Keep latest intraday snapshot per date |
| 3 | Medium | `src/live/features.py` | Align live tenor grid with `TENOR_DAYS_DEFAULT` |
| 4 | Medium | `src/live/signal_generator.py` | Clarify lookback semantics (timestamp periods) |
| 5 | Medium | `src/models/dataset.py` | Update dataset docs — `X` shape is `lookback+1` |

---

## Audit Pass 5 — 2026-03-06 (this commit)

Full pre-ablation pipeline audit. Findings and fixes below.

### Critical

| # | File(s) | Issue | Fix |
|---|---------|-------|-----|
| C1 | `src/features/node/iv_features.py:112-132`, `src/features/node/microstructure.py:126-132` | Rolling windows include current observation — IV z-scores shrunk toward 0, volume ratios biased toward 1.0 | Add `.shift(1)` before `.rolling()` for vol, z-score, and volume ratio |
| C2 | `src/live/features.py:87-94` | `RollingFeatureProvider` computes 6 features (wrong names) vs 29 required | Rebuild to use `DEFAULT_FEATURE_COLS` from training, delegating to feature generators |
| C3 | `src/live/features.py:418-440` | `DatabaseFeatureProvider` hardcodes `feature_version="v1.0"` and renames to non-existent columns | Accept `feature_version` param, remove stale column renames |

### High

| # | File(s) | Issue | Fix |
|---|---------|-------|-----|
| H1 | `src/features/macro/transforms.py:195-199` | Macro change uses `diff(lag_days)` on row count, not calendar days | Use `ts_utc`-based calendar day lookup for correct period diff |
| H2 | `src/models/dataset.py:843` | DHR `252/H` annualization creates 17.6x horizon weighting in MSE | Document as intentional; add comment explaining trade-off |
| H3 | `src/config/constants.py:28-29` | `TENOR_DAYS_LIVE` has 45d (no training node), lacks 120d | Remove `TENOR_DAYS_LIVE` — all paths use `TENOR_DAYS_DEFAULT` |
| H4 | `src/models/train/trainer.py:359-360` | All runs save to `best_model.pt` — dry-runs clobber production checkpoints | Include `run_id` in checkpoint filename |
| H5 | `src/features/macro/transforms.py:224-225` | Macro z-score `min_periods=2` produces degenerate early values (always ±0.707) | Set `min_periods` to `min(window, max(10, window//10))` for statistical validity (capped at window size) |

### Medium

| # | File(s) | Issue | Fix |
|---|---------|-------|-----|
| M1 | `src/models/eval/metrics.py:128-131` | Spearman IC assigns arbitrary ranks to ties | Use `scipy.stats.rankdata` with `method='average'` for proper tie handling |
| M2 | `src/features/node/surface.py:20-26` | Skew regression excludes P40/C40 deep wing buckets | Use `DELTA_BUCKETS_GRAPH` (all 7) for skew computation |
| M3 | `src/data/storage/repository.py:676-692` | No NaN/inf sanitization before DB write | Add `np.nan_to_num` for inf→NaN before `to_sql` |
| M4 | `src/features/node/iv_features.py:183-191` (and 3 others) | `_infer_steps_per_day` uses global median — fragile for irregular data | Document limitation; no code change (correct for current daily data) |
| M5 | `src/features/global_/realized_vol.py:84` | `pct_change(steps_per_day)` inconsistent at data gaps | Document limitation; no code change (underlying bars are gap-free) |
| M6 | `src/data/storage/repository.py:30-42` | Upsert-replace doesn't null out dropped columns | Document as known limitation; mitigated by `clear --derived` before rebuild |
| M7 | `src/backtest/data_pipeline.py:79`, `src/live/signal_generator.py:115` | Lookback default mismatch: backtest=21 vs live=22 | Align both to 22 (matches dataset convention) |
| M8 | `src/backtest/data_pipeline.py:424` | `build_surface_graph()` uses default config, not checkpoint config | Load graph config from checkpoint metadata |
| M9 | Build logs | 28 leakage warnings per chunk (rolling features lack early NaNs) | Root cause is cross-chunk data loading; documented as expected behavior |

### Low

| # | File(s) | Issue | Fix |
|---|---------|-------|-----|
| L1 | `src/models/dataset.py:507` | `np.std` uses ddof=0 (population std) | No change — bias negligible with 100+ obs per node |
| L2 | `src/models/train/trainer.py:571` | Validation loss not `.detach()`ed | Add `.detach()` for consistency |
| L3 | `src/models/eval/metrics.py:193-204` | XS-demeaning before Spearman is a no-op | Remove redundant demeaning step |
| L4 | `src/config/constants.py:98` | Fixed EST close hour ignores EDT (1hr TTE bias in summer) | Document limitation with comment |
| L5 | `scripts/train_model.py:584-691` | Standalone wrappers override `state_dict()` with unprefixed keys | Remove manual overrides — `nn.Module` handles submodule prefixing |
| L6 | `artifacts/valorem.db` | Empty 0-byte phantom file | Delete |

---

## Audit Pass 6 — 2026-03-06 (this commit)

Post-fix verification sweep. Most agent findings were false positives; 3 real issues found.

### High

| # | File(s) | Issue | Fix |
|---|---------|-------|-----|
| H1 | `src/features/node/iv_features.py:177` | `compute_iv_volatility()` standalone utility missing `.shift(1)` — inconsistent with main `generate()` path | Add `iv_shifted = df["iv_mid"].shift(1)` before rolling in utility |

### Medium

| # | File(s) | Issue | Fix |
|---|---------|-------|-----|
| M1 | `src/data/storage/repository.py:567` | `write_surface_snapshots()` missing inf→NaN sanitization (node_panel write has it, surface_snapshots doesn't) | Add same `replace([np.inf, -np.inf], np.nan)` before `to_sql` |
| M2 | `src/surface/builder.py:426,467` | Risk-free rate fallback (0.05) applied silently when FRED data missing or incomplete | Add `logger.warning()` for both empty-FRED and fillna fallback paths |
| M3 | `tests/integration/surface/test_surface_builder_integration.py`, `tests/integration/features/test_feature_engine_integration.py` | Integration tests write underlying bars with `timeframe="1h"`/`"1d"` but `read_underlying_bars()` defaults to `"1m"` — builder/engine get empty data, 4 tests always fail | Change test data to use `timeframe="1m"` to match production read path |

---

## Audit Pass 7 — 2026-03-06

Deep audit targeting features, models, backtest/live, and data/storage. 16 findings.

### High

| # | File(s) | Issue | Fix |
|---|---------|-------|-----|
| H1 | `src/models/train/trainer.py:438` | `TrainingResults.model_state_dict` captures last-epoch weights, not best-epoch | Reload best checkpoint before capturing `state_dict()` |
| H2 | `src/backtest/engine.py:513`, `src/live/loop.py:623` | `mark_closed` P&L sign mismatch — callers pass raw `fill.net_premium` but formula expects positive=received | Callers negate: pass `-fill.net_premium` to `record_exit` |
| H3 | `src/backtest/engine.py:449-455` | `_check_expirations` expires entire position when any single leg expires | Require `all()` legs expired |
| H4 | `src/live/loop.py:564` | `days_to_expiry -= 1` runs every loop iteration (~5s), not daily | Compute from actual `leg.expiry` dates |
| H5 | `src/live/loop.py:500` | Entry premium payment recorded as daily P&L loss | Remove — entry is cash outflow, not realized P&L |
| H6 | `src/surface/builder.py:600-602` | `GroupBy.first()` returns first non-null per column, creating chimeric rows | Use `drop_duplicates(keep="first")` |

### Medium

| # | File(s) | Issue | Fix |
|---|---------|-------|-----|
| M1 | `src/live/loop.py:616` | Exit P&L uses stale `position.unrealized_pnl` instead of fill-derived value | Compute from actual fill: `-net_premium - entry_price` |
| M2 | `src/data/providers/fred.py:230` | `CPIAUCSL` classified as percent series — CPI is index level (~300) | Remove from `percent_prefixes` |
| M3 | `src/data/providers/fred.py:222-238` | `FEDFUNDS` not matched by any percent prefix | Add `"FEDFUNDS"` to `percent_prefixes` |
| M4 | `src/models/train/trainer.py:416` | Training history `lr` column records final LR for all epochs | Record per-epoch LR in `lr_history` list during loop |
| M5 | `src/live/loop.py:657` | Spread `(ask-bid)/bid` produces `inf` when bid=0 (deep OTM) | Use mid-price denominator with NaN guard |
| M6 | `src/data/providers/fred.py:93` | `observation_end` documented as exclusive but FRED API treats as inclusive | Subtract 1 day from end before API call |
| M7 | `src/data/quality/validators.py:497` | Negative price `sample_rows` returns first 5 indices of full Series | Filter to `mask[mask].index` |

### Low

| # | File(s) | Issue | Fix |
|---|---------|-------|-----|
| L1 | `src/surface/iv/black_scholes.py:108-117` | Newton-Raphson updates already-converged options | Track per-element convergence; freeze converged |
| L2 | `src/live/monitoring.py`, `src/live/state.py` | Unbounded `_metrics_history`, `_alert_history`, and state snapshots | Cap at 10k/5k entries; keep 10 most recent snapshots |
| L3 | `src/models/patchtst/model.py` | Sinusoidal PE crashes with odd `d_model` | Add `__post_init__` validation: `d_model` must be even |

---

## Audit Pass 8 — 2026-03-06

Data flow, edge cases, and test coverage sweep. 10 findings.

### High

| # | File(s) | Issue | Fix |
|---|---------|-------|-----|
| H1 | `src/backtest/engine.py:400` | Capital check inverted: `net_premium < 0` (credit) checked instead of `> 0` (debit) | Change to `net_premium > 0` — debit trades need capital |
| H2 | `src/live/features.py:248-249,266-267` | Missing data returns `0.0` — indistinguishable from genuine zero-change signal | Return `float("nan")` — downstream `nan_to_num` handles it |

### Medium

| # | File(s) | Issue | Fix |
|---|---------|-------|-----|
| M1 | `src/config/constants.py:102`, `src/features/engine.py:348` | `DEFAULT_FRED_SERIES` missing DGS2, but model expects `DGS2_level`/`DGS2_change_1w` | Add `"DGS2"` to `DEFAULT_FRED_SERIES`; remove hardcoded `+ ["DGS2"]` |
| M2 | `src/models/dataset.py:507` vs `src/live/features.py:274` | Training uses `ddof=0`, live uses `ddof=1` for normalization std | Use `ddof=1` consistently |
| M3 | `src/backtest/engine.py:706` | `_get_underlying_price` crashes on empty surface (`iloc[0]` on empty DF) | Add `surface.empty` guard |
| M4 | `src/backtest/engine.py:452` | Expiration `<=` prevents trading on expiry day | Use strict `<` — expire day after |

### Low

| # | File(s) | Issue | Fix |
|---|---------|-------|-----|
| L1 | `src/backtest/results.py:273` | Short backtests (<21 days) produce misleading annualized returns | Return `total_return_pct` for runs < 1/12 year |
| L2 | `src/backtest/results.py:278` | `daily_pnl.std()` returns NaN for single-element Series | Guard `len > 1` |
| L3 | `src/backtest/results.py:295` | `running_max` division by zero if equity hits 0 | Replace 0 with 1e-10 |

---

## Audit Pass 9 — 2026-03-06

Final correctness sweep. 1 finding; all other areas (Greeks, buckets, GNN, PatchTST, signals, stress, sizing) verified correct.

### High

| # | File(s) | Issue | Fix |
|---|---------|-------|-----|
| H1 | `src/backtest/execution.py:196` | Fee sign error: `net_premium = gross - fees` grants fee bonus instead of charging | Change to `gross + fees` — fees always increase cost |

---

## Audit Pass 10 — 2026-03-06

Verification sweep of all pass 7-9 changes. **No regressions found.** All fixes confirmed internally consistent.
