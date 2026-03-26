# Model Infrastructure Roadmap

Next-phase improvements to the PatchTST + GNN ensemble architecture, informed by
[GRformer](https://openreview.net/forum?id=lmShn57DRD) (Liang et al., ICLR 2024 submission)
and its peer review discussion.

---

## Completed Milestones

### M1. Ablation Studies ✅ (2026-02-02)

Quantified each component's contribution with three ablation variants (`--ablation {patchtst,gnn,ensemble}`).

| Variant | Params | Best Epoch | IC | Rank IC | RMSE | MAE |
|---------|--------|------------|------|---------|------|-----|
| **ensemble** | 68,995 | 8 | **0.4919** | **0.5128** | 0.8809 | 0.6719 |
| **patchtst** | 60,227 | 9 | 0.4809 | 0.5033 | **0.8691** | **0.6579** |
| **gnn** | 2,435 | 7 | 0.4471 | 0.4899 | 0.9568 | 0.7399 |

Key finding: Ensemble achieves best Rank IC (+1.9% over PatchTST-only), confirming GNN
captures complementary spatial patterns.

### M2. Batched GNN Forward Pass ✅ (2026-02-02)

Eliminated per-sample Python loop via batched super-graph approach — single GNN forward pass
over all batch samples with offset node indices.

### M3. Learnable Edge Weights ✅ (2026-02-07)

Stored `edge_attr` as `nn.Parameter` in ensemble for gradient-based edge weight adaptation.

Results (dev window 2018-05 → 2019-12, 50 epochs):

| Variant | Best Val Metric | Final Rank IC | Train/Val Gap |
|---------|-----------------|---------------|---------------|
| **gnn** | **0.6972** | **0.6970** | 0.11 |
| **patchtst** | 0.6790 | 0.6756 | 0.18 |
| **ensemble (fixed)** | 0.6833 | 0.6580 | 0.28 |
| **ensemble (learnable)** | 0.6842 | 0.6375 | 0.38 |

Key finding: GNN dominates in the calm 2018–2019 window — regime-specific, not architectural.
Learnable edges destabilize in stationary regimes. Ranking expected to flip with regime
transitions (2020+ COVID, rate hikes). Full-period training required.

### M3b. Volume Feature Integration ✅ (2026-03-01)

Integrated option OHLCV bar data (Databento `ohlcv-1d` from OPRA.PILLAR) into the model
architecture across four dimensions:

**1. Input features** — Added `log_volume` and `volume_ratio_5d` to `DEFAULT_FEATURE_COLS`
(input_dim 16→18). Volume features are computed by `MicrostructureFeatureGenerator` and
flow through the feature engine into the model.

**2. Volume-weighted loss** (`--volume-weight`) — `VolumeWeightedMaskedLoss` upweights
high-liquidity nodes in the training objective. Extracts `log_volume` at the last timestep,
applies softplus for a smooth positive floor, and normalizes so mean weight across valid
nodes = 1.0 per sample. Prevents the model from wasting capacity fitting illiquid noise.

**3. Quality-flag mask refinement** — `MaskingConfig` in schema propagates surface quality
flags (`FLAG_LOW_VOLUME`, `FLAG_CROSSED`, `FLAG_WIDE_SPREAD`, `FLAG_STALE`) to `is_masked`,
excluding unreliable nodes from training loss. Gated by `masking.enabled` in feature config.

**4. Dynamic volume-based GNN edge attributes** (`--dynamic-volume-edges`) — Computes
per-sample `|log_volume_src - log_volume_dst|` for each edge, concatenated with static
`[delta_distance, tenor_distance]` to produce 3D edge_attr `(batch, num_edges, 3)`.
GNN `forward()` handles both 2D (static) and 3D (dynamic per-sample) edge_attr tensors.
Gives the GNN direct visibility into liquidity gradients between adjacent surface nodes.

**Files modified**:
- `src/models/dataset.py` — volume features in DEFAULT_FEATURE_COLS
- `src/models/train/loss.py` — `VolumeWeightedMaskedLoss` subclass
- `src/models/train/trainer.py` — `_compute_volume_weights()`, conditional loss wiring
- `src/config/schema.py` — `MaskingConfig` model
- `src/features/engine.py` — `_apply_quality_masks()` method
- `src/models/gnn/model.py` — configurable `edge_dim`, 2D/3D edge_attr dispatch
- `src/models/ensemble.py` — `_augment_edges_with_volume()`, `volume_feature_idx` param
- `scripts/train_model.py` — `--volume-weight`, `--dynamic-volume-edges` CLI flags

**Data status**: May 2018 test ingestion validated (26,327 rows, 2,100 symbols). Full
ingestion (June 2018 – December 2021) pending at ~$290 estimated Databento cost.

---

## Next: Full-Period Baseline

Before advancing to M4+, we need to run the full-period baseline that has been
deferred since M3. The dev window (2018–2019) is a calm, surface-dominant regime
that cannot distinguish temporal from cross-sectional architectures.

### Steps

1. **Ingest remaining option bar data** — June 2018 through December 2021 (~$290)
2. **Rebuild surfaces and features** — pick up volume data, compute `log_volume`,
   `volume_ratio_5d`, quality flag masks
3. **Full-period ablation** — all variants on 2018-01 → 2021-12 split:
   - `ensemble` (baseline)
   - `ensemble --volume-weight`
   - `ensemble --dynamic-volume-edges`
   - `ensemble --volume-weight --dynamic-volume-edges`
   - `patchtst` and `gnn` baselines for reference
4. **Evaluate volume impact** — compare Rank IC with and without volume features
   across the full window that includes COVID regime transition

### Split design

The full period must include regime transitions for meaningful evaluation:

| Split | Period | Regime character |
|-------|--------|-----------------|
| Train | 2018-05 → 2020-06 | Calm → COVID crash → initial recovery |
| Val | 2020-07 → 2021-03 | Post-COVID normalization, meme stock era |
| Test | 2021-04 → 2021-12 | Rate hike anticipation, supply chain vol |

This ensures val/test contain stressed days and the train→val boundary doesn't
land in a calm stretch.

---

## M4. Regime-Stratified Evaluation

**Goal**: Report model performance by volatility regime, enabling informed decisions
about M6 (interleaving) and volume feature value.

### Regime signal

Compute **trailing 10-trading-day realized volatility** from SPY underlying returns:

```
rv_trailing_10d(t) = annualized std of SPY close-to-close returns over [sid(t)-9, sid(t)]
```

### Bucketing

Compute thresholds on **training set only** (no distribution leakage):

| Bucket | Threshold | Purpose |
|--------|-----------|---------|
| Calm | <= p50 | Normal market conditions |
| Normal | (p50, p80] | Elevated but not extreme |
| Stressed | > p80 | High-vol episodes (~20% of train days) |

### Metrics per regime

- Mean Rank IC, Std Rank IC, % positive Rank IC
- Uplift vs baseline per regime
- **Volume feature uplift per regime** — volume weighting and dynamic edges may
  disproportionately help in stressed regimes where liquidity distribution shifts
  dramatically (flight to ATM, wide spreads on wings)

Aggregate metrics **per trading day** (not per sample) to account for lookback overlap.

### Files to modify

- `src/models/train/data_pipeline.py` — compute `rv_trailing_10d` and `regime_bucket`
- `src/models/train/trainer.py` — regime-stratified validation metrics
- `scripts/train_model.py` — per-regime metrics in training summary

### Success criteria

- Regime labels attached to every val/test sample
- Training summary includes per-regime Rank IC breakdown
- Metrics aggregated per-day for valid statistical comparison

---

## M5. RevIN (Reversible Instance Normalization)

**Goal**: Replace global z-score normalization with per-instance normalization to handle
non-stationary vol regimes without carrying training-set statistics at inference time.

### Design decisions

**Replace global z-score, don't stack on top.** Targets are `y_gap_Xd = log(RV) - log(IV²)`,
a relative measure that doesn't need output denormalization. RevIN is input-only:

```
Input: (batch*nodes, time, features)
  -> RevIN.normalize(x)       # subtract instance mean, divide by instance std
  -> PatchEmbedding
  -> TransformerEncoder
  -> Mean pool
  -> Prediction head           # outputs gap predictions directly, no denorm
```

**Masked RevIN for sparse nodes.** Compute instance mean/std over valid (non-zero) timesteps
only. Fallback: if `valid_count < lookback_days / 2` -> no-op (mean=0, std=1).

**Interaction with volume features.** `log_volume` and `volume_ratio_5d` have very different
distributional properties than IV-derived features (heavy right tail, zero-inflated for
illiquid nodes). RevIN's per-instance normalization should handle this naturally, but verify
that volume features don't dominate the instance statistics when most other features are
IV-derived.

### Files to create/modify

- `src/models/patchtst/revin.py` — `RevIN` module with masked stats + fallback
- `src/models/patchtst/model.py` — integrate RevIN before encoder
- `src/models/dataset.py` — remove global z-score path (or gate behind flag)

### Success criteria

- Training converges with RevIN enabled
- Compare val metrics vs global z-score baseline
- `feature_stats` plumbing no longer required for val/test datasets
- Volume features don't distort instance normalization statistics

---

## M6. Interleaved GNN-Transformer

**Goal**: Mix cross-sectional information at every transformer layer, not just at the end.

### Approach

Replace single `TransformerEncoder` stack with alternating temporal/spatial layers:

```
for each transformer layer:
    1. Self-attention + FFN on (batch*nodes, patches, d_model)    # temporal
    2. Reshape to (batch, nodes, patches * d_model)
    3. GAT message-passing across nodes                            # cross-sectional
    4. Reshape back to (batch*nodes, patches, d_model)
```

### Interaction with volume features

Dynamic volume edges (M3b) become especially relevant here — interleaving means the GNN
sees evolving representations at every layer, and liquidity gradients can influence how
temporal features are propagated. The volume edge signal should be consistent across layers
(same `log_volume` at last timestep), but GAT attention weights will differ as the node
representations evolve.

### Prerequisites

- **M2** ✅ — batched GNN eliminates per-sample loop
- **M3** ✅ — learnable edges tested (destabilize in calm regimes)
- **M3b** ✅ — dynamic volume edges provide per-sample edge features
- **M4** — regime eval needed to verify interleaving doesn't overfit calm regimes
- **DropEdge regularization** — edge dropout to prevent co-adaptation

### Staged training schedule

1. **Warmup with fixed edges** for N epochs (interleaved architecture, edge weights frozen)
2. **Unfreeze learnable edges** after temporal encoder stabilizes
3. **Optionally increase DropEdge** to prevent late-stage co-adaptation

### Success criteria

- Model trains end-to-end with interleaved architecture
- Val Rank IC improves over sequential ensemble baseline
- **Stressed-regime Rank IC does not degrade** vs sequential ensemble
- Training time overhead < 2x sequential ensemble

---

## Ordering

```
M1 (ablation) ✅ -> M2 (batched GNN) ✅ -> M3 (learnable edges) ✅ -> M3b (volume) ✅
  -> Full-period ingestion + baseline (all variants with/without volume features)
    -> M4 (regime-stratified eval)
      -> M5 (RevIN)
        -> M6 (interleaved, if regime results warrant)
```

**Current gate**: Full-period data ingestion. The dev window (2018–2019) is a calm,
surface-dominant regime that cannot distinguish architectures or demonstrate volume
feature value. Implementing M4/M5 on this window would produce misleading results.
Volume features are expected to show their value in regime transitions where liquidity
distribution shifts (COVID crash, meme stock mania, rate hike vol).
