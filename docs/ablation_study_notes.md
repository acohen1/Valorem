# M1 Ablation Study Notes

Working notes for the M1 ablation study. Not a formal writeup — rough observations and metric interpretations to inform the eventual paper/notebook.

## Important Status (Post-Cleanup)

As of March 2026, the historical ablation metrics in this document should be treated as **invalid** for model selection and comparison. Three pipeline fixes changed the effective training/evaluation setup:

1. **Rank dtype fix (`metrics.py`)**
   - Rank arrays now use integer ranks (`np.int64`) in Spearman computations.
   - This removes float32 rank precision artifacts in temporal and XS-demeaned IC paths.

2. **Label boundary leakage fix (`dataset.py`)**
   - Added `_extract_trading_dates()` and `_nullify_boundary_labels()` in `DatasetBuilder`.
   - Train labels crossing into validation and validation labels crossing into test are now nullified before dataset construction.
   - This removes forward-looking contamination near split boundaries.

3. **Per-node normalization (`dataset.py`)**
   - Feature normalization stats are now computed per node (shape `(num_nodes,)`) instead of globally across all 42 nodes.
   - This prevents node identity leakage via globally-normalized node-specific features.

**Implication:** All prior ablation tables/interpretations below reflect the pre-clean pipeline and must be rerun before making architecture decisions.

---

## 1. Experimental Setup

### Models

Five variants, each isolating a different source of predictive signal:

| Variant | Architecture | Params | Feature Set | What It Tests |
|---------|-------------|--------|-------------|---------------|
| Naive gamma | Raw gamma at last timestep, broadcast to 3 horizons | 0 | gamma only | Statistical floor — the trivial cross-sectional signal |
| Linear | Shared `Linear(29, 3)` on last timestep, per-node | ~90 | DEFAULT (29) | Feature-value floor — signal in features alone, no inductive bias |
| PatchTST | Per-node patched transformer over 22 timesteps, no cross-node communication | ~60K | DEFAULT (29) | Value of temporal inductive bias (patch attention over history) |
| GNN | GAT message passing on last timestep only | ~2.4K | GNN_ABLATION (20) | Value of cross-sectional inductive bias (surface topology) |
| Ensemble | PatchTST encoder -> GNN propagation -> linear head | ~69K | DEFAULT (29) | Whether temporal + cross-sectional signals are complementary |

### Feature Sets

**DEFAULT_FEATURE_COLS** (29 features, including 9 global): Used by linear, PatchTST, and ensemble. Includes greeks, IV dynamics, volume, realized vol, VIX, and rates.

**GNN_ABLATION_FEATURE_COLS** (20 node-specific features): Used by GNN-only ablation. Strips 9 global features that are identical across all 42 nodes at each timestamp:

- Realized vol: underlying_rv_5d, underlying_rv_10d, underlying_rv_21d
- VIX: VIXCLS_level, VIXCLS_change_1w
- Rates: DGS10_level, DGS10_change_1w, DGS2_level, DGS2_change_1w

**Why strip globals for GNN?** Message passing trivially averages 42 identical copies of global features, denoising without learning — an advantage PatchTST doesn't have (it processes nodes independently). The ensemble's GNN layer receives PatchTST embeddings (d_model=128), not raw features, so global-feature denoising doesn't apply there.

### Data

Data period: 2018-05-01 to 2021-12-31.

| Split | Period | Regime Character |
|-------|--------|-----------------|
| Train | 2018-05 to 2020-07 | Calm -> COVID crash -> initial recovery |
| Val | 2020-07 to 2021-04 | Post-COVID normalization, meme stock era |
| Test | 2021-04 to 2021-12 | Rate hike anticipation, supply chain vol |

---

## 2. Metrics

### Temporal IC

Per-node Spearman rank correlation of predictions vs actuals **across timestamps**, averaged over all nodes and horizons. Each node must have >= 5 valid timestamps.

**Question answered**: "For a given surface point (e.g., 30-delta 3-month call), does the model predict WHEN delta-hedged return is large vs small?"

### XS-Demeaned IC

At each timestamp, demean both predictions and targets across nodes (removing the common temporal component), then compute Spearman rank correlation on the residuals. Averaged over timestamps and horizons. Each timestamp must have >= 3 valid nodes.

**Question answered**: "At a given time, does the model predict WHICH surface nodes are more mispriced than average?"

### RMSE / MAE

Standard regression metrics. Useful for calibration and scale, but less informative than IC for ranking-based applications.

### Why Not Pooled IC?

Pooled IC (Spearman across all sample-node pairs) is dominated by the trivial cross-sectional relationship inherent in DHR targets. Since DHR = 0.5 * gamma * dS^2, and gamma varies dramatically across the surface while dS^2 is shared across nodes at each timestamp, any model that simply outputs something correlated with gamma gets high pooled IC for free. Pooled IC was removed from training metrics for this reason.

### Cross-Model Comparability

Temporal IC and XS-demeaned IC are both rank-based (Spearman), scale-invariant, and computed over the same evaluation set — same nodes, same timestamps, same targets. A temporal IC of 0.25 means "the model ranks timestamps by DHR magnitude for a given node with Spearman rho = 0.25" regardless of how the model produced its predictions.

**Primary comparisons** are ablations against the ensemble. These are nested — the ensemble contains both temporal and cross-sectional components, so removing one cleanly isolates its contribution:

- Ensemble vs PatchTST-only = value of adding cross-sectional propagation (+16% temporal IC)
- Ensemble vs GNN-only = value of adding temporal context (+26% temporal IC)
- Ensemble vs linear = value of any architectural inductive bias (+36% temporal IC)
- All trained models vs naive gamma = value of learning anything at all

**The PatchTST vs GNN head-to-head** is valid but requires caveats. They have different information sets (22 timesteps vs 1), different feature sets (29 vs 20), and arrive at their temporal IC through completely different mechanisms — PatchTST via attention over temporal history, GNN via cross-sectional regime detection from a single snapshot. The comparison reveals relative signal magnitude, not mechanistic equivalence.

The GNN's 20-feature restriction is a deliberate experimental choice (preventing message-passing denoising of globally-identical features), not a confound. A supplementary GNN run with all 29 features could demonstrate the artificial inflation this prevents.

---

## 3. Results

All metrics are from the best epoch (matching the saved checkpoint). RMSE/MAE were not logged per-epoch in the original runs and need re-verification with the fixed reporting code.

| Variant      | Temporal IC | XS-Demeaned IC | Best Epoch | Params |
|-------------|-------------|----------------|------------|--------|
| Naive gamma  | -0.0978     | 1.0000         | N/A        | 0      |
| Linear       | 0.2519      | 0.8201         | 4          | ~90    |
| PatchTST     | 0.2944      | 0.8404         | 3          | ~60K   |
| GNN          | 0.2728      | 0.8910         | 12         | ~2.4K  |
| **Ensemble** | **0.3424**  | **0.8325**     | 3          | ~69K   |

### Per-Model Interpretation

**Naive gamma**: Temporal IC is negative (-0.098) because gamma is relatively stable for a given node while DHR timing depends on realized vol (dS^2), which gamma doesn't capture. XS-Demeaned IC is 1.0 because gamma IS the cross-sectional determinant of DHR — it perfectly ranks nodes. Any trained model with temporal IC below this is learning nothing.

**Linear**: Temporal IC of 0.2519 shows that a simple linear combination of current features (greeks, IV dynamics, volume, macro) can predict DHR timing. The gap from naive gamma (0.2519 vs -0.098) quantifies the value of features beyond gamma alone. XS-Demeaned IC of 0.8201 is lower than naive gamma's perfect 1.0 — the linear model trades some cross-sectional precision for temporal prediction.

**PatchTST**: Temporal IC of 0.2944 is the highest among single-architecture models — the transformer extracts temporal patterns in IV dynamics, volume trends, and greek evolution that predict when DHR will be large. XS-Demeaned IC of 0.8404 is surprisingly high for a model with no cross-node communication; it learns "higher gamma input -> higher DHR prediction" per-node, producing correct cross-sectional ranking without message passing.

**GNN**: Temporal IC of 0.2728 is achieved with no temporal context — each timestamp is processed independently. The GNN reads the current cross-sectional state as a regime indicator: if neighboring nodes all show elevated IV changes, that signals a high-vol regime, predicting DHR timing without explicit temporal modeling. XS-Demeaned IC of 0.8910 is the highest among trained models — message passing propagates information across the surface topology for superior cross-sectional ranking.

**Ensemble**: Temporal IC of 0.3424 is the best across all variants. The GNN receives d_model=128 dimensional temporal embeddings (not raw features), so it propagates learned temporal representations across the surface — a qualitatively different operation from the GNN-only ablation. XS-Demeaned IC of 0.8325 is the lowest among trained models, revealing a tradeoff: optimizing for temporal IC sacrifices some cross-sectional precision.

---

## 4. Analysis

### Architectural Complementarity

The +16% temporal IC lift over PatchTST (and +26% over GNN) is not simply additive — it reflects genuinely complementary information channels.

**PatchTST captures temporal dynamics that the GNN cannot see.** Per-node, PatchTST attends over 22 timesteps of IV evolution, greek trajectories, and volume trends. It learns patterns like "IV term structure has been steepening for 2 weeks -> elevated DHR ahead." The GNN, seeing only the current snapshot, has no access to these trajectories.

**GNN captures cross-sectional context that PatchTST cannot see.** PatchTST processes each of the 42 nodes independently — it has no mechanism to learn that when ATM IV spikes, OTM nodes are also affected. The GNN propagates information across the surface topology, learning relationships like "if 25-delta put gamma is anomalously high relative to ATM, the entire wing is mispriced."

**The ensemble operates on temporal embeddings, not raw features.** The GNN layer in the ensemble receives PatchTST's learned representations, not raw features. It propagates temporal patterns across the surface — the ensemble's GNN can learn "node A's temporal IV pattern is similar to node B's, so their DHR predictions should be correlated" — something neither component can express alone.

**Empirical evidence**: If the two signals were redundant, the ensemble temporal IC would plateau near PatchTST's baseline (~0.294). Instead it jumps to 0.342, suggesting the temporal embeddings contain information that only becomes useful when contextualized cross-sectionally, and vice versa.

### Temporal IC vs XS-Demeaned IC Tradeoff

The corrected best-epoch metrics reveal a tradeoff that wasn't visible in the original (buggy) last-epoch reporting:

- GNN has the best XS-Demeaned IC (0.8910) but lower temporal IC (0.2728)
- Ensemble has the best temporal IC (0.3424) but the lowest XS-Demeaned IC among trained models (0.8325)
- PatchTST sits between on both axes (0.2944 / 0.8404)

When optimizing for temporal IC (early stopping on temporal IC), the model cannot simultaneously maximize cross-sectional ranking. The ensemble appears to trade some cross-sectional precision for temporal prediction power. Whether this tradeoff is fundamental or an artifact of the optimization strategy is an open question.

### Training Dynamics

PatchTST and ensemble both peak early (epoch 3) then overfit. GNN trains more stably to epoch 12. The temporal encoder drives the training dynamics; the GNN layer in the ensemble converges quickly on top of PatchTST embeddings.

### PatchTST vs GNN Head-to-Head

PatchTST clearly beats GNN on temporal IC (0.2944 vs 0.2728) — the 22-step temporal context provides a genuine edge over single-snapshot regime detection for predicting DHR timing. GNN dominates on XS-Demeaned IC (0.8910 vs 0.8404) — message passing provides a clear +0.05 edge over PatchTST's indirect gamma-as-input strategy for cross-sectional ranking. Each architecture excels at what it was designed for.

---

## 5. Volume Feature Experiments (M3b)

### Setup

Four ensemble variants tested on the same data splits using identical hyperparameters (cosine scheduler, LR 5e-04, patience 15) to isolate the effect of volume features. All use DEFAULT_FEATURE_COLS (29 features including `log_volume` and `volume_ratio_5d`).

| Variant | What Changes | Hypothesis |
|---------|-------------|------------|
| Baseline ensemble | Static 2D edge_attr `[delta_distance, tenor_distance]` | Control |
| + volume-weight | `VolumeWeightedMaskedLoss` upweights liquid nodes | Reduces noise from illiquid contracts |
| + dynamic-edges | 3D edge_attr `[delta_dist, tenor_dist, \|log_vol_src - log_vol_dst\|]` per sample | GNN sees liquidity gradients between adjacent nodes |
| + both | Volume-weighted loss + dynamic edge attributes | Combined effect |

### Results

| Variant | Best Epoch | Temporal IC | XS-Demeaned IC | RMSE | MAE |
|---------|-----------|-------------|----------------|------|-----|
| **Baseline** | 3 | **0.3424** | 0.8325 | — | — |
| + volume-weight | 10 | 0.3340 | 0.8335 | 0.2314 | 0.0750 |
| + dynamic-edges | 22 | 0.3341 | **0.8799** | 0.2303 | 0.0737 |
| + both | 38 | 0.3311 | 0.8588 | 0.2297 | 0.0775 |

### Analysis

**Temporal IC: volume features do not help.** All three volume variants underperform the baseline (0.3424). The combined variant is the weakest (0.3311). Volume information — whether through loss weighting or edge features — is neutral-to-slightly-negative for predicting *when* DHR is large. On this calm-dominated evaluation window, the temporal signal comes from IV dynamics and greek evolution, not liquidity.

**XS-Demeaned IC: dynamic edges are a clear win (+5.7%).** The `|log_volume_src - log_volume_dst|` edge feature gives the GNN direct visibility into liquidity gradients between adjacent surface nodes, improving cross-sectional ranking from 0.8325 to 0.8799. This makes mechanistic sense: during normal markets, ATM nodes are liquid and wings are thin — the GNN can use this gradient to weight message passing, routing more information from liquid nodes.

**Volume weighting hurts when combined with dynamic edges.** The combined variant's XS IC (0.8588) is worse than dynamic edges alone (0.8799). The volume-weighted loss reshapes the optimization landscape — upweighting liquid nodes may cause the model to over-index on ATM behaviour at the expense of the cross-sectional ranking that dynamic edges provide.

**Training dynamics shift dramatically.** Best epoch moves later with each volume addition: 3 → 10 → 22 → 38. The baseline ensemble finds its optimum during warmup; the combined variant needs the full cosine cycle to converge (best at LR ~9.6e-05). Volume features make the loss landscape harder to navigate, requiring lower learning rates to find good solutions. This is where the cosine scheduler's slow decay actually helped — the dynamic-edges variant (best at epoch 22, LR ~3.4e-04) and combined variant (best at epoch 38, LR ~9.6e-05) both benefited from late-cycle LR decay that was wasted on the baseline.

**Volume weighting alone is not justified.** It doesn't improve temporal IC, doesn't improve XS IC, and slows convergence. The feature is available in DEFAULT_FEATURE_COLS as an input — the model can learn from `log_volume` without the loss function being distorted by it.

### Recommendation

Use **baseline ensemble + dynamic-volume-edges** (no volume weighting) as the default configuration going forward. This preserves the XS IC improvement while avoiding the temporal IC and convergence penalties of volume-weighted loss. The temporal IC question — whether volume helps predict *when* mispricing occurs — needs M4 regime-stratified evaluation to resolve, since volume's temporal value likely manifests during regime transitions (COVID crash, meme stock mania) rather than in aggregate metrics dominated by calm periods.

---

## 6. Limitations

- **Regime coverage**: Current results are from the dev window (2018-2021), which includes the COVID crash (2020-03) but is mostly a calm, low-rate environment. Full-period results spanning rate hikes (2022-2023) are needed for definitive conclusions about model robustness across regimes.
- **Volume feature temporal value is unresolved**: Volume features are neutral for temporal IC in aggregate, but may help in stressed regimes where liquidity distribution shifts dramatically. Regime-stratified evaluation (M4) is needed to test this.
- **RMSE/MAE not yet verified for architectural ablation**: Baseline RMSE/MAE values were from the last epoch due to a reporting bug (now fixed). Volume variant RMSE/MAE are from best-epoch and are comparable across variants (~0.230 RMSE, ~0.075 MAE).
- **LR scheduling is suboptimal**: Cosine annealing with T_max=45 decays too slowly for early-peaking models (baseline best at epoch 3). The volume variants inadvertently benefited from late-cycle decay. A plateau scheduler or shorter cosine cycle may improve all variants.

---

## 7. Open Questions

- PatchTST's 0.8404 XS-Demeaned IC seems high for a model with no cross-node communication. Is this entirely from gamma-as-input, or is there feature normalization leakage (features normalized jointly across nodes)?
- GNN trains more stably — is PatchTST capacity too high? Would a smaller PatchTST (d_model=64) close the gap?
- The ensemble optimizes temporal IC at the cost of XS-Demeaned IC (0.8325, lowest among trained models). Is there a Pareto-optimal training strategy that balances both, or is this tradeoff fundamental?
- Would dynamic edges help the GNN-only ablation? The current GNN ablation uses static edges — a GNN + dynamic-edges run could show whether the XS IC boost is architecture-general or ensemble-specific.
- Does ReduceLROnPlateau improve temporal IC for the baseline? The cosine scheduler barely decays before early stopping; plateau scheduling tied to the monitored metric could help all variants converge better.
