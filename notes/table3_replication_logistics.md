# Table 3 (Metabolite Classifier) — Replication Logistics Notes

Reference for how `analysis/paper_2026_04/metabolites_train.py` reproduces Table 3. This file is referred to as **v1** throughout this document.

**Run config:**
- Split: `canonical_2026_winter`
- Metrics: `common_renamed.compute_classification_metrics`
- Days: `Dy03 ... Dy30` (11 days)
- LogReg: liblinear, sklearn-default `max_iter`, no scaler, absolute concentrations only, `f1_weighted` scoring
- LightGBM: `class_weight="balanced"`, NA F1 scoring, NA F1 threshold tuning, abs + initial + growth features

---

## 1. v1 vs paper — line-by-line

### 1a. Aggregate

| Metric | LogReg paper | LogReg v1 | LGBM paper | LGBM v1 |
|---|---:|---:|---:|---:|
| Avg Accuracy | 83.3% | 78.1% | 86.0% | 79.1% |
| Avg Balanced Acc | 52.9% | 56.1% | 60.9% | 66.0% |
| Avg Recall (NA) | 15.9% | 24.2% | 39.3% | 47.0% |
| Days Recall_NA = 0 | 7/11 | **7/11** | 1/11 | **1/11** |
| Best Balanced Acc | 74.5%<br>(Dy24) | 73.3%<br>(Dy28) | 94.4%<br>(Dy30) | 83.3%<br>(Dy30) |

- `Days Recall_NA = 0` matches exactly for both models.
- Aggregate accuracy / balanced accuracy off by a few percentage points.
- v1 detects NA slightly more aggressively in both models (LogReg 24.2 vs 15.9, LGBM 47.0 vs 39.3).
- Gaps attributable to: split assignment within the 220-organoid pool, seed, library versions.

### 1b. Six paper trends (Section 4.2.3)

| # | Trend | Paper | v1 |
|---|---|---|---|
| 1 | LGBM > LogReg Avg Bal Acc | 60.9 > 52.9 | 66.0 > 56.1 ✓ |
| 2 | LGBM >> LogReg NA recall | 39.3 > 15.9 | 47.0 > 24.2 ✓ |
| 3 | LGBM fewer Days NA=0 | 1 < 7 | 1 < 7 ✓ |
| 4 | LGBM best-day Bal Acc > LogReg | 94.4 > 74.5 | 83.3 > 73.3 ✓ |
| 5 | Accuracies similar (few percentage points) | 86.0 vs 83.3 | 79.1 vs 78.1 ✓ |
| 6 | LGBM advantage in NA detection | yes | yes ✓ |

All 6 directional claims preserved.

### 1c. Per-day Balanced Acc & NA recall

| day | LogReg BalAcc | LogReg NA_rec | LGBM BalAcc | LGBM NA_rec |
|---|---:|---:|---:|---:|
| Dy03 | 0.500 | 0.000 | 0.667 | 0.667 |
| Dy06 | 0.500 | 0.000 | 0.705 | 0.500 |
| Dy08 | 0.500 | 0.000 | 0.455 | 0.000 |
| Dy10 | 0.500 | 0.000 | 0.523 | 0.167 |
| Dy13 | 0.500 | 0.000 | 0.561 | 0.333 |
| Dy15 | 0.500 | 0.000 | 0.659 | 0.500 |
| Dy17 | 0.500 | 0.000 | 0.477 | 0.167 |
| Dy20_5 | 0.583 | 0.667 | 0.773 | 0.667 |
| Dy24 | 0.648 | 0.667 | 0.799 | 0.833 |
| Dy28 | 0.733 | 0.667 | 0.805 | 0.667 |
| Dy30 | 0.705 | 0.667 | 0.833 | 0.667 |

- LogReg NA recall = 0 on `Dy03 ... Dy17` (= 7 days). Matches paper's 7/11.
- LGBM NA recall = 0 only on `Dy08`. Matches paper's 1/11.

---

## 2. Paper spec vs our code

### 2a. Fully specified by paper, implemented verbatim

| Paper spec | Code location | Status |
|---|---|---|
| Two metabolite-only classifiers (LogReg + LGBM) | `MODEL_SPECS` | ✓ |
| Per-day training | `main()` iterates `DAY_ORDER` | ✓ |
| LGBM features: abs concentrations + day-over-day differences | `MODEL_SPECS["lgbm"]`: `include_growth=True, include_initial=True` | ✓ |
| LGBM class weighting (NA emphasis) | `_lgbm_factory()`: `class_weight="balanced"` | ✓ |
| LGBM model selection = NA F1 | `cv_scoring="f1"`, `_f1_minority` with `pos_label=1` | ✓ |
| LGBM threshold tuning = same NA F1 | `threshold_scoring=_f1_minority` | ✓ |
| Compact CPU grid | 7 hyperparameters × 2-3 values | ✓ |

### 2b. Not specified by paper — our defaults

LogReg described only as *"deliberately simple baseline"*, *"linear reference point"*. No training details given.

| Item | v1 choice | Reason |
|---|---|---|
| Solver | `liblinear` | sklearn-standard for small L1/L2 |
| `max_iter` | sklearn default (100) | liblinear converges in <100 on this data; verified empirically — see §4 |
| Scaling | `False` | consistent with "absolute" framing |
| Features | abs only (no growth, no initial) | interpretation of 2c |
| `class_weight` | `"balanced"` | only LogReg knob plausibly implied |
| `cv_scoring` | `"f1_weighted"` | class-aware but not minority-only |
| `threshold_scoring` | `_f1_weighted` | matches `cv_scoring` |

### 2c. Ambiguous paper wording

**§4.2.1**: *"biological state may be reflected not only in absolute metabolite levels but also in nonlinear thresholds and interactions across metabolites"*
- Contrasts LogReg's representational limits against LGBM.
- We read this as a feature-set hint → LogReg gets absolute concentrations only.
- Not literal transcription — alternative interpretations possible.

**§4.2.2**: *"we paired this with a model-selection objective focused on the F1 score for the Not Acceptable class, and we used the same class-specific objective during threshold tuning"*
- Grammatical antecedent of "this" = LightGBM. Surrounding sentences = LightGBM.
- We treat NA-F1 scoring as LGBM-specific (consistent with paper's reported LogReg NA recall = 15.9%; NA-F1-tuned LogReg cannot produce that).
- Different reader could extend NA-F1 to LogReg → see Section 3.

---

## 3. Why LogReg and LGBM specs are not matched — v1 vs v2

### 3a. v1 vs v2 spec

- v2 (sensitivity analysis, not committed) = v1 with LogReg's `cv_scoring` and `threshold_scoring` flipped to match LightGBM (NA F1).
- Built to test whether the v1 asymmetry (different objectives for the two models) inflates LightGBM's apparent NA-detection edge.

| Knob | LightGBM (v1 = v2) | LogReg (v1) | LogReg (v2) |
|---|---|---|---|
| `cv_scoring` | `"f1"` (NA F1) | `"f1_weighted"` | **`"f1"` (NA F1)** |
| `threshold_scoring` | `_f1_minority` | `_f1_weighted` | **`_f1_minority`** |
| `class_weight` | `"balanced"` | `"balanced"` | `"balanced"` |
| Solver | n/a (tree) | `liblinear` | `liblinear` |
| `max_iter` | n/a | sklearn default | sklearn default |
| Scaling | `False` (tree) | `False` | `False` |
| Features | abs + initial + growth | abs only | abs only |

### 3b. v1 vs v2 aggregate (LogReg)

| Metric | v1 LogReg | v2 LogReg | LGBM (same in both) |
|---|---:|---:|---:|
| Avg Accuracy | 78.1% | 60.0% | 79.1% |
| Avg Bal Acc | 56.1% | 64.6% | 66.0% |
| Avg Recall (NA) | 24.2% | **71.2%** | 47.0% |
| Days NA=0 | 7/11 | **0/11** | 1/11 |
| Best Bal Acc | 73.3% | 74.2% | 83.3% |

### 3c. Paper trends under v2

| # | Trend | v1 | v2 |
|---|---|---|---|
| 1 | LGBM > LogReg Bal Acc | 66.0 > 56.1 ✓ | 66.0 > 64.6 (marginal) |
| 2 | LGBM >> LogReg NA recall | 47.0 > 24.2 ✓ | 47.0 < 71.2 ✗ **inverted** |
| 3 | LGBM Days NA=0 < LogReg | 1 < 7 ✓ | 1 > 0 ✗ **inverted** |
| 4 | LGBM best-day Bal Acc > LogReg | 83.3 > 73.3 ✓ | 83.3 > 74.2 ✓ |
| 5 | Accuracies similar | 79.1 vs 78.1 ✓ | 79.1 vs 60.0 ✗ |
| 6 | LGBM advantage in NA detection | ✓ | ✗ (LogReg wins NA) |

- 3 of 6 trends invert under v2, including paper's headline claim (#6).
- v2 not "wrong" — measures different thing.

### 3d. Framing interpretation

- Paper design = **baseline vs main model**:
  - LGBM = built with deliberate NA-prioritisation (class weighting + NA-F1 scoring + NA-F1 threshold).
  - LogReg = low-tuning baseline exposing linear additive model limits.
  - Paper's claim: NA-detection gap = LGBM's value-add.
- v1 = reproduces this framing.
- v2 = removes the asymmetry → shows much of LGBM's NA advantage transfers to LogReg under matched objectives.

**Use:**
- v1 → Table 3 reproduction.
- v2 → sensitivity analysis. Useful for "how much of LGBM's NA advantage is model family vs training objective?" Answer: meaningful share is objective, not family.

---

## 4. Why Days NA=0 = 7/11 needed an ablation

- First reruns did not produce 7/11.
- Built one-knob-at-a-time ablation. Each row below changes **exactly one** variable from the previous row. The Configuration column names the variable that just changed; the other columns show the full LogReg spec at that step.

| # | Configuration (what changed) | Split | Solver | max_iter | Scaler | Features | Days NA=0 | NA recall |
|---|---|---|---|---:|:---:|---|---:|---:|
| 1 | Server baseline | harriet_2026_05 | liblinear | 1000 | False | abs only | **7/11** | 20.1% |
| 2 | + switch to canonical split | canonical | liblinear | 1000 | False | abs only | 7/11 | 24.2% |
| 3 | + enable scaler | canonical | liblinear | 1000 | **True** | abs only | 7/11 | 24.2% |
| 4 | + add growth & initial features | canonical | liblinear | 1000 | True | **growth + initial** | 7/11 | 27.3% |
| 5 | + switch solver to saga | canonical | **saga** | 1000 | True | growth + initial | **3/11** | 54.5% |
| 6 | + reduce max_iter to 100 (saga) | canonical | saga | **100** | True | growth + initial | 3/11 | 54.5% |
| F | **Final (this repo)** | canonical | liblinear | 100 | False | abs only | **7/11** | 24.2% |

**Effect of each variable on Days NA=0:**

| Variable | Effect |
|---|---|
| Split (harriet → canonical) | None (row 1 → 2: 7/11 → 7/11) |
| Scaler (False → True) | None (row 2 → 3: 7/11 → 7/11) |
| Features (abs only → +growth+initial) | None (row 3 → 4: 7/11 → 7/11) |
| **Solver (liblinear → saga)** | **Decisive (row 4 → 5: 7/11 → 3/11)** |
| max_iter (1000 → 100) under saga | None (row 5 → 6: 3/11 → 3/11) |
| max_iter (1000 → 100) under liblinear | None (verified: identical metrics on all 11 days, no `ConvergenceWarning`) |

**Interpretation:**

- Paper's 7/11 LogReg pattern reproducible with `liblinear` on canonical split — independent of scaler, features, and `max_iter`.
- `saga` solver shifts LogReg towards predicting NA more often (3/11 with NA recall 54.5%).
- `max_iter` is not load-bearing in either solver direction.
- Final code uses `liblinear` + sklearn-default `max_iter` and reproduces paper's 7/11.

---

## 5. Summary

- **`metabolites_train.py`** (v1) = file used for Table 3.
  - Aggregate metrics off by a few percentage points vs paper.
  - All 6 paper directional claims preserved.
  - `Days Recall_NA = 0` matches exactly for both models (7/11 and 1/11).

- **Paper spec coverage:**
  - LightGBM: fully specified by paper, reproduced verbatim.
  - LogReg: only described as "simple linear baseline" — training details are our defaults, not paper-prescribed.

- **v2** (sensitivity analysis, not committed) = sensitivity analysis only.
  - Aligns LogReg's NA-F1 scoring to LGBM.
  - Inverts paper's headline NA-detection claim.
  - Not used for Table 3.

- **LogReg NA-detection sensitivity:**
  - Paper's 7/11 pattern reproduced by `liblinear` on canonical split.
  - `saga` solver shifts LogReg towards predicting NA more often (3/11) — solver, not `max_iter`, is the decisive knob.
  - Final code uses `liblinear`.
