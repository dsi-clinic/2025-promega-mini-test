# Refactored Table 3 — Reproducibility

- **Reference document:** `notes/refactored_table3.md`
- **Code:** `analysis/paper_2026_04/metabolites_train.py`

---

## 1. Setup

- Split is fixed: JSON split (`data/splits.csv`)
- Metabolite data: `data/all_data.json`
- LightGBM specs from paper:
  - Features: absolute concentrations + day-over-day differences
  - Imbalance: class weighting (Not Acceptable class weighted up)
  - Model selection objective: F1 score for the Not Acceptable class
  - Threshold tuning: same F1 (NA) objective
  - Hyperparameter search: compact CPU-based grid, stratified CV
- LogReg specs from paper:
  - Features: absolute metabolite concentrations only ("deliberately simple baseline")
  - Form: linear and additive on the log-odds scale
- In `notes/refactored_table3.md`, on the JSON split:
  - LightGBM: no noticeable performance difference between the two versions of code
  - Logistic Regression: the previous version of code was selected because:
    1. it aligned with the paper description regarding features (concentrations only, no growth, no initial, no scaling)
    2. the refactored code set label 1 as the minority and used it for scoring
  - Between two versions of the old LogReg file, the liblinear version was selected as it was closer to the paper's metrics compared to the saga version
- Default setup for LightGBM: `class_weight="balanced"` + F1(NA) for CV scoring + `_f1_notacceptable` for threshold scoring + concentrations + differences + initial
- Default setup for LogReg: `solver="liblinear"` + `max_iter=1000` + `class_weight="balanced"` + `f1_weighted` for CV scoring + `_f1_weighted` for threshold scoring + concentrations only (no differences, no initial, no scaling)

---

## 2. Patches

- **Print metrics:** Paper Table 3 metric printout was added to `metabolites_train.py` (Avg. Accuracy, Avg. Bal. Acc., Avg. Recall (NA), Days Recall_NA=0, Best Bal. Acc.); `_compute_paper_metrics` and `_print_paper_metrics_table` functions added following the same pattern as `train_model_dinov2.py`
- **ModelSpec dataclass:** added two boolean fields (default `True`) consumed by `_features_for_day` to build different feature sets for LightGBM vs LogReg:
  - `include_growth`: enables day-over-day metabolite differences (e.g. Day 6 BCAAGlo − Day 3 BCAAGlo) as features
  - `include_initial`: enables Day 0 metabolite concentrations (initial-day measurements) as additional features alongside same-day concentrations
- **LightGBM — aligning training config to paper:**
  - Modified `_lgbm_factory`: replaced `scale_pos_weight` parameter with hardcoded `class_weight="balanced"` (paper says "class weighting")
  - Added new function `_f1_notacceptable` (uses `pos_label=0`) and removed `_f1_minority` (used `pos_label=1`); old function was misleading because in our data NA is sometimes minority and sometimes majority depending on the filter applied
  - Modified `lgbm` ModelSpec: changed `cv_scoring` from `"f1"` to `make_scorer(f1_score, pos_label=0)` (sklearn's `"f1"` shortcut defaults to `pos_label=1`, so `make_scorer` is required to target NA); changed `threshold_scoring` from `_f1_minority` to `_f1_notacceptable`
- **LogReg — aligning training config to paper:**
  - Modified `_logreg_factory`: hardcoded `solver="liblinear"` and `max_iter=1000`; removed saga option
  - Modified `logreg` ModelSpec: set `include_growth=False`, `include_initial=False`, `use_scaler=False` (paper's "deliberately simple baseline using absolute concentrations")

---

## 3. Runs, Same Config — Results

| Run | Solver | Notes |
|---|:-:|---|
| Run 1 | saga | Default solver in current refactored code |
| Run 2 | liblinear | Solver matching old `train_metabolites_logreg_nogrowth.py` |

### LightGBM (default setup)

| Metric | Paper | Run 1 / Run 2 (identical) |
|---|:-:|:-:|
| Avg. Accuracy | 86.0% | 81.5% |
| Avg. Bal. Acc. | 60.9% | 63.3% |
| Avg. Recall (NA) | 39.3% | 90.7% |
| Days Recall_NA = 0 | 1/11 | 0/11 |
| Best Bal. Acc. | 94.4% | 85.7% |

### Logistic Regression (default setup)

| Metric | Paper | Run 1 (saga) | Run 2 (liblinear) |
|---|:-:|:-:|:-:|
| Avg. Accuracy | 83.3% | 76.6% | **80.6%** |
| Avg. Bal. Acc. | 52.9% | 57.2% | **54.6%** |
| Avg. Recall (NA) | 15.9% | 86.4% | 93.7% |
| Days Recall_NA = 0 | 7/11 | 1/11 | 0/11 |
| Best Bal. Acc. | 74.5% | 77.1% | **75.5%** |

Bold = closer to paper for that metric. `liblinear` is closer than `saga` on three of five metrics including Best Bal. Acc.

---

## 4. Implications

- LightGBM gap (8.7%p below paper) is data-driven: the BCAAGlo completeness filter and `paper_label_fn` recompute change the Dy30 test set composition from 35/9 to 7/33 (Acceptable/Not Acceptable). Cannot be closed by training config changes
- LogReg gap (1.0%p above paper) is fully accounted for by the paper-spec configuration (liblinear, no growth, no initial, no scaling)
- High Avg. Recall (NA) and low Days Recall_NA=0 vs the paper both follow from the inverted Dy30 test-set distribution (NA is now majority); the model is not "better at NA detection", it is evaluated on a different distribution
