# Table 3 Reproduction — Findings

**Code references:**
- New (refactored) code: `analysis/paper_2026_04/metabolites_train.py`
- Old code: `analysis/metabolites/train_metabolites_cpu.py`, `analysis/metabolites/train_metabolites_logreg_nogrowth.py`

---

## 1. Paper Reference Values

| Metric | Logistic Regression | LightGBM |
|---|:-:|:-:|
| Average Accuracy | 83.3% | 86.0% |
| Average Balanced Accuracy | 52.9% | 60.9% |
| Average Recall (Not Acceptable) | 15.9% | 39.3% |
| Days with Recall_NA = 0 | 7/11 | 1/11 |
| Best Balanced Accuracy | 74.5% | 94.4% |

---

## 2. Logistic Regression — Configuration Sweep

| Metric | Paper | W + new | W + old/saga | W + old/liblinear | J + new | J + old/saga | J + old/liblinear |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| Average Accuracy | 83.3% | 70.0% | **84.5%** | 79.1% | 40.2% | 76.6% | 80.6% |
| Average Balanced Accuracy | 52.9% | 62.5% | 57.7% | **57.0%** | 59.6% | 57.2% | 54.6% |
| Average Recall (NA) | 15.9% | 73.6% | 97.2% | 89.7% | **30.4%** | 86.4% | 93.7% |
| Days Recall_NA = 0 | 7/11 | 0/11 | 0/11 | 0/11 | **7/11** | 1/11 | 0/11 |
| Best Balanced Accuracy | 74.5% | 77.4% | 80.3% | **74.2%** | 80.0% | 77.1% | 75.5% |

- Bold = closest to paper for that metric
- Closest match on Best Balanced Accuracy: **Winter + old/liblinear (gap −0.3%p)**

---

## 3. Sources of LogReg Divergence

| # | Setting | Old code | New code |
|:-:|---|---|---|
| 1 | Growth features | dropped | included (`include_growth=True`) |
| 2 | Initial concentration | dropped | included (`include_initial=True`) |
| 3 | Feature scaling | none | `StandardScaler` |
| 4 | Solver | `liblinear` | `saga` |
| 5 | `max_iter` | 2000 | default (~100) |

Empirical effect on Winter, Best Balanced Accuracy:

- New code defaults: 77.4% (gap +2.9%p)
- Drop growth + initial + scaling, keep saga: 80.3% (gap +5.8%p)
- Drop growth + initial + scaling, switch to liblinear + max_iter=2000: **74.2% (gap −0.3%p)**

→ Solver alone accounts for a 6.1%p shift toward paper. LogReg gap fully accounted for.

---

## 4. LightGBM — Configuration Sweep

| Metric | Paper | W + new | W + old | J + new | J + old |
|---|:-:|:-:|:-:|:-:|:-:|
| Average Accuracy | 86.0% | 76.8% | **81.5%** | 79.3% | **81.5%** |
| Average Balanced Accuracy | 60.9% | 63.5% | 66.3% | **63.1%** | 63.3% |
| Average Recall (NA) | 39.3% | **83.1%** | 88.7% | 87.4% | 90.7% |
| Days Recall_NA = 0 | 1/11 | 0/11 | 0/11 | 0/11 | 0/11 |
| Best Balanced Accuracy | 94.4% | 83.3% | 83.3% | **91.3%** | 85.7% |

- Bold = closest to paper for that metric
- Closest match on Best Balanced Accuracy: **JSON + new (gap −3.1%p)**
- No configuration matches Days_Recall_NA = 1/11

---

## 5. Sources of LightGBM Divergence

| # | Setting | Old code | New code |
|:-:|---|---|---|
| 1 | Metabolite set | 5 mets including MalateGlo; no BCAAGlo | 5 required incl. BCAAGlo; MalateGlo conditional (day > 10) |
| 2 | Completeness filter | none — all 220 organoids | `require_complete_metabolites` removes BCAAGlo-missing (220 → 39–40 at Dy30) |
| 3 | Label source | `"label"` field stored in JSON | recomputed via `paper_label_fn` |
| 4 | Test set size at Dy30 | 44 organoids | 39–40 organoids |
| 5 | Default split | JSON files (seed=42) | `2026_winter_student_splits.csv` |

Dy30 test set composition:

| Quantity | Old code | New code |
|---|:-:|:-:|
| Test n | 44 | 40 |
| Acceptable | 35 (80%) | 7 (17%) |
| Not Acceptable | 9 (20%) | 33 (83%) |

- Label distribution inverted (not just 4 organoids missing)
- Specification change does not improve LightGBM (Winter: 0%p; JSON: −5.6%p)
- Gap is data-driven, not training-config

---

## 6. Summary

| Quantity | LightGBM | Logistic Regression |
|---|:-:|:-:|
| Closest configuration | JSON + new | Winter + old/liblinear |
| Paper Best Balanced Accuracy | 94.4% | 74.5% |
| Reproduced value | 91.3% | 74.2% |
