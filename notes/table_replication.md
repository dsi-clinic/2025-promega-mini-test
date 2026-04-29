# Paper Replication Check

Comparison of paper tables against current code outputs.

**Date:** 2026-04-24
**Code version:** `nross/separation-of-concerns`
**Data:** `data/all_data.json` (5,168 records, regenerated 2026-04-17), `2026_winter_student_splits.csv` (organoid-level)

**Note:** This file is a historical comparison snapshot. After the 2026-04 schema refactor, paper scripts moved under `analysis/paper_2026_04/` and the trainer is `train_model_accuracy.py` (the legacy `run_study.py` and `data_reorg/identifiers/all_data.json` paths referenced below no longer exist).

---

## Summary

| Table | Paper Section | Code Output | Status |
|-------|---------------|-------------|--------|
| Table 1 | 3.2 | `analysis_output/figures/metabolite_summary_table.csv` | Minor differences |
| Table 2 | 4.1.2 | `analysis_output/images/perday_results.json` | EfficientNet roughly matches; ViT/ResNet not comparable (different data & split) |
| Table 3 | 4.2.3 | `analysis_output/metabolites/results.json` | Mismatch — see below; Best Bal. Acc. (LightGBM) was 0.9444 in April snapshot, now **0.7849** on 2026-04-24 rerun against fresh `all_data.json` |

---

## Table 1: Metabolite Summary Statistics

- **Paper location:** Section 3.2
- **Code:** `analysis.paper_2026_04.descriptive_stats` → `analysis_output/figures/metabolite_summary_table.csv`
- **Status:** Partial match
- **Differences:**
  1. `N` column exists in code output but not in paper
  2. Malate concentration mean — paper: 0.120, code: 0.121
  3. Malate concentration Std. Dev — paper: 0.772, code: 0.736

---

## Table 2: Three-Backbone Image Classifier Comparison

- **Paper location:** Section 4.1.2
- **Paper models:** ViT (DINOv2-base), ResNet50, EfficientNet-B0
- **Status:** Only EfficientNet is directly comparable; ViT and ResNet use different data and splits

### EfficientNet: Paper vs Code

Only EfficientNet (`run_study.py`) uses the current project setup:
- **Data:** `data/all_data.json` (original schema, 220 organoids)
- **Split:** fixed `2026_winter_student_splits.csv`
- **Output:** `analysis_output/images/perday_results.json`

**Label convention issue:** `run_study.py` encodes labels as Acceptable=0, Not Acceptable=1 — opposite to the paper. Its reported TNR is actually Acceptable recall. The table below uses its **TPR** as the paper's TNR.

| Model | Avg. TNR (All) | Early TNR (Dy3-10) | Bal. Acc. | Days TNR=0 | F1 (N.A.) |
|-------|:-:|:-:|:-:|:-:|:-:|
| EfficientNet (paper) | 29.1% | 12.5% | 59.0% | 2/11 | 29.2% |
| EfficientNet (code) | 28.3% | 16.7% | 60.6% | 4/11 | 28.6% |

Roughly consistent. Remaining differences likely due to different organoid counts (paper: 265, current: 220).

### ViT & ResNet: Not Comparable

ViT and ResNet results come from `train_model_accuracy.py`, which uses a completely different setup:

| | `run_study.py` (EfficientNet) | `train_model_accuracy.py` (ViT, ResNet) |
|---|---|---|
| Data | `data/all_data.json` (original schema) | `data_reorg/identifiers/all_data.json` (reorganized schema) |
| Split | Fixed winter CSV | Random stratified 80/10/10, seed=1 |
| Labels | Acceptable=0, Not Acceptable=1 | Accepted=1, Not Accepted=0 |
| Output metrics | TNR, bal acc, F1 directly computed | Only accuracy, F1, ROC-AUC; TNR/bal acc must be calculated from raw predictions |

To produce a valid Table 2, all three models need to run from one script with the same data, split, and label convention.

---

## Table 3: LightGBM vs Logistic Regression (Metabolite Classifiers)

- **Paper location:** Section 4.2.3
- **Code:** `analysis.paper_2026_04.metabolites_train` → `analysis_output/metabolites/results.json`
- **Status:** Mismatch; only Best Bal. Acc. (LightGBM) matches
- **Label convention:** Consistent — both models use the same encoding within one script
- **Split:** Winter split (train=158, val=18, test=44)

### Aggregate Comparison

| Metric | LightGBM (paper) | LightGBM (code) | LogReg (paper) | LogReg (code) |
|--------|:-:|:-:|:-:|:-:|
| Avg. Accuracy | 86.0% | 78.1% | 83.3% | 60.7% |
| Avg. Bal. Acc. | 60.9% | 65.7% | 52.9% | 58.0% |
| Avg. Recall (NA) | 39.3% | 47.6% | 15.9% | 54.2% |
| Days Recall_NA=0 | 1/11 | 0/11 | 7/11 | 1/11 |
| Best Bal. Acc. | 94.4% | **94.4%** | 74.5% | 67.9% |

### Organoid Count Difference

Paper used an older split with 265 organoids. Current code uses winter split with 220 organoids. This is the likely cause of the numeric differences — the same models trained/evaluated on a smaller, differently composed dataset will produce different aggregate metrics.

---

