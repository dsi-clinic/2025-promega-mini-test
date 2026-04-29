# Paper Replication Check

Comparison of paper tables against current code outputs.

**Date:** 2026-04-09 (Table 2 content updated 2026-04-16; paths updated 2026-04-29)
**Code version:** `main` (post separation-of-concerns rebuild)
**Data:** `data/all_data.json` (5,168 records), `2026_winter_student_splits.csv` (220 organoids: train=158, val=18, test=44)

**Note on paths:** After the 2026-04 restructure, paper-replication scripts live under `analysis/paper_2026_04/`; the DINOv2 trainer was renamed `train_model_accuracy_tony_dinov2.py` → `train_model_dinov2.py`. Older script-name references below are kept verbatim as historical pointers.

---

## Summary

| Table | Paper Section | Code Output | Status |
|-------|---------------|-------------|--------|
| Table 1 | 3.2 | `analysis_output/figures/metabolite_summary_table.csv` | Minor differences |
| Table 2 | 4.1.2 | Script C (`train_model_accuracy_tony_dinov2.py`) + Script D (`legacy_image_backbone.py`) | Script C reproduces paper with +2-8pp offset; Script D (winter split) diverges further (+24-32pp on Avg TNR) |
| Table 3 | 4.2.3 | `analysis_output/metabolites/results.json` | Mismatch; Best Bal. Acc. only match |

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

### Four Table 2 scripts — split method

| | Script A | Script B | Script C | Script D |
|---|---|---|---|---|
| **Filename** | `train_model_accuracy.py` | `train_model_deep_ensemble.py` | `train_model_accuracy_tony_dinov2.py` | `legacy_image_backbone.py` |
| **Split method** | Random per-day (`train_test_split`, seed=1) | Random per-day (`train_test_split`, seed=1) | Fixed JSON splits | Fixed CSV split (winter) via `OrganoidDataset` |

### Element-by-element comparison against the paper

| # | Element | Script A | Script B | Script C | Script D |
|---|---|:---:|:---:|:---:|:---:|
| 1a | ViT = DINOv2-base | ✗ `vit_base_patch16_224` | ✗ `vit_base_patch16_224` | ✓ `facebook/dinov2-base` | ✓ `facebook/dinov2-base` |
| 1b | ResNet50 | ✓ | ✓ | ✓ | ✓ |
| 1c | EfficientNet-B0 | ✗ SmallCNN | ✓ | ✓ | ✓ |
| 2 | Label (Acc=1, NAcc=0) | ✓ | ✓ | ✓ | ✓ |
| 3 | Fixed split JSON | ✗ random | ✗ random | ✓ (JSON) | ✓ (winter CSV) |
| 4 | Per-day input | ✓ | ✓ | ✓ | ✓ |
| 5 | Image 384×512 | ✓ | ✓ | ✓ | ✓ |
| 6 | Batch size 16 | ✓ | ✓ | ✓ | ✓ |
| 7 | Training augmentation (flip + color jitter) | ✗ `augment=False` | ✗ `augment=False` | ✓ `augment=True` | ✓ `augment=True` |
| 8a | Focal loss (γ=2.0, α=0.25) | ✗ BCE only | ✗ BCE only | ✓ | ✓ |
| 8b | Balanced class weights | ✓ | ✓ | ✓ | ✓ |
| 9 | Two-phase (P1 LR 1e-3, P2 LR 1e-4) | ✓ | ✓ | ✓ | ✓ |
| 10 | ReduceLROnPlateau | ✗ | ✗ | ✓ | ✓ |
| 11 | Early stopping | ✓ | ✓ | ✓ | ✓ |
| 12 | Best by val accuracy | ✓ | ✓ | ✓ | ✓ |
| 13 | DINOv2 CLS token | N/A | N/A | ✓ | ✓ |
| 14 | ResNet50 global avg pool + MLP | ✓ | ✓ | ✓ | ✓ |
| 15 | EfficientNet conv + head | N/A | ✓ | ✓ | ✓ |

Script C and Script D are functionally equivalent in all modelling elements. The only real difference is the source of the train/val/test split.

### Replication results — 3-way comparison

Script C uses the JSON split. Script D uses the winter split CSV. Both splits draw from the same 220-organoid pool but disagree on 45 organoids (all in BA2 batch).

| Metric | Model | Paper | Script C (JSON) | Script D (winter) |
|---|---|:---:|:---:|:---:|
| **Avg TNR** | ViT (DINOv2) | 23.7% | 31.2% | 50.5% |
| | ResNet50 | 20.5% | 25.5% | 53.5% |
| | EfficientNet-B0 | 29.1% | 33.3% | 57.6% |
| **Early TNR (Dy3-10)** | ViT (DINOv2) | 4.2% | 12.5% | 37.5% |
| | ResNet50 | 0.0% | 0.0% | 45.8% |
| | EfficientNet-B0 | 12.5% | 12.5% | 41.7% |
| **Balanced Accuracy** | ViT (DINOv2) | 58.0% | 62.1% | 66.1% |
| | ResNet50 | 57.2% | 60.3% | 63.1% |
| | EfficientNet-B0 | 59.0% | 61.4% | 66.3% |
| **Days TNR=0** | ViT (DINOv2) | 4/11 | 3/11 | 0/11 |
| | ResNet50 | 5/11 | 5/11 | 1/11 |
| | EfficientNet-B0 | 2/11 | 2/11 | 0/11 |
| **F1 (Not Acceptable)** | ViT (DINOv2) | 24.7% | 33.1% | 41.2% |
| | ResNet50 | 21.4% | 27.5% | 36.2% |
| | EfficientNet-B0 | 29.2% | 30.6% | 41.6% |

### ROC AUC — is Script D better?

TNR and F1 are measured at a fixed cutoff (0.5), which means a model can raise these metrics just by predicting Not Acceptable more often. ROC AUC evaluates the model across all cutoffs, which means it reflects the model's ranking ability independent of any one cutoff. For DINOv2:

| | ROC AUC | PR AUC |
|---|:-:|:-:|
| Script C (JSON) | 0.678 | 0.898 |
| Script D (winter) | 0.675 | 0.482 |

ROC AUC is essentially identical, which implies the two models rank organoids with equal skill. Script D has higher TNR at the 0.5 cutoff, which implies it leans toward Not Acceptable more than Script C does. Higher paper metrics on Script D therefore reflect a different cutoff choice rather than a better model.

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
