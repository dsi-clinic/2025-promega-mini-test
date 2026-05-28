# Table 2 (Image Classifier) — Replication Logistics Notes

Reference for how `analysis/imagequality_classification/train_model_dinov2.py` reproduces Table 2.
 
**Run config (baseline):**
- Backbones: DINOv2-base, ResNet50, EfficientNet-B0
- Split: `canonical_2026_winter`
- Image source: `cm_source_image_abs` (AR-conserved 575×575 from `resized_575_square/`)
- Model input size: (384, 512)
- Augmentation: HorizontalFlip + ColorJitter (paper spec)
- Loss: Focal (γ=2.0, α=0.25) — paper spec, see §3 for label-convention caveat
- Class weight: sklearn `compute_class_weight("balanced")`
- Two-phase training: Phase 1 lr=1e-3 (patience=20), Phase 2 lr=1e-4 (patience=30)
- ImageNet normalize: DINOv2 only
---
 
## 1. Our setup vs paper
 
### Aggregate per backbone
 
| Backbone | Avg.TNR (paper) | Avg.TNR | Bal.Acc (paper) | Bal.Acc | DaysTNR=0 (paper) | DaysTNR=0 | F1(NA) (paper) | F1(NA) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DINOv2 | 23.7% | 27.3% | 58.0% | 60.9% | 4/11 | 4/11 | 24.7% | 29.2% |
| ResNet | 20.5% | 4.5% | 57.2% | 52.3% | 5/11 | 9/11 | 21.4% | 7.1% |
| EfficientNet | 29.1% | 21.2% | 59.0% | 57.9% | 2/11 | 5/11 | 29.2% | 21.2% |
 
### EarlyTNR (Dy3-10)
 
| Backbone | EarlyTNR (paper) | EarlyTNR |
|---|---:|---:|
| DINOv2 | 4.2% | 8.3% |
| ResNet | 0.0% | 0.0% |
| EfficientNet | 12.5% | 4.2% |
 
- DINOv2 matches paper within ~3-4 percentage points across all metrics.
- ResNet underperforms paper by ~16 percentage points on Avg.TNR and shows collapse pattern (9/11 days predict no NA).
- EfficientNet underperforms paper by ~8 percentage points on Avg.TNR.
---
 
## 2. Paper spec vs our code
 
### 2a. Specified by paper, implemented verbatim
 
| Paper spec | Code location |
|---|---|
| 3 backbones: DINOv2-base, ResNet50, EfficientNet-B0 | `models.py:BACKBONES_DINOV2` |
| Focal loss γ=2.0, α=0.25 | `train_model_dinov2.py:FocalLoss(γ=2.0, α=0.25)` |
| Balanced class weights | `compute_class_weight("balanced", ...)` |
| Augmentation: random horizontal flip + mild color jitter | `data.py:ImagePathDataset` |
| Per-day binary classification (Acc vs NA) | `main()` iterates `ds.days` |
| Two-phase training: frozen backbone → unfreeze | `train.py:run_phases` |
 
### 2b. Not specified by paper — our defaults
 
| Item | Our choice | Source |
|---|---|---|
| Phase 1 / 2 epochs | 100 / 300 | `train_model_accuracy_tony_dinov2.py` |
| Phase 1 / 2 LR | 1e-3 / 1e-4 | Same |
| Phase 1 / 2 patience | 20 / 30 | Same |
| LR scheduler | ReduceLROnPlateau(factor=0.5, patience=10) | Same |
| Batch size | 16 | Same |
| Seed | 1 | Same |
| ImageNet normalize for ResNet / EfficientNet | none | Same (paper doesn't specify) |
| Image source | `cm_source_image_abs` (AR-conserved 575×575) | `perday_image_study.py` |
| Model input size | (384, 512) | `perday_image_study.py` |
| Filter | none (splits only) | `perday_image_study.py` |
 
### 2c. Ambiguous paper wording
 
- **Image preprocessing pipeline**: paper says "images" without specifying mean-filled vs AR-conserved vs raw. Using `cm_source_image_abs` per `perday_image_study.py`.
- **Focal loss positive-label convention**: see §3.
---
 
## 3. Open issue — focal loss α under flipped label convention
 
- Paper spec: focal loss γ=2.0, α=0.25.
- α in focal loss multiplies the *positive class* loss → effect depends on which class is encoded as 1.
- Paper-era code: `LABEL_TO_INT = {"Acceptable": 1, ...}` → α=0.25 upweights minority Not Acceptable (weight 0.75 vs 0.25).
- Current pipeline: `LABEL_TO_INT = {"Not Acceptable": 1, "Acceptable": 0}` → label convention is **inverted** vs paper-era.
- Under current convention, α=0.25 instead upweights majority Acceptable — paper's intended class emphasis silently reversed.
- α=0.75 under current convention restores paper-equivalent per-class weighting.
Ablation run with α=0.75 only — identical setup otherwise.
 
### 3a. Aggregate: paper vs α=0.25 vs α=0.75
 
| Backbone | Metric | Paper | α=0.25 | α=0.75 |
|---|---|---:|---:|---:|
| **DINOv2** | Avg.TNR | 23.7% | 27.3% | 59.1% |
| | EarlyTNR | 4.2% | 8.3% | 58.3% |
| | Bal.Acc | 58.0% | 60.9% | 60.9% |
| | DaysTNR=0 | 4/11 | 4/11 | 0/11 |
| | F1(NA) | 24.7% | 29.2% | 33.1% |
| **ResNet** | Avg.TNR | 20.5% | 4.5% | 62.1% |
| | EarlyTNR | 0.0% | 0.0% | 70.8% |
| | Bal.Acc | 57.2% | 52.3% | 61.8% |
| | DaysTNR=0 | 5/11 | 9/11 | 1/11 |
| | F1(NA) | 21.4% | 7.1% | 34.5% |
| **EfficientNet** | Avg.TNR | 29.1% | 21.2% | 37.9% |
| | EarlyTNR | 12.5% | 4.2% | 8.3% |
| | Bal.Acc | 59.0% | 57.9% | 60.9% |
| | DaysTNR=0 | 2/11 | 5/11 | 3/11 |
| | F1(NA) | 29.2% | 21.2% | 29.2% |
 
### 3b. Findings
 
- **α=0.75 raises NA detection across all 3 backbones, but overshoots paper for DINOv2 and ResNet.** Avg.TNR jumps DINOv2 27.3% → 59.1% (paper 23.7%) and ResNet 4.5% → 62.1% (paper 20.5%).
- **ResNet is the most sensitive** to α. Under α=0.25 it collapses (9/11 days predict no NA); under α=0.75 it predicts NA on 10/11 days. Label-convention flip hits ResNet hardest.
- **EfficientNet is the least sensitive.** Both α settings stay within ~8 percentage points of paper Avg.TNR. F1(NA) under α=0.75 exactly matches paper (29.2%).
- **Neither α=0.25 nor α=0.75 reproduces paper's TNR numbers for all 3 backbones simultaneously.** Paper's TNR range (20–30%) sits between our two settings — α=0.25 too low, α=0.75 too high — suggesting paper's run differs from ours in additional unspecified ways beyond the focal α convention (e.g. optimizer, init, augmentation magnitude, library versions).
- **F1(NA) shows a precision-recall trade-off under α=0.75 in late days.** On DINOv2 Dy20.5-Dy30, α=0.75 has higher TNR but *lower* F1(NA) than α=0.25 — α=0.75 over-predicts NA, lifting recall but hurting precision.
### 3c. Per-day patterns
 
**Per-day TNR (recall on Not Acceptable):**
 
![Per-day TNR comparison](../paper/images/per_day_tnr_alpha_comparison.png)
 
α=0.75 lifts TNR almost everywhere across all 3 backbones, most dramatically on ResNet early days (0% → 30–100%). DINOv2 and ResNet under α=0.25 collapse to TNR=0 on most days; α=0.75 recovers them. EfficientNet shows the smallest gap between the two settings.
 
**Per-day F1 of Not Acceptable:**
 
![Per-day F1(NA) comparison](../paper/images/per_day_f1na_alpha_comparison.png)
 
α=0.75 wins on early/middle days for all backbones, especially ResNet (α=0.25 stuck near 0 through Dy24). α=0.25 retains an edge on DINOv2 late days (Dy20.5, Dy24, Dy28), where α=0.75 over-predicts NA and hurts precision. EfficientNet late days (Dy28, Dy30) are roughly tied.
 
### 3d. Decision
 
- α=0.25 ships as paper-spec verbatim in `train_model_dinov2.py`.
- α=0.75 documented here as a follow-up flag; reproducible by changing the single `FocalLoss(gamma=2.0, alpha=0.25)` call in `train_model_dinov2.py` to `alpha=0.75`.
---
 
## 4. Summary
 
- Our setup = `train_model_dinov2.py` with α=0.25, paper-spec focal loss / augmentation / class weights, `perday_image_study.py`-aligned image source / split / no filter, training hyperparameters inherited from `train_model_accuracy_tony_dinov2.py`.
- DINOv2 matches paper closely under α=0.25. ResNet and EfficientNet underperform paper.
- Focal loss α direction is sensitive to `LABEL_TO_INT` convention, which was silently flipped between paper-era and current pipeline. α=0.75 ablation lifts TNR for all 3 backbones, but overshoots paper for DINOv2/ResNet — suggesting paper's run differs from ours in additional unspecified ways beyond the focal α convention.
- F1(NA) reveals a precision-recall trade-off under α=0.75 on DINOv2 late days, where α=0.25 retains an edge.
