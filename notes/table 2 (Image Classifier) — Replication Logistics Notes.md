# Table 2 (Image Classifier) — Replication Logistics Notes

Reference for how `analysis/imagequality_classification/train_model_dinov2.py` reproduces Table 2.

**Run config:**
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

| Backbone | Avg.TNR (paper) | Avg.TNR  | Bal.Acc (paper) | Bal.Acc  | DaysTNR=0 (paper) | DaysTNR=0  | F1(NA) (paper) | F1(NA)  |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DINOv2 | 23.7% | 27.3% | 58.0% | 60.9% | 4/11 | 4/11 | 24.7% | 29.2% |
| ResNet | 20.5% | 7.6% | 57.2% | 53.8% | 5/11 | 9/11 | 21.4% | 9.9% |
| EfficientNet | 29.1% | 19.7% | 59.0% | 57.4% | 2/11 | 5/11 | 29.2% | 21.8% |

### EarlyTNR (Dy3-10)

| Backbone | EarlyTNR (paper) | EarlyTNR  |
|---|---:|---:|
| DINOv2 | 4.2% | 8.3% |
| ResNet | 0.0% | 0.0% |
| EfficientNet | 12.5% | 4.2% |

- DINOv2 matches paper within ~3-4 percentage points across all metrics.
- ResNet underperforms paper by ~13 percentage points on Avg.TNR and shows collapse pattern (9/11 days predict no NA).
- EfficientNet underperforms paper by ~10 percentage points on Avg.TNR.

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

### Empirical: ResNet with α=0.75 vs α=0.25 (identical otherwise)

| Metric | α=0.25  | α=0.75 |
|---|---:|---:|
| Avg.TNR | 7.6% | **66.7%** |
| EarlyTNR | 0.0% | **62.5%** |
| Bal.Acc | 53.8% | **60.3%** |
| DaysTNR=0 | 9/11 | **2/11** |
| F1(NA) | 9.9% | **33.0%** |

- α=0.75 overshoots paper's ResNet TNR (20.5%) by a wide margin (66.7%).
- Neither α=0.25 nor α=0.75 reproduces paper's ResNet number — α=0.25 too low, α=0.75 too high.
- α=0.25 ships as paper-spec verbatim. α=0.75 documented here as a flag for follow-up.

---

## 4. Summary

- Our setup = `train_model_dinov2.py` with α=0.25, paper-spec focal loss / augmentation / class weights, `perday_image_study.py`-aligned image source / split / no filter, training hyperparameters inherited from `train_model_accuracy_tony_dinov2.py`.
- DINOv2 matches paper closely. ResNet and EfficientNet underperform paper.
- Focal loss α direction is sensitive to `LABEL_TO_INT` convention, which was silently flipped between paper-era and current pipeline. Flagged for follow-up — α=0.75 swings ResNet TNR from 7.6% to 66.7% (overshoot), neither value reproduces paper's 20.5%.
