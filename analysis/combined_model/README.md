# Combined Model

This module implements a **Day-Adaptive Multimodal Organoid Classifier** that fuses image morphology and metabolite features to predict organoid quality (Acceptable vs. Not Acceptable) across development days.

The key idea is that early-day predictions are image-primary (metabolites supplement), while late-day predictions are metabolite-primary (images supplement), using cross-attention fusion to switch roles by day.

---

## Files

| File | Description |
|------|-------------|
| `train_adaptive_cv.py` | Main training script — day-adaptive multimodal model with 5-fold CV |
| `submit_adaptive_multimodal.slurm` | SLURM job script to run training on the cluster |
| `compute_feature_correlation.py` | Computes CCA and PCA-based correlations between image embeddings and metabolite features |
| `submit_feature_corr.slurm` | SLURM job script to run feature correlation analysis |
| `plot_model_comparison.py` | Plots balanced accuracy comparison across combined, image-only, and metabolite-only models |

---

## Setup

All scripts use `YOUR_USERNAME` as a placeholder for the project root `/home/YOUR_USERNAME/2025-promega-mini-test`.

**Before running anything**, replace `YOUR_USERNAME` with your own cluster username throughout all scripts:

```bash
sed -i 's/YOUR_USERNAME/your_actual_username/g' analysis/combined_model/train_adaptive_cv.py
sed -i 's/YOUR_USERNAME/your_actual_username/g' analysis/combined_model/submit_adaptive_multimodal.slurm
sed -i 's/YOUR_USERNAME/your_actual_username/g' analysis/combined_model/compute_feature_correlation.py
sed -i 's/YOUR_USERNAME/your_actual_username/g' analysis/combined_model/submit_feature_corr.slurm
sed -i 's/YOUR_USERNAME/your_actual_username/g' analysis/combined_model/plot_model_comparison.py
```

---

## Model Architecture

**Backbone:** EfficientNet-B0 (pretrained, frozen during training)

**Metabolite branch:** Small MLP — Linear → ReLU → Dropout → Linear → ReLU → optional projection

**Fusion:** Adaptive cross-attention (`nn.MultiheadAttention`, 4 heads) that switches direction by day:
- **Early days (Dy03–Dy17):** Image is the query (primary), metabolites supplement
- **Late days (Dy20–Dy30):** Metabolites are the query (primary), image supplements

**Training:** 5-fold cross-validation with stratified group splits (by well), weighted BCE loss, early stopping (patience=20)

---

## How to Run

### 1. Train the combined model

```bash
sbatch analysis/combined_model/submit_adaptive_multimodal.slurm
```

Outputs saved to:
```
analysis/combined_model/outputs/adaptive_multimodal/
├── results.json
├── cv_summary.png
├── cv_metrics_overlay.png
└── cv_balanced_accuracy.png
```

To run directly (without SLURM):
```bash
python3 analysis/combined_model/train_adaptive_cv.py \
    --backbone efficientnet \
    --input-mode rgb \
    --cross-attn-proj-dim 128 \
    --cross-attn-heads 4 \
    --use-projection \
    --proj-dim 256 \
    --batch-size 16 \
    --learning-rate 1e-3 \
    --num-epochs 50 \
    --early-stopping-patience 20 \
    --use-augmentation \
    --output-dir analysis/combined_model/outputs/adaptive_multimodal
```

---

### 2. Run feature correlation analysis

```bash
sbatch analysis/combined_model/submit_feature_corr.slurm
```

Outputs saved to:
```
analysis/combined_model/outputs/feature_correlation/
├── aligned_features.csv
├── cca_by_day.csv
├── metabolite_pc_correlations.csv
├── cca_by_day.png
└── metabolite_pc_heatmap.png
```

---

### 3. Plot model comparison

This script compares the combined model against the image-only and metabolite-only models. The image-only and metabolite-only results are read from the shared project directory `/net/projects2/promega/` and should already be accessible to you.

**Note:** You must complete Step 1 first, as this script reads from `outputs/adaptive_multimodal/results.json`.

```bash
python3 analysis/combined_model/plot_model_comparison.py
```

Output saved to:
```
analysis/combined_model/outputs/model_comparison/
└── balanced_accuracy_comparison.png
```

---

## Key Arguments for `train_adaptive_cv.py`

| Argument | Default | Description |
|----------|---------|-------------|
| `--backbone` | `efficientnet` | Image backbone: `efficientnet`, `resnet`, or `vit` |
| `--input-mode` | `rgb` | Image type: `rgb` or `overlay` |
| `--use-projection` | off | Enable projection layer in metabolite branch |
| `--proj-dim` | `256` | Projection dimension (if `--use-projection` is set) |
| `--images-only` | off | Train on images only (no metabolites) |
| `--metabolites-only` | off | Train on metabolites only (no images) |
| `--num-epochs` | `50` | Max training epochs per fold |
| `--early-stopping-patience` | `20` | Early stopping patience |
| `--output-dir` | `outputs_cv` | Directory to save results |

---

## Label Convention

```
Not Acceptable / Not Accepted  →  1
Acceptable / Accepted          →  0
```

F1 score is reported for the positive class (Not Acceptable).
