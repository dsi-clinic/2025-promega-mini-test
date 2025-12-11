# Image Classifier

This directory contains scripts for training deep learning classifiers to predict organoid quality (Acceptable vs Not Acceptable) based on image features. Each script trains separate models per day (Dy03, Dy06, etc.) since image patterns and data availability differ across timepoints.

## Overview

The image classifier system supports three backbone architectures: **VIT**, **ResNet**, and **EfficientNet**. Based on comprehensive evaluation across multiple metrics, **EfficientNet is the recommended backbone** for production use:

**Note on VIT:** We use **VIT (Vision Transformer)** as implemented by **DINOv2**, which is a self-supervised Vision Transformer variant. DINOv2 provides state-of-the-art self-supervised learning that produces robust Vision Transformer features. Throughout this documentation, we refer to it as "VIT" while the underlying implementation uses DINOv2 from HuggingFace (`facebook/dinov2-base`).

- **42% better TNR** than ResNet (0.2913 vs 0.2046)
- **23% better TNR** than VIT (0.2913 vs 0.2371)
- **Best balanced accuracy** (0.5897) among all three backbones
- **Only model with consistent TNR > 0** in early prediction days (Dy3-10)
- ResNet has **ZERO TNR** on all 4 early prediction days (always predicts positive)

For quality control applications requiring detection of "Not Acceptable" organoids, use `train_efficientnet_improved_tnr.py` which is specifically optimized for True Negative Rate.

---

## Core Training Scripts

### 1. `train_model_accuracy.py`

Main training script that trains all three backbone architectures (VIT, ResNet, EfficientNet) per day and selects the best one by validation accuracy. This is the general-purpose training script for model comparison.

**Note:** VIT uses DINOv2 (a self-supervised Vision Transformer variant) as the implementation. DINOv2 is a state-of-the-art self-supervised learning approach that produces robust Vision Transformer features.

**Fixed Parameters (hardcoded after testing):**
- Focal loss with alpha=0.25, gamma=2.0
- ReduceLROnPlateau scheduler
- Early stopping patience=20 (Phase 1), 30 (Phase 2)
- Random horizontal flips and color jitter augmentation
- Target size: (384, 512) in (H, W) format

**Command Line Arguments:**
```bash
--outdir PATH              # Output directory (default: analysis/images/classifier/outputs_512x384_fixed_splits)
--train-split PATH         # Train split JSON file (default: data_splits/both_train_base.json)
--val-split PATH           # Validation split JSON file (default: data_splits/both_val_base.json)
--test-split PATH          # Test split JSON file (default: data_splits/both_test_base.json)
--batch-size INT           # Training batch size (default: 16)
--val-batch-size INT       # Validation/test batch size (defaults to --batch-size)
--input-path-key STR       # Image field: "img_path" or "overlay_path" (default: "img_path")
--use-mask                 # Include segmentation masks as additional input channel (flag, no value)
```

**Usage Examples:**
```bash
# RGB images, no mask
python train_model_accuracy.py --input-path-key img_path

# Fluorescence overlay images, no mask
python train_model_accuracy.py --input-path-key overlay_path

# RGB images with mask branch
python train_model_accuracy.py --input-path-key img_path --use-mask

# Custom output directory and batch size
python train_model_accuracy.py --outdir /path/to/outputs --batch-size 32 --val-batch-size 16

# Custom split files
python train_model_accuracy.py \
    --train-split data_splits/both_train_exclude_stitch_only.json \
    --val-split data_splits/both_val_exclude_stitch_only.json \
    --test-split data_splits/both_test_exclude_stitch_only.json
```

**Output Structure:**
```
{outdir}/
├── best_per_day_summary.csv          # Best backbone per day (by validation accuracy)
├── vit/
│   ├── summary_all_days.csv
│   ├── metrics_all_days.json
│   └── Dy{XX}/
│       ├── metrics_test.json
│       ├── metrics_val.json
│       └── model.pth
├── resnet/
│   └── [same structure]
└── efficientnet/
    └── [same structure]
```

---

### 2. `train_efficientnet_improved_tnr.py`

Specialized EfficientNet training optimized for True Negative Rate (TNR) performance. **Only trains EfficientNet** (not VIT or ResNet) to save time. This script is recommended for quality control applications where detecting "Not Acceptable" organoids is critical.

**Key Improvements over `train_model_accuracy.py`:**
- Model selection by **balanced accuracy** (not raw accuracy) to better handle class imbalance
- Boosted minority class weights (2.5x multiplier for "Not Acceptable" class)
- Lower focal loss alpha (0.15 instead of 0.25) to emphasize minority class
- Tracks TNR/TPR during training and validation
- Early stopping and LR scheduling monitor balanced accuracy instead of accuracy
- More aggressive data augmentation (vertical flips, rotations, perspective transforms)

**Command Line Arguments:**
```bash
--outdir PATH              # Output directory (REQUIRED)
--train-split PATH         # Train split JSON file (REQUIRED)
--val-split PATH           # Validation split JSON file (REQUIRED)
--test-split PATH          # Test split JSON file (REQUIRED)
--batch-size INT           # Batch size (default: 16)
```

**Usage Examples:**
```bash
# Basic usage with required arguments
python train_efficientnet_improved_tnr.py \
    --outdir /path/to/outputs \
    --train-split data_splits/both_train_exclude_stitch_only.json \
    --val-split data_splits/both_val_exclude_stitch_only.json \
    --test-split data_splits/both_test_exclude_stitch_only.json

# With custom batch size
python train_efficientnet_improved_tnr.py \
    --outdir /path/to/outputs \
    --train-split data_splits/both_train_base.json \
    --val-split data_splits/both_val_base.json \
    --test-split data_splits/both_test_base.json \
    --batch-size 32
```

**Output Structure:**
```
{outdir}/
├── training_summary.json              # Summary of all days
└── efficientnet/
    └── Dy{XX}/
        ├── metrics_test.json         # Test metrics including TNR, TPR, balanced accuracy
        ├── model.pth                 # Best model checkpoint
        └── training_curves.png       # Training curves including TNR/TPR plots
```

---

## SLURM Submission Script

### 3. `run_training.s`

Unified SLURM submission script for all image classifier configurations. Replaces the previous 4 separate scripts (`run_nomask_image.s`, `run_nomask_overlay.s`, `run_mask_image.s`, `run_mask_overlay.s`).

**SLURM Resource Requirements:**
- Partition: `general`
- GPU: 1x A100
- Memory: 32G
- Time: 12:00:00

**Command Line Arguments:**
```bash
--input-path-key STR       # Image field: "img_path" or "overlay_path" (REQUIRED)
--use-mask                 # Include segmentation masks (optional flag)
--outdir PATH              # Custom output directory (optional, has defaults)
```

**Output Directory Defaults:**
- `outputs_nomask_image/` (img_path, no mask)
- `outputs_nomask_overlay/` (overlay_path, no mask)
- `outputs_mask_image/` (img_path, with mask)
- `outputs_mask_overlay/` (overlay_path, with mask)

**Usage Examples:**
```bash
# RGB images, no mask
sbatch --job-name=train-img run_training.s --input-path-key img_path

# Fluorescence overlay images, no mask
sbatch --job-name=train-overlay run_training.s --input-path-key overlay_path

# RGB images with mask branch
sbatch --job-name=train-img-mask run_training.s --input-path-key img_path --use-mask

# Fluorescence overlay images with mask branch
sbatch --job-name=train-overlay-mask run_training.s --input-path-key overlay_path --use-mask

# Custom output directory
sbatch --job-name=train-img run_training.s --input-path-key img_path --outdir /custom/path
```

**Note:** The script uses `train_model_accuracy.py` by default. To use `train_efficientnet_improved_tnr.py`, modify the `PY` variable in the script.

---

## Utility Scripts

### 4. `extract_misclassified_efficientnet.py`

Extracts misclassified samples from EfficientNet results for error analysis. Generates CSV files organized by day and error type (false positives, false negatives).

**Command Line Arguments:**
```bash
--results-dir PATH         # Directory with EfficientNet training results (REQUIRED)
--test-split PATH          # Test split JSON file (REQUIRED)
--output-dir PATH          # Output directory for misclassified samples (REQUIRED)
```

**Usage Example:**
```bash
python extract_misclassified_efficientnet.py \
    --results-dir /path/to/efficientnet/results \
    --test-split data_splits/both_test_base.json \
    --output-dir /path/to/misclassified/outputs
```

**Output:**
- CSV files with misclassified organoid IDs and predictions
- Organized by day and error type

---

### 5. `generate_efficientnet_summary.py`

Generates comprehensive summary tables for EfficientNet results across all days. Aggregates metrics from `metrics_test.json` files and includes sample/organoid counts from split files.

**Command Line Arguments:**
```bash
--results-dir PATH         # Directory with EfficientNet training results (REQUIRED)
--split-prefix STR         # Prefix for split files, e.g., "both_train_exclude_stitch_only" (REQUIRED)
--output-name STR          # Output CSV filename without .csv extension (REQUIRED)
```

**Usage Example:**
```bash
python generate_efficientnet_summary.py \
    --results-dir /path/to/efficientnet/results \
    --split-prefix "both_train_exclude_stitch_only" \
    --output-name "efficientnet_summary_EXCLUDE_STITCH_ONLY"
```

**Output:**
- CSV file saved to `~/efficientnet_summary_{output-name}.csv`
- Includes: Accuracy, F1, TPR, TNR, Precision, Balanced Accuracy, ROC-AUC, PR-AUC, confusion matrix, sample counts

**Note:** The script automatically looks for split files:
- `data_splits/{split-prefix}_train.json`
- `data_splits/{split-prefix}_val.json`
- `data_splits/{split-prefix}_test.json`

---

## Data Preparation

### 6. `split_data.py`

Generates reproducible train/val/test splits at the organoid level to prevent data leakage. This script must be run before training to create the required split JSON files.

**Command Line Arguments:**
```bash
--switch STR               # Image filtering mode (default: "exclude_nothing")
                          # Options: "exclude_stitched_only", "exclude_split_only", 
                          #          "exclude_both", "exclude_nothing"
--all                      # Generate splits for all 4 switch modes (flag)
```

**Switch Modes (Image Filtering Only):**
- `exclude_stitched_only`: Exclude stitched images only (keep split)
- `exclude_split_only`: Exclude split/presplit images only (keep stitched)
- `exclude_both`: Exclude both stitched AND split/presplit images
- `exclude_nothing`: Include all images (no filtering)

**High Quality Filters (Always Applied):**
- BA1+BA2 batches only
- 4/5 vote consensus required for labels
- Complete metabolite data required (all 4 metabolites)
- Valid processed images required

**Organoid-Level Exclusion:**
If ANY day has a problematic image type (per switch), the ENTIRE organoid is excluded from all days (including metabolite data).

**Usage Examples:**
```bash
# Generate splits for default mode (exclude_nothing)
python split_data.py

# Generate splits excluding stitched images only
python split_data.py --switch exclude_stitched_only

# Generate splits for all 4 switch modes
python split_data.py --all
```

**Output:**
- Files saved to `data_splits/` directory
- Naming pattern: `both_{train|val|test}_{switch_mode}.json`
- Example: `both_train_exclude_stitch_only.json`, `both_val_exclude_stitch_only.json`, `both_test_exclude_stitch_only.json`

**Note:** Uses fixed random seed for reproducibility. Split structure: 80% Training / 20% Test (held out). Within Training: ~72% Train / ~18% Val.

---

## Documentation

### 7. `README.md`

This file. Provides comprehensive documentation for all image classifier scripts, command line arguments, usage examples, and output structures.

---

## Workflow Examples

### Complete Training Workflow

```bash
# 1. Generate data splits
python split_data.py --switch exclude_stitched_only

# 2. Train all three backbones (for comparison)
sbatch --job-name=train-img run_training.s --input-path-key img_path

# 3. Train EfficientNet with TNR optimization (recommended for production)
python train_efficientnet_improved_tnr.py \
    --outdir /path/to/efficientnet_tnr_results \
    --train-split data_splits/both_train_exclude_stitched_only.json \
    --val-split data_splits/both_val_exclude_stitched_only.json \
    --test-split data_splits/both_test_exclude_stitched_only.json

# 4. Generate summary table
python generate_efficientnet_summary.py \
    --results-dir /path/to/efficientnet_tnr_results \
    --split-prefix "both_train_exclude_stitched_only" \
    --output-name "efficientnet_tnr_summary"

# 5. Extract misclassified samples for analysis
python extract_misclassified_efficientnet.py \
    --results-dir /path/to/efficientnet_tnr_results \
    --test-split data_splits/both_test_exclude_stitched_only.json \
    --output-dir /path/to/misclassified_samples
```

---

## Data Requirements

All training scripts expect JSON data splits in `data_splits/` directory. Each JSON file contains organoid data with:
- Image paths: `img_path` (RGB brightfield) and/or `overlay_path` (fluorescence overlay)
- Optional mask paths: `mask_path` (segmentation masks)
- Labels: "Acceptable" or "Not Acceptable"
- Timepoints: Dy03, Dy06, Dy08, Dy10, Dy13, Dy15, Dy17, Dy20_5, Dy21, Dy24, Dy28, Dy30

The splits are generated by `split_data.py` using reproducible organoid-level splitting to ensure no data leakage between train/val/test sets.

---

## Model Selection Evidence

Based on comprehensive evaluation across 11 days:

| Metric | EfficientNet | VIT (DINOv2) | ResNet | Winner |
|--------|--------------|--------------|---------|---------|
| **Average TNR** | **0.2913** | 0.2371 | 0.2046 | EfficientNet |
| **Early TNR (Dy3-10)** | **0.1250** | 0.0417 | 0.0000 | EfficientNet |
| **Days TNR=0** | **2/11** | 4/11 | 5/11 | EfficientNet |
| **Balanced Accuracy** | **0.5897** | 0.5801 | 0.5722 | EfficientNet |
| **TNR/TPR Ratio** | **0.3281** | 0.2569 | 0.2177 | EfficientNet |

**Key Findings:**
- EfficientNet is **42% better** than ResNet in TNR
- EfficientNet is **23% better** than VIT (DINOv2 variant) in TNR
- ResNet has **ZERO TNR** on all 4 early prediction days (always predicts positive)
- EfficientNet is the **only viable option** for early prediction use cases

For detailed analysis, see `EFFICIENTNET_DECISION_SUMMARY.md` in the project root.
