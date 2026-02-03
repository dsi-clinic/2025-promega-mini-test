# Metabolite Classifier

This directory contains scripts for training LightGBM classifiers to predict organoid quality (Acceptable vs Not Acceptable) based on metabolite features. Each script trains a separate model per day (Dy03, Dy06, etc.) since metabolite availability and patterns differ across timepoints.

## Training Models

### train_metabolites.py

Main per-day classifier using LightGBM with metabolite concentration and growth features. This is the primary model after extensive experimentation.

**Fixed Parameters (hardcoded after testing):**
- `boosting_type="gbdt"`: Gradient boosting (tested vs dart, goss)
- `threshold_mode="per_day"`: Threshold tuned separately for each day on validation data (tested vs global threshold)
- `weight_mode="both"`: Uses both `class_weight` and `scale_pos_weight` to handle class imbalance (tested vs each alone)
- `use_scaling=False`: No StandardScaler applied; LightGBM handles unscaled features well (tested vs scaled)

**Configurable Parameters:**
- `cv_scoring`: Metric used during GridSearchCV hyperparameter tuning
  - `f1_weighted`: Balances F1 across classes by support
  - `f1_notaccept`: Optimizes for detecting "Not Acceptable" organoids (minority class)
  - `macro_f1`: Equal weight to both classes regardless of support
- `threshold_metric`: Metric used to tune classification threshold on validation set (same options as above)

**Usage:**
```bash

python train_metabolites.py --cv_scoring f1_notaccept --threshold_metric f1_notaccept
```

---

### train_metabolites_SMOTE_THRES.py

Alternative approach using SMOTE oversampling to address class imbalance instead of class weighting. Uses an imblearn Pipeline combining SMOTE with LightGBM.

**Key Differences from main script:**
- Applies SMOTE with `k_neighbors=3` to synthetically oversample the minority class before training
- Uses StratifiedKFold (not StratifiedGroupKFold) since SMOTE cannot preserve group structure
- Threshold tuning based on macro F1 score to balance both classes
- No class weighting (relies on SMOTE instead)

**Note:** This was an experimental approach. Results were comparable but SMOTE can be problematic with small datasets and doesn't respect organoid grouping.

**Usage:**
```bash
python train_metabolites_SMOTE_THRES.py
```

---

### train_metabolites_trajectory.py

Trajectory-based classification using multi-day metabolite histories to predict the final Day 30 label. Instead of training one model per day with single-day features, this flattens multiple days into one feature vector per organoid.

**Motivation:** Test whether knowing an organoid's metabolite trajectory (change over time) improves prediction compared to single-day snapshots.

**Experiments (4 variants):**
1. `traj_late_Dy28`: Late-only trajectory (Dy24, Dy28) to predict at Dy28
2. `traj_allhist_Dy28`: All-history trajectory (all days <= 28) to predict at Dy28
3. `traj_late_Dy30`: Late-only trajectory (Dy24, Dy28, Dy30) to predict at Dy30
4. `traj_allhist_Dy30`: All-history trajectory (all days <= 30) to predict at Dy30

The "late" variants use only the final 2-3 timepoints (hypothesis: late-stage metabolites are most predictive). The "allhist" variants use the full trajectory (hypothesis: cumulative patterns matter).

**Usage:**
```bash
python train_metabolites_trajectory.py
python train_metabolites_trajectory.py --variant traj_late_Dy28
```

---

## Analysis Scripts

### analyze_results.py

Aggregates results from `outputs_metabolites` and generates comparison visualizations.

**Generates:**
- `combined_model_metrics.csv`: Combined raw metrics from all model variants
- `average_metrics_summary.csv`: Average metrics across all days per variant
- `best_models_summary.csv`: Best model for each metric
- Comparison plots for each metric:
  - `comparison_Test_F1_NotAcceptable.png`
  - `comparison_Test_F1_Acceptable.png`
  - `comparison_Test_Specificity.png`
  - `comparison_Test_ROC_AUC.png`

**Usage:**
```bash
python analyze_results.py
```

---

### extract_feature_importance.py

Retrains LightGBM models for specified days and extracts feature importances.

**Generates (per day):**
- `feature_importance.csv`: All features with importance scores
- `feature_names.json`: List of feature names used
- `feature_importance_top20.png`: Top 20 features visualization
- `feature_importance_full.png`: All features visualization (if <= 50 features)

**Cross-day outputs (if multiple days):**
- `feature_importance_comparison.csv`: Normalized importance across days
- `feature_importance_comparison.png`: Grouped bar chart comparison

**Usage:**
```bash
python extract_feature_importance.py --days 24 28 30
python extract_feature_importance.py --days all
```

---

## Output Structure

### outputs_metabolites/

Per-day classifier outputs organized by model variant:

```
outputs_metabolites/
└── lgbm_per_day_noscale_cv_{cv_scoring}_thresh_{threshold_metric}/
    ├── results_summary.csv              # Summary metrics for all days
    ├── metrics_by_day.png               # Per-day metrics visualization
    ├── calibration_bins.csv             # Calibration diagnostic data
    ├── calibration_curve.png            # Reliability curve plot
    ├── feature_importance_summary.csv   # Feature importance across days
    ├── feature_importance_comparison.png
    └── Dy{XX}/                          # Per-day subdirectory
        ├── metrics_test.json            # Detailed test metrics
        ├── organoid_predictions.csv     # Per-organoid predictions
        ├── confusion_matrix.png         # Confusion matrix plot
        ├── feature_importance.csv       # Feature importances
        └── feature_importance_top20.png # Top features visualization
```

### outputs_metabolites_trajectory/

Trajectory classifier outputs:

```
outputs_metabolites_trajectory/
└── traj_{mode}_Dy{target}/
    ├── metrics_test.json
    ├── organoid_predictions.csv
    ├── confusion_matrix.png
    ├── calibration_bins.csv
    └── calibration_curve.png
```

---

## Data Requirements

All scripts expect JSON data splits in `data_splits/`:
- `both_train_base.json`
- `both_val_base.json`
- `both_test_base.json`

Each JSON file contains organoid data with metabolite timepoints.
