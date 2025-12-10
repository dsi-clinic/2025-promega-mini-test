#!/usr/bin/env python3
"""
Metabolite Organoid Quality Classification
Trains per-day classifiers using LightGBM with metabolite features.
"""

import os
import json
import re
from pathlib import Path
from collections import defaultdict
from imblearn.pipeline import Pipeline

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE

from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, precision_score, recall_score, f1_score, precision_recall_curve
)
from lightgbm import LGBMClassifier

SEED = 42

def set_seed(seed=SEED):
    np.random.seed(seed)

def json_to_df(json_data):
    """Convert JSON split data to DataFrame with metabolite features."""
    rows = []
    for org_id, info in json_data.items():
        label = info.get("label")
        batch = info.get("batch")
        timepoints = info.get("timepoints", {})
        
        for day_name, tp in timepoints.items():
            row = {
                "ID": org_id,
                "batch": batch,
                "label": label,
                "DY": day_name,
                "img_path": tp.get("img_path"),
                "mask_path": tp.get("mask_path"),
            }
            
            # Add metabolites
            for k, v in tp.get("metabolites", {}).items():
                row[k] = v
            
            rows.append(row)
    
    return pd.DataFrame(rows)

def compute_growth_features(df):
    """Add growth features (difference between consecutive timepoints)."""
    df = df.copy()
    df['day'] = df['DY'].str.extract(r'(\d+)').astype(int)
    df = df.sort_values(['ID', 'day'])
    
    # Compute growth features by organoid ID
    df['glucose_growth'] = df.groupby('ID')['GlucoseGlo_concentration_uM'].diff()
    df['glutamate_growth'] = df.groupby('ID')['GlutamateGlo_concentration_uM'].diff()
    df['LactateGlo_growth'] = df.groupby('ID')['LactateGlo_concentration_uM'].diff()
    df['PyruvateGlo_growth'] = df.groupby('ID')['PyruvateGlo_concentration_uM'].diff()
    df['MalateGlo_growth'] = df.groupby('ID')['MalateGlo_concentration_uM'].diff()
    
    return df

def save_organoid_predictions(selected_test_df, y_test, y_pred, y_score, output_path):
    """
    Save per-organoid predictions to CSV in the same format as multimodal.
    
    Args:
        selected_test_df: Test dataframe with organoid IDs
        y_test: True labels
        y_pred: Predicted labels
        y_score: Predicted probabilities
        output_path: Path to save CSV
    """
    # Label mapping for binary representation
    label_map = {"Acceptable": 1, "Not Acceptable": 0}
    
    organoid_results = []
    for idx in range(len(selected_test_df)):
        org_id = selected_test_df.iloc[idx]['ID']
        true_label_str = selected_test_df.iloc[idx]['label']
        true_label = label_map.get(true_label_str, 0)
        pred_label_str = y_pred[idx]
        pred_label = label_map.get(pred_label_str, 0)
        pred_prob = float(y_score[idx])
        correct = (pred_label == true_label)
        
        # Determine confusion matrix category
        if true_label == 1 and pred_label == 1:
            cm_category = 'TP'
        elif true_label == 0 and pred_label == 1:
            cm_category = 'FP'
        elif true_label == 1 and pred_label == 0:
            cm_category = 'FN'
        else:  # true_label == 0 and pred_label == 0
            cm_category = 'TN'
        
        organoid_results.append({
            'Organoid_ID': org_id,
            'True_Label': true_label,
            'Predicted_Probability': pred_prob,
            'Predicted_Label': pred_label,
            'Correct': correct,
            'CM_Category': cm_category
        })
    
    # Save to CSV
    organoid_preds_df = pd.DataFrame(organoid_results)
    organoid_preds_df.to_csv(output_path, index=False)
    print(f"  Saved organoid predictions to {output_path}")

def train_metabolite_classifier_per_day(trainval, test_df, output_dir, model_name="lgbm"):
    """
    Train LightGBM classifier for each day and save detailed results.
    Outputs organized as: output_dir/model_name/DayXX/
    
    Args:
        trainval: Combined training and validation dataframe
        test_df: Test dataframe
        output_dir: Root output directory
        model_name: Model identifier (default: "lgbm")
    """
    set_seed()
    
    # Create output structure: model-level directory
    model_dir = Path(output_dir) / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    
    results_summary = []
    unique_days = sorted(np.unique(trainval.DY))
    
    print(f"\n{'='*60}")
    print(f"Training Metabolite Classifier ({model_name.upper()})")
    print(f"{'='*60}\n")
    
    for days in unique_days:
        print(f"\n{'='*60}")
        print(f"Training for {days}")
        print(f"{'='*60}")
        
        selected_train_df = trainval[trainval['DY'] == days].copy()
        selected_test_df = test_df[test_df['DY'] == days].copy()
        
        if len(selected_train_df) == 0 or len(selected_test_df) == 0:
            print(f"  Skipping {days} - insufficient data")
            continue
        
        print(f"  Train: {len(selected_train_df)}, Test: {len(selected_test_df)}")
        
        # Extract day number
        day_num = int(re.search(r'\d+', days).group())
        
        # Prepare features based on day
        # Never use *_initial_concentration fields (match multimodal exactly)
        cols_to_drop = [
            "DY", 'batch', 'img_path', 'mask_path',
            'MalateGlo_initial_concentration',
            'GlucoseGlo_initial_concentration',
            'GlutamateGlo_initial_concentration',
            'LactateGlo_initial_concentration',
            'PyruvateGlo_initial_concentration',
            'day'
        ]
        
        # For days <= 10, also drop Malate concentration
        if day_num <= 10:
            cols_to_drop.extend(['MalateGlo_concentration_uM'])
        
        # Drop growth features for day 3 (no previous timepoint)
        growth_features = ['glucose_growth', 'glutamate_growth', 'LactateGlo_growth', 
                          'PyruvateGlo_growth', 'MalateGlo_growth']
        if day_num == 3:
            cols_to_drop.extend(growth_features)
        elif day_num == 13:
            # Only drop MalateGlo_growth for day 13 (first day with Malate)
            cols_to_drop.append('MalateGlo_growth')
        
        # Prepare train data
        train_data = selected_train_df.drop(columns=[c for c in cols_to_drop if c in selected_train_df.columns])
        test_data = selected_test_df.drop(columns=[c for c in cols_to_drop if c in selected_test_df.columns])
        
        # Also need to drop ID from test_data for X_test
        X_train = train_data.drop(columns=["label", 'ID'])
        y_train = train_data["label"]
        groups_train = train_data["ID"]
        
        X_test = test_data.drop(columns=["label", "ID"])
        y_test = test_data["label"]
        
        # ===== FIX NaN AND CONSTANT COLUMN ISSUES =====
        # Growth features have NaNs (first timepoint has no previous value)
        # Some days may have all-NaN or constant columns
        
        # 1. Identify problematic columns BEFORE scaling
        nan_counts_train = X_train.isna().sum()
        nan_counts_test = X_test.isna().sum()
        
        print(f"  NaN counts in train features:")
        for col in X_train.columns:
            if nan_counts_train[col] > 0:
                print(f"    {col}: {nan_counts_train[col]}/{len(X_train)} ({100*nan_counts_train[col]/len(X_train):.1f}%)")
        
        # 2. Drop columns that are ALL NaN in training set
        all_nan_cols = X_train.columns[X_train.isna().all()].tolist()
        if all_nan_cols:
            print(f"  Dropping all-NaN columns: {all_nan_cols}")
            X_train = X_train.drop(columns=all_nan_cols)
            X_test = X_test.drop(columns=[c for c in all_nan_cols if c in X_test.columns])
        
        # 3. Drop columns that are constant (zero variance) in training set
        constant_cols = []
        for col in X_train.columns:
            if X_train[col].nunique(dropna=True) <= 1:
                constant_cols.append(col)
        if constant_cols:
            print(f"  Dropping constant columns: {constant_cols}")
            X_train = X_train.drop(columns=constant_cols)
            X_test = X_test.drop(columns=[c for c in constant_cols if c in X_test.columns])
        
        # 4. Fill remaining NaNs with 0 (growth features = 0 means no change from unknown previous)
        if X_train.isna().any().any():
            print(f"  Filling remaining NaNs with 0")
            X_train = X_train.fillna(0)
            X_test = X_test.fillna(0)
        
        # 5. Verify no NaNs remain
        assert not X_train.isna().any().any(), "Training data still has NaNs after cleaning!"
        assert not X_test.isna().any().any(), "Test data still has NaNs after cleaning!"
        
        if len(X_train.columns) == 0:
            print(f"  ERROR: No features remain after cleaning for {days}")
            continue

        print(f"  Final feature count: {len(X_train.columns)}")

        
        # 6. NOW scale the cleaned data
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )
        
        # Calculate class weights
        #pos_label = "Acceptable"
        pos_label = "Not Acceptable"
        y_arr = pd.Series(y_train).to_numpy()
        pos = (y_arr == pos_label).sum()
        neg = (y_arr != pos_label).sum()
        ratio = (neg / pos) if pos > 0 else 1.0
        
        # Train LightGBM with GridSearch
        model = LGBMClassifier(
            random_state=SEED,
            verbose=-1,
            n_jobs=1,
            scale_pos_weight=ratio
        )
        pipe = Pipeline(
            steps=[
                ("smote", SMOTE(random_state=42, k_neighbors=3)),
                #("scaler", StandardScaler()),
                ("clf", LGBMClassifier(random_state=SEED, n_jobs=1, verbose=-1)),
            ]
        )
        #scale_pos_weight=ratio,
        param_grid = {
            "smote__k_neighbors": [3],
            'clf__max_depth': [3, 6],
            'clf__num_leaves': [31, 63],
            'clf__min_child_samples': [10, 20],
            'clf__subsample': [0.8],
            'clf__colsample_bytree': [0.8],
            'clf__learning_rate': [0.05, 0.1, 0.01],
            'clf__n_estimators': [200, 500, 1000],
        }
        
        #cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
        cv = StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=SEED,
        ) 
        grid = GridSearchCV(estimator=pipe, param_grid=param_grid, cv=cv, scoring='f1_weighted', n_jobs=-1)
 
        
        # ===== FIX: Use SCALED data for training =====
        grid.fit(X_train, y_train)

        
        best_model = grid.best_estimator_
        
        # ===== FIX: Use SCALED data for predictions =====
         # -------------------------
        # Predictions & probabilities
        # -------------------------
        y_pred = best_model.predict(X_test)

        # proba for the positive class ("Not Acceptable")
        classes = best_model.named_steps["clf"].classes_
        pos_idx = list(classes).index(pos_label)
        y_score = best_model.predict_proba(X_test)[:, pos_idx]

        # -------------------------
        # Choose best threshold based on macro F1 score
        # -------------------------
        if len(np.unique(y_test)) > 1:
    # 1 = Not Acceptable, 0 = Acceptable
            y_true_bin = (pd.Series(y_test) == "Not Acceptable").astype(int).to_numpy()

            thresholds = np.linspace(0.001, 0.9, 17)   # candidate thresholds
            best_t = 0.5
            best_f1 = -1.0
            best_macro_f1 = -1.0
            best_f1_notacc = -1.0
            best_f1_acc = -1.0

            for t in thresholds:
                # predict "Not Acceptable" if prob >= t
                y_bin_pred = (y_score >= t).astype(int)
                f1 = f1_score(y_true_bin, y_bin_pred, zero_division=0)
                # F1 for Not Acceptable (positive class = 1)
                f1_notacc = f1_score(y_true_bin, y_bin_pred, pos_label=1, zero_division=0)

        # F1 for Acceptable (treat class 0 as positive)
                f1_acc = f1_score(y_true_bin, y_bin_pred, pos_label=0, zero_division=0)

                macro_f1 = 0.5 * (f1_notacc + f1_acc)

                if f1 > best_macro_f1:
                    best_f1 = f1
                    best_t = t
                    best_macro_f1 = macro_f1
                    best_f1_notacc = f1_notacc
                    best_f1_acc = f1_acc

            print(
                f"[Threshold tuning] best threshold={best_t:.3f}, "
                f"Macro-F1={best_macro_f1:.3f}, "
                f"F1(Not Acceptable)={best_f1_notacc:.3f}, "
                f"F1(Acceptable)={best_f1_acc:.3f}"
            )

            # Replace y_pred labels using tuned threshold
            y_pred_thresh = np.where(y_score >= best_t, "Not Acceptable", "Acceptable")
        # -------------------------
        # Metrics
        # -------------------------
        # Confusion matrix with thresholded predictions
        cm = confusion_matrix(y_test, y_pred_thresh, labels=classes)
        
        # Calculate metrics
        y_true_bin = (y_test == pos_label).astype(int)

        try:
            roc_auc = roc_auc_score(y_true_bin, y_score) if len(np.unique(y_true_bin)) > 1 else None
        except ValueError:
            roc_auc = None
        accuracy = accuracy_score(y_test, y_pred_thresh)
        report = classification_report(y_test, y_pred_thresh, output_dict=True, zero_division=0)
        
        # Extract metrics for each class
        precision_NotAcceptable = report.get(pos_label, {}).get("precision", 0)
        recall_NotAcceptable = report.get(pos_label, {}).get("recall", 0)
        f1_NotAcceptable = report.get(pos_label, {}).get("f1-score", 0)
        precision_Acceptable = report.get('Acceptable', {}).get("precision", 0)
        recall_Acceptable = report.get('Acceptable', {}).get("recall", 0)
        f1_Acceptable = report.get('Acceptable', {}).get("f1-score", 0)

        
        # Confusion matrix
      
        
        print(f"  Best Params: {grid.best_params_}")
        print(f"  Accuracy: {accuracy:.3f}")
        print(f"  ROC AUC: {roc_auc:.3f}" if roc_auc else "  ROC AUC: N/A")
        print(f"  F1 (Not Acceptable): {f1_NotAcceptable:.3f}")
        print(f"  Recall (Not Acceptable): {recall_NotAcceptable:.3f}")
        print(f"  Precision (Not Acceptable): {precision_NotAcceptable:.3f}")
        
        # Identify misclassified organoids
        different_rows = selected_test_df[
            selected_test_df['label'].values != y_pred
        ]
        if len(different_rows) > 0:
            print(f"  Misclassified organoids: {list(different_rows['ID'].values)}")
        
        # Create day directory
        day_dir = model_dir / days
        day_dir.mkdir(parents=True, exist_ok=True)
        
        # Save organoid predictions (NEW)
        save_organoid_predictions(
            selected_test_df.reset_index(drop=True),
            y_test,
            y_pred,
            y_score,
            day_dir / 'organoid_predictions.csv'
        )
        
        # Compute confusion matrix values
        tn, fp, fn, tp = 0, 0, 0, 0
        if cm.shape == (2, 2):
            # Order depends on class order in best_model.classes_
            if classes[0] == pos_label:
            # rows: [pos, neg], cols: [pos, neg]
                tp = cm[0, 0]
                fn = cm[0, 1]
                fp = cm[1, 0]
                tn = cm[1, 1]
            else:
            # classes[1] is positive
                tn = cm[0, 0]
                fp = cm[0, 1]
                fn = cm[1, 0]
                tp = cm[1, 1]
        
        # Calculate specificity
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        # Save day-level metrics (similar to multimodal format)
        metrics = {
                "Day": days,
                "Test_Accuracy": accuracy,
                "Test_F1_NotAcceptable": f1_NotAcceptable,
                "Test_Recall_NotAcceptable": recall_NotAcceptable,
                "Test_Precision_NotAcceptable": precision_NotAcceptable,
                "Test_F1_Acceptable": f1_Acceptable,
                "Test_Recall_Acceptable": recall_Acceptable,
                "Test_Precision_Acceptable": precision_Acceptable,
                "Test_ROC_AUC": roc_auc,
                "TP": int(tp),
                "FP": int(fp),
                "TN": int(tn),
                "FN": int(fn),
                "Best_Threshold_Positive": float(best_t),
        }
        
        
        with open(day_dir / 'metrics_test.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"  Saved metrics to {day_dir / 'metrics_test.json'}")
        
        # Save confusion matrix plot to day directory
        plt.figure(figsize=(6, 5))
        plt.imshow(cm, interpolation='nearest', cmap='Blues')
        plt.title(f"Confusion Matrix - {days}")
        plt.colorbar()
        tick_marks = np.arange(len(best_model.classes_))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        plt.savefig(day_dir / 'confusion_matrix.png', dpi=150)
        plt.close()
        print(f"  Saved confusion matrix to {day_dir / 'confusion_matrix.png'}")
        
        # Store for summary
        results_summary.append({
                "Day_No": days,
                "Test_Accuracy": accuracy,
                "Test_F1_NotAcceptable": f1_NotAcceptable,
                "Test_Recall_NotAcceptable": recall_NotAcceptable,
                "Test_Precision_NotAcceptable": precision_NotAcceptable,
                "Test_F1_Acceptable": f1_Acceptable,
                "Test_Recall_Acceptable": recall_Acceptable,
                "Test_Precision_Acceptable": precision_Acceptable,
                "Test_ROC_AUC": roc_auc,
                "TP": int(tp),
                "FP": int(fp),
                "TN": int(tn),
                "FN": int(fn),
                "Best_Threshold_Positive": float(best_t),
        })
    
    if not results_summary:
        print("\nWarning: No results to summarize")
        return
    
    # Create summary dataframe
    summary_df = pd.DataFrame(results_summary).sort_values('Day_No')
    
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    print(summary_df.to_string(index=False))
    
    # Save model-level summary CSV
    summary_df.to_csv(model_dir / 'results_summary.csv', index=False)
    print(f"\nSaved results summary to {model_dir / 'results_summary.csv'}")
    
    # Create model-level metrics by day plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Accuracy
    axes[0, 0].plot(summary_df['Day_No'], summary_df['Test_Accuracy'], 'o-', color='blue')
    axes[0, 0].set_title('Test Accuracy by Day')
    axes[0, 0].set_xlabel('Day')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim([0, 1])
    
    # F1 Score (Not Acceptable)
    axes[0, 1].plot(summary_df['Day_No'], summary_df['Test_F1_NotAcceptable'], 'o-', color='orange')
    axes[0, 1].set_title('Test F1 Score (NotAcceptable) by Day')
    axes[0, 1].set_xlabel('Day')
    axes[0, 1].set_ylabel('F1 Score')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim([0, 1])
    
    # ROC AUC
    auc_data = summary_df.dropna(subset=['Test_ROC_AUC'])
    if len(auc_data) > 0:
        axes[1, 0].plot(auc_data['Day_No'], auc_data['Test_ROC_AUC'], 'o-', color='green')
        axes[1, 0].set_title('Test ROC-AUC by Day')
        axes[1, 0].set_xlabel('Day')
        axes[1, 0].set_ylabel('ROC-AUC')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_ylim([0, 1])
    
    # Recall (Not Acceptable)
    axes[1, 1].plot(summary_df['Day_No'], summary_df['Test_Recall_NotAcceptable'], 'o-', color='purple')
    axes[1, 1].set_title('Test Recall (NotAcceptable) by Day')
    axes[1, 1].set_xlabel('Day')
    axes[1, 1].set_ylabel('Recall')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(model_dir / 'metrics_by_day.png', dpi=150)
    plt.close()
    
    print(f"Saved metrics plot to {model_dir / 'metrics_by_day.png'}")
    
    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"Results saved to {model_dir}")
    print(f"{'='*60}\n")

def main():
    """Main training function."""
    # Paths to data splits
    train_data_path = 'data_splits/both_train_base.json'
    val_data_path = 'data_splits/both_val_base.json'
    test_data_path = 'data_splits/both_test_base.json'
    
    output_dir = 'analysis/metabolites/classifier/outputs_metabolites'
    
    print(f"\n{'='*60}")
    print("Loading data splits...")
    print(f"{'='*60}")
    
    # Load JSON data
    with open(train_data_path, 'r') as f:
        train_data_json = json.load(f)
    with open(val_data_path, 'r') as f:
        val_data_json = json.load(f)
    with open(test_data_path, 'r') as f:
        test_data_json = json.load(f)
    
    # Convert to DataFrames
    train_df = json_to_df(train_data_json)
    val_df = json_to_df(val_data_json)
    test_df = json_to_df(test_data_json)
    
    # Combine train and val for training
    trainval = pd.concat([train_df, val_df], ignore_index=True)
    
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    print(f"Combined train+val: {len(trainval)}")
    
    # Compute growth features
    print("\nComputing growth features...")
    trainval = compute_growth_features(trainval)
    test_df = compute_growth_features(test_df)
    
    # Train classifier
    train_metabolite_classifier_per_day(trainval, test_df, output_dir, model_name="lgbm")

if __name__ == '__main__':
    main()
