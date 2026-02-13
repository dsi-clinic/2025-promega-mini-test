#!/usr/bin/env python3
"""
Extract Feature Importance from Metabolite Models

This script retrains the LightGBM models for specific days and extracts feature importances.
It replicates the exact training process from train_metabolites.py but focuses on
feature extraction and visualization.

Usage:
    python3 extract_feature_importance.py --days 24 28 30
    python3 extract_feature_importance.py --days all
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.utils.class_weight import compute_class_weight
from lightgbm import LGBMClassifier

SEED = 42


def set_seed(seed=SEED):
    """Set random seed for reproducibility.

    Args:
        seed: Random seed value to use.
    """
    np.random.seed(seed)


def json_to_df(json_data):
    """Convert JSON split data to DataFrame with metabolite features.

    Args:
        json_data: Dictionary mapping organoid IDs to organoid data with timepoints.

    Returns:
        DataFrame with one row per (organoid, day) combination, including metabolite features.
    """
    data = json_data["data"]
    columns = json_data["columns"]
    df = pd.DataFrame(data, columns=columns)

    # Convert concentration columns to float
    conc_cols = [
        c
        for c in df.columns
        if "concentration" in c or "Glo" in c or c in ["glucose", "glutamate"]
    ]
    for col in conc_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def compute_growth_features(df):
    """Add growth features (difference between consecutive timepoints).

    Computes the difference in metabolite concentrations between consecutive days
    for each organoid, creating growth rate features.

    Args:
        df: DataFrame with metabolite concentration columns and 'ID' and 'DY' columns.

    Returns:
        DataFrame with additional growth feature columns (e.g., 'glucose_growth').
    """
    df = df.sort_values(["ID", "day"]).reset_index(drop=True)

    metabolites = ["glucose", "glutamate", "LactateGlo", "PyruvateGlo", "MalateGlo"]

    for met in metabolites:
        col_name = (
            f"{met}_concentration_uM"
            if met != "glucose" and met != "glutamate"
            else met
        )
        if col_name in df.columns:
            df[f"{met}_growth"] = df.groupby("ID")[col_name].diff()

    return df


def prepare_data_for_day(df, day_num, cols_to_drop_base):
    """Helper to prepare X, y, groups for a specific day."""
    df_day = df.copy()
    cols_to_drop = cols_to_drop_base.copy()

    # Drop concentration column for MalateGlo if day < 13
    if day_num < 13 and "MalateGlo_concentration_uM" in df_day.columns:
        cols_to_drop.append("MalateGlo_concentration_uM")

    # Drop growth features for day 3 (no previous timepoint)
    growth_features = [
        "glucose_growth",
        "glutamate_growth",
        "LactateGlo_growth",
        "PyruvateGlo_growth",
        "MalateGlo_growth",
    ]
    if day_num == 3:
        cols_to_drop.extend([g for g in growth_features if g in df_day.columns])
    elif day_num == 13 and "MalateGlo_growth" in df_day.columns:
        cols_to_drop.append("MalateGlo_growth")

    df_day = df_day.drop(columns=[c for c in cols_to_drop if c in df_day.columns])

    X = df_day.drop(columns=["label", "ID"])
    y = df_day["label"]
    groups = df_day["ID"]

    return X, y, groups


def clean_data_no_scaling(X_train, X_val=None, X_test=None):
    """Clean NaNs/constants but DO NOT scale."""
    # Drop all-NaN columns
    all_nan_cols = X_train.columns[X_train.isna().all()].tolist()
    if all_nan_cols:
        print(f"  Dropping all-NaN columns: {all_nan_cols}")
        X_train = X_train.drop(columns=all_nan_cols)
        if X_val is not None:
            X_val = X_val.drop(columns=[c for c in all_nan_cols if c in X_val.columns])
        if X_test is not None:
            X_test = X_test.drop(
                columns=[c for c in all_nan_cols if c in X_test.columns]
            )

    # Drop constant columns
    constant_cols = []
    for col in X_train.columns:
        if X_train[col].nunique(dropna=True) <= 1:
            constant_cols.append(col)
    if constant_cols:
        print(f"  Dropping constant columns: {constant_cols}")
        X_train = X_train.drop(columns=constant_cols)
        if X_val is not None:
            X_val = X_val.drop(columns=[c for c in constant_cols if c in X_val.columns])
        if X_test is not None:
            X_test = X_test.drop(
                columns=[c for c in constant_cols if c in X_test.columns]
            )

    # Drop near-constant columns
    near_constant_cols = []
    for col in X_train.columns:
        col_std = X_train[col].std(skipna=True)
        if np.isfinite(col_std) and col_std < 1e-6:
            near_constant_cols.append(col)
    if near_constant_cols:
        print(f"  Dropping near-constant columns: {near_constant_cols}")
        X_train = X_train.drop(columns=near_constant_cols)
        if X_val is not None:
            X_val = X_val.drop(
                columns=[c for c in near_constant_cols if c in X_val.columns]
            )
        if X_test is not None:
            X_test = X_test.drop(
                columns=[c for c in near_constant_cols if c in X_test.columns]
            )

    # Fill NaNs
    if X_train.isna().any().any():
        print("  Filling remaining NaNs with 0")
        X_train = X_train.fillna(0)
        if X_val is not None:
            X_val = X_val.fillna(0)
        if X_test is not None:
            X_test = X_test.fillna(0)

    return X_train, X_val, X_test


def extract_feature_importance_for_day(train_df, val_df, day_str, output_dir):
    """Extract feature importance for a specific day."""
    set_seed()

    day_num = int(re.search(r"\d+", day_str).group())
    print(f"\n{'=' * 60}")
    print(f"Extracting Feature Importance for {day_str} (day_num={day_num})")
    print(f"{'=' * 60}")

    # Base columns to drop
    cols_to_drop_base = [
        "DY",
        "batch",
        "img_path",
        "mask_path",
        "MalateGlo_initial_concentration",
        "GlucoseGlo_initial_concentration",
        "GlutamateGlo_initial_concentration",
        "LactateGlo_initial_concentration",
        "PyruvateGlo_initial_concentration",
        "day",
    ]

    # Get data for this day
    day_train = train_df[train_df["DY"] == day_str].copy()
    day_val = val_df[val_df["DY"] == day_str].copy()

    if len(day_train) == 0:
        print(f"No training data for {day_str}, skipping.")
        return None

    print(f"Train: {len(day_train)}, Val: {len(day_val)}")

    # Prepare data
    X_train, y_train, groups_train = prepare_data_for_day(
        day_train, day_num, cols_to_drop_base
    )
    X_val, y_val, _ = prepare_data_for_day(day_val, day_num, cols_to_drop_base)

    # Clean data
    X_train_clean, X_val_clean, _ = clean_data_no_scaling(X_train, X_val=X_val)

    if X_train_clean.shape[1] == 0:
        print(f"No features left after cleaning; skipping {day_str}.")
        return None

    print(f"Number of features: {X_train_clean.shape[1]}")

    # Compute class weights
    classes = np.unique(y_train)
    if len(classes) < 2:
        class_weight_dict = None
        scale_pos_weight = 1.0
    else:
        class_weights = compute_class_weight("balanced", classes=classes, y=y_train)
        class_weight_dict = {cls: float(w) for cls, w in zip(classes, class_weights)}

        y_arr = pd.Series(y_train).to_numpy()
        pos = (y_arr == "Acceptable").sum()
        neg = (y_arr != "Acceptable").sum()
        scale_pos_weight = (neg / pos) if pos > 0 else 1.0

    # Hyperparameter tuning
    print("Running hyperparameter tuning...")
    model = LGBMClassifier(
        random_state=SEED,
        verbose=-1,
        n_jobs=1,
        scale_pos_weight=scale_pos_weight,
        class_weight=class_weight_dict,
        boosting_type="gbdt",
    )

    param_grid = {
        "max_depth": [3, 6],
        "num_leaves": [31, 63],
        "min_child_samples": [10, 20],
        "subsample": [0.8],
        "colsample_bytree": [0.8],
        "learning_rate": [0.05, 0.1],
        "n_estimators": [200, 500],
    }

    from sklearn.metrics import f1_score, make_scorer

    scoring_obj = make_scorer(f1_score, pos_label="Not Acceptable")

    cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED)
    grid = GridSearchCV(
        model, param_grid, cv=cv, scoring=scoring_obj, n_jobs=4, verbose=0
    )
    grid.fit(X_train_clean, y_train, groups=groups_train)
    best_params = grid.best_params_
    print(f"Best Params: {best_params}")

    # Train final model on train+val combined
    day_combined = pd.concat([day_train, day_val], ignore_index=True)
    X_combined, y_combined, _ = prepare_data_for_day(
        day_combined, day_num, cols_to_drop_base
    )
    X_combined_clean, _, _ = clean_data_no_scaling(X_combined)

    # Recompute weights on combined
    classes_comb = np.unique(y_combined)
    if len(classes_comb) < 2:
        weight_dict_comb = None
        ratio_comb = 1.0
    else:
        weights_comb = compute_class_weight(
            "balanced", classes=classes_comb, y=y_combined
        )
        weight_dict_comb = {cls: float(w) for cls, w in zip(classes_comb, weights_comb)}

        y_arr_comb = pd.Series(y_combined).to_numpy()
        pos_comb = (y_arr_comb == "Acceptable").sum()
        neg_comb = (y_arr_comb != "Acceptable").sum()
        ratio_comb = (neg_comb / pos_comb) if pos_comb > 0 else 1.0

    final_model = LGBMClassifier(
        random_state=SEED,
        verbose=-1,
        n_jobs=1,
        scale_pos_weight=ratio_comb,
        class_weight=weight_dict_comb,
        boosting_type="gbdt",
        **best_params,
    )

    print("Training final model on train+val combined...")
    final_model.fit(X_combined_clean, y_combined)

    # Extract feature importances
    feature_importance = final_model.feature_importances_
    importance_df = pd.DataFrame(
        {
            "feature": X_combined_clean.columns,
            "importance": feature_importance,
            "importance_normalized": feature_importance / feature_importance.sum(),
        }
    ).sort_values("importance", ascending=False)

    # Create output directory for this day
    day_output_dir = Path(output_dir) / day_str
    day_output_dir.mkdir(parents=True, exist_ok=True)

    # Save CSV
    csv_path = day_output_dir / "feature_importance.csv"
    importance_df.to_csv(csv_path, index=False)
    print(f"Saved feature importance to {csv_path}")

    # Save feature names
    feature_names_path = day_output_dir / "feature_names.json"
    with open(feature_names_path, "w") as f:
        json.dump(list(X_combined_clean.columns), f, indent=2)

    # Visualize top 20 features
    top_n = min(20, len(importance_df))
    plt.figure(figsize=(10, 8))
    top_importance = importance_df.head(top_n)

    plt.barh(range(top_n), top_importance["importance"].values, color="steelblue")
    plt.yticks(range(top_n), top_importance["feature"].values)
    plt.xlabel("Importance (split count)", fontweight="bold")
    plt.ylabel("Feature", fontweight="bold")
    plt.title(
        f"Top {top_n} Feature Importances - {day_str}", fontweight="bold", fontsize=14
    )
    plt.gca().invert_yaxis()
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    top20_path = day_output_dir / "feature_importance_top20.png"
    plt.savefig(top20_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved top 20 visualization to {top20_path}")

    # Visualize all features (if reasonable number)
    if len(importance_df) <= 50:
        plt.figure(figsize=(10, max(12, len(importance_df) * 0.3)))

        plt.barh(
            range(len(importance_df)),
            importance_df["importance"].values,
            color="steelblue",
        )
        plt.yticks(
            range(len(importance_df)), importance_df["feature"].values, fontsize=8
        )
        plt.xlabel("Importance (split count)", fontweight="bold")
        plt.ylabel("Feature", fontweight="bold")
        plt.title(
            f"All Feature Importances - {day_str}", fontweight="bold", fontsize=14
        )
        plt.gca().invert_yaxis()
        plt.grid(axis="x", alpha=0.3)
        plt.tight_layout()

        full_path = day_output_dir / "feature_importance_full.png"
        plt.savefig(full_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved full visualization to {full_path}")

    print(f"\nTop 10 Features for {day_str}:")
    print(importance_df.head(10).to_string(index=False))

    return importance_df


def main():
    parser = argparse.ArgumentParser(
        description="Extract Feature Importance from Metabolite Models"
    )
    parser.add_argument(
        "--days",
        nargs="+",
        default=["24", "28", "30"],
        help="Days to extract feature importance for (e.g., 24 28 30 or 'all')",
    )

    args = parser.parse_args()

    # Data paths
    train_data_path = "data_splits/both_train_base.json"
    val_data_path = "data_splits/both_val_base.json"
    output_dir = "analysis/metabolites/reports/feature_importance"

    print(f"\n{'=' * 60}")
    print("Loading data splits...")
    print(f"{'=' * 60}")

    with open(train_data_path, "r") as f:
        train_data_json = json.load(f)
    with open(val_data_path, "r") as f:
        val_data_json = json.load(f)

    train_df = json_to_df(train_data_json)
    val_df = json_to_df(val_data_json)

    print(f"Train: {len(train_df)}, Val: {len(val_df)}")

    print("\nComputing growth features...")
    train_df = compute_growth_features(train_df)
    val_df = compute_growth_features(val_df)

    # Determine which days to process
    if args.days == ["all"]:
        unique_days = sorted(np.unique(train_df.DY))
    else:
        # Map day numbers to DY format
        day_mapping = {
            str(d): f
            for f in np.unique(train_df.DY)
            for d in [int(re.search(r"\d+", f).group())]
            if str(d) in args.days
        }
        unique_days = [day_mapping[d] for d in args.days if d in day_mapping]

    print(f"\nProcessing days: {unique_days}")

    # Extract feature importance for each day
    all_importance = {}
    for day_str in unique_days:
        importance_df = extract_feature_importance_for_day(
            train_df, val_df, day_str, output_dir
        )
        if importance_df is not None:
            all_importance[day_str] = importance_df

    # Create comparison visualization if multiple days
    if len(all_importance) > 1:
        print(f"\n{'=' * 60}")
        print("Creating cross-day comparison...")
        print(f"{'=' * 60}")

        # Get union of all features
        all_features = set()
        for df in all_importance.values():
            all_features.update(df["feature"].values)

        # Create comparison dataframe
        comparison_data = []
        for feature in all_features:
            row = {"feature": feature}
            for day_str, imp_df in all_importance.items():
                day_num = int(re.search(r"\d+", day_str).group())
                feature_row = imp_df[imp_df["feature"] == feature]
                if len(feature_row) > 0:
                    row[f"Day_{day_num}"] = feature_row["importance_normalized"].values[
                        0
                    ]
                else:
                    row[f"Day_{day_num}"] = 0.0
            comparison_data.append(row)

        comparison_df = pd.DataFrame(comparison_data)

        # Average importance across days
        day_cols = [c for c in comparison_df.columns if c.startswith("Day_")]
        comparison_df["avg_importance"] = comparison_df[day_cols].mean(axis=1)
        comparison_df = comparison_df.sort_values("avg_importance", ascending=False)

        # Save comparison
        comparison_path = Path(output_dir) / "feature_importance_comparison.csv"
        comparison_df.to_csv(comparison_path, index=False)
        print(f"Saved comparison to {comparison_path}")

        # Visualize top features across days
        top_features = comparison_df.head(15)

        fig, ax = plt.subplots(figsize=(12, 8))
        x = np.arange(len(top_features))
        width = 0.8 / len(day_cols)

        for i, day_col in enumerate(day_cols):
            offset = (i - len(day_cols) / 2) * width + width / 2
            ax.bar(
                x + offset,
                top_features[day_col].values,
                width,
                label=day_col.replace("_", " "),
            )

        ax.set_xlabel("Feature", fontweight="bold", fontsize=12)
        ax.set_ylabel("Normalized Importance", fontweight="bold", fontsize=12)
        ax.set_title(
            "Top 15 Features: Importance Across Days", fontweight="bold", fontsize=14
        )
        ax.set_xticks(x)
        ax.set_xticklabels(top_features["feature"].values, rotation=45, ha="right")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        comparison_plot_path = Path(output_dir) / "feature_importance_comparison.png"
        plt.savefig(comparison_plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved comparison visualization to {comparison_plot_path}")

    print(f"\n{'=' * 60}")
    print("Feature Importance Extraction Complete!")
    print(f"Results saved to {output_dir}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
