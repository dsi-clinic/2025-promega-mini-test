#!/usr/bin/env python3
"""
Combined Image + Metabolite Organoid Quality Classification
============================================================
Uses exactly 12 metabolite features per day (5 concentrations + 5 growth rates
+ 2 acceleration; missing/NaN filled with 0) and 12 PCA-reduced image features
(1280-dim EfficientNet-B0 -> PCA -> 12), for 24 features total. Trains per-day
LightGBM classifiers with GridSearchCV.

Pipeline:
    1. Load same JSON splits (both_train_base.json / val / test)
    2. Compute metabolite features (concentrations + growth rates) -> ~12 features
    3. Extract EfficientNet-B0 image embeddings (1280-dim) for every image
    4. PCA: fit on train+val embeddings, reduce to 12 dims
    5. Concatenate: 12 metabolite + 12 image = 24 features per sample
    6. Train LightGBM per day with GridSearchCV (CPU)

Usage:
    python train_combined_lgbm.py
    python train_combined_lgbm.py --scoring f1_weighted --n_folds 5
    python train_combined_lgbm.py --day_filter Dy03       # smoke test one day
    python train_combined_lgbm.py --n_image_features 16   # override PCA dims
"""

import json
import re
import argparse
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    average_precision_score,
    f1_score,
    recall_score,
    make_scorer,
)
from sklearn.utils.class_weight import compute_class_weight
from lightgbm import LGBMClassifier

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import timm
import torchvision.transforms as T
from PIL import Image
from image_classifier.preprocessing.stitched_preprocessing import (
    preprocess_stitched_pil,
)

try:
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.over_sampling import SMOTE

    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False

SEED = 42
TARGET_SIZE = (384, 512)  # (H, W) matching the image classifier
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Thresholds for warnings
MIN_DAY_SAMPLES = 20
MIN_MINORITY_SAMPLES = 8


# =====================================================================
#  Logger (copied from metabolite script for consistency)
# =====================================================================
class Logger:
    def __init__(self, verbose=False):
        self.verbose = verbose
        self._warnings = set()

    def info(self, msg):
        print(msg)

    def debug(self, msg):
        if self.verbose:
            print(f"  [DEBUG] {msg}")

    def warn(self, code, msg, once=True):
        key = f"{code}:{msg}" if once else None
        if once and key in self._warnings:
            return
        print(f"  [W{code}] {msg}")
        if once:
            self._warnings.add(key)

    def day_line(
        self,
        day,
        n_samples,
        n_minority,
        n_folds,
        cv_score,
        threshold,
        recall_na,
        fpr,
        n_metab_feat,
        n_img_feat,
        saved=True,
    ):
        status = "saved" if saved else "SKIPPED"
        fpr_str = f"{fpr:.2f}" if fpr is not None else "NA"
        print(
            f"[{day}] n={n_samples} (NA={n_minority}) | folds={n_folds} | "
            f"cv={cv_score:.2f} | thr={threshold:.2f} | recNA={recall_na:.2f} | "
            f"FPR={fpr_str} | feat={n_metab_feat}m+{n_img_feat}i | {status}"
        )


logger = Logger()


# =====================================================================
#  Seed
# =====================================================================
def set_seed(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =====================================================================
#  JSON -> DataFrame (reused from metabolite script)
# =====================================================================
def json_to_df(json_data):
    rows = []
    for org_id, info in json_data.items():
        label = info.get("label")
        batch = info.get("batch")
        for day_name, tp in info.get("timepoints", {}).items():
            row = {
                "ID": org_id,
                "batch": batch,
                "label": label,
                "DY": day_name,
                "img_path": tp.get("img_path"),
                "mask_path": tp.get("mask_path"),
            }
            for k, v in tp.get("metabolites", {}).items():
                row[k] = v
            rows.append(row)
    return pd.DataFrame(rows)


# =====================================================================
#  Growth features (reused from metabolite script)
# =====================================================================
def compute_growth_features(df, use_second_order=False):
    df = df.copy()
    df["day"] = df["DY"].str.extract(r"(\d+)").astype(int)
    df = df.sort_values(["ID", "day"])

    metabolites = [
        ("GlucoseGlo_concentration_uM", "glucose_growth"),
        ("GlutamateGlo_concentration_uM", "glutamate_growth"),
        ("LactateGlo_concentration_uM", "LactateGlo_growth"),
        ("PyruvateGlo_concentration_uM", "PyruvateGlo_growth"),
        ("MalateGlo_concentration_uM", "MalateGlo_growth"),
    ]
    for conc_col, growth_col in metabolites:
        if conc_col in df.columns:
            df[growth_col] = df.groupby("ID")[conc_col].diff()

    if use_second_order:
        growth_cols = [gc for _, gc in metabolites if gc in df.columns]
        for gc in growth_cols:
            df[f"{gc}_accel"] = df.groupby("ID")[gc].diff()

    return df


# =====================================================================
#  EfficientNet Feature Extractor
# =====================================================================
class EfficientNetExtractor:
    """Wraps timm EfficientNet-B0 for 1280-dim embedding extraction."""

    def __init__(self, device=DEVICE):
        self.device = device
        self.model = timm.create_model(
            "efficientnet_b0",
            pretrained=True,
            num_classes=0,
            global_pool="avg",
        )
        self.model.eval().to(self.device)
        self.transform = T.Compose(
            [
                T.Resize(TARGET_SIZE),
                T.ToTensor(),
                T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ]
        )

    @torch.no_grad()
    def extract_batch(self, img_paths, batch_size=32):
        """Extract 1280-dim features for a list of image paths.

        Returns:
            np.ndarray of shape (n_images, 1280).
            For missing/broken images the row is all-zeros.
        """
        n = len(img_paths)
        features = np.zeros((n, 1280), dtype=np.float32)
        valid_indices = []
        valid_tensors = []

        for i, p in enumerate(img_paths):
            try:
                img = Image.open(str(p)).convert("RGB")
                img = preprocess_stitched_pil(img, str(p))
                valid_tensors.append(self.transform(img))
                valid_indices.append(i)
            except Exception as e:
                logger.warn("IMG", f"Cannot load {p}: {e}")

        if not valid_tensors:
            return features

        dataset = torch.stack(valid_tensors)
        for start in range(0, len(valid_indices), batch_size):
            end = min(start + batch_size, len(valid_indices))
            batch = dataset[start:end].to(self.device)
            emb = self.model(batch).cpu().numpy()
            for j, idx in enumerate(valid_indices[start:end]):
                features[idx] = emb[j]

        return features


# =====================================================================
#  Metabolite feature prep: exactly 12 features per day
#  5 concentrations + 5 growth rates + 2 acceleration (0-filled when missing)
# =====================================================================
COLS_TO_DROP_BASE = [
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

# Fixed 12 metabolite features: same set every day; missing/NaN filled with 0
METAB_FEATURES_12 = [
    "GlucoseGlo_concentration_uM",
    "GlutamateGlo_concentration_uM",
    "LactateGlo_concentration_uM",
    "PyruvateGlo_concentration_uM",
    "MalateGlo_concentration_uM",
    "glucose_growth",
    "glutamate_growth",
    "LactateGlo_growth",
    "PyruvateGlo_growth",
    "MalateGlo_growth",
    "glucose_growth_accel",
    "glutamate_growth_accel",
]


def prepare_metabolite_features(df, day_num, use_second_order=False):
    """Return metabolite feature matrix with exactly 12 columns, labels, groups, img_paths.

    Uses METAB_FEATURES_12; if a feature is missing for this day (e.g. growth on day 3,
    Malate conc for day<=10), it is filled with 0 so we always have 12 metabolite features.
    """
    df_day = df.copy()
    img_paths = df_day["img_path"].tolist()
    y = df_day["label"]
    groups = df_day["ID"]

    # Build 12-column frame: take value if present else 0
    rows = []
    for i in range(len(df_day)):
        row = {}
        for col in METAB_FEATURES_12:
            if col in df_day.columns:
                val = df_day.iloc[i][col]
                row[col] = 0.0 if pd.isna(val) else float(val)
            else:
                row[col] = 0.0
        rows.append(row)

    X = pd.DataFrame(rows, columns=METAB_FEATURES_12)
    return X, y, groups, img_paths


def clean_data(X_train, X_test=None):
    """No-op for fixed 12: we already have 12 columns and no NaN. Kept for API compatibility."""
    if X_train.isna().any().any():
        X_train = X_train.fillna(0)
        if X_test is not None:
            X_test = X_test.fillna(0)
    return X_train, X_test


# =====================================================================
#  LightGBM helpers (mirrors metabolite-only script)
# =====================================================================
def get_scoring_function(scoring):
    if scoring == "f1_notaccept":
        return make_scorer(f1_score, pos_label="Not Acceptable")
    elif scoring == "f1_weighted":
        return "f1_weighted"
    elif scoring == "recall_notaccept":
        return make_scorer(recall_score, pos_label="Not Acceptable")
    elif scoring == "macro_f1":
        return "f1_macro"
    raise ValueError(f"Unknown scoring: {scoring}")


def get_param_grid():
    return {
        "max_depth": [3, 6],
        "num_leaves": [31, 63],
        "min_child_samples": [10, 20],
        "subsample": [0.8],
        "colsample_bytree": [0.8],
        "learning_rate": [0.05, 0.1],
        "n_estimators": [200, 500],
    }


def build_model(imbalance, y_train):
    classes = np.unique(y_train)
    if imbalance == "class_weight":
        weights = compute_class_weight("balanced", classes=classes, y=y_train)
        cw = {cls: float(w) for cls, w in zip(classes, weights)}
        return LGBMClassifier(
            random_state=SEED,
            verbose=-1,
            n_jobs=1,
            device="cpu",
            class_weight=cw,
            boosting_type="gbdt",
        ), False
    elif imbalance == "scale_pos_weight":
        y_arr = np.asarray(y_train)
        pos = (y_arr == "Not Acceptable").sum()
        neg = (y_arr != "Not Acceptable").sum()
        ratio = neg / pos if pos > 0 else 1.0
        return LGBMClassifier(
            random_state=SEED,
            verbose=-1,
            n_jobs=1,
            device="cpu",
            scale_pos_weight=ratio,
            boosting_type="gbdt",
        ), False
    elif imbalance == "smote":
        if not SMOTE_AVAILABLE:
            raise ImportError("SMOTE requires imbalanced-learn")
        pipe = ImbPipeline(
            [
                ("scaler", StandardScaler()),
                ("smote", SMOTE(random_state=SEED, k_neighbors=5)),
                (
                    "model",
                    LGBMClassifier(
                        random_state=SEED,
                        verbose=-1,
                        n_jobs=1,
                        device="cpu",
                        boosting_type="gbdt",
                    ),
                ),
            ]
        )
        return pipe, True
    raise ValueError(f"Unknown imbalance: {imbalance}")


def save_organoid_predictions(df_test, y_test, y_pred, y_score, path):
    label_map = {"Acceptable": 1, "Not Acceptable": 0}
    rows = []
    for i in range(len(df_test)):
        org = df_test.iloc[i]["ID"]
        tl = label_map.get(df_test.iloc[i]["label"], 0)
        pl = label_map.get(y_pred[i], 0)
        if tl == 1 and pl == 1:
            cat = "TP"
        elif tl == 0 and pl == 1:
            cat = "FP"
        elif tl == 1 and pl == 0:
            cat = "FN"
        else:
            cat = "TN"
        rows.append(
            {
                "Organoid_ID": org,
                "True_Label": tl,
                "Predicted_Probability": float(y_score[i]),
                "Predicted_Label": pl,
                "Correct": tl == pl,
                "CM_Category": cat,
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"  Saved organoid predictions -> {path}")


# =====================================================================
#  Main training loop
# =====================================================================
def train_combined_per_day(
    train_df,
    val_df,
    test_df,
    output_dir,
    model_name,
    scoring,
    imbalance,
    n_folds,
    use_second_order,
    n_image_features,
    day_filter=None,
):
    set_seed()

    model_dir = Path(output_dir) / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    # ------ Step 1: extract ALL image features once (GPU) ------
    print(f"\n{'=' * 65}")
    print(
        f"  Combined Model: Metabolite ({scoring}) + EfficientNet-B0 PCA->{n_image_features}"
    )
    print(f"  Imbalance: {imbalance} | Folds: {n_folds} | Device(CNN): {DEVICE}")
    print(f"  Output: {model_dir}")
    print(f"{'=' * 65}\n")

    extractor = EfficientNetExtractor(device=DEVICE)

    all_dfs = {"train": train_df, "val": val_df, "test": test_df}
    img_features = {}
    for split_name, df in all_dfs.items():
        paths = df["img_path"].tolist()
        print(f"Extracting {split_name} image features ({len(paths)} images) ...")
        feats = extractor.extract_batch(paths, batch_size=32)
        img_features[split_name] = feats
        print(f"  {split_name}: {feats.shape}")

    del extractor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ------ Step 2: per-day loop ------
    results_summary = []
    unique_days = sorted(np.unique(train_df.DY))
    if day_filter is not None:
        if day_filter in unique_days:
            unique_days = [day_filter]
            print(f"Day filter: {day_filter}")
        else:
            print(
                f"WARNING: day_filter '{day_filter}' not in data. Available: {unique_days}"
            )

    try:
        git_hash = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        git_hash = None

    run_config = {
        "model_type": "combined_metabolite_image",
        "scoring": scoring,
        "imbalance": imbalance,
        "n_folds": n_folds,
        "use_second_order": use_second_order,
        "n_image_features": n_image_features,
        "dim_reduction": "PCA",
        "image_backbone": "efficientnet_b0",
        "image_embedding_dim": 1280,
        "seed": SEED,
        "git_hash": git_hash,
    }
    with open(model_dir / "run_config.json", "w") as f:
        json.dump(run_config, f, indent=2)

    for days in unique_days:
        day_train = train_df[train_df["DY"] == days].copy()
        day_val = val_df[val_df["DY"] == days].copy()
        day_test = test_df[test_df["DY"] == days].copy()

        if len(day_train) == 0 or len(day_test) == 0:
            logger.warn("010", f"{days}: insufficient data, skipping")
            continue

        day_num = int(re.search(r"\d+", days).group())

        # Combine train + val
        day_trainval = pd.concat([day_train, day_val], ignore_index=True)

        # Check minority
        minority_count = (day_trainval["label"] == "Not Acceptable").sum()
        n_total = len(day_trainval)
        if n_total < MIN_DAY_SAMPLES:
            logger.warn("001", f"{days}: only {n_total} samples")
        if minority_count < MIN_MINORITY_SAMPLES:
            logger.warn(
                "002", f"{days}: only {minority_count} 'Not Acceptable' samples"
            )

        # ---- Metabolite features ----
        X_tv_metab, y_tv, groups_tv, _ = prepare_metabolite_features(
            day_trainval, day_num, use_second_order
        )
        X_te_metab, y_test, _, _ = prepare_metabolite_features(
            day_test, day_num, use_second_order
        )

        X_tv_metab, X_te_metab = clean_data(X_tv_metab, X_te_metab)
        metab_cols = list(X_tv_metab.columns)
        n_metab = len(metab_cols)

        # ---- Image features (slice from pre-extracted arrays) ----
        tv_train_mask = train_df["DY"] == days
        tv_val_mask = val_df["DY"] == days
        te_mask = test_df["DY"] == days

        img_tv = np.concatenate(
            [
                img_features["train"][tv_train_mask.values],
                img_features["val"][tv_val_mask.values],
            ],
            axis=0,
        )
        img_te = img_features["test"][te_mask.values]

        # PCA: fit on trainval, transform both
        n_components = min(n_image_features, img_tv.shape[0], img_tv.shape[1])
        scaler_img = StandardScaler()
        img_tv_scaled = scaler_img.fit_transform(img_tv)
        img_te_scaled = scaler_img.transform(img_te)

        pca = PCA(n_components=n_components, random_state=SEED)
        img_tv_pca = pca.fit_transform(img_tv_scaled)
        img_te_pca = pca.transform(img_te_scaled)

        explained = pca.explained_variance_ratio_.sum()
        logger.debug(
            f"{days}: PCA {n_components} components explain {explained:.2%} variance"
        )

        img_cols = [f"img_pca_{i}" for i in range(n_components)]

        # ---- Combine: metabolite + image ----
        X_tv_img_df = pd.DataFrame(img_tv_pca, columns=img_cols, index=X_tv_metab.index)
        X_te_img_df = pd.DataFrame(img_te_pca, columns=img_cols, index=X_te_metab.index)

        X_trainval = pd.concat([X_tv_metab, X_tv_img_df], axis=1)
        X_test_combined = pd.concat([X_te_metab, X_te_img_df], axis=1)

        total_features = X_trainval.shape[1]
        logger.debug(
            f"{days}: {n_metab} metab + {n_components} img = {total_features} features"
        )

        if total_features == 0:
            logger.warn("011", f"{days}: no features after merge, skipping")
            continue

        # ---- LightGBM ----
        model, is_pipeline = build_model(imbalance, y_tv)
        param_grid = get_param_grid()
        if is_pipeline:
            param_grid = {f"model__{k}": v for k, v in param_grid.items()}
            param_grid["smote__k_neighbors"] = [3, 5]

        cv = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
        scoring_fn = get_scoring_function(scoring)

        grid = GridSearchCV(
            model, param_grid, scoring=scoring_fn, cv=cv, n_jobs=4, verbose=0
        )
        try:
            if is_pipeline:
                grid.fit(X_trainval, y_tv)
            else:
                grid.fit(X_trainval, y_tv, groups=groups_tv)
        except Exception as e:
            logger.warn("009", f"{days}: GridSearchCV failed ({e}), using defaults")
            model.fit(X_trainval, y_tv)
            grid = type(
                "obj",
                (object,),
                {"best_estimator_": model, "best_score_": 0, "best_params_": {}},
            )()

        best_model = grid.best_estimator_
        best_params = grid.best_params_
        best_cv_score = grid.best_score_

        # ---- Predict ----
        if hasattr(best_model, "predict_proba"):
            proba = best_model.predict_proba(X_test_combined)
            actual_model = (
                best_model.named_steps["model"] if is_pipeline else best_model
            )
            classes_order = list(actual_model.classes_)
            if "Acceptable" in classes_order:
                acc_idx = classes_order.index("Acceptable")
                y_score_test = proba[:, acc_idx]
            else:
                y_score_test = proba[:, 1]
        else:
            y_score_test = np.ones(len(y_test)) * 0.5

        y_pred_test = np.where(y_score_test >= 0.5, "Acceptable", "Not Acceptable")

        # ---- Metrics ----
        y_true_bin = (y_test == "Acceptable").astype(int).to_numpy()
        try:
            pr_auc = average_precision_score(y_true_bin, y_score_test)
        except ValueError:
            pr_auc = None

        accuracy = accuracy_score(y_test, y_pred_test)
        report = classification_report(
            y_test, y_pred_test, output_dict=True, zero_division=0
        )

        f1_accept = report.get("Acceptable", {}).get("f1-score", 0)
        f1_notaccept = report.get("Not Acceptable", {}).get("f1-score", 0)
        recall_accept = report.get("Acceptable", {}).get("recall", 0)
        recall_notaccept = report.get("Not Acceptable", {}).get("recall", 0)
        precision_accept = report.get("Acceptable", {}).get("precision", 0)
        precision_notaccept = report.get("Not Acceptable", {}).get("precision", 0)

        actual_model = best_model.named_steps["model"] if is_pipeline else best_model
        cm = confusion_matrix(y_test, y_pred_test, labels=actual_model.classes_)

        tn, fp, fn, tp = 0, 0, 0, 0
        if cm.shape == (2, 2):
            classes_cm = actual_model.classes_
            if "Acceptable" in classes_cm:
                pos_idx = list(classes_cm).index("Acceptable")
                neg_idx = 1 - pos_idx
                tp = cm[pos_idx, pos_idx]
                fn = cm[pos_idx, neg_idx]
                fp = cm[neg_idx, pos_idx]
                tn = cm[neg_idx, neg_idx]
            else:
                tn, fp, fn, tp = cm.ravel()

        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else None

        logger.day_line(
            day=days,
            n_samples=n_total,
            n_minority=minority_count,
            n_folds=n_folds,
            cv_score=best_cv_score,
            threshold=0.5,
            recall_na=recall_notaccept,
            fpr=fpr,
            n_metab_feat=n_metab,
            n_img_feat=n_components,
            saved=True,
        )

        # ---- Save ----
        day_dir = model_dir / days
        day_dir.mkdir(parents=True, exist_ok=True)

        save_organoid_predictions(
            day_test.reset_index(drop=True),
            y_test,
            y_pred_test,
            y_score_test,
            day_dir / "organoid_predictions.csv",
        )

        if not is_pipeline:
            imp = best_model.feature_importances_
            imp_df = pd.DataFrame(
                {
                    "feature": X_trainval.columns,
                    "importance": imp,
                }
            ).sort_values("importance", ascending=False)
            imp_df.to_csv(day_dir / "feature_importance.csv", index=False)

        # Save PCA info (metadata)
        pca_info = {
            "n_components": int(n_components),
            "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
            "total_explained_variance": float(explained),
        }
        with open(day_dir / "pca_info.json", "w") as f:
            json.dump(pca_info, f, indent=2)

        # Lock state: save fitted scaler and PCA for inference (fit was on trainval only)
        joblib.dump(scaler_img, day_dir / "image_scaler.joblib")
        joblib.dump(pca, day_dir / "image_pca.joblib")

        metrics = {
            "day": days,
            "day_no": day_num,
            "n_metabolite_features": n_metab,
            "n_image_features": n_components,
            "n_total_features": total_features,
            "test_accuracy": float(accuracy),
            "test_f1_acceptable": float(f1_accept),
            "test_f1_notacceptable": float(f1_notaccept),
            "test_recall_acceptable": float(recall_accept),
            "test_recall_notacceptable": float(recall_notaccept),
            "test_precision_acceptable": float(precision_accept),
            "test_precision_notacceptable": float(precision_notaccept),
            "test_specificity": float(specificity),
            "test_pr_auc": float(pr_auc) if pr_auc is not None else None,
            "best_params": best_params,
            "cv_score": float(best_cv_score),
            "pca_explained_variance": float(explained),
            "confusion_matrix": {
                "TP": int(tp),
                "FP": int(fp),
                "TN": int(tn),
                "FN": int(fn),
            },
        }
        with open(day_dir / "metrics_test.json", "w") as f:
            json.dump(metrics, f, indent=2)

        # Confusion matrix plot
        plt.figure(figsize=(6, 5))
        plt.imshow(cm, interpolation="nearest", cmap="Blues")
        plt.title(f"Confusion Matrix - {days} (combined)")
        plt.colorbar()
        ticks = np.arange(len(actual_model.classes_))
        plt.xticks(ticks, actual_model.classes_, rotation=45)
        plt.yticks(ticks, actual_model.classes_)
        thresh_cm = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(
                    j,
                    i,
                    format(cm[i, j], "d"),
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > thresh_cm else "black",
                )
        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        plt.tight_layout()
        plt.savefig(day_dir / "confusion_matrix.png", dpi=150)
        plt.close()

        results_summary.append(
            {
                "Day": days,
                "Day_No": day_num,
                "N_Metab_Features": n_metab,
                "N_Image_Features": n_components,
                "N_Total_Features": total_features,
                "PCA_Explained_Var": explained,
                "Test_Accuracy": accuracy,
                "Test_F1_Acceptable": f1_accept,
                "Test_F1_NotAcceptable": f1_notaccept,
                "Test_Recall_Acceptable": recall_accept,
                "Test_Recall_NotAcceptable": recall_notaccept,
                "Test_Precision_Acceptable": precision_accept,
                "Test_Precision_NotAcceptable": precision_notaccept,
                "Test_Specificity": specificity,
                "Test_PR_AUC": pr_auc,
                "TP": int(tp),
                "FP": int(fp),
                "TN": int(tn),
                "FN": int(fn),
                "CV_Score": best_cv_score,
            }
        )

    if not results_summary:
        logger.warn("012", "No results to summarize")
        return

    # ---- Summary ----
    summary_df = pd.DataFrame(results_summary).sort_values("Day_No")
    summary_df.to_csv(model_dir / "results_summary.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(
        summary_df["Day_No"], summary_df["Test_Accuracy"], "o-", color="blue"
    )
    axes[0, 0].set_title("Test Accuracy")
    axes[0, 0].set_xlabel("Day")
    axes[0, 0].set_ylabel("Accuracy")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim([0, 1])

    axes[0, 1].plot(
        summary_df["Day_No"],
        summary_df["Test_F1_Acceptable"],
        "o-",
        color="green",
        label="Acceptable",
    )
    axes[0, 1].plot(
        summary_df["Day_No"],
        summary_df["Test_F1_NotAcceptable"],
        "o--",
        color="red",
        label="Not Acceptable",
    )
    axes[0, 1].set_title("Test F1 Score")
    axes[0, 1].set_xlabel("Day")
    axes[0, 1].set_ylabel("F1")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim([0, 1])

    axes[1, 0].plot(
        summary_df["Day_No"], summary_df["Test_Specificity"], "o-", color="purple"
    )
    axes[1, 0].set_title("Test Specificity (TNR)")
    axes[1, 0].set_xlabel("Day")
    axes[1, 0].set_ylabel("Specificity")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim([0, 1])

    pr_data = summary_df.dropna(subset=["Test_PR_AUC"])
    if len(pr_data) > 0:
        axes[1, 1].plot(pr_data["Day_No"], pr_data["Test_PR_AUC"], "o-", color="orange")
        axes[1, 1].set_title("Test PR-AUC")
        axes[1, 1].set_xlabel("Day")
        axes[1, 1].set_ylabel("PR-AUC")
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_ylim([0, 1])

    plt.suptitle(
        f"Combined Model: {n_metab} metab + {n_image_features} img PCA", fontsize=13
    )
    plt.tight_layout()
    plt.savefig(model_dir / "metrics_by_day.png", dpi=150)
    plt.close()

    print(f"\nDone. Results: {model_dir}")


# =====================================================================
#  CLI
# =====================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Combined Image+Metabolite LightGBM Classifier"
    )
    parser.add_argument(
        "--scoring",
        choices=["f1_notaccept", "f1_weighted", "recall_notaccept", "macro_f1"],
        default="f1_notaccept",
    )
    parser.add_argument(
        "--imbalance",
        choices=["class_weight", "scale_pos_weight", "smote"],
        default="class_weight",
    )
    parser.add_argument("--n_folds", type=int, choices=[3, 5], default=3)
    parser.add_argument("--use_second_order_growth", action="store_true")
    parser.add_argument(
        "--n_image_features",
        type=int,
        default=12,
        help="Number of PCA components for image features (default: 12)",
    )
    parser.add_argument(
        "--day_filter",
        type=str,
        default=None,
        help="Run only specified day (e.g. Dy03)",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Path to directory containing data_splits/ (default: auto-detect)",
    )

    args = parser.parse_args()

    global logger
    logger = Logger(verbose=args.verbose)

    # Locate data splits - try several paths
    if args.data_dir:
        base = Path(args.data_dir)
    else:
        candidates = [
            Path(__file__).resolve().parent.parent,  # repo root
            Path(__file__).resolve().parent,
        ]
        base = None
        for c in candidates:
            if (c / "data_splits" / "both_train_base.json").exists():
                base = c
                break
        if base is None:
            raise FileNotFoundError(
                "Cannot find data_splits/. Use --data_dir to specify the project root."
            )

    train_path = base / "data_splits" / "both_train_base.json"
    val_path = base / "data_splits" / "both_val_base.json"
    test_path = base / "data_splits" / "both_test_base.json"

    print(f"Data root: {base}")
    for p in [train_path, val_path, test_path]:
        assert p.exists(), f"Missing split: {p}"

    with open(train_path) as f:
        train_json = json.load(f)
    with open(val_path) as f:
        val_json = json.load(f)
    with open(test_path) as f:
        test_json = json.load(f)

    train_df = json_to_df(train_json)
    val_df = json_to_df(val_json)
    test_df = json_to_df(test_json)

    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    print("Computing growth features ...")
    train_df = compute_growth_features(train_df, args.use_second_order_growth)
    val_df = compute_growth_features(val_df, args.use_second_order_growth)
    test_df = compute_growth_features(test_df, args.use_second_order_growth)

    model_name = (
        f"combined_lgbm_{args.scoring}_{args.imbalance}_pca{args.n_image_features}"
    )
    if args.use_second_order_growth:
        model_name += "_accel"

    output_dir = Path(__file__).resolve().parent / "outputs"

    train_combined_per_day(
        train_df,
        val_df,
        test_df,
        output_dir=str(output_dir),
        model_name=model_name,
        scoring=args.scoring,
        imbalance=args.imbalance,
        n_folds=args.n_folds,
        use_second_order=args.use_second_order_growth,
        n_image_features=args.n_image_features,
        day_filter=args.day_filter,
    )


if __name__ == "__main__":
    main()
