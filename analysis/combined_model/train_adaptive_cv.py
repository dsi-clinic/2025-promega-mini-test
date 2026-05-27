
#!/usr/bin/env python3
"""
Day-Adaptive Multimodal Organoid Classification with 5-Fold Cross Validation.

Early days (Dy03-Dy17): Image is primary, metabolite supplements.
Late days  (Dy20-Dy30): Metabolite is primary, image supplements.

Uses:
  - Current-day metabolite concentrations only
  - Weighted BCE loss
  - Label convention matching paper scripts:
      Not Acceptable = 1
      Acceptable = 0
  - Splits loaded through pipeline.splits.Splits.from_csv()
"""

import json
import re
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import timm

from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
    roc_auc_score,
)

SEED = 1

EARLY_DAYS = {"Dy03", "Dy06", "Dy08", "Dy10", "Dy13", "Dy15", "Dy17"}
LATE_DAYS = {"Dy20_5", "Dy24", "Dy28", "Dy30"}

BASE_MET_FEATURES = [
    "GlucoseGlo_concentration_uM",
    "GlutamateGlo_concentration_uM",
    "LactateGlo_concentration_uM",
    "PyruvateGlo_concentration_uM",
]
MALATE_FEATURE = "MalateGlo_concentration_uM"

BACKBONE_MODELS = {
    "vit": "vit_base_patch16_224",
    "resnet": "resnet50",
    "efficientnet": "efficientnet_b0",
}

DAY_ORDER_LABELS = [
    "Dy03", "Dy06", "Dy08", "Dy10", "Dy13",
    "Dy15", "Dy17", "Dy20_5", "Dy24", "Dy28", "Dy30",
]

N_FOLDS = 5
VAL_FRACTION = 0.1

LABEL_MAP = {
    "Not Acceptable": 1,
    "Not Accepted": 1,
    "Acceptable": 0,
    "Accepted": 0,
}


def set_seed(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def day_to_int(day_str):
    m = re.search(r"[Dd][Yy](\d+)", day_str)
    return int(m.group(1)) if m else -1


class OrganoidDataset(Dataset):
    def __init__(self, df, config, transform=None, scaler=None, fit_scaler=False):
        self.df = df.reset_index(drop=True)
        self.config = config
        self.transform = transform
        self.label_map = LABEL_MAP
        self.img_key = "overlay_path" if "overlay" in config["input_mode"] else "img_path"
        self.use_metabolites = config["use_metabolites"]

        valid = []
        for i in range(len(self.df)):
            if config["use_images"]:
                img = self.df.iloc[i][self.img_key]
                if pd.isna(img) or not Path(img).exists():
                    continue
            valid.append(i)

        self.df = self.df.iloc[valid].reset_index(drop=True)

        if self.use_metabolites:
            self.meta_features_list = self._extract_metabolite_features()

            if fit_scaler:
                max_dim = max(len(f) for f in self.meta_features_list) if self.meta_features_list else 5
                padded = [f + [0.0] * (max_dim - len(f)) for f in self.meta_features_list]
                self.scaler = StandardScaler()
                self.meta_features = np.array(padded, dtype=np.float32)
                self.meta_features = self.scaler.fit_transform(self.meta_features)

            elif scaler is not None:
                scaler_dim = scaler.mean_.shape[0]
                padded = [f + [0.0] * (scaler_dim - len(f)) for f in self.meta_features_list]
                self.scaler = scaler
                self.meta_features = np.array(padded, dtype=np.float32)
                self.meta_features = self.scaler.transform(self.meta_features)

            else:
                max_dim = max(len(f) for f in self.meta_features_list) if self.meta_features_list else 5
                padded = [f + [0.0] * (max_dim - len(f)) for f in self.meta_features_list]
                self.scaler = None
                self.meta_features = np.array(padded, dtype=np.float32)
        else:
            self.scaler = scaler

    def _extract_metabolite_features(self):
        """Extract current-day metabolite concentrations only."""
        features = []
        self.meta_dims = []

        for idx in range(len(self.df)):
            row = self.df.iloc[idx]
            day_str = row.get("day", "Dy00")
            day_num = day_to_int(day_str)

            met_names = BASE_MET_FEATURES.copy()
            if day_num > 10:
                met_names.append(MALATE_FEATURE)

            vals = []
            for m in met_names:
                v = row.get(m, np.nan)
                vals.append(0.0 if pd.isna(v) else float(v))

            features.append(vals)
            self.meta_dims.append(len(vals))

        return features

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        label = torch.tensor(self.label_map.get(row["label"], 0), dtype=torch.float32)

        items = []

        if self.config["use_images"]:
            img = Image.open(row[self.img_key]).convert("RGB")
            if self.transform:
                img = self.transform(img)
            items.append(img)

        if self.use_metabolites:
            meta = torch.tensor(self.meta_features[idx], dtype=torch.float32)
            items.append(meta)

        items.append(label)
        return tuple(items)


class MetaboliteBranch(nn.Module):
    """Metabolite MLP with optional projection."""

    def __init__(self, input_dim=5, hidden_dim=64, proj_dim=None):
        super().__init__()
        self.input_dim = input_dim

        layers = [
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        ]

        if proj_dim is not None:
            layers += [
                nn.Linear(hidden_dim, proj_dim),
                nn.LayerNorm(proj_dim),
                nn.ReLU(),
            ]
            self.out_dim = proj_dim
        else:
            self.out_dim = hidden_dim

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        if x.shape[1] < self.input_dim:
            padding = torch.zeros(
                x.shape[0],
                self.input_dim - x.shape[1],
                device=x.device,
            )
            x = torch.cat([x, padding], dim=1)
        return self.net(x)


class AdaptiveCrossAttentionFusion(nn.Module):
    """
    Adaptive cross-modal fusion.

    early_mode=True:
        image is primary, metabolite supplements.

    early_mode=False:
        metabolite is primary, image supplements.
    """

    def __init__(
        self,
        img_dim,
        meta_dim,
        proj_dim=128,
        num_heads=4,
        dropout=0.1,
        early_mode=True,
        use_projection=False,
    ):
        super().__init__()
        assert proj_dim % num_heads == 0, "proj_dim must be divisible by num_heads"

        self.early_mode = early_mode
        self.use_projection = use_projection

        if use_projection:
            self.img_pre_proj = nn.Sequential(
                nn.Linear(img_dim, proj_dim),
                nn.LayerNorm(proj_dim),
                nn.ReLU(),
            )
            img_dim = proj_dim

        self.img_proj = nn.Linear(img_dim, proj_dim)
        self.meta_proj = nn.Linear(meta_dim, proj_dim)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=proj_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm = nn.LayerNorm(proj_dim)
        self.out_dim = proj_dim * 2

    def forward(self, img_feats, meta_feats):
        if self.use_projection:
            img_feats = self.img_pre_proj(img_feats)

        img_p = self.img_proj(img_feats).unsqueeze(1)
        meta_p = self.meta_proj(meta_feats).unsqueeze(1)

        if self.early_mode:
            q, k, v = img_p, meta_p, meta_p
        else:
            q, k, v = meta_p, img_p, img_p

        attended, _ = self.cross_attn(q, k, v)
        attended = attended.squeeze(1)

        primary_normed = self.norm(q.squeeze(1) + attended)
        return torch.cat([primary_normed, attended], dim=1)


class AdaptiveMultimodalClassifier(nn.Module):
    def __init__(self, config, early_mode=True):
        super().__init__()

        self.config = config
        self.early_mode = early_mode
        self.use_images = config.get("use_images", True)
        self.use_metabolites = config.get("use_metabolites", True)

        use_projection = config.get("use_projection", False)
        proj_dim = config.get("proj_dim", 256)

        backbone_name = BACKBONE_MODELS[config["backbone"]]
        extra = {"img_size": config["target_size"]} if "vit" in backbone_name else {}

        self.backbone = timm.create_model(
            backbone_name,
            pretrained=True,
            num_classes=0,
            **extra,
        )
        img_dim = self.backbone.num_features

        if self.use_images and self.use_metabolites:
            self.meta_branch = MetaboliteBranch(
                input_dim=5,
                hidden_dim=64,
                proj_dim=proj_dim if use_projection else None,
            )

            meta_dim = self.meta_branch.out_dim

            self.fusion = AdaptiveCrossAttentionFusion(
                img_dim=img_dim,
                meta_dim=meta_dim,
                proj_dim=config.get("cross_attn_proj_dim", 128),
                num_heads=config.get("cross_attn_heads", 4),
                dropout=0.1,
                early_mode=early_mode,
                use_projection=use_projection,
            )

            head_in = self.fusion.out_dim

        elif self.use_images:
            self.meta_branch = None
            self.fusion = None
            head_in = img_dim

        else:
            self.meta_branch = MetaboliteBranch(input_dim=5, hidden_dim=64)
            self.fusion = None
            head_in = self.meta_branch.out_dim

        self.head = nn.Sequential(
            nn.Linear(head_in, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
        )

    def forward(self, *args):
        if self.use_images and self.use_metabolites:
            img, meta = args[0], args[1]
            img_feats = self.backbone(img)
            meta_feats = self.meta_branch(meta)
            fused = self.fusion(img_feats, meta_feats)
            return self.head(fused).squeeze(1)

        elif self.use_images:
            img = args[0]
            img_feats = self.backbone(img)
            return self.head(img_feats).squeeze(1)

        else:
            meta = args[0]
            meta_feats = self.meta_branch(meta)
            return self.head(meta_feats).squeeze(1)


class EarlyStopping:
    def __init__(self, patience=20):
        self.patience = patience
        self.best = -np.inf
        self.counter = 0

    def __call__(self, score):
        if score > self.best + 1e-4:
            self.best = score
            self.counter = 0
            return False

        self.counter += 1
        return self.counter >= self.patience


def get_transforms(config, augment=False):
    t = [T.Resize(config["target_size"])]

    if augment and config["use_augmentation"]:
        t.extend([
            T.RandomHorizontalFlip(0.5),
            T.RandomVerticalFlip(0.5),
            T.RandomRotation(15),
        ])

    t.extend([
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    return T.Compose(t)


def load_all_data(config):
    """Load all organoids into a single DataFrame using the provided split CSV."""
    sys.path.insert(0, str(Path("/home/YOUR_USERNAME/2025-promega-mini-test")))

    from pipeline.data_loader import OrganoidDataset as LoaderDataset
    from pipeline.data_loader import default_filters
    from pipeline.splits import Splits

    splits = Splits.canonical()

    ds = LoaderDataset(
        "/home/YOUR_USERNAME/2025-promega-mini-test/data/all_data.json",
        splits=splits,
        filters=default_filters(),
    )

    rows = []

    for org_id, info in ds.iter_organoids():
        split = info.get("split")

        for day, rec in info["records"].items():
            imgs = rec.get("images") or {}
            mets = rec.get("metabolite") or {}
            plate = rec.get("plate") or {}

            batch = (plate.get("batch") or "").replace(" ", "_")
            well = plate.get("well") or ""
            well_id = f"{batch}_{well}" if batch and well else org_id

            row = {
                "org_id": org_id,
                "well_id": well_id,
                "split": split,
                "label": info["label"],
                "day": day,
                "day_num": day_to_int(day),
                "img_path": imgs.get("img_path"),
                "mask_path": imgs.get("mask_path"),
                "overlay_path": imgs.get("overlay_path"),
            }

            for met_name in BASE_MET_FEATURES:
                met_key = met_name.replace("_concentration_uM", "")
                row[met_name] = (mets.get(met_key) or {}).get("concentration_uM", np.nan)

            malate_key = MALATE_FEATURE.replace("_concentration_uM", "")
            row[MALATE_FEATURE] = (mets.get(malate_key) or {}).get("concentration_uM", np.nan)

            rows.append(row)

    df = pd.DataFrame(rows)

    print(f"Total records loaded: {len(df)} across {df['org_id'].nunique()} organoids")
    print(df["label"].value_counts(dropna=False))
    print(df["split"].value_counts(dropna=False))

    return df


def train_epoch(model, loader, opt, crit, weights, device, config):
    model.train()
    losses = []

    use_img = config.get("use_images", True)
    use_met = config.get("use_metabolites", True)

    for batch in loader:
        if use_img and use_met:
            img = batch[0].to(device)
            meta = batch[1].to(device)
            y = batch[2].to(device)
            logits = model(img, meta)

        elif use_img:
            img = batch[0].to(device)
            y = batch[1].to(device)
            logits = model(img)

        else:
            meta = batch[0].to(device)
            y = batch[1].to(device)
            logits = model(meta)

        loss_raw = crit(logits, y)
        sample_weights = torch.tensor(
            [weights[int(label.item())] for label in y],
            device=device,
            dtype=torch.float32,
        )
        loss = (loss_raw * sample_weights).mean()

        opt.zero_grad()
        loss.backward()
        opt.step()

        losses.append(loss.item())

    return float(np.mean(losses)) if losses else np.nan


def eval_epoch(model, loader, device, config):
    model.eval()

    probs_all = []
    labels_all = []

    use_img = config.get("use_images", True)
    use_met = config.get("use_metabolites", True)

    with torch.no_grad():
        for batch in loader:
            if use_img and use_met:
                img = batch[0].to(device)
                meta = batch[1].to(device)
                y = batch[2]
                logits = model(img, meta)

            elif use_img:
                img = batch[0].to(device)
                y = batch[1]
                logits = model(img)

            else:
                meta = batch[0].to(device)
                y = batch[1]
                logits = model(meta)

            probs = torch.sigmoid(logits).cpu().numpy()
            probs_all.extend(probs)
            labels_all.extend(y.numpy())

    probs_all = np.array(probs_all)
    labels_all = np.array(labels_all).astype(int)

    preds_bin = (probs_all >= 0.5).astype(int)

    return {
        "acc": accuracy_score(labels_all, preds_bin),
        "f1": f1_score(labels_all, preds_bin, pos_label=1, zero_division=0),
        "bal_acc": balanced_accuracy_score(labels_all, preds_bin),
        "auc": roc_auc_score(labels_all, probs_all) if len(np.unique(labels_all)) > 1 else None,
    }


def train_day_cv(day, day_df, config):
    early_mode = day in EARLY_DAYS
    mode_str = "early image-primary" if early_mode else "late metabolite-primary"

    proj_str = (
        f" + projection dim={config['proj_dim']}"
        if config.get("use_projection") and config.get("use_images") and config.get("use_metabolites")
        else ""
    )

    print(f"\n{'=' * 60}")
    print(f"{day} | {mode_str}{proj_str}")
    print(f"{'=' * 60}")

    if len(day_df) == 0:
        print(f"No data for {day}, skipping.")
        return []

    org_df = (
        day_df.drop_duplicates(subset="org_id")[["org_id", "well_id", "label"]]
        .reset_index(drop=True)
    )

    y_org = np.array([LABEL_MAP.get(label, 0) for label in org_df["label"]])
    groups = org_df["well_id"].values

    if len(np.unique(y_org)) < 2:
        print(f"Only one class present at {day}, skipping.")
        return []

    sgkf = StratifiedGroupKFold(
        n_splits=N_FOLDS,
        shuffle=True,
        random_state=SEED,
    )

    fold_results = []

    for fold_idx, (train_org_idx, test_org_idx) in enumerate(
        sgkf.split(org_df, y_org, groups)
    ):
        print(f"\n  Fold {fold_idx + 1}/{N_FOLDS}")

        test_org_ids = set(org_df.iloc[test_org_idx]["org_id"])

        train_org_sub = org_df.iloc[train_org_idx].reset_index(drop=True)
        y_train_sub = y_org[train_org_idx]
        groups_train = train_org_sub["well_id"].values

        inner_cv = StratifiedGroupKFold(
            n_splits=max(2, int(1 / VAL_FRACTION)),
            shuffle=True,
            random_state=SEED + fold_idx,
        )

        inner_train_idx, inner_val_idx = next(
            inner_cv.split(train_org_sub, y_train_sub, groups_train)
        )

        inner_train_org_ids = set(train_org_sub.iloc[inner_train_idx]["org_id"])
        inner_val_org_ids = set(train_org_sub.iloc[inner_val_idx]["org_id"])

        train_df_fold = day_df[day_df["org_id"].isin(inner_train_org_ids)]
        val_df_fold = day_df[day_df["org_id"].isin(inner_val_org_ids)]
        test_df_fold = day_df[day_df["org_id"].isin(test_org_ids)]

        print(
            f"    Train: {len(train_df_fold)}, "
            f"Val: {len(val_df_fold)}, "
            f"Test: {len(test_df_fold)}"
        )

        if len(train_df_fold) == 0 or len(test_df_fold) == 0:
            print("    Insufficient data, skipping fold.")
            continue

        t_train = get_transforms(config, augment=True)
        t_eval = get_transforms(config, augment=False)

        train_ds = OrganoidDataset(
            train_df_fold,
            config,
            transform=t_train,
            fit_scaler=True,
        )
        scaler = train_ds.scaler

        val_ds = OrganoidDataset(
            val_df_fold,
            config,
            transform=t_eval,
            scaler=scaler,
        )

        test_ds = OrganoidDataset(
            test_df_fold,
            config,
            transform=t_eval,
            scaler=scaler,
        )

        if len(train_ds) == 0 or len(test_ds) == 0:
            print("    No valid samples found, skipping fold.")
            continue

        train_loader = DataLoader(
            train_ds,
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=4,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=config["batch_size"],
            shuffle=False,
            num_workers=4,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=config["batch_size"],
            shuffle=False,
            num_workers=4,
        )

        labels_list = [LABEL_MAP.get(label, 0) for label in train_df_fold["label"]]
        unique_classes = np.unique(labels_list)

        weights_arr = compute_class_weight(
            class_weight="balanced",
            classes=unique_classes,
            y=labels_list,
        )

        class_weights = {int(c): float(w) for c, w in zip(unique_classes, weights_arr)}

        for c in [0, 1]:
            class_weights.setdefault(c, 1.0)

        model = AdaptiveMultimodalClassifier(config, early_mode=early_mode).to(config["device"])

        if config.get("use_images", True):
            for param in model.backbone.parameters():
                param.requires_grad = False

        opt = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config["learning_rate"],
            weight_decay=1e-4,
        )

        crit = nn.BCEWithLogitsLoss(reduction="none")

        best_bal_acc = -np.inf
        best_state = None
        es = EarlyStopping(config["early_stopping_patience"])

        for epoch in range(config["num_epochs"]):
            train_epoch(
                model,
                train_loader,
                opt,
                crit,
                class_weights,
                config["device"],
                config,
            )

            if len(val_ds) > 0:
                val_res = eval_epoch(model, val_loader, config["device"], config)
            else:
                val_res = eval_epoch(model, train_loader, config["device"], config)

            monitor = val_res["bal_acc"]

            if monitor > best_bal_acc:
                best_bal_acc = monitor
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()
                }

            if epoch % 10 == 0:
                print(f"    Epoch {epoch:3d}: val_bal_acc={monitor:.3f}")

            if es(monitor):
                print(f"    Early stop at epoch {epoch}")
                break

        if best_state is not None:
            model.load_state_dict(best_state)

        model.to(config["device"])
        test_res = eval_epoch(model, test_loader, config["device"], config)

        auc_print = f"{test_res['auc']:.3f}" if test_res["auc"] is not None else "N/A"

        print(
            f"    Test — Acc: {test_res['acc']:.3f}, "
            f"F1_NA: {test_res['f1']:.3f}, "
            f"BalAcc: {test_res['bal_acc']:.3f}, "
            f"AUC: {auc_print}"
        )

        fold_results.append({
            "fold": fold_idx,
            "acc": float(test_res["acc"]),
            "f1": float(test_res["f1"]),
            "bal_acc": float(test_res["bal_acc"]),
            "auc": float(test_res["auc"]) if test_res["auc"] is not None else None,
        })

    return fold_results


def aggregate_folds(fold_results):
    if not fold_results:
        return None

    metrics = ["acc", "f1", "bal_acc", "auc"]
    agg = {}

    for metric in metrics:
        vals = [r[metric] for r in fold_results if r[metric] is not None]
        if vals:
            agg[f"{metric}_mean"] = float(np.mean(vals))
            agg[f"{metric}_std"] = float(np.std(vals))
        else:
            agg[f"{metric}_mean"] = None
            agg[f"{metric}_std"] = None

    return agg


def plot_summary(results, output_dir, backbone, use_projection, proj_dim):
    days_present = [d for d in DAY_ORDER_LABELS if d in results and results[d]]
    day_nos = [day_to_int(d) for d in days_present]
    modes = ["early" if d in EARLY_DAYS else "late" for d in days_present]
    colors = ["#2196F3" if m == "early" else "#FF9800" for m in modes]

    proj_label = f" + Projection dim={proj_dim}" if use_projection else ""

    def get_mean_std(metric):
        means = [results[d][f"{metric}_mean"] for d in days_present]
        stds = [results[d][f"{metric}_std"] for d in days_present]
        return np.array(means, dtype=float), np.array(stds, dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle(
        f"Day-Adaptive Multimodal Model — 5-Fold CV ({backbone}){proj_label}\n"
        f"Blue = Early image-primary | Orange = Late metabolite-primary\n"
        f"Shaded bands = ±1 std across folds",
        fontsize=12,
        fontweight="bold",
    )

    metric_titles = [
        ("acc", "Test Accuracy"),
        ("f1", "Test F1 Score for Not Acceptable"),
        ("bal_acc", "Test Balanced Accuracy"),
        ("auc", "Test ROC-AUC"),
    ]

    for (metric, title), ax in zip(metric_titles, axes.flat):
        means, stds = get_mean_std(metric)
        valid = ~np.isnan(means)

        ax.plot(
            np.array(day_nos)[valid],
            means[valid],
            color="gray",
            linewidth=1.5,
            zorder=1,
        )

        ax.fill_between(
            np.array(day_nos)[valid],
            (means - stds)[valid],
            (means + stds)[valid],
            alpha=0.2,
            color="gray",
            zorder=0,
        )

        for x, y, s, c in zip(day_nos, means, stds, colors):
            if np.isnan(y):
                continue
            ax.scatter(x, y, color=c, s=90, zorder=3)
            ax.annotate(
                f"{y:.2f}\n±{s:.2f}",
                (x, y),
                textcoords="offset points",
                xytext=(0, 9),
                ha="center",
                fontsize=7.5,
            )

        ax.axvspan(19, 31, alpha=0.07, color="#FF9800")
        ax.axhline(0.5, color="gray", linestyle="--", alpha=0.4, linewidth=1)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Day")
        ax.set_xticks(day_nos)
        ax.set_xticklabels([str(n) for n in day_nos], rotation=45)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.1)

    early_patch = mpatches.Patch(color="#2196F3", label="Early image-primary")
    late_patch = mpatches.Patch(color="#FF9800", label="Late metabolite-primary")

    fig.legend(
        handles=[early_patch, late_patch],
        loc="lower center",
        ncol=2,
        fontsize=11,
        bbox_to_anchor=(0.5, 0.01),
    )

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    out = output_dir / "cv_summary.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")

    fig, ax = plt.subplots(figsize=(13, 5))

    palette = {
        "acc": "#1f77b4",
        "f1": "#ff7f0e",
        "bal_acc": "#2ca02c",
    }

    labels = {
        "acc": "Accuracy",
        "f1": "F1 Not Acceptable",
        "bal_acc": "Balanced Accuracy",
    }

    for metric, color in palette.items():
        means, stds = get_mean_std(metric)
        valid = ~np.isnan(means)
        xs = np.array(day_nos)[valid]

        ax.plot(
            xs,
            means[valid],
            "o-",
            color=color,
            label=labels[metric],
            linewidth=2,
        )

        ax.fill_between(
            xs,
            (means - stds)[valid],
            (means + stds)[valid],
            alpha=0.15,
            color=color,
        )

    ax.axvspan(19, 31, alpha=0.07, color="#FF9800")
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.4)
    ax.set_title(
        f"Accuracy / F1 / Balanced Accuracy — 5-Fold CV ({backbone}){proj_label}\n"
        f"Orange region = Late metabolite-primary",
        fontweight="bold",
    )

    ax.set_xlabel("Day")
    ax.set_ylabel("Score")
    ax.set_xticks(day_nos)
    ax.set_xticklabels([str(n) for n in day_nos], rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    out2 = output_dir / "cv_metrics_overlay.png"
    plt.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out2}")

    fig, ax = plt.subplots(figsize=(12, 5))

    means, stds = get_mean_std("bal_acc")
    valid = ~np.isnan(means)
    xs = np.array(day_nos)[valid]

    ax.plot(
        xs,
        means[valid],
        "o-",
        color="#2ca02c",
        linewidth=2.5,
        markersize=8,
    )

    ax.fill_between(
        xs,
        (means - stds)[valid],
        (means + stds)[valid],
        alpha=0.2,
        color="#2ca02c",
        label="±1 std",
    )

    ax.axvspan(
        19,
        31,
        alpha=0.07,
        color="#FF9800",
        label="Late metabolite-primary",
    )

    ax.axhline(
        0.5,
        color="gray",
        linestyle="--",
        alpha=0.5,
        label="Random baseline",
    )

    for x, y in zip(xs, means[valid]):
        ax.annotate(
            f"{y:.2f}",
            (x, y),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=9,
        )

    ax.set_title(
        f"Balanced Accuracy Across Days — 5-Fold CV ({backbone}){proj_label}",
        fontweight="bold",
        fontsize=13,
    )

    ax.set_xlabel("Day", fontsize=12)
    ax.set_ylabel("Balanced Accuracy", fontsize=12)
    ax.set_xticks(day_nos)
    ax.set_xticklabels([str(n) for n in day_nos], rotation=45)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.3, 1.05)

    plt.tight_layout()
    out3 = output_dir / "cv_balanced_accuracy.png"
    plt.savefig(out3, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out3}")


def main():
    parser = argparse.ArgumentParser(
        description="Day-adaptive multimodal CV with weighted BCE."
    )

    parser.add_argument("--backbone", choices=["vit", "resnet", "efficientnet"], default="efficientnet")
    parser.add_argument("--input-mode", choices=["rgb", "overlay"], default="rgb")
    parser.add_argument("--cross-attn-proj-dim", type=int, default=128)
    parser.add_argument("--cross-attn-heads", type=int, default=4)

    parser.add_argument("--use-projection", action="store_true")
    parser.add_argument("--proj-dim", type=int, default=256)

    parser.add_argument("--images-only", action="store_true")
    parser.add_argument("--metabolites-only", action="store_true")

    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--num-epochs", type=int, default=50)
    parser.add_argument("--early-stopping-patience", type=int, default=20)
    parser.add_argument("--use-augmentation", action="store_true")
    parser.add_argument("--days", nargs="*", default=None)
    parser.add_argument("--output-dir", default="outputs_cv")

    args = parser.parse_args()

    if args.images_only and args.metabolites_only:
        raise ValueError("Cannot set both --images-only and --metabolites-only")

    use_images = not args.metabolites_only
    use_metabolites = not args.images_only

    config = {
        "backbone": args.backbone,
        "input_mode": args.input_mode,
        "use_images": use_images,
        "use_metabolites": use_metabolites,
        "cross_attn_proj_dim": args.cross_attn_proj_dim,
        "cross_attn_heads": args.cross_attn_heads,
        "use_projection": args.use_projection,
        "proj_dim": args.proj_dim,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "num_epochs": args.num_epochs,
        "early_stopping_patience": args.early_stopping_patience,
        "target_size": (384, 512),
        "use_augmentation": args.use_augmentation,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }

    set_seed()

    modality_str = (
        "IMAGE ONLY"
        if args.images_only
        else "METABOLITE ONLY"
        if args.metabolites_only
        else "MULTIMODAL"
    )

    print("\n" + "=" * 70)
    print(f"DAY-ADAPTIVE MODEL — 5-FOLD CV — {modality_str}")
    print("Loss: weighted BCE")
    print("Label convention: Not Acceptable = 1, Acceptable = 0")
    print("=" * 70)

    for k, v in config.items():
        print(f"  {k:30s}: {v}")

    print("=" * 70 + "\n")

    all_df = load_all_data(config)

    days_to_train = args.days if args.days else sorted(
        all_df["day"].unique(),
        key=day_to_int,
    )

    print(f"Days to train: {days_to_train}\n")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    raw_results = {}

    for day in days_to_train:
        day_df = all_df[all_df["day"] == day]

        if len(day_df) == 0:
            continue

        fold_results = train_day_cv(day, day_df, config)

        if not fold_results:
            continue

        raw_results[day] = fold_results
        agg = aggregate_folds(fold_results)

        results[day] = {
            **agg,
            "day_no": day_to_int(day),
            "mode": "early" if day in EARLY_DAYS else "late",
        }

    with open(output_dir / "results.json", "w") as f:
        json.dump(
            {
                "aggregated": results,
                "per_fold": raw_results,
                "config": {
                    k: str(v) if not isinstance(v, (int, float, bool, str, type(None))) else v
                    for k, v in config.items()
                },
            },
            f,
            indent=2,
        )

    print("\n" + "=" * 75)
    print("SUMMARY mean ± std across 5 folds")
    print("=" * 75)
    print(f"{'Day':<10} {'Mode':<8} {'Acc':>12} {'F1_NA':>12} {'BalAcc':>12} {'AUC':>12}")
    print("-" * 70)

    for day in DAY_ORDER_LABELS:
        if day not in results:
            continue

        r = results[day]

        def fmt(metric):
            if r[f"{metric}_mean"] is None:
                return "N/A"
            return f"{r[f'{metric}_mean']:.3f}±{r[f'{metric}_std']:.3f}"

        print(
            f"{day:<10} {r['mode']:<8} "
            f"{fmt('acc'):>12} {fmt('f1'):>12} "
            f"{fmt('bal_acc'):>12} {fmt('auc'):>12}"
        )

    plot_summary(
        results,
        output_dir,
        args.backbone,
        args.use_projection,
        args.proj_dim,
    )

    print(f"\nAll outputs saved to {output_dir}/")


if __name__ == "__main__":
    main()
