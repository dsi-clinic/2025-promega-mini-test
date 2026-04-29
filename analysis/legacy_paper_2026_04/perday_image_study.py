#!/usr/bin/env python3
"""
Reproduce per-day EfficientNet image classifier results from the paper.

Architecture: EfficientNet-B0 (ImageNet pretrained) → 128-dim → binary logit
Input: overlay images (RGB with green mask outline), 384×512
Training: two-phase (freeze backbone → unfreeze last 2 blocks)
Loss: BCEWithLogitsLoss with class weights

Outputs:
  - analysis/outputs/images/results.json (metrics per day)
  - analysis/outputs/figures/perday_vs_timeseries.png (if time-series also run)

Usage:
    make run ARGS="-m analysis.paper_2026_04.perday_image_study"
    make run ARGS="-m analysis.paper_2026_04.perday_image_study --days Dy30"
    make run ARGS="-m analysis.paper_2026_04.perday_image_study --input-mode img"
"""

import argparse
import json
import os
import random
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from pipeline.data_loader import ANALYSIS_OUTPUT_DIR, DAY_ORDER, FIGURE_DIR, OrganoidDataset

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

SEED = 1  # student code used seed=1 for image models
ALL_DATA_PATH = "data/all_data.json"
SPLITS_CSV = "data/2026_winter_student_splits.csv"
OUTPUT_DIR = ANALYSIS_OUTPUT_DIR / "images"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Training hyperparameters (matching student code exactly)
BATCH_SIZE = 16
MAX_EPOCHS = 100
LR_HEAD = 5e-4
LR_BACKBONE = 5e-5  # student: LR * 0.1 = 5e-5
UNFREEZE_AFTER = 4  # epochs before unfreezing backbone
PATIENCE = 15
GRAD_CLIP = 1.0
IMG_HEIGHT = 384
IMG_WIDTH = 512

# ImageNet normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class OrganoidImageDataset(Dataset):
    """PyTorch dataset for organoid images."""

    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        label = self.labels[idx]

        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            # Return a blank image on failure
            img = Image.new("RGB", (IMG_WIDTH, IMG_HEIGHT), (128, 128, 128))

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(label, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class EfficientNetClassifier(nn.Module):
    """EfficientNet-B0 with a small classifier head."""

    def __init__(self):
        super().__init__()
        import timm
        self.backbone = timm.create_model(
            "efficientnet_b0", pretrained=True, num_classes=0
        )
        feat_dim = self.backbone.num_features  # 1280

        self.head = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
        )

        # Freeze backbone initially
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_last_blocks(self):
        """Unfreeze the last 2 blocks of EfficientNet."""
        # EfficientNet blocks are in self.backbone.blocks
        blocks = list(self.backbone.blocks)
        for block in blocks[-2:]:
            for param in block.parameters():
                param.requires_grad = True
        # Also unfreeze the conv_head and bn2
        if hasattr(self.backbone, "conv_head"):
            for param in self.backbone.conv_head.parameters():
                param.requires_grad = True
        if hasattr(self.backbone, "bn2"):
            for param in self.backbone.bn2.parameters():
                param.requires_grad = True

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features).squeeze(-1)

    def get_features(self, x):
        """Extract backbone features (for combined model)."""
        with torch.no_grad():
            return self.backbone(x)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def get_transforms(train=True):
    if train:
        return T.Compose([
            T.Resize((IMG_HEIGHT, IMG_WIDTH)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    else:
        return T.Compose([
            T.Resize((IMG_HEIGHT, IMG_WIDTH)),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])


def compute_metrics(y_true, y_pred, y_prob=None) -> dict:
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )

    return {
        "accuracy": round(acc, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "tnr": round(tnr, 4),
        "tpr": round(tpr, 4),
        "precision_acceptable": round(prec[0], 4),
        "recall_acceptable": round(rec[0], 4),
        "f1_acceptable": round(f1[0], 4),
        "precision_not_acceptable": round(prec[1], 4),
        "recall_not_acceptable": round(rec[1], 4),
        "f1_not_acceptable": round(f1[1], 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "n_test": len(y_true),
    }


def train_one_day(
    ds: OrganoidDataset,
    day: str,
    input_mode: str = "overlay",
    verbose: bool = True,
) -> Optional[dict]:
    """Train per-day EfficientNet classifier."""
    set_seed(SEED)

    # Get image paths and labels for each split
    def get_data(split):
        items = ds.get_image_paths(split, day, mode=input_mode)
        paths = [p for _, _, p in items]
        labels = [0 if lbl == "Acceptable" else 1 for _, lbl, _ in items]
        return paths, labels

    train_paths, train_labels = get_data("train")
    val_paths, val_labels = get_data("val")
    test_paths, test_labels = get_data("test")

    if len(train_paths) == 0 or len(test_paths) == 0:
        if verbose:
            print(f"  Skipping {day}: no data")
        return None

    n_pos = sum(train_labels)
    n_neg = len(train_labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        if verbose:
            print(f"  Skipping {day}: single class in train")
        return None

    if verbose:
        print(f"  Train: {len(train_paths)} ({n_neg} Acc, {n_pos} NAcc)")
        print(f"  Val:   {len(val_paths)}")
        print(f"  Test:  {len(test_paths)}")

    # Datasets and loaders
    train_ds = OrganoidImageDataset(train_paths, train_labels, get_transforms(True))
    val_ds = OrganoidImageDataset(val_paths, val_labels, get_transforms(False))
    test_ds = OrganoidImageDataset(test_paths, test_labels, get_transforms(False))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Model
    model = EfficientNetClassifier().to(DEVICE)

    # Loss with class weight
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Optimizer — initially only head params (student code)
    optimizer = optim.Adam(
        [p for p in model.head.parameters() if p.requires_grad],
        lr=LR_HEAD,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5  # student: minimize val_loss
    )

    # Training loop — student code selects by val accuracy (not balanced accuracy)
    best_val_acc = 0.0
    best_model_state = None
    patience_counter = 0
    backbone_unfrozen = False

    for epoch in range(MAX_EPOCHS):
        # Unfreeze backbone after warmup (epoch 4)
        if epoch == UNFREEZE_AFTER and not backbone_unfrozen:
            model.unfreeze_last_blocks()
            backbone_unfrozen = True
            # Student code: LR * 0.1 for all params after unfreeze
            optimizer = optim.Adam(model.parameters(), lr=LR_BACKBONE)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=5
            )

        # Train
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            train_loss += loss.item() * len(labels)
            preds = (torch.sigmoid(logits) > 0.5).long()
            correct += (preds == labels.long()).sum().item()
            total += len(labels)
        train_loss /= len(train_ds)

        # Validate
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_true = []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels_v = imgs.to(DEVICE), labels.to(DEVICE)
                logits = model(imgs)
                vloss = criterion(logits, labels_v)
                val_loss += vloss.item() * len(labels_v)
                probs = torch.sigmoid(logits)
                preds = (probs >= 0.5).long()
                val_preds.extend(preds.cpu().numpy())
                val_true.extend(labels.numpy().astype(int))
        val_loss /= max(len(val_ds), 1)

        val_acc = accuracy_score(val_true, val_preds)
        scheduler.step(val_loss)

        # Early stopping on val accuracy (matching student code)
        if val_acc > best_val_acc + 1e-4:
            best_val_acc = val_acc
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            if verbose:
                print(f"  Early stopping at epoch {epoch+1}")
            break

        if verbose and (epoch + 1) % 10 == 0:
            val_bal_acc = balanced_accuracy_score(val_true, val_preds)
            print(f"  Epoch {epoch+1}: loss={train_loss:.4f}, val_acc={val_acc:.4f}, val_bal_acc={val_bal_acc:.4f}")

    # Load best model and evaluate on test
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    model = model.to(DEVICE)
    model.eval()

    test_preds = []
    test_probs_list = []
    test_true = []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(DEVICE)
            logits = model(imgs)
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).long()
            test_preds.extend(preds.cpu().numpy())
            test_probs_list.extend(probs.cpu().numpy())
            test_true.extend(labels.numpy().astype(int))

    test_true = np.array(test_true)
    test_preds = np.array(test_preds)
    test_probs_arr = np.array(test_probs_list)

    metrics = compute_metrics(test_true, test_preds, test_probs_arr)
    metrics["best_val_accuracy"] = round(best_val_acc, 4)
    metrics["threshold"] = 0.5

    if verbose:
        print(f"  Test: bal_acc={metrics['balanced_accuracy']:.4f}, "
              f"acc={metrics['accuracy']:.4f}, tnr={metrics['tnr']:.4f}")

    return metrics


def plot_perday_results(results: dict, output_path: Path):
    """Plot per-day balanced accuracy."""
    import matplotlib.pyplot as plt

    days = []
    bal_accs = []
    for day in DAY_ORDER:
        m = results.get(day)
        if m is not None:
            days.append(day)
            bal_accs.append(m["balanced_accuracy"])

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(len(days)), bal_accs, "o-", label="Per-day EfficientNet",
            color="#1f77b4", linewidth=2)
    ax.set_xticks(range(len(days)))
    ax.set_xticklabels(days, rotation=45)
    ax.set_ylabel("Balanced Accuracy")
    ax.set_xlabel("Day")
    ax.set_title("Per-Day Image Classifier: Balanced Accuracy by Day")
    ax.legend()
    ax.set_ylim(0.3, 1.0)
    ax.grid(True, alpha=0.3)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="Random")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", nargs="+", default=None)
    parser.add_argument("--input-mode", default="overlay",
                        choices=["overlay", "img", "mask"])
    args = parser.parse_args()

    set_seed(SEED)
    ds = OrganoidDataset(ALL_DATA_PATH, splits_csv=SPLITS_CSV)
    print(ds.summary())
    print(f"Device: {DEVICE}")

    days_to_train = args.days if args.days else DAY_ORDER

    results = {}
    for day in days_to_train:
        if day not in ds.days:
            print(f"\nSkipping {day} (no data)")
            continue

        print(f"\n{'='*50}")
        print(f"Image Classifier - {day}")
        print(f"{'='*50}")
        m = train_one_day(ds, day, input_mode=args.input_mode)
        if m:
            results[day] = m

    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "perday_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {OUTPUT_DIR / 'perday_results.json'}")

    # Print summary
    if results:
        print(f"\n{'='*60}")
        print("PER-DAY IMAGE RESULTS SUMMARY")
        print(f"{'='*60}")
        tnrs = []
        bal_accs = []
        f1_nas = []
        for day in DAY_ORDER:
            m = results.get(day)
            if m:
                tnrs.append(m["tnr"])
                bal_accs.append(m["balanced_accuracy"])
                f1_nas.append(m["f1_not_acceptable"])
                print(f"  {day}: bal_acc={m['balanced_accuracy']:.4f}, "
                      f"tnr={m['tnr']:.4f}, f1_NA={m['f1_not_acceptable']:.4f}")

        n = len(tnrs)
        days_zero_tnr = sum(1 for t in tnrs if t == 0.0)
        print(f"\n  Avg TNR:     {np.mean(tnrs):.1%}")
        print(f"  Avg Bal Acc: {np.mean(bal_accs):.1%}")
        print(f"  Days TNR=0:  {days_zero_tnr}/{n}")
        print(f"  Avg F1(NA):  {np.mean(f1_nas):.1%}")

        # Plot
        FIGURE_DIR.mkdir(parents=True, exist_ok=True)
        plot_perday_results(results, FIGURE_DIR / "perday_image_balanced_accuracy.png")


if __name__ == "__main__":
    main()
