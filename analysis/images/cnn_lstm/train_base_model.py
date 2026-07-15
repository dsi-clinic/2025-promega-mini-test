#!/usr/bin/env python3
"""
Baseline EfficientNet (single timepoint) for comparison with LSTM models.
Trains on each day range separately: [8, 10, 13, 15, 17, 20.5, 24, 30]
Uses the same data splits as CNN-LSTM temporal models for fair comparison.
Run: python train_baseline_effnet.py
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ----- Repo root on sys.path -----
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0
from tqdm import tqdm

from analysis.images.cnn_lstm.organoid_dataset import make_idor_series_splits
from pipeline.data_loader import (
    LABEL_TO_INT,
    get_clipped_meanfill_image_path,
    get_day_float,
)

# -------- Config --------
DAY_RANGES = [3, 6, 8, 10, 13, 15, 17, 20.5, 24, 30]  # Same as LSTM
BATCH_SIZE = 16
NUM_WORKERS = 0
MAX_EPOCHS = 100
PATIENCE = 15
LR = 5e-4
GRAD_CLIP = 1.0
SEED = 42
TARGET_SIZE = (384, 512)  # (H, W) to match coworker's code
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------- Dataset ----------
class SingleDayOrganoidDataset(Dataset):
    """
    Dataset for single timepoint organoid images.
    Uses the LSTM processed images (same as LSTM but picks one timepoint).
    """
    def __init__(self, organoid_ids, dataset, target_day, transform=None, image_type='std'):
        self.samples = []

        for org_id in organoid_ids:
            label_str = dataset.organoid_label(org_id)
            if label_str is None:
                continue
            label = LABEL_TO_INT.get(label_str, 0)

            records = dataset.organoid_records(org_id)
            if not records:
                continue

            # Pick the day whose mdl_day is closest to target_day
            best_day = None
            best_dist = float("inf")
            for day in records:
                mdl = get_day_float(day)
                if mdl is None:
                    continue
                d = abs(mdl - target_day)
                if d < best_dist:
                    best_dist = d
                    best_day = day
            if best_day is None:
                continue
            best_rec = records[best_day]

            if image_type == "clipped":
                img_path = get_clipped_meanfill_image_path(best_rec)
            else:
                img_path = (best_rec.get("images") or {}).get("img_path")
            if img_path is None or not Path(img_path).exists():
                continue

            self.samples.append({
                "img_path": img_path,
                "label": label,
                "org_id": org_id,
                "actual_day": get_day_float(best_day),
            })

        self.transform = transform
        print(f"  Loaded {len(self.samples)} samples for day ~{target_day}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load image (same as LSTM)
        from skimage.io import imread
        img = imread(sample["img_path"])

        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)

        img = img.astype(np.float32) / 255.0  # Normalize to [0,1]

        # Apply transforms (if any)
        if self.transform:
            img_pil = Image.fromarray((img * 255).astype(np.uint8))
            img_pil = self.transform(img_pil)
            img = np.array(img_pil).astype(np.float32) / 255.0

        # Convert to tensor and apply ImageNet normalization (SAME AS LSTM!)
        img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
        img = torch.from_numpy(img).float()

        imagenet_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
        imagenet_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)
        img = (img - imagenet_mean) / imagenet_std

        label = torch.tensor(sample["label"], dtype=torch.float32)
        return img, label, sample["org_id"]


# ---------- Model ----------
class BaselineEfficientNet(nn.Module):
    """Single image classifier using EfficientNet-B0."""

    def __init__(self):
        super().__init__()
        # Load pretrained EfficientNet
        eff = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        eff.classifier = nn.Identity()
        self.backbone = eff

        # Freeze backbone initially
        for p in self.backbone.parameters():
            p.requires_grad = False

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(1280, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
        )

    def unfreeze_backbone(self, n_blocks=2):
        """Unfreeze last n blocks of EfficientNet."""
        feats = getattr(self.backbone, "features", None)
        if feats is None:
            return
        start = max(0, len(feats) - n_blocks)
        for i in range(start, len(feats)):
            for p in feats[i].parameters():
                p.requires_grad = True
        print(f"  Unfroze last {n_blocks} blocks of backbone")

    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features).squeeze(1)
        return logits


# ---------- Evaluation ----------
@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    all_probs, all_labels, all_ids = [], [], []
    losses = []

    for imgs, labels, ids in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)

        logits = model(imgs)
        loss = criterion(logits, labels)
        losses.append(loss.item())

        probs = torch.sigmoid(logits)
        all_probs.append(probs.cpu())
        all_labels.append(labels.cpu())
        all_ids.extend(ids)

    if len(all_probs) == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, float('nan'), float('nan'), [], []

    probs = torch.cat(all_probs)
    labels = torch.cat(all_labels)
    preds = (probs > 0.5).int()

    acc = (preds == labels.int()).float().mean().item()

    prec, rec, f1, _ = precision_recall_fscore_support(
        labels.numpy(), preds.numpy(), average="binary", zero_division=0
    )

    try:
        auc = roc_auc_score(labels.numpy(), probs.numpy())
    except ValueError:
        auc = float("nan")

    try:
        ap = average_precision_score(labels.numpy(), probs.numpy())
    except ValueError:
        ap = float("nan")

    # Get false positives/negatives
    fp_ids = [all_ids[i] for i in range(len(all_ids)) if preds[i] == 1 and labels[i] == 0]
    fn_ids = [all_ids[i] for i in range(len(all_ids)) if preds[i] == 0 and labels[i] == 1]

    return (
        float(np.mean(losses)),
        acc,
        float(prec),
        float(rec),
        float(f1),
        float(auc),
        float(ap),
        fp_ids,
        fn_ids,
    )


# ---------- Training ----------
def train_for_day(target_day, train_ids, val_ids, test_ids,
                  dataset, device, output_dir, image_type='std'):
    print(f"\n{'='*70}\nTRAINING BASELINE for DAY {target_day}\n{'='*70}")

    train_tf = T.Compose([
        T.Resize(TARGET_SIZE),
        T.RandomHorizontalFlip(0.5),
        T.RandomVerticalFlip(0.5),
        T.ColorJitter(0.2, 0.2, 0.2, 0.1),
    ])

    eval_tf = T.Compose([
        T.Resize(TARGET_SIZE),
    ])

    train_dataset = SingleDayOrganoidDataset(train_ids, dataset, target_day, transform=train_tf, image_type=image_type)
    val_dataset   = SingleDayOrganoidDataset(val_ids,   dataset, target_day, transform=eval_tf, image_type=image_type)
    test_dataset  = SingleDayOrganoidDataset(test_ids,  dataset, target_day, transform=eval_tf, image_type=image_type)

    if len(train_dataset) == 0:
        print(f"  ⚠ No training samples for day {target_day}, skipping")
        return None

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                             num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                           num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)

    # Class balance per rule #9: label 1 = Not Acceptable (minority).
    train_labels = [s["label"] for s in train_dataset.samples]
    n_pos = sum(train_labels)
    n_neg = len(train_labels) - n_pos
    if n_pos == 0: n_pos = 1
    if n_neg == 0: n_neg = 1
    pos_weight = torch.tensor([n_neg / n_pos], device=device)
    print(f"  Class balance: NotAcceptable={n_pos}, Acceptable={n_neg}, pos_weight={pos_weight.item():.3f}")

    # Model
    model = BaselineEfficientNet().to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.classifier.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    best_val_acc = -1.0
    best_state = None
    bad_epochs = 0
    history = []  # track per-epoch metrics for plotting

    # Training loop
    for epoch in range(1, MAX_EPOCHS + 1):
        # Unfreeze backbone after 3 epochs
        if epoch == 4:
            model.unfreeze_backbone()
            optimizer = optim.Adam(model.parameters(), lr=LR * 0.1)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for imgs, labels, _ in tqdm(train_loader, desc=f"Epoch {epoch:02d}", leave=False):
            imgs = imgs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            preds = (torch.sigmoid(logits) > 0.5).long()
            correct += (preds == labels.long()).sum().item()
            total += labels.size(0)

        train_loss = running_loss / max(1, total)
        train_acc = correct / max(1, total)

        val_loss, val_acc, val_prec, val_rec, val_f1, val_auc, val_ap, _, _ = evaluate(
            model, val_loader, criterion, device
        )

        scheduler.step(val_loss)

        print(
            f"Epoch {epoch:02d} | Train {train_acc:.3f}/{train_loss:.4f} | "
            f"Val {val_acc:.3f}/{val_loss:.4f} (P {val_prec:.3f} R {val_rec:.3f} F1 {val_f1:.3f})"
        )

        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
        })

        if val_acc > best_val_acc + 1e-4:
            best_val_acc = val_acc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            bad_epochs = 0
            print("  * new best")
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                print(f"  Early stopping at epoch {epoch}")
                break

    # Test with best model
    if best_state is None:
        best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state, strict=True)

    test_loss, test_acc, test_prec, test_rec, test_f1, test_auc, test_ap, test_fp, test_fn = evaluate(
        model, test_loader, criterion, device
    )

    # Save model
    model_dir = output_dir
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"model_day_{target_day}.pth"
    torch.save({
        "state_dict": best_state,
        "target_day": target_day,
        "best_val_acc": best_val_acc,
    }, model_path)

    print("\nFinal TEST results:")
    print(f"  Acc {test_acc:.3f} | F1 {test_f1:.3f} | P {test_prec:.3f} | R {test_rec:.3f}")
    print(f"  Saved → {model_path}")

    # Save confusion matrix
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels, _ in test_loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            preds = (torch.sigmoid(logits) > 0.5).int().cpu()
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.int().cpu().numpy())

    if len(all_preds) > 0:
        cm = confusion_matrix(all_labels, all_preds)
        print("\nConfusion Matrix (Test Set):")
        print("                       Predicted")
        print("                Acceptable   Not Acceptable")
        print(f"Acceptable        {cm[0,0]:4d}            {cm[0,1]:4d}")
        print(f"Not Acceptable    {cm[1,0]:4d}            {cm[1,1]:4d}")

        # --- Save confusion matrix image ---
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.colorbar(im, ax=ax)
        classes = ['Acceptable (0)', 'Not Acceptable (1)']
        ax.set(xticks=[0, 1], yticks=[0, 1],
               xticklabels=classes, yticklabels=classes,
               xlabel='Predicted', ylabel='Actual',
               title=f'Confusion Matrix – Day {target_day} (Test)')
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                        color='white' if cm[i, j] > thresh else 'black')
        plt.tight_layout()
        cm_path = model_dir / f'confusion_matrix_day_{target_day}.png'
        plt.savefig(cm_path, dpi=150)
        plt.close(fig)
        print(f"  Confusion matrix saved → {cm_path}")

    # --- Save accuracy & loss plot ---
    if history:
        epochs = [h['epoch'] for h in history]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        ax1.plot(epochs, [h['train_acc'] for h in history], label='Train Acc')
        ax1.plot(epochs, [h['val_acc'] for h in history], label='Val Acc')
        ax1.set(xlabel='Epoch', ylabel='Accuracy', title=f'Accuracy – Day {target_day}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.plot(epochs, [h['train_loss'] for h in history], label='Train Loss')
        ax2.plot(epochs, [h['val_loss'] for h in history], label='Val Loss')
        ax2.set(xlabel='Epoch', ylabel='Loss', title=f'Loss – Day {target_day}')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = model_dir / f'training_curves_day_{target_day}.png'
        plt.savefig(plot_path, dpi=150)
        plt.close(fig)
        print(f"  Training curves saved → {plot_path}")

    del model, train_loader, val_loader, test_loader
    torch.cuda.empty_cache()

    return {
        "target_day": target_day,
        "best_val_acc": float(best_val_acc),
        "test_acc": float(test_acc),
        "test_precision": float(test_prec),
        "test_recall": float(test_rec),
        "test_f1": float(test_f1),
        "test_auc": float(test_auc),
        "test_ap": float(test_ap),
        "model_path": str(model_path),
        "test_false_positives": test_fp,
        "test_false_negatives": test_fn,
    }


# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser(description='Train single-day EfficientNet baseline')
    parser.add_argument('--output-dir', type=str, default='outputs/base_models/base_effnet',
                        help='Output directory for model checkpoints and results')
    parser.add_argument('--image-type', type=str, default='std', choices=['clipped', 'std'],
                        help='Image variant: std (512x384) or clipped (575x575 AR meanfill)')
    args = parser.parse_args()

    set_seed(SEED)
    device = torch.device(DEVICE)
    print(f"Using device: {device}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out_dir}")

    # Load data (same splits as LSTM!)
    print("\n" + "="*70)
    print("LOADING DATA")
    print("="*70)

    ds, train_ids, val_ids, test_ids = make_idor_series_splits()

    print(f"Splits: train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}")

    print("\n" + "="*70)
    print("STARTING BASELINE TRAINING")
    print("="*70)

    # Train for each day range (same as LSTM)
    results = []
    for target_day in DAY_RANGES:
        result = train_for_day(
            target_day, train_ids, val_ids, test_ids,
            ds, device,
            out_dir / f"day_{target_day}",
            image_type=args.image_type
        )
        if result:
            results.append(result)

    # Save all results (matching LSTM format)
    results_path = out_dir / "baseline_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*70)
    print("BASELINE TRAINING SUMMARY")
    print("="*70)
    print(f"{'Day':<15} {'Val Acc':<12} {'Test Acc':<12} {'Test F1':<12}")
    print("-"*70)
    for r in results:
        print(f"{str(r['target_day']):<15} {r['best_val_acc']:<12.3f} {r['test_acc']:<12.3f} {r['test_f1']:<12.3f}")

    best = max(results, key=lambda x: x["test_acc"]) if results else None
    if best:
        print(f"\nBest on test (day {best['target_day']}): Acc={best['test_acc']:.3f}, F1={best['test_f1']:.3f}")
    print(f"Results saved → {results_path}")


if __name__ == "__main__":
    main()
