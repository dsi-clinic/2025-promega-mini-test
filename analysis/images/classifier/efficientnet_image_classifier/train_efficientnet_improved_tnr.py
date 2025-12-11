#!/usr/bin/env python3
"""
Improved EfficientNet training with Phase 2 TNR optimizations.
ONLY trains EfficientNet (not DINOv2 or ResNet) to save time.

Key Improvements for TNR:
1. Model selection by BALANCED ACCURACY (not accuracy)
2. Boosted minority class weights (2.5x)
3. Lower focal loss alpha (0.15 instead of 0.25)
4. Track TNR/TPR during training
5. Early stopping on balanced accuracy
6. LR scheduler monitors balanced accuracy
7. More aggressive augmentation
"""

import os, json, argparse, re, csv
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.utils.class_weight import compute_class_weight
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score
import timm
from torchvision import transforms as T

# -------- Config --------
BACKBONE_NAME = "efficientnet_b0"
BACKBONE_KEY = "efficientnet"
TARGET_SIZE = (384, 512)  # (H, W)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 0
SEED = 1
BATCH_SIZE = 16

# IMPROVED: TNR-focused parameters
MINORITY_CLASS_BOOST = 2.5  # Boost "Not Acceptable" weight by this factor
FOCAL_LOSS_ALPHA = 0.15  # Lower alpha = more weight on minority class (was 0.25)
FOCAL_LOSS_GAMMA = 2.0
# -------------------------------------------------------------

def set_seed(seed=SEED):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def day_to_int(day_str: str) -> int:
    m = re.search(r"[Dd][Yy](\d+)", day_str)
    return int(m.group(1)) if m else -1

class EarlyStopping:
    def __init__(self, patience=20, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best = -np.inf
        self.bad = 0
    def step(self, score):
        if score > self.best + self.min_delta:
            self.best = score
            self.bad = 0
            return False
        self.bad += 1
        return self.bad >= self.patience

# ---------- Data ----------
class OrganoidDataset(Dataset):
    def __init__(self, img_paths, labels, augment=False):
        self.img_paths = img_paths
        self.labels = labels
        
        # IMPROVED: More aggressive augmentation
        t = [T.Resize(TARGET_SIZE)]
        if augment:
            t += [
                T.RandomHorizontalFlip(0.5),
                T.RandomVerticalFlip(0.3),  # NEW: vertical flips
                T.RandomRotation(15),  # NEW: rotations
                T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),  # IMPROVED: more aggressive
                T.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # NEW: translations
                T.RandomPerspective(distortion_scale=0.2, p=0.3),  # NEW: perspective
            ]
        t += [T.ToTensor()]
        self.t_img = T.Compose(t)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert("RGB")
        img = self.t_img(img)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return img, label

# ---------- Model ----------
class ImageOnlyClassifier(nn.Module):
    def __init__(self, backbone_name, target_size):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=True,
            num_classes=0,
            global_pool="avg",
        )
        out_dim = self.backbone.num_features
        
        # Freeze backbone initially
        for p in self.backbone.parameters():
            p.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Linear(out_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
        )

    def unfreeze_backbone(self):
        for name, p in self.backbone.named_parameters():
            if "blocks." in name or "layer" in name:
                p.requires_grad = True

    def forward(self, img):
        f = self.backbone(img)
        return self.classifier(f).squeeze(1)

def make_loader(imgs, labels, augment, batch_size):
    ds = OrganoidDataset(imgs, labels, augment=augment)
    return DataLoader(ds, batch_size=batch_size, shuffle=augment, num_workers=NUM_WORKERS)

# ---------- Focal Loss ----------
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        probs = torch.sigmoid(inputs)
        bce = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        p_t = targets * probs + (1 - targets) * (1 - probs)
        modulating_factor = torch.pow(1 - p_t, self.gamma)
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        focal_loss = alpha_t * modulating_factor * bce
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# ---------- IMPROVED Train/Eval ----------
def epoch_loop(model, loader, optimizer, class_weights, train=True, focal_loss_fn=None):
    """
    IMPROVED: Now returns balanced_acc, tnr, tpr for monitoring.
    """
    model.train() if train else model.eval()
    
    if focal_loss_fn is not None:
        loss_fn = focal_loss_fn
    else:
        loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    
    losses, preds, trues = [], [], []

    for batch in loader:
        img, label = batch
        img, label = img.to(DEVICE), label.to(DEVICE)
        logit = model(img)
        
        if focal_loss_fn is not None:
            loss = loss_fn(logit, label)
            weight = torch.tensor([class_weights[int(l.item())] for l in label], device=label.device)
            loss = (loss * weight).mean()
        else:
            loss = loss_fn(logit, label)
            weight = torch.tensor([class_weights[int(l.item())] for l in label], device=label.device)
            loss = (loss * weight).mean()

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        losses.append(loss.item())
        preds.extend(torch.sigmoid(logit).detach().cpu().numpy())
        trues.extend(label.cpu().numpy())

    preds = np.array(preds)
    trues = np.array(trues)
    preds_bin = (preds > 0.5).astype(int)
    acc = accuracy_score(trues, preds_bin)
    
    # IMPROVED: Calculate TNR, TPR, balanced accuracy
    tn = ((preds_bin == 0) & (trues == 0)).sum()
    fp = ((preds_bin == 1) & (trues == 0)).sum()
    fn = ((preds_bin == 0) & (trues == 1)).sum()
    tp = ((preds_bin == 1) & (trues == 1)).sum()
    
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    balanced_acc = (tpr + tnr) / 2.0
    
    return np.mean(losses), acc, balanced_acc, tnr, tpr, preds_bin, trues

def evaluate_on_loader(model, loader):
    """Run inference and compute all metrics including TNR/TPR."""
    model.eval()
    preds_bin, trues, probs = [], [], []
    with torch.no_grad():
        for batch in loader:
            img, lbl = batch
            img = img.to(DEVICE)
            logit = model(img)
            prob = torch.sigmoid(logit).cpu().numpy()
            probs.extend(prob)
            preds_bin.extend((prob > 0.5).astype(int))
            trues.extend(lbl.numpy())
    
    preds_bin = np.array(preds_bin)
    trues = np.array(trues)
    probs = np.array(probs)
    
    acc = accuracy_score(trues, preds_bin)
    f1 = f1_score(trues, preds_bin)
    
    # Calculate TNR, TPR
    tn = ((preds_bin == 0) & (trues == 0)).sum()
    fp = ((preds_bin == 1) & (trues == 0)).sum()
    fn = ((preds_bin == 0) & (trues == 1)).sum()
    tp = ((preds_bin == 1) & (trues == 1)).sum()
    
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    balanced_acc = (tpr + tnr) / 2.0
    
    return preds_bin, trues, float(acc), float(f1), probs, float(tpr), float(tnr), float(balanced_acc)

def load_split_data(train_file: Path, val_file: Path, test_file: Path):
    with open(train_file) as f:
        train_data = json.load(f)
    with open(val_file) as f:
        val_data = json.load(f)
    with open(test_file) as f:
        test_data = json.load(f)
    return train_data, val_data, test_data

def extract_samples_by_day(split_data, day_str):
    img_paths = []
    labels = []
    label_map = {"Acceptable": 1, "Not Acceptable": 0}
    
    for organoid_id, org_data in split_data.items():
        label_str = org_data.get('label')
        if label_str not in label_map:
            continue
        
        label = label_map[label_str]
        timepoints = org_data.get('timepoints', {})
        
        if day_str not in timepoints:
            continue
        
        tp_data = timepoints[day_str]
        img_path = tp_data.get('img_path')
        if not img_path or not Path(img_path).exists():
            continue
        
        img_paths.append(img_path)
        labels.append(label)
    
    return np.array(img_paths), np.array(labels, dtype=int)

def run_training_for_day(day_str: str, train_data: dict, val_data: dict, test_data: dict,
                         train_bs: int, val_bs: int, out_root: Path):
    """Train EfficientNet for one day with IMPROVED TNR-focused training."""
    
    # Extract samples
    train_imgs, train_labels = extract_samples_by_day(train_data, day_str)
    val_imgs, val_labels = extract_samples_by_day(val_data, day_str)
    test_imgs, test_labels = extract_samples_by_day(test_data, day_str)
    
    if len(train_imgs) == 0 or len(val_imgs) == 0 or len(test_imgs) == 0:
        print(f"Skipping {day_str}: insufficient data")
        return None
    
    # IMPROVED: Compute class weights and BOOST minority class
    weights = compute_class_weight("balanced", classes=np.unique(train_labels), y=train_labels)
    class_weights = {int(k): float(w) for k, w in zip(np.unique(train_labels), weights)}
    
    if 0 in class_weights:  # Class 0 = "Not Acceptable"
        class_weights[0] *= MINORITY_CLASS_BOOST
        print(f"[OK] BOOSTED minority class weight by {MINORITY_CLASS_BOOST}x")
    print(f"  Class weights: {class_weights}")
    
    # Loaders
    train_loader = make_loader(train_imgs, train_labels, augment=True, batch_size=train_bs)
    val_loader = make_loader(val_imgs, val_labels, augment=False, batch_size=val_bs)
    test_loader = make_loader(test_imgs, test_labels, augment=False, batch_size=val_bs)
    
    # Model
    model = ImageOnlyClassifier(BACKBONE_NAME, TARGET_SIZE).to(DEVICE)
    model_dir = out_root / BACKBONE_KEY / day_str
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "model.pth"
    
    # IMPROVED: Use focal loss with lower alpha
    focal_loss_fn = FocalLoss(gamma=FOCAL_LOSS_GAMMA, alpha=FOCAL_LOSS_ALPHA)
    print(f"[OK] Using Focal Loss (gamma={FOCAL_LOSS_GAMMA}, alpha={FOCAL_LOSS_ALPHA})")
    
    opt = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    # IMPROVED: Scheduler monitors balanced accuracy
    scheduler = ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=10, min_lr=1e-7, verbose=True)
    es = EarlyStopping(patience=20)
    history = defaultdict(list)
    best_balanced_acc = -np.inf  # IMPROVED: Track best balanced accuracy
    
    print(f"\n{'='*80}")
    print(f"Training {day_str} - Phase 1 (Frozen Backbone)")
    print(f"{'='*80}")
    
    # Phase 1 — frozen backbone
    for epoch in range(100):
        tl, tacc, tbal, ttnr, ttpr, _, _ = epoch_loop(model, train_loader, opt, class_weights, train=True, focal_loss_fn=focal_loss_fn)
        vl, vacc, vbal, vtnr, vtpr, _, _ = epoch_loop(model, val_loader, opt, class_weights, train=False, focal_loss_fn=focal_loss_fn)
        
        history["train_loss"].append(tl)
        history["val_loss"].append(vl)
        history["train_acc"].append(tacc)
        history["val_acc"].append(vacc)
        history["train_balanced_acc"].append(tbal)
        history["val_balanced_acc"].append(vbal)
        history["train_tnr"].append(ttnr)
        history["val_tnr"].append(vtnr)
        history["train_tpr"].append(ttpr)
        history["val_tpr"].append(vtpr)
        
        # IMPROVED: Print TNR/TPR metrics
        print(f"[{day_str}][P1][{epoch:02d}] loss {tl:.4f}/{vl:.4f} | "
              f"acc {tacc:.3f}/{vacc:.3f} | bal {tbal:.3f}/{vbal:.3f} | "
              f"TNR {ttnr:.3f}/{vtnr:.3f} | TPR {ttpr:.3f}/{vtpr:.3f}")
        
        # IMPROVED: Save model based on balanced accuracy
        if vbal > best_balanced_acc:
            best_balanced_acc = vbal
            torch.save(model.state_dict(), model_path)
            print(f"  → New best balanced acc: {vbal:.4f} (saved)")
        
        # IMPROVED: Step scheduler and early stopping on balanced accuracy
        scheduler.step(vbal)
        if es.step(vbal):
            print(f"  → Early stopping at epoch {epoch}")
            break
    
    # Phase 2 — unfreeze backbone
    print(f"\n{'='*80}")
    print(f"Training {day_str} - Phase 2 (Unfrozen Backbone)")
    print(f"{'='*80}")
    
    model.unfreeze_backbone()
    opt = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    scheduler = ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=10, min_lr=1e-7, verbose=True)
    es = EarlyStopping(patience=30)
    
    for epoch in range(300):
        tl, tacc, tbal, ttnr, ttpr, _, _ = epoch_loop(model, train_loader, opt, class_weights, train=True, focal_loss_fn=focal_loss_fn)
        vl, vacc, vbal, vtnr, vtpr, _, _ = epoch_loop(model, val_loader, opt, class_weights, train=False, focal_loss_fn=focal_loss_fn)
        
        history["train_loss"].append(tl)
        history["val_loss"].append(vl)
        history["train_acc"].append(tacc)
        history["val_acc"].append(vacc)
        history["train_balanced_acc"].append(tbal)
        history["val_balanced_acc"].append(vbal)
        history["train_tnr"].append(ttnr)
        history["val_tnr"].append(vtnr)
        history["train_tpr"].append(ttpr)
        history["val_tpr"].append(vtpr)
        
        print(f"[{day_str}][P2][{epoch:03d}] loss {tl:.4f}/{vl:.4f} | "
              f"acc {tacc:.3f}/{vacc:.3f} | bal {tbal:.3f}/{vbal:.3f} | "
              f"TNR {ttnr:.3f}/{vtnr:.3f} | TPR {ttpr:.3f}/{vtpr:.3f}")
        
        if vbal > best_balanced_acc:
            best_balanced_acc = vbal
            torch.save(model.state_dict(), model_path)
            print(f"  → New best balanced acc: {vbal:.4f} (saved)")
        
        scheduler.step(vbal)
        if es.step(vbal):
            print(f"  → Early stopping at epoch {epoch}")
            break
    
    # Save training curves with TNR/TPR
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    axes[0, 0].plot(history["train_acc"], label="Train")
    axes[0, 0].plot(history["val_acc"], label="Val")
    axes[0, 0].set_title("Accuracy")
    axes[0, 0].legend()
    
    axes[0, 1].plot(history["train_loss"], label="Train")
    axes[0, 1].plot(history["val_loss"], label="Val")
    axes[0, 1].set_title("Loss")
    axes[0, 1].legend()
    
    axes[0, 2].plot(history["train_balanced_acc"], label="Train")
    axes[0, 2].plot(history["val_balanced_acc"], label="Val")
    axes[0, 2].set_title("Balanced Accuracy")
    axes[0, 2].legend()
    
    axes[1, 0].plot(history["train_tnr"], label="Train")
    axes[1, 0].plot(history["val_tnr"], label="Val")
    axes[1, 0].set_title("TNR (True Negative Rate)")
    axes[1, 0].legend()
    
    axes[1, 1].plot(history["train_tpr"], label="Train")
    axes[1, 1].plot(history["val_tpr"], label="Val")
    axes[1, 1].set_title("TPR (True Positive Rate / Recall)")
    axes[1, 1].legend()
    
    axes[1, 2].plot(history["val_tpr"], history["val_tnr"], 'o-', alpha=0.5)
    axes[1, 2].set_xlabel("TPR")
    axes[1, 2].set_ylabel("TNR")
    axes[1, 2].set_title("TPR vs TNR Evolution")
    axes[1, 2].plot([0, 1], [0, 1], 'k--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(model_dir / "training_curves.png")
    plt.close()
    print(f"[OK] Saved training curves to {model_dir/'training_curves.png'}")
    
    # Evaluate on test set
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    
    _, val_trues, val_acc, val_f1, val_probs, val_tpr, val_tnr, val_balanced_acc = evaluate_on_loader(model, val_loader)
    preds_bin, test_trues, test_acc, test_f1, test_probs, test_tpr, test_tnr, test_balanced_acc = evaluate_on_loader(model, test_loader)
    
    try:
        val_roc_auc = float(roc_auc_score(val_trues, val_probs))
    except:
        val_roc_auc = None
    val_pr_auc = float(average_precision_score(val_trues, val_probs)) if len(val_trues) > 0 else None
    
    try:
        test_roc_auc = float(roc_auc_score(test_trues, test_probs))
    except:
        test_roc_auc = None
    test_pr_auc = float(average_precision_score(test_trues, test_probs)) if len(test_trues) > 0 else None
    
    day_no = day_to_int(day_str)
    actual_good = int(test_trues.sum())
    predicted_good = int(preds_bin.sum())
    
    # Save metrics
    test_metrics = {
        "day": day_str,
        "day_no": day_no,
        "backbone_key": BACKBONE_KEY,
        "split": "test",
        "accuracy": test_acc,
        "f1": test_f1,
        "tpr": test_tpr,
        "tnr": test_tnr,
        "balanced_accuracy": test_balanced_acc,
        "roc_auc": test_roc_auc,
        "pr_auc": test_pr_auc,
        "val_balanced_accuracy_for_selection": float(best_balanced_acc),
        "test_n": int(len(test_trues)),
        "actual_good": actual_good,
        "predicted_good": predicted_good,
        "improvements_used": {
            "minority_class_boost": MINORITY_CLASS_BOOST,
            "focal_loss_alpha": FOCAL_LOSS_ALPHA,
            "selection_criterion": "balanced_accuracy",
            "augmentation": "aggressive"
        }
    }
    
    with (model_dir / "metrics_test.json").open("w") as f:
        json.dump(test_metrics, f, indent=2)
    
    print(f"\n[OK] Test Results:")
    print(f"  Accuracy: {test_acc:.4f}, F1: {test_f1:.4f}")
    print(f"  TNR: {test_tnr:.4f}, TPR: {test_tpr:.4f}")
    print(f"  Balanced Acc: {test_balanced_acc:.4f}")
    print(f"  ROC-AUC: {test_roc_auc:.4f}" if test_roc_auc else "  ROC-AUC: N/A")
    
    return test_metrics

# ---------- Main ----------
def main():
    set_seed()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", required=True, help="Output directory")
    parser.add_argument("--train-split", required=True, help="Train split JSON")
    parser.add_argument("--val-split", required=True, help="Val split JSON")
    parser.add_argument("--test-split", required=True, help="Test split JSON")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size")
    args = parser.parse_args()
    
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    train_data, val_data, test_data = load_split_data(
        Path(args.train_split),
        Path(args.val_split),
        Path(args.test_split)
    )
    
    print("="*80)
    print("IMPROVED EFFICIENTNET TRAINING (TNR-FOCUSED)")
    print("="*80)
    print(f"\nImprovements:")
    print(f"  [OK] Model selection by BALANCED ACCURACY (not accuracy)")
    print(f"  [OK] Minority class weight boosted by {MINORITY_CLASS_BOOST}x")
    print(f"  [OK] Focal loss alpha = {FOCAL_LOSS_ALPHA} (was 0.25)")
    print(f"  [OK] Track TNR/TPR during training")
    print(f"  [OK] Early stopping on balanced accuracy")
    print(f"  [OK] More aggressive augmentation")
    print(f"\nLoaded:")
    print(f"  Train: {len(train_data)} organoids")
    print(f"  Val: {len(val_data)} organoids")
    print(f"  Test: {len(test_data)} organoids")
    print(f"\nDevice: {DEVICE}")
    print(f"Output: {out_dir}\n")
    
    # Get all days
    all_days = set()
    for org_data in list(train_data.values()) + list(val_data.values()) + list(test_data.values()):
        all_days.update(org_data.get('timepoints', {}).keys())
    all_days = sorted(all_days, key=day_to_int)
    
    print(f"Days to train: {', '.join(all_days)}\n")
    
    all_results = []
    for day_str in all_days:
        print(f"\n{'#'*80}")
        print(f"# DAY: {day_str}")
        print(f"{'#'*80}\n")
        
        result = run_training_for_day(
            day_str, train_data, val_data, test_data,
            args.batch_size, args.batch_size, out_dir
        )
        if result:
            all_results.append(result)
    
    # Save summary
    summary_path = out_dir / "training_summary.json"
    with summary_path.open("w") as f:
        json.dump({
            "results": all_results,
            "backbone": BACKBONE_KEY,
            "improvements": {
                "minority_class_boost": MINORITY_CLASS_BOOST,
                "focal_loss_alpha": FOCAL_LOSS_ALPHA,
                "selection_criterion": "balanced_accuracy"
            }
        }, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"[OK] TRAINING COMPLETE")
    print(f"={'='*80}")
    print(f"Results saved to: {out_dir}")
    print(f"Summary: {summary_path}")

if __name__ == "__main__":
    main()

