#!/usr/bin/env python3
"""
Baseline EfficientNet (single timepoint) for comparison with LSTM models.
Trains on each day range separately: [8, 10, 13, 15, 17, 20.5, 24, 30]
Uses the same data splits as CNN-LSTM temporal models for fair comparison.
Run: python train_baseline_effnet.py
"""

import sys, json, random
import os
from pathlib import Path

# ----- Repo root on sys.path -----
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

from sklearn.metrics import (
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)

from config import OUTPUT_FOLDER
from analysis.images.cnn_lstm.organoid_dataset import load_data_and_create_splits

# -------- Config --------
DAY_RANGES = [3, 6, 8, 10, 13, 15, 17, 20.5, 24, 30]  # Same as LSTM
BATCH_SIZE = 16
NUM_WORKERS = 0
MAX_EPOCHS = 100
PATIENCE = 15
LR = 5e-4
GRAD_CLIP = 1.0
SEED = 1
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
    def __init__(self, organoid_ids, series_metadata, data, target_day, transform=None):
        self.samples = []
        
        for org_id in organoid_ids:
            metadata = series_metadata.get(org_id, {})
            label_str = str(metadata.get("label", "")).strip().lower()
            label = 1 if label_str in ("good", "acceptable", "accepted") else 0
            
            # Get entry_keys and days (SAME AS LSTM!)
            entry_keys = metadata.get('entry_keys', [])
            days = metadata.get('days', [])
            
            if not entry_keys or not days:
                continue
            
            # Find the entry closest to target_day
            best_idx = None
            min_diff = float('inf')
            for i, day in enumerate(days):
                diff = abs(day - target_day)
                if diff < min_diff:
                    min_diff = diff
                    best_idx = i
            
            if best_idx is None:
                continue
            
            # Get the entry for that day
            entry_key = entry_keys[best_idx]
            entry = data.get(entry_key, {})
            
            # Get processed image path (512x384 resized images)
            processed = entry.get('processed', {})
            img_path = processed.get('img_path')
            
            if img_path is None or not Path(img_path).exists():
                continue
            
            self.samples.append({
                "img_path": img_path,
                "label": label,
                "org_id": org_id,
                "actual_day": days[best_idx],
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
                  series_metadata, data, device, output_dir):
    print(f"\n{'='*70}\nTRAINING BASELINE for DAY {target_day}\n{'='*70}")
    
    # Transforms (no ToTensor - we do that in __getitem__)
    train_tf = T.Compose([
        T.Resize(TARGET_SIZE),
        T.RandomHorizontalFlip(0.5),
        T.RandomVerticalFlip(0.5),
        T.ColorJitter(0.2, 0.2, 0.2, 0.1),
    ])
    
    eval_tf = T.Compose([
        T.Resize(TARGET_SIZE),
    ])
    
    # Datasets
    train_dataset = SingleDayOrganoidDataset(
        train_ids, series_metadata, data, target_day, transform=train_tf
    )
    val_dataset = SingleDayOrganoidDataset(
        val_ids, series_metadata, data, target_day, transform=eval_tf
    )
    test_dataset = SingleDayOrganoidDataset(
        test_ids, series_metadata, data, target_day, transform=eval_tf
    )
    
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
    
    # Class balance
    train_labels = [s["label"] for s in train_dataset.samples]
    n_good = sum(train_labels)
    n_bad = len(train_labels) - n_good
    if n_good == 0: n_good = 1
    if n_bad == 0: n_bad = 1
    pos_weight = torch.tensor([n_bad / n_good], device=device)
    print(f"  Class balance: good={n_good}, bad={n_bad}, pos_weight={pos_weight.item():.3f}")
    
    # Model
    model = BaselineEfficientNet().to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.classifier.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    best_val_acc = -1.0
    best_state = None
    bad_epochs = 0
    
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
    
    print(f"\nFinal TEST results:")
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
        print(f"              Predicted")
        print(f"              Good   Bad")
        print(f"Actual Good   {cm[1,1]:<6} {cm[1,0]:<6}")
        print(f"Actual Bad    {cm[0,1]:<6} {cm[0,0]:<6}")
    
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
    set_seed(SEED)
    device = torch.device(DEVICE)
    print(f"Using device: {device}")
    
    out_dir = OUTPUT_FOLDER / "base_models" / "base_effnet"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out_dir}")
    
    # Load data (same splits as LSTM!)
    print("\n" + "="*70)
    print("LOADING DATA")
    print("="*70)
    
    series_metadata_path = OUTPUT_FOLDER / "complete_series_metadata_no_blanks.json"
    data_path = OUTPUT_FOLDER / "complete_series_data_no_blanks.json"
    
    train_ids, val_ids, test_ids, series_metadata, data = load_data_and_create_splits(
        series_metadata_path, data_path, random_seed=SEED
    )
    
    print(f"Splits: train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}")
    
    print("\n" + "="*70)
    print("STARTING BASELINE TRAINING")
    print("="*70)
    
    # Train for each day range (same as LSTM)
    results = []
    for target_day in DAY_RANGES:
        result = train_for_day(
            target_day, train_ids, val_ids, test_ids,
            series_metadata, data, device,
            out_dir / f"day_{target_day}"
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