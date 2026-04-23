#!/usr/bin/env python3
"""
Tony's version with DINOv2 instead of ViT:
- Uses reproducible splits from split_data_reproducible.py (both_train_base.json, both_val_base.json)
- Includes focal loss and learning rate scheduling (from survey classifier improvements)
- Trains DINOv2, ResNet, and EfficientNet models
"""

import os, json, argparse, re, csv
from pathlib import Path
from collections import defaultdict

# -------- Path remapping --------
_DATA_DIR = Path(os.environ.get("DATA_DIR", "/net/projects2/promega/2026_04_15_data"))
_IMG_DIR = _DATA_DIR / "intermediate" / "resized_512x384"
_SPLIT_DIR = _DATA_DIR / "intermediate" / "data_splits"

_SUFFIX_RE = re.compile(r"_(nosplit|presplit)_(nostitch|stitch)$")

def _remap_img_path(img_path: str) -> str:
    """Remap stale img_path to 2026_04_15_data resized_512x384 flat directory."""
    stem = Path(img_path).stem
    # Strip _nosplit_nostitch / _nosplit_stitch / _presplit_nostitch etc.
    stem = _SUFFIX_RE.sub("", stem)
    candidate = _IMG_DIR / f"{stem}.png"
    return str(candidate) if candidate.exists() else img_path

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score, balanced_accuracy_score
import timm
from torchvision import transforms as T
from transformers import AutoModel

# -------- Config (defaults; can be overridden by CLI) --------
BACKBONES = {
    "dinov2": "facebook/dinov2-base",  # DINOv2 instead of ViT
    "resnet": "resnet50",
    "efficientnet": "efficientnet_b0",
}
# Prefer split files from the data directory if present, fall back to repo copy
_CLUSTER_SPLIT_DIR = _DATA_DIR / "intermediate" / "data_splits"
SPLIT_DATA_DIR = _CLUSTER_SPLIT_DIR if _CLUSTER_SPLIT_DIR.exists() else Path("data_splits")
TRAIN_SPLIT_FILE = SPLIT_DATA_DIR / "both_train_base.json"
VAL_SPLIT_FILE = SPLIT_DATA_DIR / "both_val_base.json"
TEST_SPLIT_FILE = SPLIT_DATA_DIR / "both_test_base.json"
OUT_ROOT = Path("analysis/images/classifier/outputs_512x384_tony_dinov2_fixed_splits")
BATCH_SIZE = 16
# IMPORTANT: torchvision Resize expects (H, W). We want 512x384 images => (H=384, W=512)
TARGET_SIZE = (384, 512)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 0
SEED = 1
# -------------------------------------------------------------

# ---------- Utils ----------
def set_seed(seed=SEED):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def day_to_int(day_str: str) -> int:
    # "Dy28" -> 28, fallback -1
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
    """Dataset that can optionally return mask tensors alongside images."""

    def __init__(self, img_paths, labels, mask_paths=None, augment=False, use_mask=False, use_dinov2=False):
        self.img_paths = img_paths
        self.labels = labels
        self.mask_paths = mask_paths
        self.augment = augment
        self.use_mask = use_mask
        t = [T.Resize(TARGET_SIZE)]
        if augment:
            t += [
                T.RandomHorizontalFlip(0.5),
                T.ColorJitter(0.2, 0.2, 0.2, 0.1),
            ]
        t += [T.ToTensor()]
        # DINOv2 expects ImageNet normalization
        if use_dinov2:
            t += [T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        self.t_img = T.Compose(t)

        if self.use_mask:
            if self.mask_paths is None:
                raise ValueError("mask_paths must be provided when use_mask=True")
            self.t_mask = T.Compose([
                T.Resize(TARGET_SIZE, interpolation=T.InterpolationMode.NEAREST),
                T.ToTensor(),
            ])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert("RGB")
        img = self.t_img(img)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        if self.use_mask:
            mask = Image.open(self.mask_paths[idx]).convert("L")
            mask = self.t_mask(mask)
            return img, mask, label

        return img, label

# ---------- Model ----------
class MaskBranch(nn.Module):
    """Compact branch to encode binary masks into a feature vector."""
    def __init__(self, out_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, out_dim),
            nn.ReLU(inplace=True),
        )
        self.out_dim = out_dim
    def forward(self, mask):
        return self.encoder(mask)

class ImageOnlyClassifier(nn.Module):
    def __init__(self, backbone_key, backbone_name, target_size, use_mask=False):
        super().__init__()
        self.use_mask = use_mask
        self.backbone_key = backbone_key
        self._is_dinov2 = (backbone_key == "dinov2")

        if self._is_dinov2:
            # DINOv2 from HuggingFace
            self.backbone = AutoModel.from_pretrained(backbone_name)
            # Freeze backbone initially
            for p in self.backbone.parameters():
                p.requires_grad = False
            out_dim = self.backbone.config.hidden_size  # 768 for base
        else:
            # Timm models (ResNet, EfficientNet)
            extra_args = {}
            self.backbone = timm.create_model(
                backbone_name,
                pretrained=True,
                num_classes=0,          # feature extractor
                global_pool="avg",
                **extra_args
            )
            out_dim = self.backbone.num_features
            # freeze backbone initially
            for p in self.backbone.parameters():
                p.requires_grad = False

        if self.use_mask:
            self.mask_branch = MaskBranch(out_dim=64)
            head_in = out_dim + self.mask_branch.out_dim
        else:
            self.mask_branch = None
            head_in = out_dim

        self.classifier = nn.Sequential(
            nn.Linear(head_in, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
        )

    def unfreeze_backbone(self):
        if self._is_dinov2:
            # Unfreeze the last few layers of DINOv2
            for name, p in self.backbone.named_parameters():
                if "encoder.layer" in name:
                    try:
                        layer_match = re.search(r'layer\.(\d+)', name)
                        if layer_match:
                            layer_num = int(layer_match.group(1))
                            # Unfreeze last 4 layers (layers 8-11 for base)
                            if layer_num >= 8:
                                p.requires_grad = True
                    except:
                        pass
        else:
            # Timm models: unfreeze blocks/layers for fine-tuning
            for name, p in self.backbone.named_parameters():
                if "blocks." in name or "layer" in name:
                    p.requires_grad = True

    def forward(self, img, mask=None):
        if self._is_dinov2:
            # DINOv2 returns a dict with 'last_hidden_state'
            outputs = self.backbone(img)
            # Use CLS token (first token) from last hidden state
            f = outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]
        else:
            # Timm models
            f = self.backbone(img)
        
        if self.use_mask:
            if mask is None:
                raise ValueError("mask tensor must be provided when use_mask=True")
            f_mask = self.mask_branch(mask)
            f = torch.cat([f, f_mask], dim=1)
        return self.classifier(f).squeeze(1)

def make_loader(imgs, labels, augment, batch_size, mask_paths=None, use_mask=False, use_dinov2=False):
    ds = OrganoidDataset(imgs, labels, mask_paths=mask_paths, augment=augment, use_mask=use_mask, use_dinov2=use_dinov2)
    return DataLoader(ds, batch_size=batch_size, shuffle=augment, num_workers=NUM_WORKERS)

# ---------- Focal Loss (PyTorch version) ----------
class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance and hard examples.
    Adapted from survey classifier improvements (gamma=2.0, alpha=0.25).
    """
    def __init__(self, gamma=2.0, alpha=0.25, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs are logits, convert to probabilities
        probs = torch.sigmoid(inputs)
        
        # Calculate binary cross entropy
        bce = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # Calculate p_t (probability of true class)
        p_t = targets * probs + (1 - targets) * (1 - probs)
        
        # Calculate modulating factor
        modulating_factor = torch.pow(1 - p_t, self.gamma)
        
        # Apply alpha weighting
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        
        # Calculate focal loss
        focal_loss = alpha_t * modulating_factor * bce
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# ---------- Train/Eval ----------
def epoch_loop(model, loader, optimizer, class_weights, train=True, use_mask=False, focal_loss_fn=None):
    """
    Run one pass over a DataLoader for training or evaluation.

    Args:
        model: The classifier model.
        loader: PyTorch DataLoader yielding (img, [mask,] label) batches.
        optimizer: Optimizer to step when train=True (ignored when train=False).
        class_weights: Dict mapping class index -> weight for imbalance handling.
        train: If True, run in training mode and update model parameters.
        use_mask: If True, expect masks in the batches and pass them to the model.
        focal_loss_fn: Optional FocalLoss instance. If None, use weighted BCE loss.

    Returns:
        mean_loss (float), accuracy (float), binary_predictions (np.ndarray), true_labels (np.ndarray).
    """
    model.train() if train else model.eval()
    
    # Use focal loss if provided, otherwise use weighted BCE
    if focal_loss_fn is not None:
        loss_fn = focal_loss_fn
    else:
        loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    
    losses, preds, trues = [], [], []

    for batch in loader:
        if use_mask:
            img, mask, label = batch
            img, mask, label = img.to(DEVICE), mask.to(DEVICE), label.to(DEVICE)
            logit = model(img, mask)
        else:
            img, label = batch
            img, label = img.to(DEVICE), label.to(DEVICE)
            logit = model(img)
        
        if focal_loss_fn is not None:
            loss = loss_fn(logit, label)
            # Apply class weights
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

    preds_bin = (np.array(preds) > 0.5).astype(int)
    acc = accuracy_score(trues, preds_bin)
    return np.mean(losses), acc, preds_bin, np.array(trues)

def evaluate_on_loader(model, loader, use_mask=False):
    """Run inference (no grad) and compute accuracy & F1. Return preds_bin, trues, acc, f1, probs."""
    model.eval()
    preds_bin, trues, probs = [], [], []
    with torch.no_grad():
        for batch in loader:
            if use_mask:
                img, mask, lbl = batch
                img = img.to(DEVICE)
                mask = mask.to(DEVICE)
                prob = torch.sigmoid(model(img, mask)).cpu().numpy()
            else:
                img, lbl = batch
                img = img.to(DEVICE)
                prob = torch.sigmoid(model(img)).cpu().numpy()
            probs.extend(prob)
            preds_bin.extend((prob > 0.5).astype(int))
            trues.extend(lbl.numpy())
    preds_bin = np.array(preds_bin); trues = np.array(trues); probs = np.array(probs)
    acc = accuracy_score(trues, preds_bin)
    f1 = f1_score(trues, preds_bin)
    return preds_bin, trues, float(acc), float(f1), probs

def load_split_data(train_file: Path, val_file: Path, test_file: Path):
    """Load train/val/test splits from split_data_reproducible.py output."""
    with open(train_file) as f:
        train_data = json.load(f)
    with open(val_file) as f:
        val_data = json.load(f)
    with open(test_file) as f:
        test_data = json.load(f)
    return train_data, val_data, test_data

def extract_samples_by_day(split_data, day_str, input_key='img_path', use_mask=False):
    """Extract samples for a specific day from split data.
    
    Args:
        split_data: dict from split JSON (organoid_id -> {label, timepoints, ...})
        day_str: e.g., 'Dy03', 'Dy28'
        input_key: 'img_path' or 'overlay_path'
        use_mask: whether to extract mask_path
    
    Returns:
        (img_paths, labels, mask_paths) where mask_paths is None if use_mask=False
    """
    img_paths = []
    labels = []
    mask_paths = [] if use_mask else None
    
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
        img_path = _remap_img_path(tp_data.get(input_key, ""))
        if not img_path:
            continue

        if not Path(img_path).exists():
            continue
        
        if use_mask:
            mask_path = tp_data.get('mask_path')
            if not mask_path or not Path(mask_path).exists():
                continue
            mask_paths.append(mask_path)
        
        img_paths.append(img_path)
        labels.append(label)
    
    labels = np.array(labels, dtype=int)
    img_paths = np.array(img_paths)
    if use_mask:
        mask_paths = np.array(mask_paths)
    
    return img_paths, labels, mask_paths

def run_training_for_day(day_str: str, backbone_key: str, backbone_name: str,
                         train_data: dict, val_data: dict, test_data: dict,
                         train_bs: int, val_bs: int,
                         out_root: Path, input_key: str, use_mask: bool):
    """Train + validate using fixed splits; select by VAL acc, report on TEST."""
    model_dir = out_root / backbone_key / day_str
    if (model_dir / "metrics_test.json").exists():
        print(f"Skipping {day_str}/{backbone_key}: metrics_test.json already exists")
        with open(model_dir / "metrics_test.json") as f:
            m = json.load(f)
        return {
            "day": day_str,
            "day_no": m.get("day_no", -1),
            "backbone_key": backbone_key,
            "val_accuracy": m.get("val_accuracy_for_selection", 0.0),
            "test_accuracy": m.get("accuracy", 0.0),
            "test_f1": m.get("f1", 0.0),
            "val_roc_auc": m.get("val_roc_auc"),
            "test_roc_auc": m.get("roc_auc"),
            "val_num": m.get("val_n", 0),
            "test_num": m.get("test_n", 0),
            "test_actual_good": m.get("actual_good", 0),
            "test_pred_good": m.get("predicted_good", 0),
        }

    # Extract samples for this day from all splits
    train_imgs, train_labels, train_masks = extract_samples_by_day(
        train_data, day_str, input_key, use_mask
    )
    
    # Extract samples for this day from val split
    val_imgs, val_labels, val_masks = extract_samples_by_day(
        val_data, day_str, input_key, use_mask
    )
    
    # Extract samples for this day from test split
    test_imgs, test_labels, test_masks = extract_samples_by_day(
        test_data, day_str, input_key, use_mask
    )
    
    if len(train_imgs) == 0:
        print(f"Skipping {day_str}: no training samples")
        return None
    
    if len(val_imgs) == 0:
        print(f"Skipping {day_str}: no validation samples")
        return None
    
    if len(test_imgs) == 0:
        print(f"Skipping {day_str}: no test samples")
        return None
    
    # Use val and test directly (no splitting needed)
    X_val, y_val = val_imgs, val_labels
    X_test, y_test = test_imgs, test_labels
    if use_mask:
        M_val = val_masks
        M_test = test_masks
    else:
        M_val = M_test = None
    
    # class weights (train only)
    weights = compute_class_weight("balanced", classes=np.unique(train_labels), y=train_labels)
    class_weights = {int(k): float(w) for k, w in zip(np.unique(train_labels), weights)}
    
    # loaders (use_dinov2 flag for proper normalization)
    use_dinov2_flag = (backbone_key == "dinov2")
    train_loader = make_loader(
        train_imgs, train_labels,
        mask_paths=train_masks if use_mask else None,
        augment=True,  # Use augmentation for training
        batch_size=train_bs,
        use_mask=use_mask,
        use_dinov2=use_dinov2_flag,
    )
    val_loader = make_loader(
        X_val, y_val,
        mask_paths=M_val if use_mask else None,
        augment=False,
        batch_size=val_bs,
        use_mask=use_mask,
        use_dinov2=use_dinov2_flag,
    )
    test_loader = make_loader(
        X_test, y_test,
        mask_paths=M_test if use_mask else None,
        augment=False,
        batch_size=val_bs,
        use_mask=use_mask,
        use_dinov2=use_dinov2_flag,
    )
    
    # model/opt
    model = ImageOnlyClassifier(backbone_key, backbone_name, TARGET_SIZE, use_mask=use_mask).to(DEVICE)
    model_dir = out_root / backbone_key / day_str
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "model.pth"
    
    # ENHANCED: Use focal loss (from survey classifier improvements)
    focal_loss_fn = FocalLoss(gamma=2.0, alpha=0.25)
    
    opt = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    # ENHANCED: Add learning rate scheduler (from survey classifier improvements)
    scheduler = ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=10, min_lr=1e-7, verbose=True)
    es = EarlyStopping(patience=20)
    history = defaultdict(list)
    best_acc = -np.inf
    
    # Phase 1 — frozen backbone.
    # Use an upper bound on epochs; EarlyStopping (patience=20) plus ReduceLROnPlateau
    # will typically stop much earlier once validation accuracy stops improving.
    for epoch in range(100):
        tl, tacc, _, _ = epoch_loop(model, train_loader, opt, class_weights, train=True, use_mask=use_mask, focal_loss_fn=focal_loss_fn)
        vl, vacc, _, _ = epoch_loop(model, val_loader,   opt, class_weights, train=False, use_mask=use_mask, focal_loss_fn=focal_loss_fn)
        history["train_loss"].append(tl); history["val_loss"].append(vl)
        history["train_acc"].append(tacc); history["val_acc"].append(vacc)
        print(f"[{day_str}][{backbone_key}][P1][{epoch:02d}][bs={train_bs}/{val_bs}] loss {tl:.4f}/{vl:.4f} acc {tacc:.3f}/{vacc:.3f}")
        if vacc > best_acc:
            best_acc = vacc
            torch.save(model.state_dict(), model_path)
        # ENHANCED: Step scheduler based on validation accuracy
        scheduler.step(vacc)
        if es.step(vacc):
            break
    
    # Phase 2 — unfreeze partial backbone
    model.unfreeze_backbone()
    opt = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    # ENHANCED: Add learning rate scheduler for phase 2
    scheduler = ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=10, min_lr=1e-7, verbose=True)
    es = EarlyStopping(patience=30)
    for epoch in range(300):
        tl, tacc, _, _ = epoch_loop(model, train_loader, opt, class_weights, train=True, use_mask=use_mask, focal_loss_fn=focal_loss_fn)
        vl, vacc, _, _ = epoch_loop(model, val_loader,   opt, class_weights, train=False, use_mask=use_mask, focal_loss_fn=focal_loss_fn)
        history["train_loss"].append(tl); history["val_loss"].append(vl)
        history["train_acc"].append(tacc); history["val_acc"].append(vacc)
        print(f"[{day_str}][{backbone_key}][P2][{epoch:03d}][bs={train_bs}/{val_bs}] loss {tl:.4f}/{vl:.4f} acc {tacc:.3f}/{vacc:.3f}")
        if vacc > best_acc:
            best_acc = vacc
            torch.save(model.state_dict(), model_path)
        # ENHANCED: Step scheduler based on validation accuracy
        scheduler.step(vacc)
        if es.step(vacc):
            break
    
    # Save per-day training curves
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1); plt.plot(history["train_acc"], label="Train"); plt.plot(history["val_acc"], label="Val"); plt.title("Accuracy"); plt.legend()
    plt.subplot(1, 2, 2); plt.plot(history["train_loss"], label="Train"); plt.plot(history["val_loss"], label="Val"); plt.title("Loss"); plt.legend()
    plt.tight_layout()
    plt.savefig(model_dir / "training_curves.png")
    plt.close()
    print(f"Saved training curves to {model_dir/'training_curves.png'}")
    
    # ---- Evaluate with best VAL checkpoint
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    
    # Val metrics
    _, val_trues, val_acc, val_f1, val_probs = evaluate_on_loader(model, val_loader, use_mask=use_mask)
    try:
        val_roc_auc = float(roc_auc_score(val_trues, val_probs))
    except Exception:
        val_roc_auc = None
    val_pr_auc = float(average_precision_score(val_trues, val_probs)) if len(val_trues) > 0 else None
    
    # Test metrics (final reporting)
    preds_bin, trues, test_acc, test_f1, test_probs = evaluate_on_loader(model, test_loader, use_mask=use_mask)
    try:
        test_roc_auc = float(roc_auc_score(trues, test_probs))
    except Exception:
        test_roc_auc = None
    test_pr_auc = float(average_precision_score(trues, test_probs)) if len(trues) > 0 else None
    
    day_no = day_to_int(day_str)
    num_in_sample = int(len(trues))
    actual_good = int(trues.sum())
    predicted_good = int(preds_bin.sum())
    
    val_metrics = {
        "day": day_str,
        "split": "val",
        "accuracy": float(val_acc),
        "f1": float(val_f1),
        "roc_auc": val_roc_auc,
        "pr_auc": val_pr_auc,
        "n": int(len(y_val)),
        "batch_size": int(val_bs),
        "input_key": input_key,
        "use_mask": use_mask,
    }
    with (model_dir / "metrics_val.json").open("w") as f:
        json.dump(val_metrics, f, indent=2)
    
    test_metrics = {
        "day": day_str,
        "day_no": day_no,
        "split": "test",
        "accuracy": float(test_acc),
        "balanced_accuracy": float(balanced_accuracy_score(trues, preds_bin)),
        "f1": float(test_f1),
        "roc_auc": test_roc_auc,
        "pr_auc": test_pr_auc,
        "val_accuracy_for_selection": float(best_acc),
        "val_n": int(len(y_val)),
        "test_n": num_in_sample,
        "actual_good": actual_good,
        "predicted_good": predicted_good,
        "batch_size_train": int(train_bs),
        "batch_size_valtest": int(val_bs),
        "backbone_key": backbone_key,
        "input_key": input_key,
        "use_mask": use_mask,
    }
    with (model_dir / "metrics_test.json").open("w") as f:
        json.dump(test_metrics, f, indent=2)
    print(f"Saved metrics to {model_dir/'metrics_val.json'} and {model_dir/'metrics_test.json'}")
    
    return {
        "day": day_str,
        "day_no": day_no,
        "backbone_key": backbone_key,
        "val_accuracy": float(best_acc),
        "test_accuracy": float(test_acc),
        "test_f1": float(test_f1),
        "val_roc_auc": val_roc_auc,
        "test_roc_auc": test_roc_auc,
        "val_num": int(len(y_val)),
        "test_num": num_in_sample,
        "test_actual_good": actual_good,
        "test_pred_good": predicted_good,
    }

# ---------- Orchestration ----------
def main():
    """
    Entry point for fixed-split image classifier training with DINOv2/ResNet/EfficientNet.

    Loads reproducible train/val/test splits, trains each backbone per day using focal
    loss, early stopping, and LR scheduling, then writes per-day CSV summaries and
    per-backbone JSON metrics to the configured output directory.
    """
    set_seed()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default=OUT_ROOT, help="Where to save outputs")
    parser.add_argument("--train-split", default=TRAIN_SPLIT_FILE, help="Path to train split JSON")
    parser.add_argument("--val-split", default=VAL_SPLIT_FILE, help="Path to val split JSON")
    parser.add_argument("--test-split", default=TEST_SPLIT_FILE, help="Path to test split JSON")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Train batch size")
    parser.add_argument("--val-batch-size", type=int, default=None, help="Val/Test batch size (defaults to train batch size)")
    parser.add_argument(
        "--use-mask",
        action="store_true",
        help="Include mask_path tensors and a mask branch in the classifier",
    )
    parser.add_argument(
        "--input-path-key",
        choices=["img_path", "overlay_path"],
        default="img_path",
        help="Which JSON field to use as the primary image input",
    )
    args = parser.parse_args()
    
    out_dir = Path(args.outdir); out_dir.mkdir(parents=True, exist_ok=True)
    train_split_file = Path(args.train_split)
    val_split_file = Path(args.val_split)
    test_split_file = Path(args.test_split)
    
    if not train_split_file.exists():
        raise FileNotFoundError(f"Train split file not found: {train_split_file}")
    if not val_split_file.exists():
        raise FileNotFoundError(f"Val split file not found: {val_split_file}")
    if not test_split_file.exists():
        raise FileNotFoundError(f"Test split file not found: {test_split_file}")
    
    # Load split data
    print(f"Loading splits from {train_split_file}, {val_split_file}, and {test_split_file}")
    train_data, val_data, test_data = load_split_data(train_split_file, val_split_file, test_split_file)
    print(f"Loaded {len(train_data)} train organoids, {len(val_data)} val organoids, {len(test_data)} test organoids")
    
    train_bs = int(args.batch_size)
    val_bs = int(args.val_batch_size) if args.val_batch_size is not None else train_bs
    use_mask = bool(args.use_mask)
    input_key = str(args.input_path_key)
    
    print(f"Using batch sizes — train: {train_bs}, val/test: {val_bs}")
    print(f"Target size (HxW): {TARGET_SIZE}")
    print(f"Input field: {input_key}; masks enabled: {use_mask}")
    print(f"Using fixed train/val/test splits from split_data_reproducible.py")
    
    # Get all unique days from the splits
    all_days = set()
    for org_data in list(train_data.values()) + list(val_data.values()) + list(test_data.values()):
        all_days.update(org_data.get('timepoints', {}).keys())
    all_days = sorted(all_days, key=day_to_int)
    print(f"Found {len(all_days)} days: {', '.join(all_days)}")
    
    # Collect results: pick the best backbone per day by **validation accuracy**
    per_day_best = {}
    per_model_results = {bk: {} for bk in BACKBONES}
    for day_str in all_days:
        best = None
        for backbone_key, backbone_name in BACKBONES.items():
            res = run_training_for_day(
                day_str, backbone_key, backbone_name,
                train_data, val_data, test_data,
                train_bs, val_bs,
                out_dir, input_key=input_key, use_mask=use_mask
            )
            if res is None:
                continue
            per_model_results[backbone_key][day_str] = res
            if (best is None) or (res["val_accuracy"] > best["val_accuracy"]):
                best = res
        if best:
            per_day_best[day_str] = best
            print(
                f"Best for {day_str} (by VAL): {best['backbone_key']} | "
                f"val acc={best['val_accuracy']:.3f} | "
                f"TEST acc={best['test_accuracy']:.3f}, f1={best['test_f1']:.3f}"
            )
        else:
            print(f"No valid result for {day_str}")
    
    if not per_day_best:
        print("No days produced results; aborting summary.")
        return
    
    # ---- Build 4-column table (based on TEST)
    rows = []
    days_sorted = sorted(per_day_best.keys(), key=day_to_int)
    for d in days_sorted:
        r = per_day_best[d]
        rows.append({
            "Day No": r["day_no"],
            "Num in Sample": r["test_num"],
            "Actual Good": r["test_actual_good"],
            "Predicted Good": r["test_pred_good"],
        })
    
    # Save CSV table
    table_path = out_dir / "day_summary.csv"
    with table_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["Day No", "Num in Sample", "Actual Good", "Predicted Good"]
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved per-day summary table to {table_path}")
    
    # ---- Per-model charts and summary
    day_numbers = {}
    for day_res in per_model_results.values():
        for day, res in day_res.items():
            day_numbers[day] = res["day_no"]
    
    if day_numbers:
        unique_day_nos = sorted(set(day_numbers.values()))
        
        def plot_metric(metric_key, ylabel, title, filename):
            plt.figure(figsize=(9, 4))
            plotted_any = False
            for backbone_key, day_res in per_model_results.items():
                if not day_res:
                    continue
                pairs = [
                    (day_numbers[day], day_res[day].get(metric_key))
                    for day in sorted(day_res.keys(), key=lambda d: day_numbers[d])
                    if day_res[day].get(metric_key) is not None
                ]
                if not pairs:
                    continue
                xs, ys = zip(*pairs)
                plt.plot(xs, ys, marker="o", label=backbone_key)
                plotted_any = True
            if plotted_any:
                plt.xlabel("Day")
                plt.ylabel(ylabel)
                plt.title(title)
                plt.xticks(unique_day_nos)
                plt.ylim(0.0, 1.0)
                plt.legend()
                plt.tight_layout()
                out_path = out_dir / filename
                plt.savefig(out_path)
                print(f"Saved {title.lower()} plot to {out_path}")
            plt.close()
        
        plot_metric("test_accuracy", "Accuracy (test)", "Per-day Test Accuracy by Backbone", "accuracy_by_model.png")
        plot_metric("test_f1", "F1 score (test)", "Per-day Test F1 by Backbone", "f1_by_model.png")
        plot_metric("test_roc_auc", "ROC AUC (test)", "Per-day Test ROC AUC by Backbone", "rocauc_by_model.png")
    
    # ---- Final TEST summary JSON (per model)
    per_model_summary = {}
    for backbone_key, day_res in per_model_results.items():
        per_model_summary[backbone_key] = {
            "per_day": {
                day: {
                    "day_no": int(day_numbers.get(day, res["day_no"])),
                    "test_accuracy": float(res["test_accuracy"]),
                    "test_f1": float(res["test_f1"]),
                    "test_roc_auc": (None if res["test_roc_auc"] is None else float(res["test_roc_auc"])),
                    "val_accuracy": float(res["val_accuracy"]),
                    "val_roc_auc": (None if res["val_roc_auc"] is None else float(res["val_roc_auc"])),
                    "test_num": int(res["test_num"]),
                }
                for day, res in day_res.items()
            }
        }
    
    summary = {
        "per_model": per_model_summary,
        "batch_size_train": int(train_bs),
        "batch_size_valtest": int(val_bs),
        "used_fixed_splits": True,
        "train_split_file": str(train_split_file),
        "val_split_file": str(val_split_file),
        "test_split_file": str(test_split_file),
    }
    summary_path = out_dir / "final_test_summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved final test summary to {summary_path}")
    
    # ---- Also print the 4-column table to stdout
    print("\n=== Summary Table (TEST) ===")
    print(f"{'Day No':>6} | {'Num in Sample':>13} | {'Actual Good':>11} | {'Predicted Good':>14}")
    print("-" * 54)
    for row in rows:
        print(f"{row['Day No']:>6} | {row['Num in Sample']:>13} | {row['Actual Good']:>11} | {row['Predicted Good']:>14}")

if __name__ == "__main__":
    main()

