#!/usr/bin/env python3
"""
Temporal Transformer-based image classifier for organoid sequences.

Processes image sequences (Dy03→Dy06→...→Dy30) with EfficientNet backbone + Transformer
to capture growth patterns and predict final Dy30 outcome.

Architecture:
- EfficientNet-B0 backbone (frozen initially) extracts 256-dim features per day
- Day embeddings for positional encoding
- 2-layer Transformer encoder (4 heads, 512 FF dim)
- Mean pooling over time steps (with masking)
- MLP classifier head

Key features:
- Handles missing days with masking
- Class-weighted focal loss for imbalanced data
- Balanced accuracy for model selection
- TNR/TPR tracking
"""

import json
import argparse
import re
from pathlib import Path
from collections import defaultdict
from typing import List, Tuple, Dict, Optional

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.utils.class_weight import compute_class_weight
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)
import timm
from torchvision import transforms as T

# -------- Config --------
BACKBONE_NAME = "efficientnet_b0"
BACKBONE_KEY = "temporal_transformer"
TARGET_SIZE = (384, 512)  # (H, W)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 0
SEED = 1
BATCH_SIZE = 8  # Smaller batch for sequences

# Temporal config
FEATURE_DIM = 256  # EfficientNet feature dimension after projection
TRANSFORMER_D_MODEL = 256
TRANSFORMER_N_HEADS = 4
TRANSFORMER_N_LAYERS = 2
TRANSFORMER_D_FF = 512
TRANSFORMER_DROPOUT = 0.2

# Expected days in order
EXPECTED_DAYS = ["Dy03", "Dy06", "Dy08", "Dy10", "Dy13", "Dy15", "Dy17", "Dy20_5", "Dy24", "Dy28", "Dy30"]
MAX_SEQ_LEN = len(EXPECTED_DAYS)

# TNR-focused parameters
MINORITY_CLASS_BOOST = 2.5
FOCAL_LOSS_ALPHA = 0.15
FOCAL_LOSS_GAMMA = 2.0


def set_seed(seed=SEED):
    """Set random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def day_to_int(day_str: str) -> float:
    """Extract day number from day string."""
    if day_str == "Dy20_5":
        return 20.5
    m = re.search(r"[Dd][Yy](\d+)", day_str)
    return float(m.group(1)) if m else -1.0


class EarlyStopping:
    """Early stopping callback."""
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


# ---------- Temporal Dataset ----------
class TemporalOrganoidDataset(Dataset):
    """Dataset for temporal organoid sequences."""
    
    def __init__(self, organoid_data: Dict, augment=False):
        """
        Args:
            organoid_data: Dict mapping organoid_id to {
                'label': 0 or 1,
                'timepoints': {day_str: {'img_path': ...}}
            }
            augment: If True, apply augmentation to each image.
        """
        self.organoid_data = organoid_data
        self.organoid_ids = list(organoid_data.keys())
        self.augment = augment
        
        # Image transforms
        t = [T.Resize(TARGET_SIZE)]
        if augment:
            t += [
                T.RandomHorizontalFlip(0.5),
                T.RandomVerticalFlip(0.3),
                T.RandomRotation(15),
                T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                T.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                T.RandomPerspective(distortion_scale=0.2, p=0.3),
            ]
        t += [T.ToTensor()]
        self.t_img = T.Compose(t)
    
    def __len__(self):
        return len(self.organoid_ids)
    
    def __getitem__(self, idx):
        org_id = self.organoid_ids[idx]
        org_info = self.organoid_data[org_id]
        label = torch.tensor(org_info['label'], dtype=torch.float32)
        timepoints = org_info['timepoints']
        
        # Build sequence: [MAX_SEQ_LEN, 3, H, W]
        # Build mask: [MAX_SEQ_LEN] (1 = valid, 0 = missing)
        # Build day indices: [MAX_SEQ_LEN] (for positional encoding)
        images = []
        mask = []
        day_indices = []
        
        for i, day in enumerate(EXPECTED_DAYS):
            if day in timepoints and timepoints[day].get('img_path'):
                img_path = Path(timepoints[day]['img_path'])
                if img_path.exists():
                    img = Image.open(img_path).convert("RGB")
                    img = self.t_img(img)
                    images.append(img)
                    mask.append(1.0)
                    day_indices.append(i)
                else:
                    images.append(torch.zeros(3, TARGET_SIZE[0], TARGET_SIZE[1]))
                    mask.append(0.0)
                    day_indices.append(i)
            else:
                images.append(torch.zeros(3, TARGET_SIZE[0], TARGET_SIZE[1]))
                mask.append(0.0)
                day_indices.append(i)
        
        # Stack: [MAX_SEQ_LEN, 3, H, W]
        image_sequence = torch.stack(images)
        mask_tensor = torch.tensor(mask, dtype=torch.float32)
        day_indices_tensor = torch.tensor(day_indices, dtype=torch.long)
        
        return image_sequence, mask_tensor, day_indices_tensor, label, org_id


# ---------- Temporal Model ----------
class TemporalTransformerClassifier(nn.Module):
    """Temporal classifier: EfficientNet + Transformer + Mean Pooling."""
    
    def __init__(self, backbone_name, target_size, feature_dim, d_model, n_heads, n_layers, d_ff, dropout):
        super().__init__()
        
        # Image backbone (frozen initially)
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=True,
            num_classes=0,
            global_pool="avg",
        )
        backbone_dim = self.backbone.num_features
        
        # Project to d_model
        self.feature_proj = nn.Linear(backbone_dim, d_model)
        
        # Freeze backbone initially
        for p in self.backbone.parameters():
            p.requires_grad = False
        
        # Day embeddings (positional encoding)
        self.day_embedding = nn.Embedding(MAX_SEQ_LEN, d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='relu',
            batch_first=False,  # (seq_len, batch, features)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
        )
    
    def unfreeze_backbone(self):
        """Unfreeze backbone for fine-tuning."""
        for name, p in self.backbone.named_parameters():
            if "blocks." in name or "layer" in name:
                p.requires_grad = True
    
    def forward(self, image_sequence, mask, day_indices):
        """
        Args:
            image_sequence: [B, T, 3, H, W]
            mask: [B, T] (1 = valid, 0 = missing)
            day_indices: [B, T] (day index for positional encoding)
        
        Returns:
            logits: [B]
        """
        B, T, C, H, W = image_sequence.shape
        
        # Extract features for each day: [B*T, 3, H, W] -> [B*T, backbone_dim]
        images_flat = image_sequence.view(B * T, C, H, W)
        features_flat = self.backbone(images_flat)  # [B*T, backbone_dim]
        features_flat = self.feature_proj(features_flat)  # [B*T, d_model]
        
        # Reshape: [T, B, d_model]
        features_seq = features_flat.view(B, T, -1).transpose(0, 1)
        
        # Add day embeddings: [T, B, d_model]
        day_emb = self.day_embedding(day_indices.transpose(0, 1))  # [T, B, d_model]
        features_seq = features_seq + day_emb
        
        # Create attention mask: [B*T, T] (True = mask out, False = attend)
        # We want to mask out missing days
        # Transformer expects mask where True positions are ignored
        attn_mask = (mask == 0).transpose(0, 1)  # [T, B] -> transpose to [T, B]
        # Convert to [T*B] for transformer (it will reshape internally)
        # Actually, transformer expects [T, T] mask or [B*T] for key_padding_mask
        # Use key_padding_mask: [B, T] where True = mask out
        key_padding_mask = (mask == 0)  # [B, T]
        
        # Transformer: [T, B, d_model] -> [T, B, d_model]
        transformer_out = self.transformer(features_seq, src_key_padding_mask=key_padding_mask)
        
        # Transpose back: [B, T, d_model]
        transformer_out = transformer_out.transpose(0, 1)
        
        # Mean pooling with mask: [B, d_model]
        # Sum over time, divide by number of valid days
        valid_counts = mask.sum(dim=1, keepdim=True)  # [B, 1]
        pooled = (transformer_out * mask.unsqueeze(-1)).sum(dim=1) / (valid_counts + 1e-8)
        
        # Classify
        logits = self.classifier(pooled).squeeze(1)  # [B]
        
        return logits


def make_loader(organoid_data, augment, batch_size):
    """Create DataLoader for temporal sequences."""
    ds = TemporalOrganoidDataset(organoid_data, augment=augment)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=augment, num_workers=NUM_WORKERS
    )


# ---------- Focal Loss ----------
class FocalLoss(nn.Module):
    """Focal loss for class imbalance."""
    def __init__(self, gamma=2.0, alpha=0.25, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        probs = torch.sigmoid(inputs)
        bce = nn.functional.binary_cross_entropy_with_logits(
            inputs, targets, reduction="none"
        )
        p_t = targets * probs + (1 - targets) * (1 - probs)
        modulating_factor = torch.pow(1 - p_t, self.gamma)
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        focal_loss = alpha_t * modulating_factor * bce

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


# ---------- Train/Eval ----------
def epoch_loop(model, loader, optimizer, class_weights, train=True, focal_loss_fn=None):
    """Training/evaluation loop for temporal model."""
    model.train() if train else model.eval()
    
    if focal_loss_fn is not None:
        loss_fn = focal_loss_fn
    else:
        loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    
    losses, preds, trues = [], [], []
    
    for batch in loader:
        image_seq, mask, day_indices, label, _ = batch
        image_seq = image_seq.to(DEVICE)
        mask = mask.to(DEVICE)
        day_indices = day_indices.to(DEVICE)
        label = label.to(DEVICE)
        
        logit = model(image_seq, mask, day_indices)
        
        if focal_loss_fn is not None:
            loss = loss_fn(logit, label)
            weight = torch.tensor(
                [class_weights[int(label_item.item())] for label_item in label],
                device=label.device,
            )
            loss = (loss * weight).mean()
        else:
            loss = loss_fn(logit, label)
            weight = torch.tensor(
                [class_weights[int(label_item.item())] for label_item in label],
                device=label.device,
            )
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
    
    # Calculate TNR, TPR, balanced accuracy
    tn = ((preds_bin == 0) & (trues == 0)).sum()
    fp = ((preds_bin == 1) & (trues == 0)).sum()
    fn = ((preds_bin == 0) & (trues == 1)).sum()
    tp = ((preds_bin == 1) & (trues == 1)).sum()
    
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    balanced_acc = (tpr + tnr) / 2.0
    
    return np.mean(losses), acc, balanced_acc, tnr, tpr, preds_bin, trues


def evaluate_on_loader(model, loader):
    """Evaluate model and return all metrics."""
    model.eval()
    preds_bin, trues, probs = [], [], []
    with torch.no_grad():
        for batch in loader:
            image_seq, mask, day_indices, label, _ = batch
            image_seq = image_seq.to(DEVICE)
            mask = mask.to(DEVICE)
            day_indices = day_indices.to(DEVICE)
            logit = model(image_seq, mask, day_indices)
            prob = torch.sigmoid(logit).cpu().numpy()
            probs.extend(prob)
            preds_bin.extend((prob > 0.5).astype(int))
            trues.extend(label.numpy())
    
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
    
    return (
        preds_bin,
        trues,
        float(acc),
        float(f1),
        probs,
        float(tpr),
        float(tnr),
        float(balanced_acc),
    )


def load_split_data(train_file: Path, val_file: Path, test_file: Path):
    """Load train/val/test splits."""
    with open(train_file) as f:
        train_data = json.load(f)
    with open(val_file) as f:
        val_data = json.load(f)
    with open(test_file) as f:
        test_data = json.load(f)
    return train_data, val_data, test_data


def prepare_organoid_data(split_data):
    """Convert split data to format expected by TemporalOrganoidDataset."""
    label_map = {"Acceptable": 1, "Not Acceptable": 0}
    organoid_data = {}
    
    for organoid_id, org_data in split_data.items():
        label_str = org_data.get("label")
        if label_str not in label_map:
            continue
        
        organoid_data[organoid_id] = {
            'label': label_map[label_str],
            'timepoints': org_data.get("timepoints", {})
        }
    
    return organoid_data


def main():
    """Main training function."""
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
        Path(args.train_split), Path(args.val_split), Path(args.test_split)
    )
    
    # Prepare organoid data
    train_orgs = prepare_organoid_data(train_data)
    val_orgs = prepare_organoid_data(val_data)
    test_orgs = prepare_organoid_data(test_data)
    
    print("=" * 80)
    print("TEMPORAL TRANSFORMER IMAGE CLASSIFIER")
    print("=" * 80)
    print(f"\nLoaded:")
    print(f"  Train: {len(train_orgs)} organoids")
    print(f"  Val: {len(val_orgs)} organoids")
    print(f"  Test: {len(test_orgs)} organoids")
    print(f"\nDevice: {DEVICE}")
    print(f"Output: {out_dir}\n")
    
    # Compute class weights
    train_labels = [org['label'] for org in train_orgs.values()]
    weights = compute_class_weight("balanced", classes=np.unique(train_labels), y=train_labels)
    class_weights = {int(k): float(w) for k, w in zip(np.unique(train_labels), weights)}
    
    if 0 in class_weights:
        class_weights[0] *= MINORITY_CLASS_BOOST
        print(f"[OK] BOOSTED minority class weight by {MINORITY_CLASS_BOOST}x")
    print(f"  Class weights: {class_weights}")
    
    # Loaders
    train_loader = make_loader(train_orgs, augment=True, batch_size=args.batch_size)
    val_loader = make_loader(val_orgs, augment=False, batch_size=args.batch_size)
    test_loader = make_loader(test_orgs, augment=False, batch_size=args.batch_size)
    
    # Model
    model = TemporalTransformerClassifier(
        BACKBONE_NAME, TARGET_SIZE, FEATURE_DIM,
        TRANSFORMER_D_MODEL, TRANSFORMER_N_HEADS, TRANSFORMER_N_LAYERS,
        TRANSFORMER_D_FF, TRANSFORMER_DROPOUT
    ).to(DEVICE)
    model_path = out_dir / "model.pth"
    
    # Loss
    focal_loss_fn = FocalLoss(gamma=FOCAL_LOSS_GAMMA, alpha=FOCAL_LOSS_ALPHA)
    print(f"[OK] Using Focal Loss (gamma={FOCAL_LOSS_GAMMA}, alpha={FOCAL_LOSS_ALPHA})")
    
    # Optimizer
    opt = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    scheduler = ReduceLROnPlateau(
        opt, mode="max", factor=0.5, patience=10, min_lr=1e-7, verbose=True
    )
    es = EarlyStopping(patience=20)
    history = defaultdict(list)
    best_balanced_acc = -np.inf
    
    print(f"\n{'=' * 80}")
    print("Phase 1: Training (Frozen Backbone)")
    print(f"{'=' * 80}")
    
    # Phase 1: Frozen backbone
    for epoch in range(100):
        tl, tacc, tbal, ttnr, ttpr, _, _ = epoch_loop(
            model, train_loader, opt, class_weights, train=True, focal_loss_fn=focal_loss_fn
        )
        vl, vacc, vbal, vtnr, vtpr, _, _ = epoch_loop(
            model, val_loader, opt, class_weights, train=False, focal_loss_fn=focal_loss_fn
        )
        
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
        
        print(
            f"[P1][{epoch:02d}] loss {tl:.4f}/{vl:.4f} | "
            f"acc {tacc:.3f}/{vacc:.3f} | bal {tbal:.3f}/{vbal:.3f} | "
            f"TNR {ttnr:.3f}/{vtnr:.3f} | TPR {ttpr:.3f}/{vtpr:.3f}"
        )
        
        if vbal > best_balanced_acc:
            best_balanced_acc = vbal
            torch.save(model.state_dict(), model_path)
            print(f"  → New best balanced acc: {vbal:.4f} (saved)")
        
        scheduler.step(vbal)
        if es.step(vbal):
            print(f"  → Early stopping at epoch {epoch}")
            break
    
    # Phase 2: Unfreeze backbone
    print(f"\n{'=' * 80}")
    print("Phase 2: Fine-tuning (Unfrozen Backbone)")
    print(f"{'=' * 80}")
    
    model.unfreeze_backbone()
    opt = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    scheduler = ReduceLROnPlateau(
        opt, mode="max", factor=0.5, patience=10, min_lr=1e-7, verbose=True
    )
    es = EarlyStopping(patience=30)
    
    for epoch in range(300):
        tl, tacc, tbal, ttnr, ttpr, _, _ = epoch_loop(
            model, train_loader, opt, class_weights, train=True, focal_loss_fn=focal_loss_fn
        )
        vl, vacc, vbal, vtnr, vtpr, _, _ = epoch_loop(
            model, val_loader, opt, class_weights, train=False, focal_loss_fn=focal_loss_fn
        )
        
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
        
        print(
            f"[P2][{epoch:03d}] loss {tl:.4f}/{vl:.4f} | "
            f"acc {tacc:.3f}/{vacc:.3f} | bal {tbal:.3f}/{vbal:.3f} | "
            f"TNR {ttnr:.3f}/{vtnr:.3f} | TPR {ttpr:.3f}/{vtpr:.3f}"
        )
        
        if vbal > best_balanced_acc:
            best_balanced_acc = vbal
            torch.save(model.state_dict(), model_path)
            print(f"  → New best balanced acc: {vbal:.4f} (saved)")
        
        scheduler.step(vbal)
        if es.step(vbal):
            print(f"  → Early stopping at epoch {epoch}")
            break
    
    # Save training curves
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
    
    axes[1, 2].plot(history["val_tpr"], history["val_tnr"], "o-", alpha=0.5)
    axes[1, 2].set_xlabel("TPR")
    axes[1, 2].set_ylabel("TNR")
    axes[1, 2].set_title("TPR vs TNR Evolution")
    axes[1, 2].plot([0, 1], [0, 1], "k--", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_dir / "training_curves.png")
    plt.close()
    print(f"[OK] Saved training curves to {out_dir / 'training_curves.png'}")
    
    # Evaluate on test set
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    
    (
        preds_bin,
        test_trues,
        test_acc,
        test_f1,
        test_probs,
        test_tpr,
        test_tnr,
        test_balanced_acc,
    ) = evaluate_on_loader(model, test_loader)
    
    try:
        test_roc_auc = float(roc_auc_score(test_trues, test_probs))
    except ValueError as e:
        print(f"  WARNING: Could not compute test ROC-AUC: {e}")
        test_roc_auc = None
    test_pr_auc = (
        float(average_precision_score(test_trues, test_probs))
        if len(test_trues) > 0
        else None
    )
    
    actual_good = int(test_trues.sum())
    predicted_good = int(preds_bin.sum())
    
    # Save metrics
    test_metrics = {
        "model_type": "temporal_transformer",
        "backbone": BACKBONE_NAME,
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
        "model_config": {
            "feature_dim": FEATURE_DIM,
            "d_model": TRANSFORMER_D_MODEL,
            "n_heads": TRANSFORMER_N_HEADS,
            "n_layers": TRANSFORMER_N_LAYERS,
            "d_ff": TRANSFORMER_D_FF,
            "dropout": TRANSFORMER_DROPOUT,
            "max_seq_len": MAX_SEQ_LEN,
        },
        "training_config": {
            "minority_class_boost": MINORITY_CLASS_BOOST,
            "focal_loss_alpha": FOCAL_LOSS_ALPHA,
            "focal_loss_gamma": FOCAL_LOSS_GAMMA,
            "selection_criterion": "balanced_accuracy",
        },
    }
    
    with (out_dir / "metrics_test.json").open("w") as f:
        json.dump(test_metrics, f, indent=2)
    
    print("\n[OK] Test Results:")
    print(f"  Accuracy: {test_acc:.4f}, F1: {test_f1:.4f}")
    print(f"  TNR: {test_tnr:.4f}, TPR: {test_tpr:.4f}")
    print(f"  Balanced Acc: {test_balanced_acc:.4f}")
    print(f"  ROC-AUC: {test_roc_auc:.4f}" if test_roc_auc else "  ROC-AUC: N/A")
    
    # Save predictions for analysis
    predictions = {
        "organoid_ids": [org_id for org_id in test_orgs.keys()],
        "true_labels": test_trues.tolist(),
        "predicted_labels": preds_bin.tolist(),
        "predicted_probs": test_probs.tolist(),
    }
    with (out_dir / "predictions.json").open("w") as f:
        json.dump(predictions, f, indent=2)
    
    print(f"\n[OK] Predictions saved to {out_dir / 'predictions.json'}")


if __name__ == "__main__":
    main()
