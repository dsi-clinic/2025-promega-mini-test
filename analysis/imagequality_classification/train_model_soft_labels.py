#!/usr/bin/env python3
"""
train_soft_labels.py — Image classifier trained on *soft labels* from crowd votes.

What this does:
  • Reads per-day JSONs (e.g., Dy10.json) from a directory. Prefer the `raw_votes/` split
    produced by your preprocessor, where each record has:
        img_path, good_votes, num_votes, good_fraction (may be None if no votes)
  • Builds targets as soft probabilities (y = good_fraction in [0,1]).
  • Trains a single-logit model with BCEWithLogitsLoss against soft targets.
  • Optionally weights each sample by its number of votes (aleatoric trust).
  • Early-stops by *validation Brier score* (lower is better).
  • Reports uncertainty-aware metrics: Brier, RMSE, Pearson r, plus thresholded accuracy/F1
    at 0.5 (for reference) and at a validation-chosen threshold t* (max F1 by default).
  • Saves curves and per-day summaries like your original script.

Example:
    python train_soft_labels.py \
        --data_dir <path/to/raw_votes_per_day_jsons> \
        --outdir   <path/to/outdir> \
        --batch-size 16 --val-batch-size 16 --min-votes 1 --weight-by-votes \
        --th-method max_f1

Note: this script consumes per-day vote JSONs that the current pipeline no
longer emits. If you need this analysis, generate the per-day JSONs first
(see find_misclassified_images.py for the historical format).
"""

import os, json, argparse, re, math, csv
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, average_precision_score,
    precision_recall_curve, roc_curve, balanced_accuracy_score
)
from scipy.stats import pearsonr

import timm
from torchvision import transforms as T

# -------- Config (defaults; can be overridden by CLI) --------
BACKBONES = {
     "vit": "vit_base_patch16_224",   # we will set img_size=(384,512) if enabled
    "resnet": "resnet50",
     "efficientnet": "efficientnet_b0"
}
DATA_DIR = None  # required: pass via --data-dir
OUT_ROOT = None  # required: pass via --outdir
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
    """
    Parse day tokens like "Dy28" or "Dy20_5" -> scaled int for sorting.
    Example: "Dy20_5" -> 2050 (represents 20.5).
    """
    m = re.search(r"[Dd][Yy](\d+(_\d+)?)", day_str)
    if not m:
        return -1
    token = m.group(1).replace("_", ".")
    try:
        return int(round(float(token) * 100))
    except Exception:
        return -1

class EarlyStoppingMin:
    """Early stop on a metric to be minimized (e.g., val Brier)."""
    def __init__(self, patience=20, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best = math.inf
        self.bad = 0
    def step(self, value):
        if value + self.min_delta < self.best:
            self.best = value
            self.bad = 0
            return False
        self.bad += 1
        return self.bad >= self.patience

def brier_score(probs: np.ndarray, targets: np.ndarray) -> float:
    """Mean squared error between predicted probability and soft target."""
    return float(np.mean((probs - targets) ** 2))

def rmse(probs: np.ndarray, targets: np.ndarray) -> float:
    return float(np.sqrt(np.mean((probs - targets) ** 2)))

def safe_pearsonr(a: np.ndarray, b: np.ndarray) -> float:
    try:
        r, _ = pearsonr(a, b)
        return float(r)
    except Exception:
        return float("nan")

# ---------- Thresholding helpers ----------
def pick_threshold(y_true_bin: np.ndarray, y_prob: np.ndarray, method="max_f1", min_precision=None) -> float:
    """Choose a decision threshold on validation data."""
    method = method or "max_f1"
    if method == "max_bal_acc":
        fpr, tpr, ts = roc_curve(y_true_bin, y_prob)
        bal_acc = (tpr + (1 - fpr)) / 2
        return float(ts[np.nanargmax(bal_acc)])
    elif method == "prec_at_recall" and (min_precision is not None):
        # maximize recall subject to precision >= min_precision
        prec, rec, ts = precision_recall_curve(y_true_bin, y_prob)
        # precision_recall_curve returns len(ts) == len(prec)-1
        idx = np.where(prec[:-1] >= min_precision)[0]
        if len(idx) == 0:
            return 0.5
        best = idx[np.nanargmax(rec[idx])]
        return float(ts[best])
    else:  # "max_f1" default via dense scan
        ts = np.linspace(0.0, 1.0, 1001)
        best_t, best_f1 = 0.5, -1.0
        for t in ts:
            pred = (y_prob >= t).astype(int)
            f1 = f1_score(y_true_bin, pred)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        return float(best_t)

def add_more_global_metrics(y_true_bin: np.ndarray, y_prob: np.ndarray) -> dict:
    return {
        "roc_auc": float(roc_auc_score(y_true_bin, y_prob)),
        "pr_auc": float(average_precision_score(y_true_bin, y_prob)),
    }

# ---------- Data ----------
def load_soft_records(day_json_path: Path, min_votes: int):
    """
    Returns (imgs, y_soft, weights, k_votes, n_votes).
    - If file has raw_votes fields, uses them (and filters by min_votes).
    - If not, falls back to label: "Accepted"/"Not Accepted" -> y = 1.0/0.0, n=1.
    """
    records = json.loads(day_json_path.read_text())
    if not isinstance(records, list):
        return [], [], [], [], []

    imgs, y_soft, weights, ks, ns = [], [], [], [], []
    for r in records:
        img = r.get("img_path", "")
        if not img:
            continue

        # Prefer raw vote fields if present
        k = r.get("good_votes", None)
        n = r.get("num_votes", None)
        gf = r.get("good_fraction", None)

        if (k is not None) and (n is not None):
            # filter by min_votes
            if n is None or n < min_votes or n <= 0:
                continue
            if gf is None:
                gf = float(k) / float(n) if n > 0 else None
            if gf is None:
                continue
            imgs.append(img)
            y_soft.append(float(gf))
            weights.append(float(n))  # weight by number of votes (can be disabled later)
            ks.append(int(k))
            ns.append(int(n))
        else:
            # Fallback to majority/hard labels folder
            lbl = r.get("label", None)
            if lbl not in ("Accepted", "Not Accepted"):
                continue
            y = 1.0 if lbl == "Accepted" else 0.0
            imgs.append(img)
            y_soft.append(y)
            weights.append(1.0)
            ks.append(int(round(y)))
            ns.append(1)

    return np.array(imgs), np.array(y_soft, dtype=np.float32), np.array(weights, dtype=np.float32), np.array(ks, dtype=np.int64), np.array(ns, dtype=np.int64)

class OrganoidSoftDataset(Dataset):
    """Image-only dataset with soft targets y∈[0,1] and optional per-sample weight."""
    def __init__(self, img_paths, y_soft, weights=None, augment=False):
        self.img_paths = img_paths
        self.y_soft = y_soft
        self.weights = weights if weights is not None else np.ones_like(y_soft, dtype=np.float32)
        self.augment = augment
        t = [T.Resize(TARGET_SIZE)]
        if augment:
            t += [
                T.RandomHorizontalFlip(0.5),
                T.ColorJitter(0.2, 0.2, 0.2, 0.1),
            ]
        t += [T.ToTensor()]
        self.t_img = T.Compose(t)

    def __len__(self):
        return len(self.y_soft)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert("RGB")
        img = self.t_img(img)
        y = torch.tensor(self.y_soft[idx], dtype=torch.float32)   # soft target in [0,1]
        w = torch.tensor(self.weights[idx], dtype=torch.float32)  # sample weight
        return img, y, w

# ---------- Model ----------
class ImageOnlyClassifier(nn.Module):
    def __init__(self, backbone_name, target_size):
        super().__init__()
        extra_args = {}
        if "vit" in backbone_name:
            extra_args["img_size"] = target_size  # timm handles interpolation

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

        self.classifier = nn.Sequential(
            nn.Linear(out_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),  # single logit -> sigmoid -> p(good)
        )

    def unfreeze_backbone(self):
        for name, p in self.backbone.named_parameters():
            if "blocks." in name or "layer" in name:
                p.requires_grad = True

    def forward(self, img):
        f = self.backbone(img)
        return self.classifier(f).squeeze(1)  # (B,)

def make_loader(imgs, y_soft, weights, augment, batch_size):
    ds = OrganoidSoftDataset(imgs, y_soft, weights, augment)
    return DataLoader(ds, batch_size=batch_size, shuffle=augment, num_workers=NUM_WORKERS)

# ---------- Train/Eval ----------
def epoch_loop(model, loader, optimizer, weight_by_votes=True, train=True):
    model.train() if train else model.eval()
    bce = nn.BCEWithLogitsLoss(reduction="none")
    losses, probs_all, trues_all = [], [], []

    for batch in loader:
        img, y, w = batch
        img, y, w = img.to(DEVICE), y.to(DEVICE), w.to(DEVICE)
        logit = model(img)                 # (B,)
        loss_elems = bce(logit, y)         # (B,)
        if weight_by_votes:
            loss = (loss_elems * w).mean()
        else:
            loss = loss_elems.mean()

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        losses.append(loss.item())
        probs = torch.sigmoid(logit).detach().cpu().numpy()
        probs_all.extend(probs)
        trues_all.extend(y.detach().cpu().numpy())

    probs_all = np.array(probs_all, dtype=np.float32)
    trues_all = np.array(trues_all, dtype=np.float32)
    brier = brier_score(probs_all, trues_all)
    rmse_val = rmse(probs_all, trues_all)
    # For reference only: convert soft target to hard via 0.5 to compute accuracy/F1
    hard_true = (trues_all >= 0.5).astype(int)
    hard_pred = (probs_all >= 0.5).astype(int)
    acc = accuracy_score(hard_true, hard_pred)
    f1 = f1_score(hard_true, hard_pred)
    return np.mean(losses), brier, rmse_val, acc, f1, probs_all, trues_all

def evaluate_on_loader(model, loader):
    model.eval()
    with torch.no_grad():
        _, brier, rmse_val, acc, f1, probs, trues = epoch_loop(
            model, loader, optimizer=None, weight_by_votes=False, train=False
        )
    corr = safe_pearsonr(probs, trues)
    return {
        "brier": float(brier),
        "rmse": float(rmse_val),
        "acc@0.5": float(acc),
        "f1@0.5": float(f1),
        "corr": float(corr),
        "probs": probs,
        "trues": trues
    }

def stratify_bins(y_soft: np.ndarray, n_bins: int = 6):
    """
    Use vote-count bins for stratification: round(y*5) ∈ {0..5} by default.
    If y is continuous without counts, we still use 6 bins via rounding.
    """
    bins = np.clip(np.round(y_soft * 5).astype(int), 0, 5)
    return bins

def run_training_for_day(
    day_json_path: Path,
    backbone_key: str,
    backbone_name: str,
    train_bs: int,
    val_bs: int,
    test_frac: float,
    val_frac: float,
    min_votes: int,
    weight_by_votes: bool,
    lr1: float,
    lr2: float,
    patience1: int,
    patience2: int,
    th_method: str,
    min_precision: float | None
):
    """Train + validate; select by VAL Brier (lower is better), report on TEST."""
    imgs, y_soft, wts, k_votes, n_votes = load_soft_records(day_json_path, min_votes=min_votes)
    if len(imgs) == 0:
        print(f"⚠ Skipping {day_json_path.name} — no usable records (after min_votes={min_votes})")
        return None

    # ---- Split: TEST then VAL with stratification on bins (approx counts 0..5)
    y_bins = stratify_bins(y_soft, n_bins=6)
    X_tmp, X_test, y_tmp, y_test, w_tmp, w_test = train_test_split(
        imgs, y_soft, wts, test_size=test_frac, stratify=y_bins, random_state=SEED
    )
    y_bins_tmp = stratify_bins(y_tmp, n_bins=6)
    val_frac_cond = val_frac / (1.0 - test_frac)
    X_tr, X_val, y_tr, y_val, w_tr, w_val = train_test_split(
        X_tmp, y_tmp, w_tmp, test_size=val_frac_cond, stratify=y_bins_tmp, random_state=SEED
    )

    # loaders (configurable batch sizes; val/test use val_bs)
    train_loader = make_loader(X_tr, y_tr, w_tr, augment=True,  batch_size=train_bs)
    val_loader   = make_loader(X_val, y_val, w_val, augment=False, batch_size=val_bs)
    test_loader  = make_loader(X_test, y_test, w_test, augment=False, batch_size=val_bs)

    # model/opt
    model = ImageOnlyClassifier(backbone_name, TARGET_SIZE).to(DEVICE)
    model_dir = OUT_ROOT / ("soft_" + backbone_key) / day_json_path.stem
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "model.pth"

    opt = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr1)
    es = EarlyStoppingMin(patience=patience1)
    history = defaultdict(list)
    best_val_brier = math.inf

    # Phase 1 — frozen backbone
    for epoch in range(100):
        tl, tbrier, tRMSE, tacc, tf1, _, _ = epoch_loop(
            model, train_loader, opt, weight_by_votes=weight_by_votes, train=True
        )
        vl_dict = evaluate_on_loader(model, val_loader)
        vl_brier = vl_dict["brier"]
        history["train_loss"].append(tl); history["val_brier"].append(vl_brier)
        history["train_brier"].append(tbrier); history["val_rmse"].append(vl_dict["rmse"])
        print(f"[{day_json_path.stem}][{backbone_key}][P1][{epoch:02d}] "
              f"loss {tl:.4f} | brier {tbrier:.4f}/{vl_brier:.4f} | "
              f"rmse {tRMSE:.3f}/{vl_dict['rmse']:.3f} | acc@0.5 {tacc:.3f}/{vl_dict['acc@0.5']:.3f}")

        if vl_brier < best_val_brier:
            best_val_brier = vl_brier
            torch.save(model.state_dict(), model_path)
        if es.step(vl_brier):
            break

    # Phase 2 — unfreeze partial backbone
    model.unfreeze_backbone()
    opt = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr2)
    es = EarlyStoppingMin(patience=patience2)
    for epoch in range(300):
        tl, tbrier, tRMSE, tacc, tf1, _, _ = epoch_loop(
            model, train_loader, opt, weight_by_votes=weight_by_votes, train=True
        )
        vl_dict = evaluate_on_loader(model, val_loader)
        vl_brier = vl_dict["brier"]
        history["train_loss"].append(tl); history["val_brier"].append(vl_brier)
        history["train_brier"].append(tbrier); history["val_rmse"].append(vl_dict["rmse"])
        print(f"[{day_json_path.stem}][{backbone_key}][P2][{epoch:03d}] "
              f"loss {tl:.4f} | brier {tbrier:.4f}/{vl_brier:.4f} | "
              f"rmse {tRMSE:.3f}/{vl_dict['rmse']:.3f} | acc@0.5 {tacc:.3f}/{vl_dict['acc@0.5']:.3f}")

        if vl_brier < best_val_brier:
            best_val_brier = vl_brier
            torch.save(model.state_dict(), model_path)
        if es.step(vl_brier):
            break

    # Save training curves (Brier + loss)
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1); plt.plot(history["train_brier"], label="Train Brier"); plt.plot(history["val_brier"], label="Val Brier"); plt.title("Brier Score"); plt.legend()
    plt.subplot(1, 2, 2); plt.plot(history["train_loss"], label="Train Loss"); plt.plot(history["val_rmse"], label="Val RMSE"); plt.title("Loss / RMSE"); plt.legend()
    plt.tight_layout()
    plt.savefig(model_dir / "training_curves.png")
    plt.close()
    print(f"📈 Saved curves → {model_dir/'training_curves.png'}")

    # ---- Evaluate with best VAL checkpoint
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))

    # Validation metrics & pick threshold on VAL
    val_dict = evaluate_on_loader(model, val_loader)
    val_true_bin = (val_dict["trues"] >= 0.5).astype(int)
    t_star = pick_threshold(val_true_bin, val_dict["probs"], method=th_method, min_precision=min_precision)
    val_more = add_more_global_metrics(val_true_bin, val_dict["probs"])

    val_metrics = {
        "day": day_json_path.stem,
        "split": "val",
        "brier": float(val_dict["brier"]),
        "rmse": float(val_dict["rmse"]),
        "acc@0.5": float(val_dict["acc@0.5"]),
        "f1@0.5": float(val_dict["f1@0.5"]),
        "corr": float(val_dict["corr"]),
        "roc_auc": val_more["roc_auc"],
        "pr_auc": val_more["pr_auc"],
        "chosen_threshold": float(t_star),
        "threshold_method": th_method,
        "min_precision": (None if min_precision is None else float(min_precision)),
        "n": int(len(val_dict["trues"])),
        "batch_size": int(val_bs),
    }
    with (model_dir / "metrics_val.json").open("w") as f:
        json.dump(val_metrics, f, indent=2)

    # Test metrics（final reporting）
    test_dict = evaluate_on_loader(model, test_loader)
    test_true_bin = (test_dict["trues"] >= 0.5).astype(int)
    test_pred_star = (test_dict["probs"] >= t_star).astype(int)

    test_acc_star = float(accuracy_score(test_true_bin, test_pred_star))
    test_f1_star  = float(f1_score(test_true_bin, test_pred_star))
    test_balacc   = float(balanced_accuracy_score(test_true_bin, test_pred_star))
    test_more     = add_more_global_metrics(test_true_bin, test_dict["probs"])

    day_no = day_to_int(day_json_path.stem)
    num_in_sample = int(len(test_dict["trues"]))
    # For reporting similar to before (hardening at 0.5):
    hard_true = (test_dict["trues"] >= 0.5).astype(int)
    hard_pred = (test_dict["probs"] >= 0.5).astype(int)
    actual_good_05 = int(hard_true.sum())
    predicted_good_05 = int(hard_pred.sum())

    test_metrics = {
        "day": day_json_path.stem,
        "day_no": day_no,
        "split": "test",
        "brier": float(test_dict["brier"]),
        "rmse": float(test_dict["rmse"]),
        "corr": float(test_dict["corr"]),
        "acc@0.5": float(test_dict["acc@0.5"]),
        "f1@0.5": float(test_dict["f1@0.5"]),
        "actual_good@0.5": actual_good_05,
        "predicted_good@0.5": predicted_good_05,
        "acc@t*": test_acc_star,
        "f1@t*": test_f1_star,
        "bal_acc@t*": test_balacc,
        "roc_auc": test_more["roc_auc"],
        "pr_auc": test_more["pr_auc"],
        "threshold_used": float(t_star),
        "threshold_method": th_method,
        "min_precision": (None if min_precision is None else float(min_precision)),
        "val_brier_for_selection": float(best_val_brier),
        "val_n": int(val_metrics["n"]),
        "test_n": num_in_sample,
        "batch_size_train": int(train_bs),
        "batch_size_valtest": int(val_bs),
        "backbone_key": backbone_key,
    }
    with (model_dir / "metrics_test.json").open("w") as f:
        json.dump(test_metrics, f, indent=2)
    print(f"📝 Saved metrics → {model_dir/'metrics_val.json'} and {model_dir/'metrics_test.json'}")
    print(f"⭐ Chosen threshold t*={t_star:.3f} via {th_method}"
          + (f" (min_precision={min_precision:.2f})" if (th_method=='prec_at_recall' and min_precision is not None) else ""))

    # Return: choose by val (Brier), report test
    return {
        "day": day_json_path.stem,
        "day_no": day_no,
        "backbone_key": backbone_key,
        "val_brier": float(best_val_brier),     # selection metric (lower is better)
        "test_brier": float(test_dict["brier"]),
        "test_rmse": float(test_dict["rmse"]),
        "test_corr": float(test_dict["corr"]),
        "test_acc@0.5": float(test_dict["acc@0.5"]),
        "test_f1@0.5": float(test_dict["f1@0.5"]),
        "test_acc@t*": test_acc_star,
        "test_f1@t*": test_f1_star,
        "test_bal_acc@t*": test_balacc,
        "val_roc_auc": float(val_more["roc_auc"]),
        "val_pr_auc": float(val_more["pr_auc"]),
        "test_roc_auc": float(test_more["roc_auc"]),
        "test_pr_auc": float(test_more["pr_auc"]),
        "val_num": int(val_metrics["n"]),
        "test_num": num_in_sample,
        "test_actual_good@0.5": actual_good_05,
        "test_pred_good@0.5": predicted_good_05,
        "chosen_threshold": float(t_star),
        "threshold_method": th_method,
        "min_precision": (None if min_precision is None else float(min_precision)),
    }

# ---------- Orchestration ----------
def main():
    set_seed()

    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default=OUT_ROOT, help="Where to save outputs")
    parser.add_argument("--data_dir", default=DATA_DIR, help="Directory with per-day JSONs")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Train batch size")
    parser.add_argument("--val-batch-size", type=int, default=None, help="Val/Test batch size (defaults to train batch size)")
    parser.add_argument("--test-frac", type=float, default=0.10, help="Fraction for test split (e.g., 0.10)")
    parser.add_argument("--val-frac",  type=float, default=0.10, help="Overall fraction for validation split (e.g., 0.10)")
    parser.add_argument("--min-votes", type=int, default=1, help="Minimum #votes required to include a sample (only used if raw_votes fields exist)")
    parser.add_argument("--weight-by-votes", action="store_true", help="Weight BCE loss by the sample's num_votes")
    parser.add_argument("--lr1", type=float, default=1e-3, help="LR for frozen phase")
    parser.add_argument("--lr2", type=float, default=1e-4, help="LR for unfrozen phase")
    parser.add_argument("--patience1", type=int, default=20, help="Early stopping patience (phase 1)")
    parser.add_argument("--patience2", type=int, default=30, help="Early stopping patience (phase 2)")
    parser.add_argument("--th-method", choices=["max_f1","max_bal_acc","prec_at_recall"], default="max_f1",
                        help="How to pick decision threshold on validation set")
    parser.add_argument("--min-precision", type=float, default=None,
                        help="Only used if --th-method=prec_at_recall: precision floor (e.g., 0.9)")
    args = parser.parse_args()

    out_dir = Path(args.outdir); out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)

    train_bs = int(args.batch_size)
    val_bs = int(args.val_batch_size) if args.val_batch_size is not None else train_bs
    test_frac = float(args.test_frac)
    val_frac = float(args.val_frac)

    assert 0.0 < test_frac < 0.5, "test-frac must be in (0, 0.5)"
    assert 0.0 < val_frac  < 0.5, "val-frac must be in (0, 0.5)"
    assert val_frac + test_frac < 0.9, "Sum of val-frac and test-frac too large."
    print(f"🧪 Using batch sizes — train: {train_bs}, val/test: {val_bs}")
    print(f"🔀 Split fractions — train: {1.0 - test_frac - val_frac:.2f}, val: {val_frac:.2f}, test: {test_frac:.2f}")
    print(f"🖼️ Target size (HxW): {TARGET_SIZE}")
    print(f"📂 Data dir: {data_dir}")

    # Collect results: pick the best backbone per day by **validation Brier**
    per_day_best = {}
    per_model_results = {bk: {} for bk in BACKBONES}
    files = sorted(data_dir.glob("Dy*.json"), key=lambda p: day_to_int(p.stem))
    if not files:
        print(f"❌ No Dy*.json files in {data_dir}")
        return

    for json_file in files:
        day = json_file.stem
        best = None
        for backbone_key, backbone_name in BACKBONES.items():
            res = run_training_for_day(
                json_file, backbone_key, backbone_name,
                train_bs, val_bs, test_frac, val_frac,
                min_votes=args.min_votes,
                weight_by_votes=bool(args.weight_by_votes),
                lr1=args.lr1, lr2=args.lr2,
                patience1=args.patience1, patience2=args.patience2,
                th_method=args.th_method,
                min_precision=args.min_precision
            )
            if res is None:
                continue
            per_model_results[backbone_key][day] = res
            if (best is None) or (res["val_brier"] < best["val_brier"]):
                best = res
        if best:
            per_day_best[day] = best
            print(
                f"✅ Best for {day} (by VAL Brier): {best['backbone_key']} | "
                f"VAL brier={best['val_brier']:.4f} | "
                f"TEST brier={best['test_brier']:.4f}, rmse={best['test_rmse']:.3f}, corr={best['test_corr']:.3f}, "
                f"acc@0.5={best['test_acc@0.5']:.3f}, f1@0.5={best['test_f1@0.5']:.3f}, "
                f"acc@t*={best['test_acc@t*']:.3f}, f1@t*={best['test_f1@t*']:.3f}, bal_acc@t*={best['test_bal_acc@t*']:.3f} "
                f"(t*={best['chosen_threshold']:.3f}, method={best['threshold_method']})"
            )
        else:
            print(f"⚠ No valid result for {day}")

    if not per_day_best:
        print("❌ No days produced results; aborting summary.")
        return

    # ---- Build 4-column table (keep similar to your original, based on @0.5 hardening)
    rows = []
    days_sorted = sorted(per_day_best.keys(), key=lambda d: day_to_int(d))
    for d in days_sorted:
        r = per_day_best[d]
        rows.append({
            "Day No": r["day_no"]/100.0,  # undo *100 scaling for human-friendly display
            "Num in Sample": r["test_num"],
            "Actual Good@0.5": r["test_actual_good@0.5"],
            "Predicted Good@0.5": r["test_pred_good@0.5"],
        })

    # Save CSV table (exactly 4 columns)
    table_path = out_dir / "day_summary.csv"
    with table_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["Day No", "Num in Sample", "Actual Good@0.5", "Predicted Good@0.5"]
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"🧾 Saved table → {table_path}")

    # ---- Per-model charts (Brier / F1 / ROC AUC)
    day_numbers = {}
    for day_res in per_model_results.values():
        for day, res in day_res.items():
            day_numbers[day] = res["day_no"] / 100.0

    if day_numbers:
        unique_day_vals = sorted(set(day_numbers.values()))

        def plot_metric(metric_key, ylabel, title, filename, bounded=True):
            plt.figure(figsize=(9, 4))
            plotted = False
            all_ys = []
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
                plotted = True
                all_ys.extend(ys)
            if plotted:
                plt.xlabel("Day")
                plt.ylabel(ylabel)
                plt.title(title)
                plt.xticks(unique_day_vals)
                if bounded:
                    plt.ylim(0.0, 1.0)
                elif all_ys:
                    y_min = min(all_ys)
                    y_max = max(all_ys)
                    if y_max == y_min:
                        margin = max(0.05, y_max * 0.1 if y_max != 0 else 0.05)
                        plt.ylim(y_min - margin, y_max + margin)
                    else:
                        margin = 0.05 * (y_max - y_min)
                        plt.ylim(max(0.0, y_min - margin), y_max + margin)
                plt.legend()
                plt.tight_layout()
                out_path = out_dir / filename
                plt.savefig(out_path)
                print(f"📊 Saved {title.lower()} → {out_path}")
            plt.close()

        plot_metric("test_brier", "Brier score (test) ↓", "Per-day Test Brier by Backbone", "brier_by_model.png", bounded=False)
        plot_metric("test_f1@t*", "F1 score @t* (test)", "Per-day Test F1 by Backbone", "f1_by_model.png", bounded=True)
        plot_metric("test_roc_auc", "ROC AUC (test)", "Per-day Test ROC AUC by Backbone", "rocauc_by_model.png", bounded=True)

    # ---- Final TEST summary JSON (per model)
    per_model_summary = {}
    for backbone_key, day_res in per_model_results.items():
        per_model_summary[backbone_key] = {
            "per_day": {
                day: {
                    "day_no": float(day_numbers.get(day, res["day_no"]/100.0)),
                    "test_brier": float(res["test_brier"]),
                    "test_rmse": float(res["test_rmse"]),
                    "test_corr": float(res["test_corr"]),
                    "test_acc@0.5": float(res["test_acc@0.5"]),
                    "test_f1@0.5": float(res["test_f1@0.5"]),
                    "test_acc@t*": float(res["test_acc@t*"]),
                    "test_f1@t*": float(res["test_f1@t*"]),
                    "test_bal_acc@t*": float(res["test_bal_acc@t*"]),
                    "test_roc_auc": (None if res.get("test_roc_auc") is None else float(res["test_roc_auc"])),
                    "test_pr_auc": (None if res.get("test_pr_auc") is None else float(res["test_pr_auc"])),
                    "val_roc_auc": (None if res.get("val_roc_auc") is None else float(res["val_roc_auc"])),
                    "val_pr_auc": (None if res.get("val_pr_auc") is None else float(res["val_pr_auc"])),
                    "val_brier": float(res["val_brier"]),
                    "test_num": int(res["test_num"]),
                    "chosen_threshold": float(res["chosen_threshold"]),
                }
                for day, res in day_res.items()
            }
        }

    summary = {
        "per_model": per_model_summary,
        "batch_size_train": int(train_bs),
        "batch_size_valtest": int(val_bs),
        "split_fractions": {
            "train": float(1.0 - test_frac - val_frac),
            "val": float(val_frac),
            "test": float(test_frac),
        },
        "settings": {
            "min_votes": int(args.min_votes),
            "weight_by_votes": bool(args.weight_by_votes),
            "lr1": float(args.lr1),
            "lr2": float(args.lr2),
            "patience1": int(args.patience1),
            "patience2": int(args.patience2),
            "threshold_method": args.th_method,
            "min_precision": (None if args.min_precision is None else float(args.min_precision)),
        }
    }
    summary_path = out_dir / "final_test_summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"✅ Saved final test summary → {summary_path}")

    # ---- Also print the 4-column table to stdout
    print("\n=== Summary Table (TEST, @0.5 hardening) ===")
    print(f"{'Day':>6} | {'Num in Sample':>13} | {'Actual Good':>12} | {'Pred Good':>10}")
    print("-" * 56)
    for row in rows:
        print(f"{row['Day No']:>6} | {row['Num in Sample']:>13} | {row['Actual Good@0.5']:>12} | {row['Predicted Good@0.5']:>10}")

if __name__ == "__main__":
    main()
