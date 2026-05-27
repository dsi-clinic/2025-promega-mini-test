#!/usr/bin/env python3

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import timm
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

_DATA_DIR = Path(os.environ.get("DATA_DIR", "data"))
_IMG_DIR = _DATA_DIR / "lstm" / "lstm_ready" / "images"


def _remap_img_path(img_path: str) -> str:
    """Remap stale img_path to lstm_ready using filename only."""
    fname = Path(img_path).name
    candidate = _IMG_DIR / fname
    return str(candidate) if candidate.exists() else img_path


# -------- Config (defaults; can be overridden by CLI) --------
BACKBONES = {
    "efficientnet": "efficientnet_b0",
}

DATA_DIR = Path("analysis/images/classifier/data/preprocessed/512x384/majority/")
OUT_ROOT = Path("analysis/images/classifier/per_day_study/efficientnet_ensemble")
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
    """Image-only dataset (mask ignored)."""

    def __init__(self, img_paths, labels, augment=False):
        self.img_paths = img_paths
        self.labels = labels
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
        # If it's a ViT-like model, tell timm the image size.
        # timm will handle positional embedding interpolation for non-224 sizes.
        extra_args = {}
        if "vit" in backbone_name:
            extra_args["img_size"] = target_size  # (H, W) tuple is supported

        self.backbone = timm.create_model(
            backbone_name,
            pretrained=True,
            num_classes=0,  # feature extractor
            global_pool="avg",
            **extra_args,
        )
        out_dim = self.backbone.num_features

        # freeze backbone initially
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
            # unfreeze blocks/layers for fine-tuning
            if "blocks." in name or "layer" in name:
                p.requires_grad = True

    def forward(self, img):
        f = self.backbone(img)
        return self.classifier(f).squeeze(1)


def make_loader(imgs, labels, augment, batch_size):
    ds = OrganoidDataset(imgs, labels, augment)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=augment, num_workers=NUM_WORKERS
    )


# ---------- Train/Eval ----------
def epoch_loop(model, loader, optimizer, class_weights, train=True):
    model.train() if train else model.eval()
    bce = nn.BCEWithLogitsLoss(reduction="none")
    losses, preds, trues = [], [], []

    for img, label in loader:
        img, label = img.to(DEVICE), label.to(DEVICE)
        logit = model(img)
        loss = bce(logit, label)
        weight = torch.tensor(
            [class_weights[int(lbl.item())] for lbl in label], device=label.device
        )
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


def evaluate_on_loader(model, loader):
    """Run inference (no grad) and compute accuracy & F1. Return preds_bin, trues, acc, f1, probs."""
    model.eval()
    preds_bin, trues, probs = [], [], []
    with torch.no_grad():
        for img, lbl in loader:
            img = img.to(DEVICE)
            prob = torch.sigmoid(model(img)).cpu().numpy()
            probs.extend(prob)
            preds_bin.extend((prob > 0.5).astype(int))
            trues.extend(lbl.numpy())
    preds_bin = np.array(preds_bin)
    trues = np.array(trues)
    probs = np.array(probs)
    acc = accuracy_score(trues, preds_bin)
    bal_acc = balanced_accuracy_score(trues, preds_bin)
    f1 = f1_score(trues, preds_bin)
    return preds_bin, trues, float(acc), float(f1), probs, float(bal_acc)


def run_training_for_day(
    day_json_path: Path,
    backbone_key: str,
    backbone_name: str,
    train_bs: int,
    val_bs: int,
    test_frac: float,
    val_frac: float,
):
    """Train + validate with small val/test; select by VAL acc, report on TEST."""
    records = json.loads(day_json_path.read_text())
    if not records:
        print(f"⚠ Skipping {day_json_path.name} — no records")
        return None

    # labels
    label_map = {"Accepted": 1, "Not Accepted": 0}
    try:
        records = [r for r in records if r.get("label") in label_map]
    except Exception:
        print(f"⚠ Skipping {day_json_path.name} — missing 'label' field")
        return None

    # Remap paths and filter out missing files
    remapped = [_remap_img_path(r["img_path"]) for r in records]
    valid = [(img, r) for img, r in zip(remapped, records) if Path(img).exists()]
    if not valid:
        print(f"⚠ Skipping {day_json_path.name} — no valid images found")
        return None
    n_dropped = len(records) - len(valid)
    if n_dropped:
        print(f"  [{day_json_path.stem}] dropped {n_dropped} records with missing images")
    imgs = np.array([img for img, _ in valid])
    labels = np.array([label_map[r["label"]] for _, r in valid], dtype=int)

    # ---- Split: first cut TEST (test_frac), then VAL to reach overall val_frac
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        imgs, labels, test_size=test_frac, stratify=labels, random_state=SEED
    )
    val_frac_cond = val_frac / (1.0 - test_frac)  # conditional fraction from remaining
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_tmp, y_tmp, test_size=val_frac_cond, stratify=y_tmp, random_state=SEED
    )

    # class weights (train only)
    weights = compute_class_weight("balanced", classes=np.unique(y_tr), y=y_tr)
    class_weights = {int(k): float(w) for k, w in zip(np.unique(y_tr), weights)}

    # loaders (configurable batch sizes; val/test use val_bs)
    train_loader = make_loader(X_tr, y_tr, augment=False, batch_size=train_bs)
    val_loader = make_loader(X_val, y_val, augment=False, batch_size=val_bs)
    test_loader = make_loader(X_test, y_test, augment=False, batch_size=val_bs)

    # model/opt
    model = ImageOnlyClassifier(backbone_name, TARGET_SIZE).to(DEVICE)
    model_dir = OUT_ROOT / backbone_key / day_json_path.stem
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "model.pth"

    opt = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    es = EarlyStopping(patience=20)
    history = defaultdict(list)
    best_acc = -np.inf

    # Phase 1 — frozen
    for epoch in range(100):
        tl, tacc, _, _ = epoch_loop(model, train_loader, opt, class_weights, train=True)
        vl, vacc, _, _ = epoch_loop(model, val_loader, opt, class_weights, train=False)
        history["train_loss"].append(tl)
        history["val_loss"].append(vl)
        history["train_acc"].append(tacc)
        history["val_acc"].append(vacc)
        print(
            f"[{day_json_path.stem}][{backbone_key}][P1][{epoch:02d}][bs={train_bs}/{val_bs}] loss {tl:.4f}/{vl:.4f} acc {tacc:.3f}/{vacc:.3f}"
        )
        if vacc > best_acc:
            best_acc = vacc
            torch.save(model.state_dict(), model_path)
        if es.step(vacc):
            break

    # Phase 2 — unfreeze partial backbone
    model.unfreeze_backbone()
    opt = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    es = EarlyStopping(patience=30)
    for epoch in range(300):
        tl, tacc, _, _ = epoch_loop(model, train_loader, opt, class_weights, train=True)
        vl, vacc, _, _ = epoch_loop(model, val_loader, opt, class_weights, train=False)
        history["train_loss"].append(tl)
        history["val_loss"].append(vl)
        history["train_acc"].append(tacc)
        history["val_acc"].append(vacc)
        print(
            f"[{day_json_path.stem}][{backbone_key}][P2][{epoch:03d}][bs={train_bs}/{val_bs}] loss {tl:.4f}/{vl:.4f} acc {tacc:.3f}/{vacc:.3f}"
        )
        if vacc > best_acc:
            best_acc = vacc
            torch.save(model.state_dict(), model_path)
        if es.step(vacc):
            break

    # Save per-day training curves
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history["train_acc"], label="Train")
    plt.plot(history["val_acc"], label="Val")
    plt.title("Accuracy")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history["train_loss"], label="Train")
    plt.plot(history["val_loss"], label="Val")
    plt.title("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(model_dir / "training_curves.png")
    plt.close()
    print(f"📈 Saved curves → {model_dir / 'training_curves.png'}")

    # ---- Evaluate with best VAL checkpoint
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))

    # Val metrics (record only; NOT used for final reporting)
    _, _, val_acc, val_f1, _, val_bal_acc = evaluate_on_loader(model, val_loader)
    val_metrics = {
        "day": day_json_path.stem,
        "split": "val",
        "accuracy": float(val_acc),
        "f1": float(val_f1),
        "n": int(len(y_val)),
        "batch_size": int(val_bs),
    }
    with (model_dir / "metrics_val.json").open("w") as f:
        json.dump(val_metrics, f, indent=2)

    # Test metrics（final reporting）
    preds_bin, trues, test_acc, test_f1, _, test_bal_acc = evaluate_on_loader(model, test_loader)
    day_no = day_to_int(day_json_path.stem)
    num_in_sample = int(len(trues))
    actual_good = int(trues.sum())
    predicted_good = int(preds_bin.sum())

    test_metrics = {
        "day": day_json_path.stem,
        "day_no": day_no,
        "split": "test",
        "accuracy": float(test_acc),
        "balanced_accuracy": float(test_bal_acc),
        "f1": float(test_f1),
        "val_accuracy_for_selection": float(best_acc),
        "val_n": int(len(y_val)),
        "test_n": num_in_sample,
        "actual_good": actual_good,
        "predicted_good": predicted_good,
        "batch_size_train": int(train_bs),
        "batch_size_valtest": int(val_bs),
        "backbone_key": backbone_key,
    }
    with (model_dir / "metrics_test.json").open("w") as f:
        json.dump(test_metrics, f, indent=2)
    print(
        f"📝 Saved metrics → {model_dir / 'metrics_val.json'} and {model_dir / 'metrics_test.json'}"
    )

    # Return: choose by val, report test
    return {
        "day": day_json_path.stem,
        "day_no": day_no,
        "backbone_key": backbone_key,
        "val_accuracy": float(best_acc),  # selection metric
        "test_accuracy": float(test_acc),  # reporting metric
        "test_f1": float(test_f1),
        "val_num": int(len(y_val)),
        "test_num": num_in_sample,
        "test_actual_good": actual_good,
        "test_pred_good": predicted_good,
    }


# ---------- Orchestration ----------
def main():
    set_seed()

    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default=OUT_ROOT, help="Where to save outputs")
    parser.add_argument(
        "--data_dir", default=DATA_DIR, help="Directory with per-day JSONs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=BATCH_SIZE, help="Train batch size"
    )
    parser.add_argument(
        "--val-batch-size",
        type=int,
        default=None,
        help="Val/Test batch size (defaults to train batch size)",
    )
    parser.add_argument(
        "--test-frac",
        type=float,
        default=0.10,
        help="Fraction for test split (e.g., 0.10)",
    )
    parser.add_argument(
        "--val-frac",
        type=float,
        default=0.10,
        help="Overall fraction for validation split (e.g., 0.10)",
    )
    args = parser.parse_args()

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)

    train_bs = int(args.batch_size)
    val_bs = int(args.val_batch_size) if args.val_batch_size is not None else train_bs
    test_frac = float(args.test_frac)
    val_frac = float(args.val_frac)

    assert 0.0 < test_frac < 0.5, "test-frac must be in (0, 0.5)"
    assert 0.0 < val_frac < 0.5, "val-frac must be in (0, 0.5)"
    assert val_frac + test_frac < 0.9, "Sum of val-frac and test-frac too large."
    print(f"🧪 Using batch sizes — train: {train_bs}, val/test: {val_bs}")
    print(
        f"🔀 Split fractions — train: {1.0 - test_frac - val_frac:.2f}, val: {val_frac:.2f}, test: {test_frac:.2f}"
    )
    print(f"🖼️ Target size (HxW): {TARGET_SIZE}")

    # Collect results: pick the best backbone per day by **validation accuracy**
    per_day_best = {}
    for json_file in sorted(
        data_dir.glob("Dy*.json"), key=lambda p: day_to_int(p.stem)
    ):
        day = json_file.stem
        # Skip if already completed
        done_marker = out_dir / "efficientnet" / day / "metrics_test.json"
        if done_marker.exists():
            print(f"⏭ Skipping {day} — already completed")
            import json as _json
            with open(done_marker) as _f:
                _m = _json.load(_f)
            per_day_best[day] = {
                "day": day,
                "day_no": day_to_int(day),
                "backbone_key": "efficientnet",
                "val_accuracy": _m.get("val_accuracy_for_selection", 0),
                "test_accuracy": _m.get("accuracy", 0),
                "test_f1": _m.get("f1", 0),
                "val_num": _m.get("val_n", 0),
                "test_num": _m.get("test_n", 0),
                "test_actual_good": _m.get("actual_good", 0),
                "test_pred_good": _m.get("predicted_good", 0),
            }
            continue
        best = None
        for backbone_key, backbone_name in BACKBONES.items():
            res = run_training_for_day(
                json_file,
                backbone_key,
                backbone_name,
                train_bs,
                val_bs,
                test_frac,
                val_frac,
            )
            if res is None:
                continue
            if (best is None) or (res["val_accuracy"] > best["val_accuracy"]):
                best = res
        if best:
            per_day_best[day] = best
            print(
                f"✅ Best for {day} (by VAL): {best['backbone_key']} | val acc={best['val_accuracy']:.3f} | TEST acc={best['test_accuracy']:.3f}, f1={best['test_f1']:.3f}"
            )
        else:
            print(f"⚠ No valid result for {day}")

    if not per_day_best:
        print("❌ No days produced results; aborting summary.")
        return

    # ---- Build 4-column table (based on TEST)
    rows = []
    days_sorted = sorted(per_day_best.keys(), key=day_to_int)
    for d in days_sorted:
        r = per_day_best[d]
        rows.append(
            {
                "Day No": r["day_no"],
                "Num in Sample": r["test_num"],
                "Actual Good": r["test_actual_good"],
                "Predicted Good": r["test_pred_good"],
            }
        )

    # Save CSV table (exactly 4 columns)
    import csv

    table_path = out_dir / "day_summary.csv"
    with table_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["Day No", "Num in Sample", "Actual Good", "Predicted Good"]
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"🧾 Saved table → {table_path}")

    # ---- Single chart: accuracy vs day（TEST）
    xs = [per_day_best[d]["day_no"] for d in days_sorted]
    ys = [per_day_best[d]["test_accuracy"] for d in days_sorted]

    plt.figure(figsize=(8, 4))
    sns.lineplot(x=xs, y=ys, marker="o")
    plt.xlabel("Day")
    plt.ylabel("Accuracy (test)")
    plt.title("Per-day Test Accuracy (Best Image-only Model per Day)")
    plt.xticks(xs)
    plt.ylim(0.0, 1.0)
    chart_path = out_dir / "accuracy_by_day.png"
    plt.tight_layout()
    plt.savefig(chart_path)
    plt.close()
    print(f"📊 Saved accuracy chart → {chart_path}")

    # ---- Final TEST summary JSON
    acc_by_day = {d: float(per_day_best[d]["test_accuracy"]) for d in days_sorted}
    overall_best = max(per_day_best.values(), key=lambda r: r["test_accuracy"])
    summary = {
        "per_day_test_accuracy": acc_by_day,
        "overall_best": {
            "day": overall_best["day"],
            "day_no": overall_best["day_no"],
            "backbone_key": overall_best["backbone_key"],
            "test_accuracy": float(overall_best["test_accuracy"]),
            "test_f1": float(overall_best["test_f1"]),
            "selection_val_accuracy": float(overall_best["val_accuracy"]),
        },
        "batch_size_train": int(train_bs),
        "batch_size_valtest": int(val_bs),
        "split_fractions": {
            "train": float(1.0 - test_frac - val_frac),
            "val": float(val_frac),
            "test": float(test_frac),
        },
    }
    summary_path = out_dir / "final_test_summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"✅ Saved final test summary → {summary_path}")

    # ---- Also print the 4-column table to stdout
    print("\n=== Summary Table (TEST) ===")
    print(
        f"{'Day No':>6} | {'Num in Sample':>13} | {'Actual Good':>11} | {'Predicted Good':>14}"
    )
    print("-" * 54)
    for row in rows:
        print(
            f"{row['Day No']:>6} | {row['Num in Sample']:>13} | {row['Actual Good']:>11} | {row['Predicted Good']:>14}"
        )


if __name__ == "__main__":
    main()
