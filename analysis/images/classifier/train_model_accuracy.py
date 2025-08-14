#!/usr/bin/env python3

import os, json, argparse, re
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from sklearn.metrics import accuracy_score
import timm
from torchvision import transforms as T

# -------- Config --------
BACKBONES = {
    "vit": "vit_base_patch16_224",
    "resnet": "resnet50",
    "efficientnet": "efficientnet_b0"
}
DATA_DIR = Path("data/preprocessed/majority/")
OUT_ROOT = Path("outputs")
BATCH_SIZE = 16
TARGET_SIZE = (224,224)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 0
SEED = 42
# ------------------------

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
                T.ColorJitter(0.2, 0.2, 0.2, 0.1)
            ]
        t += [T.ToTensor()]
        self.t_img = T.Compose(t)

    def __len__(self): return len(self.labels)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert("RGB")
        img = self.t_img(img)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return img, label

# ---------- Model ----------
class ImageOnlyClassifier(nn.Module):
    def __init__(self, backbone_name):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=True, num_classes=0, global_pool="avg")
        out_dim = self.backbone.num_features
        # freeze backbone initially
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.classifier = nn.Sequential(
            nn.Linear(out_dim, 128), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1)
        )

    def unfreeze_backbone(self):
        for name, p in self.backbone.named_parameters():
            if "blocks." in name or "layer" in name:
                p.requires_grad = True

    def forward(self, img):
        f = self.backbone(img)
        return self.classifier(f).squeeze(1)

def make_loader(imgs, labels, augment, batch_size):
    ds = OrganoidDataset(imgs, labels, augment)
    return DataLoader(ds, batch_size=batch_size, shuffle=augment, num_workers=NUM_WORKERS)

# ---------- Train/Eval ----------
def epoch_loop(model, loader, optimizer, class_weights, train=True):
    model.train() if train else model.eval()
    bce = nn.BCEWithLogitsLoss(reduction="none")
    losses, preds, trues = [], [], []

    for img, label in loader:
        img, label = img.to(DEVICE), label.to(DEVICE)
        logit = model(img)
        loss = bce(logit, label)
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

def run_training_for_day(day_json_path: Path, backbone_key: str, backbone_name: str):
    """Train + validate on one day; return dict with metrics & paths."""
    records = json.loads(day_json_path.read_text())
    if not records:
        print(f"⚠ Skipping {day_json_path.name} — no records")
        return None

    # labels
    label_map = {"Accepted": 1, "Not Accepted": 0}
    try:
        labels = np.array([label_map[r["label"]] for r in records], dtype=int)
    except KeyError:
        print(f"⚠ Skipping {day_json_path.name} — missing 'label' field")
        return None

    imgs = np.array([r["img_path"] for r in records])

    # split
    X_tr, X_val, y_tr, y_val = train_test_split(
        imgs, labels, test_size=0.2, stratify=labels, random_state=SEED)

    # class weights
    weights = compute_class_weight("balanced", classes=np.unique(y_tr), y=y_tr)
    class_weights = {int(k): float(w) for k, w in zip(np.unique(y_tr), weights)}

    # loaders
    train_loader = make_loader(X_tr, y_tr, augment=True, batch_size=BATCH_SIZE)
    val_loader = make_loader(X_val, y_val, augment=False, batch_size=BATCH_SIZE)

    # model/opt
    model = ImageOnlyClassifier(backbone_name).to(DEVICE)
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
        history["train_loss"].append(tl); history["val_loss"].append(vl)
        history["train_acc"].append(tacc); history["val_acc"].append(vacc)
        print(f"[{day_json_path.stem}][{backbone_key}][P1][{epoch:02d}] loss {tl:.4f}/{vl:.4f} acc {tacc:.3f}/{vacc:.3f}")
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
        history["train_loss"].append(tl); history["val_loss"].append(vl)
        history["train_acc"].append(tacc); history["val_acc"].append(vacc)
        print(f"[{day_json_path.stem}][{backbone_key}][P2][{epoch:03d}] loss {tl:.4f}/{vl:.4f} acc {tacc:.3f}/{vacc:.3f}")
        if vacc > best_acc:
            best_acc = vacc
            torch.save(model.state_dict(), model_path)
        if es.step(vacc):
            break

    # Save per-day training curves (KEEPING this as requested)
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1); plt.plot(history["train_acc"], label="Train"); plt.plot(history["val_acc"], label="Val"); plt.title("Accuracy"); plt.legend()
    plt.subplot(1, 2, 2); plt.plot(history["train_loss"], label="Train"); plt.plot(history["val_loss"], label="Val"); plt.title("Loss"); plt.legend()
    plt.tight_layout()
    plt.savefig(model_dir / "training_curves.png")
    plt.close()
    print(f"📈 Saved curves → {model_dir/'training_curves.png'}")

    # Final val predictions with best weights
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    preds_bin, trues = [], []
    with torch.no_grad():
        for img, lbl in val_loader:
            img = img.to(DEVICE)
            prob = torch.sigmoid(model(img)).cpu().numpy()
            preds_bin.extend((prob > 0.5).astype(int))
            trues.extend(lbl.numpy())
    preds_bin = np.array(preds_bin); trues = np.array(trues)
    vacc = accuracy_score(trues, preds_bin)

    # Per-day summary parts
    day_no = day_to_int(day_json_path.stem)
    num_in_sample = len(trues)  # validation size
    actual_good = int(trues.sum())
    predicted_good = int(preds_bin.sum())

    return {
        "day": day_json_path.stem,
        "day_no": day_no,
        "backbone_key": backbone_key,
        "model_dir": str(model_dir),
        "val_accuracy": float(vacc),
        "val_num": int(num_in_sample),
        "val_actual_good": int(actual_good),
        "val_pred_good": int(predicted_good),
    }

# ---------- Orchestration ----------
def main():
    set_seed()

    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default=OUT_ROOT, help="Where to save outputs")
    parser.add_argument("--data_dir", default=DATA_DIR, help="Directory with per-day JSONs")
    args = parser.parse_args()

    out_dir = Path(args.outdir); out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)

    # Collect results: pick the best backbone per day by validation accuracy
    per_day_best = {}  # day -> result dict
    for json_file in sorted(data_dir.glob("Dy*.json"), key=lambda p: day_to_int(p.stem)):
        day = json_file.stem
        best = None
        for backbone_key, backbone_name in BACKBONES.items():
            res = run_training_for_day(json_file, backbone_key, backbone_name)
            if res is None:
                continue
            if (best is None) or (res["val_accuracy"] > best["val_accuracy"]):
                best = res
        if best:
            per_day_best[day] = best
            print(f"✅ Best for {day}: {best['backbone_key']} (val acc={best['val_accuracy']:.3f})")
        else:
            print(f"⚠ No valid result for {day}")

    if not per_day_best:
        print("❌ No days produced results; aborting summary.")
        return

    # ---- Build table: Day No, Num in Sample, Actual Good, Predicted Good
    rows = []
    days_sorted = sorted(per_day_best.keys(), key=day_to_int)
    for d in days_sorted:
        r = per_day_best[d]
        rows.append({
            "Day No": r["day_no"],
            "Num in Sample": r["val_num"],
            "Actual Good": r["val_actual_good"],
            "Predicted Good": r["val_pred_good"],
        })

    # Save CSV table (exactly 4 columns as requested)
    import csv
    table_path = out_dir / "day_summary.csv"
    with table_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["Day No", "Num in Sample", "Actual Good", "Predicted Good"]
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"🧾 Saved table → {table_path}")

    # ---- Single chart: accuracy vs day (best image-only per day)
    xs = [per_day_best[d]["day_no"] for d in days_sorted]
    ys = [per_day_best[d]["val_accuracy"] for d in days_sorted]

    plt.figure(figsize=(8,4))
    sns.lineplot(x=xs, y=ys, marker="o")
    plt.xlabel("Day")
    plt.ylabel("Accuracy (validation)")
    plt.title("Per-day Validation Accuracy (Best Image-only Model per Day)")
    plt.xticks(xs)
    plt.ylim(0.0, 1.0)
    chart_path = out_dir / "accuracy_by_day.png"
    plt.tight_layout()
    plt.savefig(chart_path)
    plt.close()
    print(f"📊 Saved accuracy chart → {chart_path}")

    # ---- Store final validation accuracy (JSON)
    # Per-day best accuracies
    acc_by_day = {d: float(per_day_best[d]["val_accuracy"]) for d in days_sorted}
    # Overall best (across all days)
    overall_best = max(per_day_best.values(), key=lambda r: r["val_accuracy"])
    summary = {
        "per_day_best_val_accuracy": acc_by_day,                 # { "Dy28": 0.87, ... }
        "overall_best": {
            "day": overall_best["day"],
            "day_no": overall_best["day_no"],
            "backbone_key": overall_best["backbone_key"],
            "model_dir": overall_best["model_dir"],
            "val_accuracy": float(overall_best["val_accuracy"]),
        }
    }
    summary_path = out_dir / "final_validation_summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"✅ Saved final validation accuracy → {summary_path}")

    # ---- Also print the 4-column table to stdout
    print("\n=== Summary Table (Validation) ===")
    print(f"{'Day No':>6} | {'Num in Sample':>13} | {'Actual Good':>11} | {'Predicted Good':>14}")
    print("-"*54)
    for row in rows:
        print(f"{row['Day No']:>6} | {row['Num in Sample']:>13} | {row['Actual Good']:>11} | {row['Predicted Good']:>14}")

if __name__ == "__main__":
    main()
