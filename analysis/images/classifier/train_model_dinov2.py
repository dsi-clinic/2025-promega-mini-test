#!/usr/bin/env python3

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
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score
from torchvision import transforms as T

# Import transformers for dinov2
from transformers import AutoModel

# -------- Config (defaults; can be overridden by CLI) --------
DATA_DIR = Path("analysis/images/classifier/data/preprocessed/512x384/majority/")
OUT_ROOT = Path("tony_image_classifier/result/outputs_dinov2")
BATCH_SIZE = 16
# IMPORTANT: dinov2 expects 518x518 by default, but we can resize to (384, 512)
TARGET_SIZE = (384, 512)   # (H, W)
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
    # "Dy28" -> 28, "Dy20_5" -> 20, fallback -1
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
    """Dataset for images with optional augmentation."""

    def __init__(self, img_paths, labels, augment=False):
        self.img_paths = img_paths
        self.labels = labels
        self.augment = augment
        t = [T.Resize(TARGET_SIZE)]
        if augment:
            t += [
                T.RandomHorizontalFlip(0.5),
                T.RandomVerticalFlip(0.5),
                T.RandomRotation(15),
            ]
        t += [T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        self.transform = T.Compose(t)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert("RGB")
        img = self.transform(img)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return img, label


# ---------- Model ----------
class DinoV2Classifier(nn.Module):
    def __init__(self, model_name="facebook/dinov2-base"):
        super().__init__()
        # Load dinov2 from HuggingFace
        self.backbone = AutoModel.from_pretrained(model_name)
        
        # Freeze backbone initially
        for p in self.backbone.parameters():
            p.requires_grad = False
        
        # Get feature dimension (768 for base, 1024 for large)
        # dinov2-base has hidden_size=768
        out_dim = self.backbone.config.hidden_size
        
        self.classifier = nn.Sequential(
            nn.Linear(out_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
        )

    def unfreeze_backbone(self):
        # Unfreeze the last few layers
        for name, p in self.backbone.named_parameters():
            if "encoder.layer" in name:
                # Parse layer number
                try:
                    layer_match = re.search(r'layer\.(\d+)', name)
                    if layer_match:
                        layer_num = int(layer_match.group(1))
                        # Unfreeze last 4 layers (adjust as needed)
                        if layer_num >= 8:
                            p.requires_grad = True
                except:
                    pass

    def forward(self, img):
        # dinov2 returns a dict with 'last_hidden_state' and 'pooler_output'
        outputs = self.backbone(img)
        # Use CLS token (first token) from last hidden state
        f = outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]
        return self.classifier(f).squeeze(1)


def make_loader(imgs, labels, augment, batch_size):
    ds = OrganoidDataset(imgs, labels, augment=augment)
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
    preds_bin = np.array(preds_bin); trues = np.array(trues); probs = np.array(probs)
    acc = accuracy_score(trues, preds_bin)
    f1 = f1_score(trues, preds_bin)
    return preds_bin, trues, float(acc), float(f1), probs

def run_training_for_day(day_json_path: Path, 
                         train_bs: int, val_bs: int, test_frac: float, val_frac: float,
                         out_root: Path, input_key: str):
    """Train + validate with small val/test; select by VAL acc, report on TEST."""
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

    try:
        imgs = [r[input_key] for r in records]
    except KeyError:
        print(f"⚠ Skipping {day_json_path.name} — missing '{input_key}' field")
        return None

    # Filter out entries with missing files
    filtered_imgs, filtered_labels = [], []
    missing_records = []
    for idx, img_path in enumerate(imgs):
        img_path = Path(str(img_path))
        if not img_path.exists():
            missing_records.append({"img_path": str(img_path), "reason": "missing_image"})
            continue
        filtered_imgs.append(str(img_path))
        filtered_labels.append(labels[idx])

    if missing_records:
        log_dir = out_root / "dinov2" / day_json_path.stem
        log_dir.mkdir(parents=True, exist_ok=True)
        missing_csv = log_dir / "missing_files.csv"
        with missing_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["img_path", "reason"])
            writer.writeheader()
            writer.writerows(missing_records)
        print(f"⚠ {day_json_path.name}: skipped {len(missing_records)} entries due to missing files (details → {missing_csv})")

    if not filtered_labels:
        print(f"⚠ Skipping {day_json_path.name} — no valid samples after filtering missing files")
        return None

    imgs = np.array(filtered_imgs)
    labels = np.array(filtered_labels)

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

    # loaders
    train_loader = make_loader(X_tr, y_tr, augment=True, batch_size=train_bs)
    val_loader = make_loader(X_val, y_val, augment=False, batch_size=val_bs)
    test_loader = make_loader(X_test, y_test, augment=False, batch_size=val_bs)

    # model/opt (save under dinov2/ subdirectory to match Amanda's structure)
    model = DinoV2Classifier(model_name="facebook/dinov2-base").to(DEVICE)
    model_dir = out_root / "dinov2" / day_json_path.stem
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "model_dinov2.pth"

    opt = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    es = EarlyStopping(patience=20)
    history = defaultdict(list)
    best_acc = -np.inf

    # Phase 1 — frozen backbone
    print(f"[{day_json_path.stem}][dinov2] Phase 1: Training with frozen backbone...")
    for epoch in range(100):
        tl, tacc, _, _ = epoch_loop(model, train_loader, opt, class_weights, train=True)
        vl, vacc, _, _ = epoch_loop(model, val_loader, opt, class_weights, train=False)
        history["train_loss"].append(tl); history["val_loss"].append(vl)
        history["train_acc"].append(tacc); history["val_acc"].append(vacc)
        print(f"[{day_json_path.stem}][dinov2][P1][{epoch:02d}][bs={train_bs}/{val_bs}] loss {tl:.4f}/{vl:.4f} acc {tacc:.3f}/{vacc:.3f}")
        if vacc > best_acc:
            best_acc = vacc
            torch.save(model.state_dict(), model_path)
        if es.step(vacc):
            break

    # Phase 2 — unfreeze partial backbone
    print(f"[{day_json_path.stem}][dinov2] Phase 2: Fine-tuning with unfrozen layers...")
    model.unfreeze_backbone()
    opt = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    es = EarlyStopping(patience=30)
    for epoch in range(300):
        tl, tacc, _, _ = epoch_loop(model, train_loader, opt, class_weights, train=True)
        vl, vacc, _, _ = epoch_loop(model, val_loader, opt, class_weights, train=False)
        history["train_loss"].append(tl); history["val_loss"].append(vl)
        history["train_acc"].append(tacc); history["val_acc"].append(vacc)
        print(f"[{day_json_path.stem}][dinov2][P2][{epoch:03d}][bs={train_bs}/{val_bs}] loss {tl:.4f}/{vl:.4f} acc {tacc:.3f}/{vacc:.3f}")
        if vacc > best_acc:
            best_acc = vacc
            torch.save(model.state_dict(), model_path)
        if es.step(vacc):
            break

    # Save per-day training curves
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1); plt.plot(history["train_acc"], label="Train"); plt.plot(history["val_acc"], label="Val"); plt.title("Accuracy"); plt.legend()
    plt.subplot(1, 2, 2); plt.plot(history["train_loss"], label="Train"); plt.plot(history["val_loss"], label="Val"); plt.title("Loss"); plt.legend()
    plt.tight_layout()
    plt.savefig(model_dir / "training_curves_dinov2.png")
    plt.close()
    print(f"📈 Saved curves → {model_dir/'training_curves_dinov2.png'}")

    # ---- Evaluate with best VAL checkpoint
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))

    # Val metrics
    _, val_trues, val_acc, val_f1, val_probs = evaluate_on_loader(model, val_loader)
    try:
        val_roc_auc = float(roc_auc_score(val_trues, val_probs))
    except Exception:
        val_roc_auc = None
    val_pr_auc = float(average_precision_score(val_trues, val_probs)) if len(val_trues) > 0 else None
    val_metrics = {
        "day": day_json_path.stem,
        "split": "val",
        "accuracy": float(val_acc),
        "f1": float(val_f1),
        "roc_auc": val_roc_auc,
        "pr_auc": val_pr_auc,
        "n": int(len(y_val)),
        "batch_size": int(val_bs),
        "input_key": input_key,
    }
    with (model_dir / "metrics_val_dinov2.json").open("w") as f:
        json.dump(val_metrics, f, indent=2)

    # Test metrics
    preds_bin, trues, test_acc, test_f1, test_probs = evaluate_on_loader(model, test_loader)
    try:
        test_roc_auc = float(roc_auc_score(trues, test_probs))
    except Exception:
        test_roc_auc = None
    test_pr_auc = float(average_precision_score(trues, test_probs)) if len(trues) > 0 else None
    day_no = day_to_int(day_json_path.stem)
    num_in_sample = int(len(trues))
    actual_good = int(trues.sum())
    predicted_good = int(preds_bin.sum())

    test_metrics = {
        "day": day_json_path.stem,
        "day_no": day_no,
        "split": "test",
        "accuracy": float(test_acc),
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
        "backbone_key": "dinov2",
        "input_key": input_key,
    }
    with (model_dir / "metrics_test_dinov2.json").open("w") as f:
        json.dump(test_metrics, f, indent=2)
    print(f"📝 Saved metrics → {model_dir/'metrics_val_dinov2.json'} and {model_dir/'metrics_test_dinov2.json'}")

    return {
        "day": day_json_path.stem,
        "day_no": day_no,
        "val_accuracy": float(best_acc),
        "test_accuracy": float(test_acc),
        "test_f1": float(test_f1),
        "test_roc_auc": test_roc_auc,
        "val_num": int(len(y_val)),
        "test_num": num_in_sample,
        "test_actual_good": actual_good,
        "test_pred_good": predicted_good,
    }

# ---------- Orchestration ----------
def main():
    set_seed()

    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, default=OUT_ROOT, help="Where to save outputs")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR, help="Directory with per-day JSONs")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Train batch size")
    parser.add_argument("--val-batch-size", type=int, default=None, help="Val/Test batch size")
    parser.add_argument("--test-frac", type=float, default=0.10, help="Fraction for test split")
    parser.add_argument("--val-frac", type=float, default=0.10, help="Overall fraction for validation split")
    parser.add_argument("--input-path-key", choices=["img_path", "overlay_path"], default="img_path",
                        help="Which JSON field to use as the primary image input")
    args = parser.parse_args()

    out_dir = Path(args.outdir); out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)

    train_bs = int(args.batch_size)
    val_bs = int(args.val_batch_size) if args.val_batch_size is not None else train_bs
    test_frac = float(args.test_frac)
    val_frac = float(args.val_frac)
    input_key = str(args.input_path_key)

    assert 0.0 < test_frac < 0.5, "test-frac must be in (0, 0.5)"
    assert 0.0 < val_frac < 0.5, "val-frac must be in (0, 0.5)"
    assert val_frac + test_frac < 0.9, "Sum of val-frac and test-frac too large."
    
    print(f"📁 Data dir: {data_dir}")
    print(f"📂 Output dir: {out_dir}")
    print(f"🧪 Batch sizes — train: {train_bs}, val/test: {val_bs}")
    print(f"🔀 Split fractions — train: {1.0 - test_frac - val_frac:.2f}, val: {val_frac:.2f}, test: {test_frac:.2f}")
    print(f"🖼️ Target size (HxW): {TARGET_SIZE}")
    print(f"🗂️ Input field: {input_key}")
    print(f"🤖 Using DINOv2 backbone from HuggingFace")

    # Train across all days
    per_day_results = {}
    for json_file in sorted(data_dir.glob("Dy*.json"), key=lambda p: day_to_int(p.stem)):
        day = json_file.stem
        res = run_training_for_day(json_file, train_bs, val_bs, test_frac, val_frac,
                                   out_dir, input_key=input_key)
        if res is None:
            print(f"⚠ No valid result for {day}")
            continue
        per_day_results[day] = res
        print(f"✅ Completed {day} | VAL acc={res['val_accuracy']:.3f} | TEST acc={res['test_accuracy']:.3f}, f1={res['test_f1']:.3f}")

    if not per_day_results:
        print("❌ No days produced results; aborting summary.")
        return

    # ---- Build 4-column table (based on TEST)
    rows = []
    days_sorted = sorted(per_day_results.keys(), key=day_to_int)
    for d in days_sorted:
        r = per_day_results[d]
        rows.append({
            "Day No": r["day_no"],
            "Num in Sample": r["test_num"],
            "Actual Good": r["test_actual_good"],
            "Predicted Good": r["test_pred_good"],
        })

    # Save CSV tables - both generic and dinov2-specific (matching Amanda's format)
    # Generic table (for best-per-day comparison)
    table_path = out_dir / "day_summary.csv"
    with table_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Day No", "Num in Sample", "Actual Good", "Predicted Good"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"🧾 Saved table → {table_path}")
    
    # DINOv2-specific table (matching Amanda's vit_day_summary.csv format)
    dinov2_table_path = out_dir / "dinov2_day_summary.csv"
    with dinov2_table_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Day No", "Num in Sample", "Actual Good", "Predicted Good"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"🧾 Saved dinov2 table → {dinov2_table_path}")

    # Chart: accuracy vs day (matching Amanda's format)
    xs = [per_day_results[d]["day_no"] for d in days_sorted]
    ys = [per_day_results[d]["test_accuracy"] for d in days_sorted]
    
    # Generic accuracy chart
    plt.figure(figsize=(8, 4))
    plt.plot(xs, ys, marker="o", color="blue")
    plt.xlabel("Day"); plt.ylabel("Accuracy (test)")
    plt.title("Per-day Test Accuracy (DINOv2)")
    plt.xticks(xs); plt.ylim(0.0, 1.0)
    chart_path = out_dir / "accuracy_by_day.png"
    plt.tight_layout(); plt.savefig(chart_path); plt.close()
    print(f"📊 Saved accuracy chart → {chart_path}")
    
    # DINOv2-specific accuracy chart
    dinov2_chart_path = out_dir / "dinov2_accuracy_by_day.png"
    plt.figure(figsize=(8, 4))
    plt.plot(xs, ys, marker="o", color="blue")
    plt.xlabel("Day"); plt.ylabel("Accuracy (test)")
    plt.title("DINOv2: Per-day Test Accuracy")
    plt.xticks(xs); plt.ylim(0.0, 1.0)
    plt.tight_layout(); plt.savefig(dinov2_chart_path); plt.close()
    print(f"📊 Saved dinov2 accuracy chart → {dinov2_chart_path}")

    # Combined metrics chart (accuracy + F1, matching Amanda's vit_metrics_by_day.png format)
    ys_f1 = [per_day_results[d]["test_f1"] for d in days_sorted]
    plt.figure(figsize=(10, 5))
    plt.plot(xs, ys, marker="o", label="Accuracy", color="blue")
    plt.plot(xs, ys_f1, marker="s", label="F1 Score", color="green")
    plt.xlabel("Day"); plt.ylabel("Score")
    plt.title("DINOv2: Per-day Test Metrics")
    plt.xticks(xs); plt.ylim(0.0, 1.0)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    metrics_chart_path = out_dir / "dinov2_metrics_by_day.png"
    plt.savefig(metrics_chart_path)
    plt.close()
    print(f"📊 Saved dinov2 metrics chart → {metrics_chart_path}")

    # Final summary JSON
    summary = {
        "per_day": {
            day: {
                "day_no": int(res["day_no"]),
                "test_accuracy": float(res["test_accuracy"]),
                "test_f1": float(res["test_f1"]),
                "test_roc_auc": (None if res.get("test_roc_auc") is None else float(res["test_roc_auc"])),
                "val_accuracy": float(res["val_accuracy"]),
                "test_num": int(res["test_num"]),
            }
            for day, res in per_day_results.items()
        },
        "batch_size_train": int(train_bs),
        "batch_size_valtest": int(val_bs),
        "split_fractions": {
            "train": float(1.0 - test_frac - val_frac),
            "val": float(val_frac),
            "test": float(test_frac),
        },
        "backbone": "dinov2-base",
        "input_key": input_key,
    }
    summary_path = out_dir / "final_test_summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"✅ Saved final test summary → {summary_path}")

    # Print the 4-column table
    print("\n=== Summary Table (TEST) ===")
    print(f"{'Day No':>6} | {'Num in Sample':>13} | {'Actual Good':>11} | {'Predicted Good':>14}")
    print("-" * 54)
    for row in rows:
        print(f"{row['Day No']:>6} | {row['Num in Sample']:>13} | {row['Actual Good']:>11} | {row['Predicted Good']:>14}")

if __name__ == "__main__":
    main()

