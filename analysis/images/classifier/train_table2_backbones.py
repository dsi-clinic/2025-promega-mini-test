#!/usr/bin/env python3
"""
Table 2: Train ViT and ResNet50 per-day classifiers.
EfficientNet results are read from the existing efficientnet_ensemble outputs.
Outputs aggregated Table 2 CSV to analysis/images/classifier/table2_backbone_comparison.csv
"""

import json, argparse, re, os, csv
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
from sklearn.metrics import (
    accuracy_score, f1_score, balanced_accuracy_score, recall_score
)
import timm
from torchvision import transforms as T

# -------- Config --------
BACKBONES = {
    "vit": "vit_base_patch16_224",
    "resnet": "resnet50",
}

_DATA_DIR = Path(os.environ.get("DATA_DIR", "/net/projects2/promega/2026_04_data"))
_IMG_DIR = _DATA_DIR / "lstm" / "lstm_ready" / "images"

DATA_DIR = Path("analysis/images/classifier/data/preprocessed/512x384/majority/")
OUT_ROOT = Path("analysis/images/classifier/per_day_study/table2_backbones")
EFFNET_DIR = Path("analysis/images/classifier/per_day_study/efficientnet_ensemble/efficientnet")
TABLE2_CSV = Path("analysis/images/classifier/table2_backbone_comparison.csv")

BATCH_SIZE = 16
TARGET_SIZE = (384, 512)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 0
SEED = 1


def _remap_img_path(img_path: str) -> str:
    fname = Path(img_path).name
    candidate = _IMG_DIR / fname
    return str(candidate) if candidate.exists() else img_path


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


class OrganoidDataset(Dataset):
    def __init__(self, img_paths, labels, augment=False):
        self.img_paths = img_paths
        self.labels = labels
        t = [T.Resize(TARGET_SIZE)]
        if augment:
            t += [T.RandomHorizontalFlip(0.5), T.ColorJitter(0.2, 0.2, 0.2, 0.1)]
        t += [T.ToTensor()]
        self.t_img = T.Compose(t)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert("RGB")
        img = self.t_img(img)
        return img, torch.tensor(self.labels[idx], dtype=torch.float32)


class ImageOnlyClassifier(nn.Module):
    def __init__(self, backbone_name, target_size):
        super().__init__()
        extra_args = {}
        if "vit" in backbone_name:
            extra_args["img_size"] = target_size
        self.backbone = timm.create_model(
            backbone_name, pretrained=True, num_classes=0,
            global_pool="avg", **extra_args
        )
        out_dim = self.backbone.num_features
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.classifier = nn.Sequential(
            nn.Linear(out_dim, 128), nn.ReLU(), nn.Dropout(0.5), nn.Linear(128, 1)
        )

    def unfreeze_backbone(self):
        for name, p in self.backbone.named_parameters():
            if "blocks." in name or "layer" in name:
                p.requires_grad = True

    def forward(self, img):
        return self.classifier(self.backbone(img)).squeeze(1)


def make_loader(imgs, labels, augment, batch_size):
    return DataLoader(
        OrganoidDataset(imgs, labels, augment),
        batch_size=batch_size, shuffle=augment, num_workers=NUM_WORKERS
    )


def epoch_loop(model, loader, optimizer, class_weights, train=True):
    model.train() if train else model.eval()
    bce = nn.BCEWithLogitsLoss(reduction="none")
    losses, preds, trues = [], [], []
    for img, label in loader:
        img, label = img.to(DEVICE), label.to(DEVICE)
        logit = model(img)
        loss = bce(logit, label)
        weight = torch.tensor(
            [class_weights[int(l.item())] for l in label], device=label.device
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
    acc = accuracy_score(trues, preds_bin)
    bal_acc = balanced_accuracy_score(trues, preds_bin)
    f1 = f1_score(trues, preds_bin, zero_division=0)
    # Recall for Not Acceptable (label=0)
    na_recall = recall_score(trues, preds_bin, pos_label=0, zero_division=0)
    return preds_bin, trues, float(acc), float(f1), float(bal_acc), float(na_recall)


def run_training_for_day(day_json_path, backbone_key, backbone_name, train_bs, val_bs, test_frac, val_frac):
    records = json.loads(day_json_path.read_text())
    if not records:
        return None

    label_map = {"Accepted": 1, "Not Accepted": 0}
    records = [r for r in records if r.get("label") in label_map]

    remapped = [_remap_img_path(r["img_path"]) for r in records]
    valid = [(img, r) for img, r in zip(remapped, records) if Path(img).exists()]
    if not valid:
        print(f"⚠ Skipping {day_json_path.name} — no valid images")
        return None
    n_dropped = len(records) - len(valid)
    if n_dropped:
        print(f"  [{day_json_path.stem}] dropped {n_dropped} missing images")

    imgs = np.array([img for img, _ in valid])
    labels = np.array([label_map[r["label"]] for _, r in valid], dtype=int)

    X_tmp, X_test, y_tmp, y_test = train_test_split(
        imgs, labels, test_size=test_frac, stratify=labels, random_state=SEED
    )
    val_frac_cond = val_frac / (1.0 - test_frac)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_tmp, y_tmp, test_size=val_frac_cond, stratify=y_tmp, random_state=SEED
    )

    weights = compute_class_weight("balanced", classes=np.unique(y_tr), y=y_tr)
    class_weights = {int(k): float(w) for k, w in zip(np.unique(y_tr), weights)}

    train_loader = make_loader(X_tr, y_tr, augment=False, batch_size=train_bs)
    val_loader = make_loader(X_val, y_val, augment=False, batch_size=val_bs)
    test_loader = make_loader(X_test, y_test, augment=False, batch_size=val_bs)

    model = ImageOnlyClassifier(backbone_name, TARGET_SIZE).to(DEVICE)
    model_dir = OUT_ROOT / backbone_key / day_json_path.stem
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "model.pth"

    opt = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    es = EarlyStopping(patience=20)
    best_acc = -np.inf

    for epoch in range(100):
        tl, tacc, _, _ = epoch_loop(model, train_loader, opt, class_weights, train=True)
        vl, vacc, _, _ = epoch_loop(model, val_loader, opt, class_weights, train=False)
        print(f"[{day_json_path.stem}][{backbone_key}][P1][{epoch:02d}] loss {tl:.4f}/{vl:.4f} acc {tacc:.3f}/{vacc:.3f}")
        if vacc > best_acc:
            best_acc = vacc
            torch.save(model.state_dict(), model_path)
        if es.step(vacc):
            break

    model.unfreeze_backbone()
    opt = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    es = EarlyStopping(patience=30)
    for epoch in range(300):
        tl, tacc, _, _ = epoch_loop(model, train_loader, opt, class_weights, train=True)
        vl, vacc, _, _ = epoch_loop(model, val_loader, opt, class_weights, train=False)
        print(f"[{day_json_path.stem}][{backbone_key}][P2][{epoch:03d}] loss {tl:.4f}/{vl:.4f} acc {tacc:.3f}/{vacc:.3f}")
        if vacc > best_acc:
            best_acc = vacc
            torch.save(model.state_dict(), model_path)
        if es.step(vacc):
            break

    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    preds_bin, trues, test_acc, test_f1, test_bal_acc, test_na_recall = evaluate_on_loader(model, test_loader)

    test_metrics = {
        "day": day_json_path.stem,
        "day_no": day_to_int(day_json_path.stem),
        "backbone_key": backbone_key,
        "accuracy": float(test_acc),
        "balanced_accuracy": float(test_bal_acc),
        "f1": float(test_f1),
        "na_recall": float(test_na_recall),
        "val_accuracy_for_selection": float(best_acc),
        "test_n": int(len(trues)),
    }
    with (model_dir / "metrics_test.json").open("w") as f:
        json.dump(test_metrics, f, indent=2)
    print(f"📝 Saved metrics → {model_dir / 'metrics_test.json'}")

    return test_metrics


def load_effnet_results():
    """Load existing EfficientNet results from efficientnet_ensemble outputs."""
    results = []
    if not EFFNET_DIR.exists():
        print(f"⚠ EfficientNet results not found at {EFFNET_DIR}")
        return results
    for metrics_file in sorted(EFFNET_DIR.glob("*/metrics_test.json")):
        with open(metrics_file) as f:
            m = json.load(f)
        day_no = day_to_int(metrics_file.parent.name)
        # Compute na_recall from actual_good/predicted_good if not stored
        na_recall = m.get("na_recall")
        if na_recall is None:
            # Can't recompute without confusion matrix — use 0 as placeholder
            na_recall = None
        results.append({
            "day": metrics_file.parent.name,
            "day_no": day_no,
            "backbone_key": "efficientnet",
            "accuracy": m.get("accuracy", 0),
            "balanced_accuracy": m.get("balanced_accuracy") or m.get("accuracy", 0),
            "f1": m.get("f1", 0),
            "na_recall": na_recall,
        })
    return results


def aggregate_table2(all_results):
    """Aggregate per-day results into Table 2 rows per backbone."""
    by_backbone = defaultdict(list)
    for r in all_results:
        by_backbone[r["backbone_key"]].append(r)

    rows = []
    for backbone in ["vit", "resnet", "efficientnet"]:
        day_results = by_backbone.get(backbone, [])
        if not day_results:
            continue
        accs = [r["accuracy"] for r in day_results]
        bal_accs = [r["balanced_accuracy"] for r in day_results]
        na_recalls = [r["na_recall"] for r in day_results if r.get("na_recall") is not None]
        days_recall_zero = sum(1 for r in day_results if r.get("na_recall") is not None and r["na_recall"] == 0.0)
        n_days = len(day_results)

        rows.append({
            "Model": backbone.capitalize(),
            "Avg Accuracy": f"{np.mean(accs)*100:.1f}%",
            "Avg Bal Acc": f"{np.mean(bal_accs)*100:.1f}%",
            "Avg Recall (N.A.)": f"{np.mean(na_recalls)*100:.1f}%" if na_recalls else "N/A",
            "Days Recall_NA=0": f"{days_recall_zero}/{n_days}",
            "Best Bal Acc": f"{max(bal_accs)*100:.1f}%",
        })
    return rows


def main():
    set_seed()

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=DATA_DIR)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--val-batch-size", type=int, default=None)
    parser.add_argument("--test-frac", type=float, default=0.10)
    parser.add_argument("--val-frac", type=float, default=0.10)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    train_bs = args.batch_size
    val_bs = args.val_batch_size or train_bs
    test_frac = args.test_frac
    val_frac = args.val_frac

    print(f"🖼️  Target size (HxW): {TARGET_SIZE}")
    print(f"🔀 Split: train={1-test_frac-val_frac:.0%}, val={val_frac:.0%}, test={test_frac:.0%}")

    # Train ViT and ResNet
    new_results = []
    for json_file in sorted(data_dir.glob("Dy*.json"), key=lambda p: day_to_int(p.stem)):
        for backbone_key, backbone_name in BACKBONES.items():
            done_marker = OUT_ROOT / backbone_key / json_file.stem / "metrics_test.json"
            if done_marker.exists():
                print(f"⏭ Skipping {json_file.stem}/{backbone_key} — already done")
                with open(done_marker) as f:
                    new_results.append(json.load(f))
                continue
            res = run_training_for_day(json_file, backbone_key, backbone_name, train_bs, val_bs, test_frac, val_frac)
            if res:
                new_results.append(res)
                print(f"✅ {json_file.stem}/{backbone_key} | bal_acc={res['balanced_accuracy']:.3f} | na_recall={res['na_recall']:.3f}")

    # Load EfficientNet results
    effnet_results = load_effnet_results()
    all_results = new_results + effnet_results

    # Aggregate Table 2
    rows = aggregate_table2(all_results)

    TABLE2_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(TABLE2_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Model", "Avg Accuracy", "Avg Bal Acc", "Avg Recall (N.A.)", "Days Recall_NA=0", "Best Bal Acc"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n🧾 Table 2 saved → {TABLE2_CSV}")

    print("\n=== Table 2: Backbone Comparison ===")
    print(f"{'Model':<14} {'Avg Acc':>10} {'Avg Bal Acc':>12} {'Avg Recall NA':>14} {'Days NA=0':>10} {'Best Bal Acc':>13}")
    print("-" * 65)
    for row in rows:
        print(f"{row['Model']:<14} {row['Avg Accuracy']:>10} {row['Avg Bal Acc']:>12} {row['Avg Recall (N.A.)']:>14} {row['Days Recall_NA=0']:>10} {row['Best Bal Acc']:>13}")


if __name__ == "__main__":
    main()
