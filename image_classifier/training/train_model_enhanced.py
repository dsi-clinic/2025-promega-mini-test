#!/usr/bin/env python3
# Filename: train_organoid.py

import json, argparse, re, datetime
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
from sklearn.metrics import accuracy_score, f1_score

import timm
import torchvision.transforms as T
import torchvision.transforms.functional as TF

# -------- Defaults (override via CLI) --------
BACKBONES = {
    "vit": "vit_base_patch16_224",  # we'll pass img_size=(384,512)
    "resnet": "resnet50",
    "efficientnet": "efficientnet_b0",
}
DEFAULT_DATA_DIR = Path(
    "analysis/images/classifier/data/preprocessed/512x384/majority/"
)
DEFAULT_OUT_ROOT = Path("image_classifier/training/outputs_512x384_two_level_aug")
BATCH_SIZE = 16
TARGET_SIZE = (384, 512)  # (H, W)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 0
SEED = 1
# --------------------------------------------


# ---------- Utils ----------
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


# ---------- Augmentation pieces ----------
class AddGaussianNoise(nn.Module):
    """Add i.i.d. Gaussian noise to a tensor (C,H,W) after ToTensor, before Normalize."""

    def __init__(self, std=0.01):
        super().__init__()
        self.std = float(std)

    def forward(self, tensor):
        if self.std <= 0:
            return tensor
        noise = torch.randn_like(tensor) * self.std
        return tensor + noise


def get_transforms(
    split: str,
    level: str,
    mean=(0.5, 0.5, 0.5),
    std=(0.5, 0.5, 0.5),
    geom_cfg=None,
    photo_cfg=None,
):
    """
    split: 'train' | 'val' | 'test'
    level: 'none' | 'geom' | 'geom_photo'   (val/test always 'none' regardless)
    """
    if geom_cfg is None:
        geom_cfg = {}
    if photo_cfg is None:
        photo_cfg = {}

    # Base deterministic (for all splits)
    base_resize = [T.Resize(TARGET_SIZE)]

    # Validation/Test: no randomness, only base + ToTensor + Normalize
    if split in ("val", "test") or level == "none":
        return T.Compose(base_resize + [T.ToTensor(), T.Normalize(mean, std)])

    # ---- Level 1: geometric only ----
    geom = []
    geom.append(T.RandomHorizontalFlip(p=0.5))
    geom.append(T.RandomVerticalFlip(p=0.5))

    rot_degrees = float(geom_cfg.get("rotation_degrees", 15.0))
    geom.append(T.RandomRotation(degrees=rot_degrees))

    rrc_scale = geom_cfg.get("rrc_scale", (0.9, 1.1))
    # keep ratio fixed to target aspect ratio
    aspect = TARGET_SIZE[1] / TARGET_SIZE[0]
    rrc_ratio = geom_cfg.get("rrc_ratio", (aspect, aspect))
    geom.append(T.RandomResizedCrop(size=TARGET_SIZE, scale=rrc_scale, ratio=rrc_ratio))

    aff_deg = float(geom_cfg.get("affine_degrees", 5.0))
    translate = geom_cfg.get("affine_translate", (0.02, 0.02))  # 2% shift
    scale = geom_cfg.get("affine_scale", (0.98, 1.02))  # ±2% zoom
    geom.append(T.RandomAffine(degrees=aff_deg, translate=translate, scale=scale))

    # ---- Level 2: geometric + mild photometric ----
    if level == "geom_photo":
        noise_std = float(photo_cfg.get("gaussian_noise_std", 0.01))
        # NOTE: We intentionally avoid hue/saturation and strong blur/elastic changes.
        return T.Compose(
            base_resize
            + geom
            + [T.ToTensor(), AddGaussianNoise(std=noise_std), T.Normalize(mean, std)]
        )

    # Level 1 only
    return T.Compose(base_resize + geom + [T.ToTensor(), T.Normalize(mean, std)])


# ---------- Data ----------
class OrganoidDataset(Dataset):
    """
    Image-only dataset with:
      - Optional duplication: keep both original and a deterministic H-flipped copy
      - Split-specific transforms (train/val/test)
    """

    def __init__(self, img_paths, labels, split, transform, duplicate_hflip=False):
        self.img_paths = list(map(str, img_paths))
        self.labels = list(map(int, labels))
        assert len(self.img_paths) == len(self.labels)
        self.split = split
        self.transform = transform

        # Build index (base_idx, is_flipped)
        self.items = [(i, False) for i in range(len(self.labels))]
        if duplicate_hflip and split == "train":
            self.items += [(i, True) for i in range(len(self.labels))]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        base_idx, is_flipped = self.items[idx]
        img = Image.open(self.img_paths[base_idx]).convert("RGB")

        # Deterministic horizontal flip for duplicated entries only
        if is_flipped:
            img = TF.hflip(img)

        if self.transform is not None:
            img = self.transform(img)

        label = torch.tensor(self.labels[base_idx], dtype=torch.float32)
        return img, label


# ---------- Model ----------
class ImageOnlyClassifier(nn.Module):
    def __init__(self, backbone_name, target_size):
        super().__init__()
        extra_args = {}
        if "vit" in backbone_name:
            extra_args["img_size"] = target_size  # (H,W) supported by timm ViT

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
            if "blocks." in name or "layer" in name:
                p.requires_grad = True

    def forward(self, img):
        f = self.backbone(img)
        return self.classifier(f).squeeze(1)


def make_loader(
    imgs, labels, split, transform, batch_size, duplicate_hflip=False, shuffle=None
):
    if shuffle is None:
        shuffle = split == "train"
    ds = OrganoidDataset(
        imgs, labels, split=split, transform=transform, duplicate_hflip=duplicate_hflip
    )
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, num_workers=NUM_WORKERS
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
    probs = np.array(probs)
    acc = accuracy_score(trues, preds_bin)
    f1 = f1_score(trues, preds_bin)
    return preds_bin, trues, float(acc), float(f1), probs


def run_training_for_day(
    day_json_path: Path,
    backbone_key: str,
    backbone_name: str,
    train_bs: int,
    val_bs: int,
    test_frac: float,
    val_frac: float,
    augment_level: str,
    duplicate_hflip_train: bool,
    geom_cfg: dict,
    photo_cfg: dict,
    norm_mean=(0.5, 0.5, 0.5),
    norm_std=(0.5, 0.5, 0.5),
    out_root: Path = DEFAULT_OUT_ROOT,
):
    records = json.loads(day_json_path.read_text())
    if not records:
        print(f"⚠ Skipping {day_json_path.name} — no records")
        return None

    label_map = {"Accepted": 1, "Not Accepted": 0}
    try:
        labels = np.array([label_map[r["label"]] for r in records], dtype=int)
    except KeyError:
        print(f"⚠ Skipping {day_json_path.name} — missing 'label' field")
        return None

    imgs = np.array([r["img_path"] for r in records])

    # Split: test first, then val
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        imgs, labels, test_size=test_frac, stratify=labels, random_state=SEED
    )
    val_frac_cond = val_frac / (1.0 - test_frac)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_tmp, y_tmp, test_size=val_frac_cond, stratify=y_tmp, random_state=SEED
    )

    # Class weights on train only
    weights = compute_class_weight("balanced", classes=np.unique(y_tr), y=y_tr)
    class_weights = {int(k): float(w) for k, w in zip(np.unique(y_tr), weights)}

    # Transforms
    t_train = get_transforms(
        split="train",
        level=augment_level,
        mean=norm_mean,
        std=norm_std,
        geom_cfg=geom_cfg,
        photo_cfg=photo_cfg,
    )
    t_eval = get_transforms(split="val", level="none", mean=norm_mean, std=norm_std)

    # Loaders
    train_loader = make_loader(
        X_tr,
        y_tr,
        split="train",
        transform=t_train,
        batch_size=train_bs,
        duplicate_hflip=duplicate_hflip_train,
        shuffle=True,
    )
    val_loader = make_loader(
        X_val,
        y_val,
        split="val",
        transform=t_eval,
        batch_size=val_bs,
        duplicate_hflip=False,
        shuffle=False,
    )
    test_loader = make_loader(
        X_test,
        y_test,
        split="test",
        transform=t_eval,
        batch_size=val_bs,
        duplicate_hflip=False,
        shuffle=False,
    )

    # Model/opt
    model = ImageOnlyClassifier(backbone_name, TARGET_SIZE).to(DEVICE)
    model_dir = out_root / backbone_key / day_json_path.stem
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
            f"[{day_json_path.stem}][{backbone_key}][P1][{epoch:02d}] loss {tl:.4f}/{vl:.4f} acc {tacc:.3f}/{vacc:.3f}"
        )
        if vacc > best_acc:
            best_acc = vacc
            torch.save(model.state_dict(), model_path)
        if es.step(vacc):
            break

    # Phase 2 — unfreeze parts
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
            f"[{day_json_path.stem}][{backbone_key}][P2][{epoch:03d}] loss {tl:.4f}/{vl:.4f} acc {tacc:.3f}/{vacc:.3f}"
        )
        if vacc > best_acc:
            best_acc = vacc
            torch.save(model.state_dict(), model_path)
        if es.step(vacc):
            break

    # Curves
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

    # Evaluate best checkpoint
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))

    _, _, val_acc, val_f1, _ = evaluate_on_loader(model, val_loader)
    with (model_dir / "metrics_val.json").open("w") as f:
        json.dump(
            {
                "day": day_json_path.stem,
                "split": "val",
                "accuracy": float(val_acc),
                "f1": float(val_f1),
                "n": int(len(y_val)),
                "batch_size": int(val_bs),
            },
            f,
            indent=2,
        )

    preds_bin, trues, test_acc, test_f1, _ = evaluate_on_loader(model, test_loader)
    day_no = day_to_int(day_json_path.stem)

    test_metrics = {
        "day": day_json_path.stem,
        "day_no": day_no,
        "split": "test",
        "accuracy": float(test_acc),
        "f1": float(test_f1),
        "val_accuracy_for_selection": float(best_acc),
        "val_n": int(len(y_val)),
        "test_n": int(len(trues)),
        "actual_good": int(np.array(trues).sum()),
        "predicted_good": int(np.array(preds_bin).sum()),
        "batch_size_train": int(train_bs),
        "batch_size_valtest": int(val_bs),
        "backbone_key": backbone_key,
    }
    with (model_dir / "metrics_test.json").open("w") as f:
        json.dump(test_metrics, f, indent=2)
    print(
        f"📝 Saved metrics → {model_dir / 'metrics_val.json'} and {model_dir / 'metrics_test.json'}"
    )

    return {
        "day": day_json_path.stem,
        "day_no": day_no,
        "backbone_key": backbone_key,
        "val_accuracy": float(best_acc),
        "test_accuracy": float(test_acc),
        "test_f1": float(test_f1),
        "val_num": int(len(y_val)),
        "test_num": int(len(trues)),
        "test_actual_good": int(np.array(trues).sum()),
        "test_pred_good": int(np.array(preds_bin).sum()),
    }


# ---------- Orchestration ----------
def main():
    set_seed()

    parser = argparse.ArgumentParser()
    # Directories
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory with per-day JSONs (Dy*.json)",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=DEFAULT_OUT_ROOT,
        help="Root directory to save outputs",
    )
    parser.add_argument(
        "--exp-name",
        type=str,
        default=None,
        help="Experiment name (subfolder under out-root). If omitted, a timestamped name is used.",
    )

    # Batching & splits
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
        "--test-frac", type=float, default=0.10, help="Fraction for test split"
    )
    parser.add_argument(
        "--val-frac",
        type=float,
        default=0.10,
        help="Overall fraction for validation split",
    )

    # Augment control
    parser.add_argument(
        "--augment-level",
        choices=["none", "geom", "geom_photo"],
        default="geom",
        help="Augmentation level for TRAIN split",
    )
    parser.add_argument(
        "--duplicate-train-hflip",
        action="store_true",
        help="Duplicate each training sample with a horizontally flipped copy (keep both).",
    )

    # Optional magnitudes
    parser.add_argument(
        "--rot-deg", type=float, default=15.0, help="Geom: RandomRotation degrees"
    )
    parser.add_argument(
        "--rrc-min-scale",
        type=float,
        default=0.9,
        help="Geom: RandomResizedCrop min scale",
    )
    parser.add_argument(
        "--rrc-max-scale",
        type=float,
        default=1.1,
        help="Geom: RandomResizedCrop max scale",
    )
    parser.add_argument(
        "--affine-deg", type=float, default=5.0, help="Geom: RandomAffine degrees"
    )
    parser.add_argument(
        "--affine-trans",
        type=float,
        default=0.02,
        help="Geom: RandomAffine translate fraction",
    )
    parser.add_argument(
        "--affine-min-scale",
        type=float,
        default=0.98,
        help="Geom: RandomAffine min scale",
    )
    parser.add_argument(
        "--affine-max-scale",
        type=float,
        default=1.02,
        help="Geom: RandomAffine max scale",
    )

    parser.add_argument(
        "--photo-noise-std", type=float, default=0.01, help="Photo: Gaussian noise std"
    )

    # Normalization
    parser.add_argument(
        "--norm-mean",
        type=float,
        nargs=3,
        default=[0.5, 0.5, 0.5],
        help="Normalization mean (RGB)",
    )
    parser.add_argument(
        "--norm-std",
        type=float,
        nargs=3,
        default=[0.5, 0.5, 0.5],
        help="Normalization std (RGB)",
    )

    args = parser.parse_args()

    # Decide OUT_ROOT: out-root / exp-name (or timestamp)
    if args.exp_name is None:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = f"exp_{ts}_{args.augment_level}"
    else:
        exp_name = args.exp_name
    OUT_ROOT = args.out_root / exp_name
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    # Basic prints
    train_bs = int(args.batch_size)
    val_bs = int(args.val_batch_size) if args.val_batch_size is not None else train_bs
    test_frac = float(args.test_frac)
    val_frac = float(args.val_frac)

    assert 0.0 < test_frac < 0.5, "test-frac must be in (0, 0.5)"
    assert 0.0 < val_frac < 0.5, "val-frac must be in (0, 0.5)"
    assert val_frac + test_frac < 0.9, "Sum of val-frac and test-frac too large."
    print(f"📁 Data dir: {args.data_dir}")
    print(f"📂 Output root: {OUT_ROOT}")
    print(f"🧪 Batch sizes — train: {train_bs}, val/test: {val_bs}")
    print(
        f"🔀 Split fractions — train: {1.0 - test_frac - val_frac:.2f}, val: {val_frac:.2f}, test: {test_frac:.2f}"
    )
    print(f"🖼️ Target size (HxW): {TARGET_SIZE}")
    print(
        f"🧩 Augment level: {args.augment_level} | Duplicate HFlip: {args.duplicate_train_hflip}"
    )

    geom_cfg = {
        "rotation_degrees": float(args.rot_deg),
        "rrc_scale": (float(args.rrc_min_scale), float(args.rrc_max_scale)),
        "rrc_ratio": (TARGET_SIZE[1] / TARGET_SIZE[0], TARGET_SIZE[1] / TARGET_SIZE[0]),
        "affine_degrees": float(args.affine_deg),
        "affine_translate": (float(args.affine_trans), float(args.affine_trans)),
        "affine_scale": (float(args.affine_min_scale), float(args.affine_max_scale)),
    }
    photo_cfg = {
        "gaussian_noise_std": float(args.photo_noise_std),
    }

    # Train across days/backbones, pick best per day via VAL accuracy
    per_day_best = {}
    for json_file in sorted(
        args.data_dir.glob("Dy*.json"), key=lambda p: day_to_int(p.stem)
    ):
        day = json_file.stem
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
                augment_level=args.augment_level,
                duplicate_hflip_train=args.duplicate_train_hflip,
                geom_cfg=geom_cfg,
                photo_cfg=photo_cfg,
                norm_mean=tuple(args.norm_mean),
                norm_std=tuple(args.norm_std),
                out_root=OUT_ROOT,
            )
            if res is None:
                continue
            if (best is None) or (res["val_accuracy"] > best["val_accuracy"]):
                best = res
        if best:
            per_day_best[day] = best
            print(
                f"✅ Best for {day} (by VAL): {best['backbone_key']} | VAL acc={best['val_accuracy']:.3f} | TEST acc={best['test_accuracy']:.3f}, f1={best['test_f1']:.3f}"
            )
        else:
            print(f"⚠ No valid result for {day}")

    if not per_day_best:
        print("❌ No days produced results; aborting summary.")
        return

    # ---- 4-column table (TEST)
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

    # Save CSV
    import csv

    table_path = OUT_ROOT / "day_summary.csv"
    with table_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["Day No", "Num in Sample", "Actual Good", "Predicted Good"]
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"🧾 Saved table → {table_path}")

    # Chart: accuracy vs day
    xs = [per_day_best[d]["day_no"] for d in days_sorted]
    ys = [per_day_best[d]["test_accuracy"] for d in days_sorted]
    plt.figure(figsize=(8, 4))
    sns.lineplot(x=xs, y=ys, marker="o")
    plt.xlabel("Day")
    plt.ylabel("Accuracy (test)")
    plt.title("Per-day Test Accuracy (Best Model per Day)")
    plt.xticks(xs)
    plt.ylim(0.0, 1.0)
    chart_path = OUT_ROOT / "accuracy_by_day.png"
    plt.tight_layout()
    plt.savefig(chart_path)
    plt.close()
    print(f"📊 Saved accuracy chart → {chart_path}")

    # Final summary JSON
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
        "augment_level": args.augment_level,
        "duplicate_train_hflip": bool(args.duplicate_train_hflip),
        "geom_cfg": geom_cfg,
        "photo_cfg": photo_cfg,
        "norm_mean": tuple(args.norm_mean),
        "norm_std": tuple(args.norm_std),
        "data_dir": str(args.data_dir),
        "out_root": str(OUT_ROOT),
    }
    summary_path = OUT_ROOT / "final_test_summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"✅ Saved final test summary → {summary_path}")

    # Print the 4-column table
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
