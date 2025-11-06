#!/usr/bin/env python3
# Standard imports
import argparse
import csv
import dataclasses
import datetime
import json
import random
import re
from collections import defaultdict
from pathlib import Path

# Third party imports
import matplotlib.pyplot as plt
import numpy as np
import timm
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

# -------- Config defaults --------
BACKBONES = {
    "vit": "vit_base_patch16_224",   # we will set img_size=(384,512) at create_model
    "resnet": "resnet50",
    "cnn": "cnn",
}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------------------------------------

# ---------- Classes ----------
@dataclasses.dataclass
class Config:
    out_dir: Path = dataclasses.field(metadata={
        "help": "Path to output directory where results will be saved"
    })
    epoch1: int = dataclasses.field(default=100, metadata={
        "help": "Number of training epochs for phase 1 (frozen backbone)"
    })
    epoch2: int = dataclasses.field(default=300, metadata={
        "help": "Number of training epochs for phase 2 (unfrozen backbone)"
    })
    batch_size: int = dataclasses.field(default=16, metadata={
        "help": "Training batch size"
    })
    val_batch_size: int = dataclasses.field(default=None, metadata={
        "help": "Validation/Test batch size (defaults to training batch size)"
    })
    test_frac: float = dataclasses.field(default=0.1, metadata={
        "help": "Fraction of data used for testing"
    })
    val_frac: float = dataclasses.field(default=0.1, metadata={
        "help": "Fraction of data used for validation"
    })
    use_mask: bool = dataclasses.field(default=False, metadata={
        "help": "Include mask tensors and a mask branch in the classifier"
    })
    input_path_key: str = dataclasses.field(default="img_path", metadata={
        "help": "Which JSON dataclasses.field to use as the primary image input ('img_path' or 'overlay_path')"
    })
    target_width: int = dataclasses.field(default=512, metadata={
        "help": "Target input image width (pixels)"
    })
    target_height: int = dataclasses.field(default=384, metadata={
        "help": "Target input image height (pixels)"
    })
    num_workers: int = dataclasses.field(default=0, metadata={
        "help": "Number of subprocesses for data loading (0 = main process)"
    })
    seed: int = dataclasses.field(default=1, metadata={
        "help": "Random seed for reproducibility"
    })

    def __post_init__(self):
        # Basic validation / normalization
        if not (0.0 < self.test_frac < 0.5):
            raise ValueError("test-frac must be in (0, 0.5)")
        if not (0.0 < self.val_frac < 0.5):
            raise ValueError("val-frac must be in (0, 0.5)")
        if not (self.val_frac + self.test_frac < 0.9):
            raise ValueError("Sum of val-frac and test-frac too large.")
        if self.batch_size <= 0:
            raise ValueError("train_bs must be > 0")
        # Set up
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.json_file = self.out_dir.parent.joinpath("json", "image_classifier.json")
        self.val_batch_size = int(self.val_batch_size) if self.val_batch_size is not None else self.batch_size
        self.target_size: tuple = (self.target_width, self.target_height)

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
    """Dataset that can optionally return mask tensors alongside images."""

    def __init__(self, img_paths, labels, target_size, mask_paths=None, augment=False, use_mask=False):
        self.img_paths = img_paths
        self.labels = labels
        self.mask_paths = mask_paths
        self.augment = augment
        self.use_mask = use_mask
        t = [T.Resize(target_size)]
        if augment:
            t += [
                T.RandomHorizontalFlip(0.5),
                T.ColorJitter(0.2, 0.2, 0.2, 0.1),
            ]
        t += [T.ToTensor()]
        self.t_img = T.Compose(t)

        if self.use_mask:
            if self.mask_paths is None:
                raise ValueError("mask_paths must be provided when use_mask=True")
            self.t_mask = T.Compose([
                T.Resize(target_size, interpolation=T.InterpolationMode.NEAREST),
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


class SmallCNNBackbone(nn.Module):
    """Simple CNN feature extractor used when backbone_key == 'cnn'."""

    def __init__(self, out_dim=256):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, out_dim),
            nn.ReLU(inplace=True),
        )
        self.out_dim = out_dim

    def forward(self, x):
        x = self.features(x)
        return self.proj(x)


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

# ---------- Model ----------
class ImageOnlyClassifier(nn.Module):
    def __init__(self, backbone_key, backbone_name, target_size, use_mask=False):
        super().__init__()
        self.use_mask = use_mask
        self.backbone_key = backbone_key

        if backbone_key == "cnn":
            self.backbone = SmallCNNBackbone()
            out_dim = self.backbone.out_dim
            self._is_timm = False
        else:
            # If it's a ViT-like model, tell timm the image size.
            # timm will handle positional embedding interpolation for non-224 sizes.
            extra_args = {}
            if "vit" in backbone_name:
                extra_args["img_size"] = target_size  # (H, W) tuple is supported

            self.backbone = timm.create_model(
                backbone_name,
                pretrained=True,
                num_classes=0,          # feature extractor
                global_pool="avg",
                **extra_args
            )
            out_dim = self.backbone.num_features
            self._is_timm = True

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
        if not self._is_timm:
            return
        for name, p in self.backbone.named_parameters():
            # unfreeze blocks/layers for fine-tuning
            if "blocks." in name or "layer" in name:
                p.requires_grad = True

    def forward(self, img, mask=None):
        f = self.backbone(img)
        if self.use_mask:
            if mask is None:
                raise ValueError("mask tensor must be provided when use_mask=True")
            f_mask = self.mask_branch(mask)
            f = torch.cat([f, f_mask], dim=1)
        return self.classifier(f).squeeze(1)

# ---------- Utils ----------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def print_config_stats(cfg):
    """Print information about configuration data."""
    print(f"🧪 Using batch sizes — train: {cfg.batch_size}, val/test: {cfg.val_batch_size}")
    print(f"🔀 Split fractions — train: {1.0 - cfg.test_frac - cfg.val_frac:.2f}, val: {cfg.val_frac:.2f}, test: {cfg.test_frac:.2f}")
    print(f"🖼️ Target size (HxW): {cfg.target_size}")
    print(f"🗂️ Input field: {cfg.input_path_key}; masks enabled: {cfg.use_mask}")

def create_args() -> argparse.ArgumentParser:
    """Create an ArgumentParser from the Config dataclass."""
    parser = argparse.ArgumentParser(description="Run image classifier on organoid images")

    for field in dataclasses.fields(Config):
        # Build argument flag and help message
        flags = [f"--{field.name.replace('_', '-')}"]

        kwargs = {
            "help": field.metadata.get("help", ""),
            "default": field.default
        }

        # Determine argument type
        if field.type == bool:
            kwargs["action"] = "store_true" if field.default is False else "store_false"
        else:
            kwargs["type"] = field.type

        parser.add_argument(*flags, **kwargs)

    return parser

def get_args():
    """Retrieve and return command line arguments via the Config class"""
    arg_parser = create_args()
    args = arg_parser.parse_args()
    for key,val in vars(args).items():
        print(f"{key}: {val}")
    cfg = Config(**vars(args))
    return cfg

def day_to_int(day_str: str) -> int:
    # "Dy28" -> 28, fallback -1
    m = re.search(r"[Dd][Yy](\d+)", day_str)
    return int(m.group(1)) if m else -1

# ---------- Train/Eval ----------
def collect_results(cfg):
    # Load JSON file data
    with open(cfg.json_file) as jf:
        json_data = json.load(jf)

    # Collect results: pick the best backbone per day by **validation accuracy**
    per_day_best = {}
    per_model_results = {bk: {} for bk in BACKBONES}
    for day, data in json_data["records"].items():
        if not data:
            print(f"⚠ Skipping {day} — no records")
            return None

        best = None
        for backbone_key, backbone_name in BACKBONES.items():
            res = run_training_for_day(day, data, backbone_key, backbone_name, cfg=cfg)
            if res is None:
                continue
            per_model_results[backbone_key][day] = res
            if (best is None) or (res["val_accuracy"] > best["val_accuracy"]):
                best = res

        if best:
            per_day_best[day] = best
            print(f"✅ Best for {day} (by VAL): {best['backbone_key']} | val acc={best['val_accuracy']:.3f} | TEST acc={best['test_accuracy']:.3f}, f1={best['test_f1']:.3f}")
        else:
            print(f"⚠ No valid result for {day}")

    if not per_day_best:
        print("❌ No days produced results; aborting summary.")
        return None

    return per_day_best, per_model_results

def run_training_for_day(day: str, data: dict, backbone_key: str,
                         backbone_name: str, cfg:Config):
    """Train + validate with small val/test; select by VAL acc, report on TEST."""
    labels = data.get("label", [])
    imgs = data.get(cfg.input_path_key, [])
    masks = data.get("mask_path", [])
    # Filter out entries with missing files (and record details)
    labels, imgs, masks = filter_labels_images_masks(day, labels, imgs, masks,
                                                     backbone_key, cfg)

    # ---- Split: first cut TEST (test_frac), then VAL to reach overall val_frac
    if cfg.use_mask:
        X_tmp, X_test, M_tmp, M_test, y_tmp, y_test = train_test_split(
            imgs, masks, labels, test_size=cfg.test_frac, stratify=labels, random_state=cfg.seed
        )
    else:
        X_tmp, X_test, y_tmp, y_test = train_test_split(
            imgs, labels, test_size=cfg.test_frac, stratify=labels, random_state=cfg.seed
        )
    val_frac_cond = cfg.val_frac / (1.0 - cfg.test_frac)  # conditional fraction from remaining
    if cfg.use_mask:
        X_tr, X_val, M_tr, M_val, y_tr, y_val = train_test_split(
            X_tmp, M_tmp, y_tmp, test_size=val_frac_cond, stratify=y_tmp, random_state=cfg.seed
        )
    else:
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_tmp, y_tmp, test_size=val_frac_cond, stratify=y_tmp, random_state=cfg.seed
        )

    # Class weights (train only)
    weights = compute_class_weight("balanced", classes=np.unique(y_tr), y=y_tr)
    class_weights = {int(k): float(w) for k, w in zip(np.unique(y_tr), weights)}

    # Loaders (configurable batch sizes; val/test use validation_batch_size)
    mask_path = M_tr if cfg.use_mask else None
    train_loader = make_loader(X_tr, y_tr, mask_paths=mask_path, augment=False,
                               batch_size=cfg.batch_size, cfg=cfg
    )
    val_loader = make_loader(X_val, y_val, mask_paths=mask_path, augment=False,
                             batch_size=cfg.val_batch_size, cfg=cfg
    )
    test_loader = make_loader(X_test, y_test, mask_paths=mask_path, augment=False,
                              batch_size=cfg.val_batch_size, cfg=cfg
    )

    # Define model
    model = ImageOnlyClassifier(backbone_key, backbone_name, cfg.target_size, use_mask=cfg.use_mask).to(DEVICE)
    model_dir = cfg.out_dir / backbone_key / day
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "model.pth"

    # Run training
    history, best_acc = run_phases(model, model_path, backbone_key, backbone_name,
                                   day, train_loader, val_loader,
                                   class_weights, cfg)

    # Save per-day training curves
    plot_training_curver(history, model_dir)

    # ---- Evaluate with best VAL checkpoint
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))

    # Val metrics (record only; NOT used for final reporting)
    val_metrics = get_validation_metrics(model, y_val, model_dir, val_loader,
                                         day, cfg)

    # Test metrics（final reporting）
    test_metrics = get_test_metrics(model, y_val, model_dir, test_loader,
                                    day, best_acc, backbone_key, cfg)

    # Return: choose by val, report test
    return {
        "day": day,
        "day_no": test_metrics["day_no"],
        "backbone_key": backbone_key,
        "val_accuracy": float(best_acc),     # selection metric
        "test_accuracy": test_metrics["accuracy"],    # reporting metric
        "test_f1": test_metrics["f1"],
        "val_roc_auc": val_metrics["roc_auc"],
        "test_roc_auc": test_metrics["roc_auc"],
        "val_num": int(len(y_val)),
        "test_num": test_metrics["test_n"],
        "test_actual_good": test_metrics["actual_good"],
        "test_pred_good": test_metrics["predicted_good"],
    }

def get_labels_images_masks(records, day_json_path, cfg):
    """Retrieve labels, images, and masks from records."""
    # labels
    label_map = {"Accepted": 1, "Not Accepted": 0}
    try:
        labels = np.array([label_map[r["label"]] for r in records], dtype=int)
    except KeyError:
        print(f"⚠ Skipping {day_json_path.name} — missing 'label' field")
        return None

    try:
        imgs = [r[cfg.input_path_key] for r in records]
    except KeyError:
        print(f"⚠ Skipping {day_json_path.name} — missing '{cfg.input_path_key}' field")
        return None

    if cfg.use_mask:
        try:
            masks = [r["mask_path"] for r in records]
        except KeyError:
            print(f"⚠ Skipping {day_json_path.name} — missing 'mask_path' required by --use-mask")
            return None
    else:
        masks = None
    return labels, imgs, masks

def filter_labels_images_masks(day, labels, imgs, masks, backbone_key, cfg):
    """Filter entries with missing files (and record dteails)."""
    filtered_imgs, filtered_labels = [], []
    filtered_masks = [] if cfg.use_mask else None
    missing_records = []
    for idx, img_path in enumerate(imgs):
        img_path = Path(str(img_path))
        mask_path = Path(str(masks[idx])) if (cfg.use_mask and masks is not None) else None
        if not img_path.exists():
            missing_records.append({"img_path": str(img_path),"mask_path": str(mask_path) if mask_path is not None else "","reason": "missing_image"})
            continue
        if cfg.use_mask and (mask_path is None or not mask_path.exists()):
            missing_records.append({"img_path": str(img_path),"mask_path": str(mask_path) if mask_path is not None else "","reason": "missing_mask"})
            continue
        filtered_imgs.append(str(img_path))
        filtered_labels.append(labels[idx])
        if cfg.use_mask:
            filtered_masks.append(str(mask_path))

    if missing_records:
        write_missing_csv(day, cfg, backbone_key, missing_records)

    if not filtered_labels:
        print(f"⚠ Skipping {day} — no valid samples after filtering missing files")
        return None

    imgs = np.array(filtered_imgs)
    labels = np.array(filtered_labels)
    if cfg.use_mask:
        masks = np.array(filtered_masks)
    else:
        masks = None
    return filtered_labels, filtered_imgs, filtered_masks

def write_missing_csv(day, cfg, backbone_key, missing_records):
    log_dir = cfg.out_dir / backbone_key
    log_dir.mkdir(parents=True, exist_ok=True)

    missing_csv = log_dir / "missing_files.csv"
    fieldnames = ["img_path", "mask_path", "reason"]

    combined_rows = []
    if missing_csv.exists():
        with missing_csv.open("r", newline="") as fh:
            reader = csv.DictReader(fh)
            combined_rows.extend(reader)

    combined_rows.extend(missing_records)

    with missing_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(combined_rows)

    print(f"⚠ {day}: skipped {len(missing_records)} entries due to missing files (details → {missing_csv})")

def make_loader(imgs, labels, augment, batch_size, cfg, mask_paths=None):
    ds = OrganoidDataset(imgs, labels, target_size=cfg.target_size, mask_paths=mask_paths, augment=augment, use_mask=cfg.use_mask)
    return DataLoader(ds, batch_size=batch_size, shuffle=augment, num_workers=cfg.num_workers)

def run_phases(model, model_path, backbone_key, backbone_name, day,
               train_loader, val_loader, class_weights, cfg):
    """Run epochs for each phase to train the data."""
    # Optimizer
    opt = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    es = EarlyStopping(patience=20)
    history = defaultdict(list)
    best_acc = -np.inf

    # Phase 1 — frozen
    for epoch in range(cfg.epoch1):
        tl, tacc, _, _ = epoch_loop(model, train_loader, opt, class_weights, train=True, use_mask=cfg.use_mask)
        vl, vacc, _, _ = epoch_loop(model, val_loader,   opt, class_weights, train=False, use_mask=cfg.use_mask)
        history["train_loss"].append(tl); history["val_loss"].append(vl)
        history["train_acc"].append(tacc); history["val_acc"].append(vacc)
        print(f"[{day}][{backbone_key}][P1][{epoch:02d}][bs={cfg.batch_size}/{cfg.val_batch_size}] loss {tl:.4f}/{vl:.4f} acc {tacc:.3f}/{vacc:.3f}")
        if vacc > best_acc:
            best_acc = vacc
            torch.save(model.state_dict(), model_path)
        if es.step(vacc):
            break

    # Phase 2 — unfreeze partial backbone
    model.unfreeze_backbone()
    opt = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    es = EarlyStopping(patience=30)
    for epoch in range(cfg.epoch2):
        tl, tacc, _, _ = epoch_loop(model, train_loader, opt, class_weights, train=True, use_mask=cfg.use_mask)
        vl, vacc, _, _ = epoch_loop(model, val_loader,   opt, class_weights, train=False, use_mask=cfg.use_mask)
        history["train_loss"].append(tl); history["val_loss"].append(vl)
        history["train_acc"].append(tacc); history["val_acc"].append(vacc)
        print(f"[{day}][{backbone_key}][P2][{epoch:03d}][bs={cfg.batch_size}/{cfg.val_batch_size}] loss {tl:.4f}/{vl:.4f} acc {tacc:.3f}/{vacc:.3f}")
        if vacc > best_acc:
            best_acc = vacc
            torch.save(model.state_dict(), model_path)
        if es.step(vacc):
            break
    return history, best_acc

def epoch_loop(model, loader, optimizer, class_weights, train=True, use_mask=False):
    model.train() if train else model.eval()
    bce = nn.BCEWithLogitsLoss(reduction="none")
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

def plot_training_curver(history, model_dir):
    """Plot model training curves and save to output directory."""
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1); plt.plot(history["train_acc"], label="Train"); plt.plot(history["val_acc"], label="Val"); plt.title("Accuracy"); plt.legend()
    plt.subplot(1, 2, 2); plt.plot(history["train_loss"], label="Train"); plt.plot(history["val_loss"], label="Val"); plt.title("Loss"); plt.legend()
    plt.tight_layout()
    plt.savefig(model_dir / "training_curves.png")
    plt.close()
    print(f"📈 Saved curves → {model_dir/'training_curves.png'}")

def get_validation_metrics(model, y_val, model_dir, val_loader, day, cfg):
    """Calculate and return valiation metrics."""
    _, val_trues, val_acc, val_f1, val_probs = evaluate_on_loader(model, val_loader, use_mask=cfg.use_mask)
    # Safely compute ROC AUC (may be undefined if only one class present)
    try:
        val_roc_auc = float(roc_auc_score(val_trues, val_probs))
    except Exception:
        val_roc_auc = None
    val_pr_auc = float(average_precision_score(val_trues, val_probs)) if len(val_trues) > 0 else None
    val_metrics = {
        "day": day,
        "split": "val",
        "accuracy": float(val_acc),
        "f1": float(val_f1),
        "roc_auc": val_roc_auc,
        "pr_auc": val_pr_auc,
        "n": int(len(y_val)),
        "batch_size": int(cfg.val_batch_size),
        "input_key": cfg.input_path_key,
        "use_mask": cfg.use_mask,
    }
    with (model_dir / "metrics_val.json").open("w") as f:
        json.dump(val_metrics, f, indent=2)
    return val_metrics

def get_test_metrics(model, y_val, model_dir, test_loader, day, best_acc, backbone_key, cfg):
    """Calcualte and return test metrics."""
    preds_bin, trues, test_acc, test_f1, test_probs = evaluate_on_loader(model, test_loader, use_mask=cfg.use_mask)
    # Safely compute test ROC AUC
    try:
        test_roc_auc = float(roc_auc_score(trues, test_probs))
    except Exception:
        test_roc_auc = None
    test_pr_auc = float(average_precision_score(trues, test_probs)) if len(trues) > 0 else None
    day_no = day_to_int(day)
    num_in_sample = int(len(trues))
    actual_good = int(trues.sum())
    predicted_good = int(preds_bin.sum())

    test_metrics = {
        "day": day,
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
        "batch_size_train": int(cfg.batch_size),
        "batch_size_valtest": int(cfg.val_batch_size),
        "backbone_key": backbone_key,
        "input_key": cfg.input_path_key,
        "use_mask": cfg.use_mask,
    }
    with (model_dir / "metrics_test.json").open("w") as f:
        json.dump(test_metrics, f, indent=2)
    print(f"📝 Saved metrics → {model_dir/'metrics_val.json'} and {model_dir/'metrics_test.json'}")

    return test_metrics

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

def build_results_table(per_day_best, cfg):
    """Build and save a table of results."""
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

    # Save CSV table (exactly 4 columns)
    table_path = cfg.out_dir / "day_summary.csv"
    with table_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["Day No", "Num in Sample", "Actual Good", "Predicted Good"]
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"🧾 Saved table → {table_path}")
    return rows

def create_summmary(per_model_results, rows, cfg):
    """Create a summary of results and produce plots and save JSON."""
    day_numbers = {}
    for day_res in per_model_results.values():
        for day, res in day_res.items():
            day_numbers[day] = res["day_no"]

    if day_numbers:
        unique_day_nos = sorted(set(day_numbers.values()))
        plot_metric("test_accuracy", "Accuracy (test)",
                    "Per-day Test Accuracy by Backbone", "accuracy_by_model.png",
                      per_model_results, day_numbers, unique_day_nos, cfg.out_dir)
        plot_metric("test_f1", "F1 score (test)", "Per-day Test F1 by Backbone",
                     "f1_by_model.png", per_model_results, day_numbers,
                     unique_day_nos, cfg.out_dir)
        plot_metric("test_roc_auc", "ROC AUC (test)", "Per-day Test ROC AUC by Backbone",
                    "rocauc_by_model.png", per_model_results, day_numbers,
                    unique_day_nos, cfg.out_dir)

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
        "batch_size_train": int(cfg.batch_size),
        "batch_size_valtest": int(cfg.val_batch_size),
        "split_fractions": {
            "train": float(1.0 - cfg.test_frac - cfg.val_frac),
            "val": float(cfg.val_frac),
            "test": float(cfg.test_frac),
        }
    }
    summary_path = cfg.out_dir / "final_test_summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"✅ Saved final test summary → {summary_path}")

    # ---- Also print the 4-column table to stdout
    print("\n=== Summary Table (TEST) ===")
    print(f"{'Day No':>6} | {'Num in Sample':>13} | {'Actual Good':>11} | {'Predicted Good':>14}")
    print("-" * 54)
    for row in rows:
        print(f"{row['Day No']:>6} | {row['Num in Sample']:>13} | {row['Actual Good']:>11} | {row['Predicted Good']:>14}")

def plot_metric(metric_key, ylabel, title, filename, per_model_results,
                day_numbers, unique_day_nos, out_dir):
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
                print(f"📊 Saved {title.lower()} → {out_path}")
            plt.close()

# ---------- Orchestration ----------
def main():
    start = datetime.datetime.now()
    # Set up for model training
    cfg = get_args()
    set_seed(cfg.seed)
    print_config_stats(cfg)

    # Collect results: pick the best backbone per day by **validation accuracy**
    per_day_best, per_model_results = collect_results(cfg)

    # ---- Build 4-column table (based on TEST)
    rows = build_results_table(per_day_best, cfg)

    # ---- Per-model charts and summary (no overall-best aggregation)
    create_summmary(per_model_results, rows, cfg)

    end = datetime.datetime.now()
    print(f"Execution time: {end - start}")

if __name__ == "__main__":
    main()
