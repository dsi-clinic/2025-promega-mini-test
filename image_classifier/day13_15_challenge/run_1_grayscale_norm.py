#!/usr/bin/env python3
"""
Task 1: Run effnet_ts and per_day for Days 13 and 15 only, with normalization
from grayscale-derived mean/std (not ImageNet). Uses GPU if available.
Run: python run_1_grayscale_norm.py
Requires: 01_compute_grayscale_mean_std.py run first (or we compute on the fly).
"""

import os
import sys
import json
from pathlib import Path

CHALLENGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = CHALLENGE_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(CHALLENGE_DIR))

# Env for BASE_PATH etc.
_env = REPO_ROOT / ".env"
if _env.exists():
    with open(_env) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ[k] = v
# Required by config.py when train_base_model is imported
_root = str(REPO_ROOT)
_defaults = [
    ("BASE_PATH", "/net/projects2/promega/data-analysis"),
    ("OUTPUT_FOLDER", "/net/projects2/promega/data-analysis/output"),
    ("RAW_IMAGE_DATA", _root),
    ("IMAGE_VERIFICATION_FORM", _root),
    ("PLOTS_FOLDER", _root),
    ("LOGS_FOLDER", _root),
    ("NPY_OUTPUTS", _root),
    ("PREDICTIONS_DIR", _root),
    ("SURVEY_RESULTS", _root),
    ("MANUAL_MASKS_DIR", _root),
    ("META_FILE", _root),
    ("RAW_IMAGE_MAPPING_JSON", _root),
    ("TARGET_WIDTH", "384"),
    ("TARGET_HEIGHT", "512"),
    ("TRAIN_RESIZED_DIR", _root),
    ("TRAIN_MANUAL_MAPPING_DIR", _root),
    ("TRAIN_MANUAL_PROCESSED_DIR", _root),
    ("TRAIN_SPLITS_DIR", _root),
    ("INFER_RESIZED_DIR", _root),
    ("INFER_MAPPING_TOTAL_JSON", _root),
    ("MANUAL_THRESHOLD_MAPPING", _root),
]
for k, v in _defaults:
    os.environ.setdefault(k, v)

import numpy as np
import torch
import torch.nn as optim
from torch.utils.data import DataLoader
from torchvision import transforms as T
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from image_classifier.cnn_lstm.load_split_data import load_split_data
from image_classifier.cnn_lstm.train_base_model import (
    SingleDayOrganoidDataset,
    BaselineEfficientNet,
    train_for_day,
    set_seed as set_seed_base,
    TARGET_SIZE,
    BATCH_SIZE,
    NUM_WORKERS,
    MAX_EPOCHS,
    PATIENCE,
    LR,
    GRAD_CLIP,
    SEED as BASE_SEED,
)
from image_classifier.cnn_lstm.train_organoid_lstm_single_phase import (
    collate_variable_length,
)
from image_classifier.cnn_lstm.train_temporal_ablation_attn import (
    OrganoidCNN_TAtt,
    set_seed as set_seed_attn,
    evaluate_binary as evaluate_effnet_ts,
    BATCH_SIZE as EFFNET_TS_BATCH,
    NUM_WORKERS as EFFNET_NUM_WORKERS,
    MAX_EPOCHS as EFFNET_MAX_EPOCHS,
    WARMUP_EPOCHS,
    LR_HEAD,
    LR_CNN_UNFREEZE,
    GRAD_CLIP as EFFNET_GRAD_CLIP,
    PATIENCE as EFFNET_PATIENCE,
    ATTN_DROPOUT,
)
from dataset_grayscale_norm import (
    compute_grayscale_mean_std,
    OrganoidTimeSeriesDatasetGrayscaleNorm,
    make_single_day_grayscale_norm_dataset,
)

SEED = 42
DAYS = [13.0, 15.0]
THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9]


def filter_ids_with_frames_up_to_day(organoid_ids, series_metadata, max_day):
    out = []
    for oid in organoid_ids:
        days = series_metadata.get(oid, {}).get("days", [])
        if any(d <= max_day for d in days):
            out.append(oid)
    return out


def metrics_at_threshold(probs, labels, threshold):
    preds = (probs >= threshold).astype(int)
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    if cm.size == 4:
        TN, FP, FN, TP = cm.ravel()
    else:
        TN, FP, FN, TP = 0, 0, 0, 0
    TNR = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    TPR = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    acc = (TP + TN) / len(labels) if len(labels) > 0 else 0.0
    prec, rec, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )
    balanced_acc = (TNR + TPR) / 2
    return {
        "threshold": float(threshold),
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "TNR": float(TNR),
        "TPR": float(TPR),
        "balanced_acc": float(balanced_acc),
        "TN": int(TN),
        "FP": int(FP),
        "FN": int(FN),
        "TP": int(TP),
    }


def find_best_threshold(probs, labels, min_tnr=0.5):
    best_score, best_thresh, best_metrics = -1, 0.5, None
    for th in np.linspace(0.1, 0.9, 81):
        m = metrics_at_threshold(probs, labels, th)
        if m["TNR"] < min_tnr:
            continue
        score = (m["TNR"] + m["f1"]) / 2
        if score > best_score:
            best_score, best_thresh, best_metrics = score, th, m
    return best_thresh, best_metrics


def run_effnet_ts_grayscale_norm(
    day,
    train_ids,
    val_ids,
    test_ids,
    series_metadata,
    data,
    device,
    base_dir,
    custom_mean,
    custom_std,
):
    from torchvision.transforms import InterpolationMode

    set_seed_attn(SEED)
    out_dir = (
        base_dir / "effnet_ts_grayscale_norm" / f"day_{str(day).replace('.', '_')}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ids_f = filter_ids_with_frames_up_to_day(train_ids, series_metadata, day)
    val_ids_f = filter_ids_with_frames_up_to_day(val_ids, series_metadata, day)
    test_ids_f = filter_ids_with_frames_up_to_day(test_ids, series_metadata, day)
    if len(train_ids_f) == 0:
        print(f"No train organoids with frames <= day {day}, skipping effnet_ts.")
        return None

    from torchvision import transforms as T_vis

    BILINEAR = InterpolationMode.BILINEAR
    train_tf = T_vis.Compose(
        [
            T_vis.Resize((384, 384), interpolation=BILINEAR),
            T_vis.RandomRotation(degrees=15, fill=128),
            T_vis.RandomHorizontalFlip(p=0.5),
            T_vis.RandomVerticalFlip(p=0.5),
            T_vis.RandomResizedCrop(384, scale=(0.9, 1.0)),
            T_vis.ColorJitter(brightness=0.2, contrast=0.2, saturation=0, hue=0),
        ]
    )
    eval_tf = T_vis.Compose([T_vis.Resize((384, 384), interpolation=BILINEAR)])

    train_ds = OrganoidTimeSeriesDatasetGrayscaleNorm(
        train_ids_f,
        series_metadata,
        data,
        custom_mean,
        custom_std,
        transform=train_tf,
        max_day=day,
        image_key="overlay_path",
    )
    val_ds = OrganoidTimeSeriesDatasetGrayscaleNorm(
        val_ids_f,
        series_metadata,
        data,
        custom_mean,
        custom_std,
        transform=eval_tf,
        max_day=day,
        image_key="overlay_path",
    )
    test_ds = OrganoidTimeSeriesDatasetGrayscaleNorm(
        test_ids_f,
        series_metadata,
        data,
        custom_mean,
        custom_std,
        transform=eval_tf,
        max_day=day,
        image_key="overlay_path",
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=EFFNET_TS_BATCH,
        shuffle=True,
        num_workers=EFFNET_NUM_WORKERS or 0,
        collate_fn=collate_variable_length,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=EFFNET_TS_BATCH,
        shuffle=False,
        num_workers=EFFNET_NUM_WORKERS or 0,
        collate_fn=collate_variable_length,
        pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=EFFNET_TS_BATCH,
        shuffle=False,
        num_workers=EFFNET_NUM_WORKERS or 0,
        collate_fn=collate_variable_length,
        pin_memory=(device.type == "cuda"),
    )

    train_labels = [
        1
        if str(series_metadata[oid].get("label", "")).strip().lower()
        in ("good", "acceptable", "accepted")
        else 0
        for oid in train_ids_f
    ]
    n_good, n_bad = sum(train_labels), len(train_labels) - sum(train_labels)
    w_pos = max(n_bad, 1) / max(n_good, 1)
    w_neg = max(n_good, 1) / max(n_bad, 1)
    criterion = torch.nn.BCEWithLogitsLoss(reduction="none")
    model = OrganoidCNN_TAtt(attn_dropout=ATTN_DROPOUT, in_channels=3).to(device)

    def make_opt(lr_cnn, lr_head):
        params_cnn = [p for n, p in model.cnn.named_parameters() if p.requires_grad]
        params_head = [
            p
            for n, p in model.named_parameters()
            if not n.startswith("cnn.") and p.requires_grad
        ]
        groups = []
        if params_cnn:
            groups.append({"params": params_cnn, "lr": lr_cnn})
        if params_head:
            groups.append({"params": params_head, "lr": lr_head})
        return torch.optim.Adam(groups)

    optimizer = make_opt(0.0, LR_HEAD)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    best_val_acc, best_state, train_history, bad_epochs = -1, None, [], 0

    for epoch in range(1, EFFNET_MAX_EPOCHS + 1):
        if epoch == WARMUP_EPOCHS + 1:
            model.unfreeze_last_blocks()
            optimizer = make_opt(LR_CNN_UNFREEZE, LR_HEAD)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=5
            )
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for seqs, days_n, labels, weights, _ in tqdm(
            train_loader, desc=f"Epoch {epoch}", leave=False
        ):
            seqs, days_n = seqs.to(device), days_n.to(device).float()
            labels, weights = labels.to(device).float(), weights.to(device).float()
            optimizer.zero_grad()
            logits, _ = model(seqs, days_n)
            loss_raw = criterion(logits, labels)
            cls_w = labels * w_pos + (1 - labels) * w_neg
            loss = (loss_raw * weights * cls_w).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), EFFNET_GRAD_CLIP)
            optimizer.step()
            running_loss += loss.item() * labels.size(0)
            correct += (
                ((torch.sigmoid(logits) > 0.5).long() == labels.long()).sum().item()
            )
            total += labels.size(0)
        train_acc = correct / max(total, 1)
        val_loss, val_acc, val_prec, val_rec, val_f1, _, _, _, _ = evaluate_effnet_ts(
            model, val_loader, criterion, device
        )
        scheduler.step(val_loss)
        train_history.append(
            {
                "epoch": epoch,
                "train_acc": train_acc,
                "val_acc": val_acc,
                "val_f1": val_f1,
            }
        )
        if val_acc > best_val_acc + 1e-4:
            best_val_acc, best_state = (
                val_acc,
                {k: v.cpu().clone() for k, v in model.state_dict().items()},
            )
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= EFFNET_PATIENCE:
                break

    if best_state is None:
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state, strict=True)
    torch.save(
        {"state_dict": best_state, "best_val_acc": float(best_val_acc)},
        out_dir / "best_model.pth",
    )
    np.save(
        out_dir / "grayscale_mean_std.npy", {"mean": custom_mean, "std": custom_std}
    )

    val_probs, val_labels = [], []
    test_probs, test_labels = [], []
    model.eval()
    with torch.no_grad():
        for seqs, days_n, labels, _, _ in val_loader:
            logits, _ = model(seqs.to(device), days_n.to(device).float())
            val_probs.extend(torch.sigmoid(logits).cpu().numpy().ravel())
            val_labels.extend(labels.cpu().numpy().ravel())
        for seqs, days_n, labels, _, _ in test_loader:
            logits, _ = model(seqs.to(device), days_n.to(device).float())
            test_probs.extend(torch.sigmoid(logits).cpu().numpy().ravel())
            test_labels.extend(labels.cpu().numpy().ravel())
    val_probs, val_labels = np.array(val_probs), np.array(val_labels)
    test_probs, test_labels = np.array(test_probs), np.array(test_labels)
    best_thresh, _ = find_best_threshold(val_probs, val_labels)
    test_at_05 = metrics_at_threshold(test_probs, test_labels, 0.5)
    save_result = {
        "model_type": "effnet_ts_grayscale_norm",
        "day": day,
        "best_val_acc": float(best_val_acc),
        "optimal_threshold": float(best_thresh),
        "test_at_0.5": test_at_05,
        "grayscale_mean": custom_mean.tolist(),
        "grayscale_std": custom_std.tolist(),
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(save_result, f, indent=2)
    print(
        f"Day {day} effnet_ts (grayscale norm): test balanced_acc@0.5 = {test_at_05['balanced_acc']:.3f}"
    )
    return save_result


def run_per_day_grayscale_norm(
    day,
    train_ids,
    val_ids,
    test_ids,
    series_metadata,
    data,
    device,
    base_dir,
    custom_mean,
    custom_std,
):
    set_seed_base(SEED)
    out_dir = base_dir / "per_day_grayscale_norm" / f"day_{str(day).replace('.', '_')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    eval_tf = T.Compose([T.Resize(TARGET_SIZE)])
    train_tf = T.Compose(
        [
            T.Resize(TARGET_SIZE),
            T.RandomHorizontalFlip(0.5),
            T.RandomVerticalFlip(0.5),
            T.ColorJitter(0.2, 0.2, 0.2, 0.1),
        ]
    )
    train_ds, _, _ = make_single_day_grayscale_norm_dataset(
        train_ids,
        val_ids,
        test_ids,
        series_metadata,
        data,
        day,
        custom_mean,
        custom_std,
        train_tf,
        image_key="overlay_path",
        use_rgb_mask=False,
    )
    _, val_ds, test_ds = make_single_day_grayscale_norm_dataset(
        train_ids,
        val_ids,
        test_ids,
        series_metadata,
        data,
        day,
        custom_mean,
        custom_std,
        eval_tf,
        image_key="overlay_path",
        use_rgb_mask=False,
    )

    if len(train_ds) == 0:
        print(f"No train samples for day {day}, skipping per_day.")
        return None

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS or 0,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS or 0,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS or 0,
        pin_memory=True,
    )

    train_labels = [s["label"] for s in train_ds.samples]
    n_good, n_bad = sum(train_labels), len(train_labels) - sum(train_labels)
    if n_good == 0:
        n_good = 1
    if n_bad == 0:
        n_bad = 1
    pos_weight = torch.tensor([n_bad / n_good], device=device)
    model = BaselineEfficientNet(in_channels=3).to(device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    best_val_acc, best_state, bad_epochs = -1, None, 0
    for epoch in range(1, MAX_EPOCHS + 1):
        if epoch == 4:
            model.unfreeze_backbone()
            optimizer = torch.optim.Adam(model.parameters(), lr=LR * 0.1)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=5
            )
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for imgs, labels, _ in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
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
        train_acc = correct / max(1, total)
        model.eval()
        val_probs, val_labels_list = [], []
        with torch.no_grad():
            for imgs, labels, _ in val_loader:
                logits = model(imgs.to(device))
                val_probs.extend(torch.sigmoid(logits).cpu().numpy().ravel())
                val_labels_list.extend(labels.numpy().ravel())
        val_probs = np.array(val_probs)
        val_labels_list = np.array(val_labels_list)
        val_preds = (val_probs >= 0.5).astype(int)
        val_acc = (val_preds == val_labels_list).mean()
        val_loss = 0.0
        scheduler.step(val_loss)
        if val_acc > best_val_acc + 1e-4:
            best_val_acc, best_state = (
                val_acc,
                {k: v.cpu() for k, v in model.state_dict().items()},
            )
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                break

    if best_state is None:
        best_state = {k: v.cpu() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state, strict=True)
    torch.save(
        {"state_dict": best_state, "best_val_acc": float(best_val_acc)},
        out_dir / f"model_day_{day}.pth",
    )

    val_probs, val_labels_list = [], []
    model.eval()
    with torch.no_grad():
        for imgs, labels, _ in val_loader:
            logits = model(imgs.to(device))
            val_probs.extend(torch.sigmoid(logits).cpu().numpy().ravel())
            val_labels_list.extend(labels.numpy().ravel())
    val_probs = np.array(val_probs)
    val_labels_list = np.array(val_labels_list)

    test_probs, test_labels_list = [], []
    with torch.no_grad():
        for imgs, labels, _ in test_loader:
            logits = model(imgs.to(device))
            test_probs.extend(torch.sigmoid(logits).cpu().numpy().ravel())
            test_labels_list.extend(labels.numpy().ravel())
    test_probs = np.array(test_probs)
    test_labels_list = np.array(test_labels_list)
    best_thresh, _ = find_best_threshold(val_probs, val_labels_list)
    test_at_05 = metrics_at_threshold(test_probs, test_labels_list, 0.5)
    save_result = {
        "model_type": "per_day_grayscale_norm",
        "day": day,
        "best_val_acc": float(best_val_acc),
        "optimal_threshold": float(best_thresh),
        "test_at_0.5": test_at_05,
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(save_result, f, indent=2)
    print(
        f"Day {day} per_day (grayscale norm): test balanced_acc@0.5 = {test_at_05['balanced_acc']:.3f}"
    )
    return save_result


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Step 3: Grayscale-norm training using device: {device}", flush=True)

    split_dir = REPO_ROOT / "data_splits"
    train_ids, val_ids, test_ids, series_metadata, data = load_split_data(
        split_dir / "both_train_base.json",
        split_dir / "both_val_base.json",
        split_dir / "both_test_base.json",
    )

    base_dir = CHALLENGE_DIR / "runs_grayscale_norm"
    base_dir.mkdir(parents=True, exist_ok=True)

    for day in DAYS:
        npy_path = CHALLENGE_DIR / f"grayscale_mean_std_day{int(day)}.npy"
        if npy_path.exists():
            d = np.load(npy_path, allow_pickle=True).item()
            custom_mean, custom_std = d["mean"], d["std"]
        else:
            print(f"Computing grayscale mean/std for day {day}...")
            custom_mean, custom_std = compute_grayscale_mean_std(
                train_ids, series_metadata, data, day
            )
            np.save(npy_path, {"mean": custom_mean, "std": custom_std, "max_day": day})
        print(f"Day {day}: mean={custom_mean}, std={custom_std}")

        run_effnet_ts_grayscale_norm(
            day,
            train_ids,
            val_ids,
            test_ids,
            series_metadata,
            data,
            device,
            base_dir,
            custom_mean,
            custom_std,
        )
        run_per_day_grayscale_norm(
            day,
            train_ids,
            val_ids,
            test_ids,
            series_metadata,
            data,
            device,
            base_dir,
            custom_mean,
            custom_std,
        )

    print("Done. Results under", base_dir)


if __name__ == "__main__":
    main()
