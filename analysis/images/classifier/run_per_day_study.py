#!/usr/bin/env python3
"""
Per-day study: same data, same config, 11 timepoints.
- per_day: one model per day (single image at that day).
- cnn_lstm / effnet_ts: training accumulated to that day (max_day), one model per cutoff.
Stores ALL metrics (TNR, TPR, acc, prec, rec, f1, TN, FP, FN, TP, cm, etc.) for direct comparison.
"""

import sys
import os
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

# Env and path setup
REPO_ROOT = Path(__file__).resolve().parents[3]
env_file = REPO_ROOT / ".env"
if env_file.exists():
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ[k] = v
for k, v in [
    ("BASE_PATH", "/net/projects2/promega/data-analysis"),
    ("OUTPUT_FOLDER", "/net/projects2/promega/data-analysis/output"),
]:
    os.environ.setdefault(k, v)

sys.path.insert(0, str(REPO_ROOT))

from analysis.images.cnn_lstm.load_split_data import load_split_data
from analysis.images.cnn_lstm.organoid_dataset import (
    OrganoidTimeSeriesDataset,
    compute_global_mean_from_ids,
)
from analysis.images.cnn_lstm.train_organoid_lstm_single_phase import (
    collate_variable_length,
    train_one_epoch,
    evaluate as evaluate_cnn_lstm,
    set_seed,
)
from analysis.images.cnn_lstm.organoid_model import OrganoidCNN_LSTM
from analysis.images.cnn_lstm.train_base_model import (
    SingleDayOrganoidDataset,
    train_for_day,
    set_seed as set_seed_base,
    TARGET_SIZE,
)
from analysis.images.cnn_lstm.train_temporal_ablation_attn import (
    OrganoidCNN_TAtt,
    set_seed as set_seed_attn,
    evaluate_binary as evaluate_effnet_ts,
    BATCH_SIZE as EFFNET_TS_BATCH,
    NUM_WORKERS,
    MAX_EPOCHS,
    WARMUP_EPOCHS,
    LR_HEAD,
    LR_CNN_UNFREEZE,
    GRAD_CLIP,
    PATIENCE,
    ATTN_DROPOUT,
)
from analysis.images.cnn_lstm.train_temporal_change import (
    OrganoidCNN_TChange,
    set_seed as set_seed_tchange,
    evaluate_binary as evaluate_tchange,
)
from torchvision import transforms as T
from tqdm import tqdm

# 11 timepoints for direct comparison
DAYS_11 = [6, 8, 10, 13, 15, 17, 20.5, 24, 26, 28, 30]
# Fixed thresholds for threshold study (primary metric: balanced_acc = (Sensitivity+Specificity)/2)
THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9]
SEED = 42


def filter_ids_with_frames_up_to_day(organoid_ids, series_metadata, max_day):
    """Keep only organoids that have at least one frame with day <= max_day (avoids empty sequence)."""
    out = []
    for oid in organoid_ids:
        days = series_metadata.get(oid, {}).get("days", [])
        if any(d <= max_day for d in days):
            out.append(oid)
    return out


def metrics_at_threshold(probs, labels, threshold):
    """Full metrics dict at one threshold (labels 0=Bad, 1=Good)."""
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
    balanced_acc = (TNR + TPR) / 2  # (Sensitivity + Specificity) / 2
    sensitivity, specificity = TPR, TNR
    cm_list = [[int(TN), int(FP)], [int(FN), int(TP)]]
    return {
        "threshold": float(threshold),
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "TNR": float(TNR),
        "TPR": float(TPR),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "balanced_acc": float(balanced_acc),
        "TN": int(TN),
        "FP": int(FP),
        "FN": int(FN),
        "TP": int(TP),
        "confusion_matrix": cm_list,
    }


def find_best_threshold(probs, labels, metric="tnr_f1", min_tnr=0.5):
    """Return (best_threshold, metrics_at_best)."""
    best_score, best_thresh, best_metrics = -1, 0.5, None
    for th in np.linspace(0.1, 0.9, 81):
        m = metrics_at_threshold(probs, labels, th)
        if m["TNR"] < min_tnr:
            continue
        score = (m["TNR"] + m["f1"]) / 2 if metric == "tnr_f1" else m["TNR"]
        if score > best_score:
            best_score, best_thresh, best_metrics = score, th, m
    return best_thresh, best_metrics


def build_threshold_results(
    val_probs,
    val_labels,
    test_probs,
    test_labels,
    best_thresh,
    train_probs=None,
    train_labels=None,
):
    """Build list of {threshold, threshold_value, val, test[, train]} for THRESHOLDS + optimal. For CSV export."""
    out = []
    for th in THRESHOLDS:
        entry = {
            "threshold": "fixed",
            "threshold_value": float(th),
            "val": metrics_at_threshold(val_probs, val_labels, th),
            "test": metrics_at_threshold(test_probs, test_labels, th),
        }
        if train_probs is not None and train_labels is not None:
            entry["train"] = metrics_at_threshold(train_probs, train_labels, th)
        out.append(entry)
    entry = {
        "threshold": "optimal",
        "threshold_value": float(best_thresh),
        "val": metrics_at_threshold(val_probs, val_labels, best_thresh),
        "test": metrics_at_threshold(test_probs, test_labels, best_thresh),
    }
    if train_probs is not None and train_labels is not None:
        entry["train"] = metrics_at_threshold(train_probs, train_labels, best_thresh)
    out.append(entry)
    return out


def run_per_day(
    day,
    train_ids,
    val_ids,
    test_ids,
    series_metadata,
    data,
    device,
    base_dir,
    image_key="img_path",
    use_rgb_mask=False,
    in_channels=3,
    save_model=True,
):
    """Per-day model: train one single-day EfficientNet at target_day=day. Same data. Full metrics."""
    set_seed_base(SEED)
    out_dir = base_dir / "per_day" / f"day_{str(day).replace('.', '_')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    result = train_for_day(
        day,
        train_ids,
        val_ids,
        test_ids,
        series_metadata,
        data,
        device,
        out_dir,
        image_key=image_key,
        use_rgb_mask=use_rgb_mask,
        in_channels=in_channels,
        save_model=save_model,
    )
    if result is None:
        return None
    from analysis.images.cnn_lstm.train_base_model import (
        BaselineEfficientNet,
    )

    eval_tf = T.Compose([T.Resize(TARGET_SIZE)])
    train_ds = SingleDayOrganoidDataset(
        train_ids,
        series_metadata,
        data,
        day,
        transform=eval_tf,
        image_key=image_key,
        use_rgb_mask=use_rgb_mask,
    )
    val_ds = SingleDayOrganoidDataset(
        val_ids,
        series_metadata,
        data,
        day,
        transform=eval_tf,
        image_key=image_key,
        use_rgb_mask=use_rgb_mask,
    )
    test_ds = SingleDayOrganoidDataset(
        test_ids,
        series_metadata,
        data,
        day,
        transform=eval_tf,
        image_key=image_key,
        use_rgb_mask=use_rgb_mask,
    )
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=False, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=0)
    model = BaselineEfficientNet(in_channels=in_channels).to(device)
    if save_model:
        ckpt = torch.load(out_dir / f"model_day_{day}.pth", map_location=device)
        model.load_state_dict(ckpt["state_dict"])
    else:
        model.load_state_dict(result["state_dict"], strict=True)
    model.eval()
    train_probs, train_labels_arr = [], []
    val_probs, val_labels = [], []
    test_probs, test_labels = [], []
    with torch.no_grad():
        for imgs, labels, _ in train_loader:
            logits = model(imgs.to(device))
            train_probs.extend(torch.sigmoid(logits).cpu().numpy().ravel())
            train_labels_arr.extend(labels.numpy().ravel())
        for imgs, labels, _ in val_loader:
            logits = model(imgs.to(device))
            val_probs.extend(torch.sigmoid(logits).cpu().numpy().ravel())
            val_labels.extend(labels.numpy().ravel())
        for imgs, labels, _ in test_loader:
            logits = model(imgs.to(device))
            test_probs.extend(torch.sigmoid(logits).cpu().numpy().ravel())
            test_labels.extend(labels.numpy().ravel())
    train_probs = np.array(train_probs)
    train_labels_arr = np.array(train_labels_arr)
    val_probs = np.array(val_probs)
    val_labels = np.array(val_labels)
    test_probs = np.array(test_probs)
    test_labels = np.array(test_labels)
    best_thresh, val_metrics = find_best_threshold(val_probs, val_labels)
    test_metrics = metrics_at_threshold(test_probs, test_labels, best_thresh)
    train_at_05 = metrics_at_threshold(train_probs, train_labels_arr, 0.5)
    val_at_05 = metrics_at_threshold(val_probs, val_labels, 0.5)
    test_at_05 = metrics_at_threshold(test_probs, test_labels, 0.5)
    threshold_results = build_threshold_results(
        val_probs,
        val_labels,
        test_probs,
        test_labels,
        best_thresh,
        train_probs=train_probs,
        train_labels=train_labels_arr,
    )
    save_result = {
        "model_type": "per_day",
        "day": day,
        "best_val_acc": result["best_val_acc"],
        "optimal_threshold": float(best_thresh),
        "val_at_optimal": val_metrics,
        "test_at_optimal": test_metrics,
        "train_at_0.5": train_at_05,
        "val_at_0.5": val_at_05,
        "test_at_0.5": test_at_05,
        "threshold_results": threshold_results,
        "test_false_positives": result.get("test_false_positives", []),
        "test_false_negatives": result.get("test_false_negatives", []),
        "model_path": str(out_dir / f"model_day_{day}.pth") if save_model else None,
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(save_result, f, indent=2)
    return save_result


def run_cnn_lstm_accumulated(
    day,
    train_ids,
    val_ids,
    test_ids,
    series_metadata,
    data,
    device,
    base_dir,
    image_key="image_path",
    input_rgb_mask=False,
    in_channels=3,
    save_model=True,
):
    """CNN-LSTM trained on data accumulated to max_day=day. Full metrics + train_history."""
    set_seed(SEED)
    out_dir = base_dir / "cnn_lstm" / f"day_{str(day).replace('.', '_')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    train_ids_f = filter_ids_with_frames_up_to_day(train_ids, series_metadata, day)
    val_ids_f = filter_ids_with_frames_up_to_day(val_ids, series_metadata, day)
    test_ids_f = filter_ids_with_frames_up_to_day(test_ids, series_metadata, day)
    if len(train_ids_f) == 0:
        print(f"No train organoids with frames <= day {day}, skipping.")
        return None
    print(
        f"Using {len(train_ids_f)}/{len(train_ids)} train, {len(val_ids_f)}/{len(val_ids)} val, {len(test_ids_f)}/{len(test_ids)} test (frames <= {day})"
    )
    global_mean = compute_global_mean_from_ids(train_ids_f, series_metadata, data)
    np.save(out_dir / "global_mean.npy", global_mean)
    from torchvision.transforms import InterpolationMode

    train_tf = T.Compose(
        [
            T.Resize((384, 384), interpolation=InterpolationMode.BILINEAR),
            T.RandomHorizontalFlip(0.5),
            T.RandomVerticalFlip(0.5),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0, hue=0),
        ]
    )
    eval_tf = T.Compose(
        [T.Resize((384, 384), interpolation=InterpolationMode.BILINEAR)]
    )
    train_ds = OrganoidTimeSeriesDataset(
        train_ids_f,
        series_metadata,
        data,
        global_mean=global_mean,
        max_day=day,
        image_key=image_key,
        input_rgb_mask=input_rgb_mask,
        transform=train_tf,
    )
    val_ds = OrganoidTimeSeriesDataset(
        val_ids_f,
        series_metadata,
        data,
        global_mean=global_mean,
        max_day=day,
        image_key=image_key,
        input_rgb_mask=input_rgb_mask,
        transform=eval_tf,
    )
    test_ds = OrganoidTimeSeriesDataset(
        test_ids_f,
        series_metadata,
        data,
        global_mean=global_mean,
        max_day=day,
        image_key=image_key,
        input_rgb_mask=input_rgb_mask,
        transform=eval_tf,
    )
    g = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(
        train_ds,
        batch_size=8,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_variable_length,
        generator=g,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=8,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_variable_length,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=8,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_variable_length,
    )
    train_labels = [
        1
        if str(series_metadata[oid].get("label", "")).strip().lower()
        in ("good", "acceptable", "accepted")
        else 0
        for oid in train_ids_f
    ]
    n_bad = sum(1 for l in train_labels if l == 0)
    n_good = len(train_labels) - n_bad
    weight_0 = len(train_labels) / (2 * max(n_bad, 1))
    weight_1 = len(train_labels) / (2 * max(n_good, 1))
    class_weights = torch.FloatTensor([weight_0, weight_1]).to(device)
    model = OrganoidCNN_LSTM(
        num_classes=2, lstm_hidden=256, lstm_layers=2, in_channels=in_channels
    ).to(device)
    for p in model.cnn.parameters():
        p.requires_grad = True
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5
    )
    best_val_acc, best_state_dict, train_history = 0, None, []
    for epoch in range(20):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc, val_prec, val_rec, val_f1, _, _ = evaluate_cnn_lstm(
            model, val_loader, criterion, device
        )
        scheduler.step(val_loss)
        train_history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_precision": val_prec,
                "val_recall": val_rec,
                "val_f1": val_f1,
            }
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            if save_model:
                torch.save(
                    {"model_state_dict": model.state_dict(), "val_acc": val_acc},
                    out_dir / "best_model.pth",
                )
            else:
                best_state_dict = {
                    k: v.cpu().clone() for k, v in model.state_dict().items()
                }
    if save_model:
        ckpt = torch.load(out_dir / "best_model.pth", map_location=device)
    else:
        ckpt = {
            "model_state_dict": best_state_dict or model.state_dict(),
            "val_acc": best_val_acc,
        }
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    val_probs, val_labels = [], []
    test_probs, test_labels = [], []
    with torch.no_grad():
        for imgs, days_n, labels, _, _ in val_loader:
            out = model(imgs.to(device))
            val_probs.extend(torch.softmax(out, dim=1)[:, 1].cpu().numpy())
            val_labels.extend(labels.cpu().numpy())
        for imgs, days_n, labels, _, _ in test_loader:
            out = model(imgs.to(device))
            test_probs.extend(torch.softmax(out, dim=1)[:, 1].cpu().numpy())
            test_labels.extend(labels.cpu().numpy())
    val_probs, val_labels = np.array(val_probs), np.array(val_labels)
    test_probs, test_labels = np.array(test_probs), np.array(test_labels)
    best_thresh, val_metrics = find_best_threshold(val_probs, val_labels)
    test_metrics = metrics_at_threshold(test_probs, test_labels, best_thresh)
    val_at_05 = metrics_at_threshold(val_probs, val_labels, 0.5)
    test_at_05 = metrics_at_threshold(test_probs, test_labels, 0.5)
    threshold_results = build_threshold_results(
        val_probs, val_labels, test_probs, test_labels, best_thresh
    )
    save_result = {
        "model_type": "cnn_lstm",
        "day": day,
        "best_val_acc": float(best_val_acc),
        "optimal_threshold": float(best_thresh),
        "val_at_optimal": val_metrics,
        "test_at_optimal": test_metrics,
        "val_at_0.5": val_at_05,
        "test_at_0.5": test_at_05,
        "threshold_results": threshold_results,
        "train_history": train_history,
        "model_path": str(out_dir / "best_model.pth") if save_model else None,
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(save_result, f, indent=2)
    return save_result


def run_effnet_ts_accumulated(
    day,
    train_ids,
    val_ids,
    test_ids,
    series_metadata,
    data,
    device,
    base_dir,
    image_key="image_path",
    input_rgb_mask=False,
    in_channels=3,
    save_model=True,
):
    """EfficientNet time-series trained on data accumulated to max_day=day. Full metrics + train_history."""
    set_seed_attn(SEED)
    out_dir = base_dir / "effnet_ts" / f"day_{str(day).replace('.', '_')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    train_ids_f = filter_ids_with_frames_up_to_day(train_ids, series_metadata, day)
    val_ids_f = filter_ids_with_frames_up_to_day(val_ids, series_metadata, day)
    test_ids_f = filter_ids_with_frames_up_to_day(test_ids, series_metadata, day)
    if len(train_ids_f) == 0:
        print(f"No train organoids with frames <= day {day}, skipping.")
        return None
    print(
        f"Using {len(train_ids_f)}/{len(train_ids)} train, {len(val_ids_f)}/{len(val_ids)} val, {len(test_ids_f)}/{len(test_ids)} test (frames <= {day})"
    )
    global_mean = compute_global_mean_from_ids(train_ids_f, series_metadata, data)
    np.save(out_dir / "global_mean.npy", global_mean)
    from torchvision.transforms import InterpolationMode

    train_tf = T.Compose(
        [
            T.Resize((384, 384), interpolation=InterpolationMode.BILINEAR),
            T.RandomHorizontalFlip(0.5),
            T.RandomVerticalFlip(0.5),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0, hue=0),
        ]
    )
    eval_tf = T.Compose(
        [T.Resize((384, 384), interpolation=InterpolationMode.BILINEAR)]
    )
    train_ds = OrganoidTimeSeriesDataset(
        train_ids_f,
        series_metadata,
        data,
        global_mean=global_mean,
        max_day=day,
        image_key=image_key,
        input_rgb_mask=input_rgb_mask,
        transform=train_tf,
    )
    val_ds = OrganoidTimeSeriesDataset(
        val_ids_f,
        series_metadata,
        data,
        global_mean=global_mean,
        max_day=day,
        image_key=image_key,
        input_rgb_mask=input_rgb_mask,
        transform=eval_tf,
    )
    test_ds = OrganoidTimeSeriesDataset(
        test_ids_f,
        series_metadata,
        data,
        global_mean=global_mean,
        max_day=day,
        image_key=image_key,
        input_rgb_mask=input_rgb_mask,
        transform=eval_tf,
    )
    g = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(
        train_ds,
        batch_size=EFFNET_TS_BATCH,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=collate_variable_length,
        pin_memory=(device.type == "cuda"),
        generator=g,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=EFFNET_TS_BATCH,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_variable_length,
        pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=EFFNET_TS_BATCH,
        shuffle=False,
        num_workers=NUM_WORKERS,
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
    n_total = len(train_labels)
    w_pos = n_total / (2.0 * max(n_good, 1))
    w_neg = n_total / (2.0 * max(n_bad, 1))
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    model = OrganoidCNN_TAtt(attn_dropout=ATTN_DROPOUT, in_channels=in_channels).to(
        device
    )

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
        return optim.Adam(groups)

    optimizer = make_opt(0.0, LR_HEAD)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    best_val_acc, best_state, train_history, bad_epochs = -1, None, [], 0
    for epoch in range(1, MAX_EPOCHS + 1):
        if epoch == WARMUP_EPOCHS + 1:
            model.unfreeze_last_blocks()
            optimizer = make_opt(LR_CNN_UNFREEZE, LR_HEAD)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            running_loss += loss.item() * labels.size(0)
            correct += (
                ((torch.sigmoid(logits) > 0.5).long() == labels.long()).sum().item()
            )
            total += labels.size(0)
        train_acc = correct / max(total, 1)
        train_loss = running_loss / max(total, 1)
        val_loss, val_acc, val_prec, val_rec, val_f1, _, _, _, _ = evaluate_effnet_ts(
            model, val_loader, criterion, device
        )
        scheduler.step(val_loss)
        train_history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_precision": val_prec,
                "val_recall": val_rec,
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
            if bad_epochs >= PATIENCE:
                break
    if best_state is None:
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state, strict=True)
    if save_model:
        torch.save(
            {"state_dict": best_state, "best_val_acc": float(best_val_acc)},
            out_dir / "best_model.pth",
        )
    train_eval_ds = OrganoidTimeSeriesDataset(
        train_ids_f,
        series_metadata,
        data,
        global_mean=global_mean,
        max_day=day,
        image_key=image_key,
        input_rgb_mask=input_rgb_mask,
        transform=eval_tf,
    )
    train_eval_loader = DataLoader(
        train_eval_ds,
        batch_size=EFFNET_TS_BATCH,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_variable_length,
        pin_memory=(device.type == "cuda"),
    )
    train_probs, train_labels_arr, train_organoid_ids = [], [], []
    val_probs, val_labels, val_organoid_ids = [], [], []
    test_probs, test_labels, test_organoid_ids = [], [], []
    model.eval()
    with torch.no_grad():
        for seqs, days_n, labels, _, oids in train_eval_loader:
            logits, _ = model(seqs.to(device), days_n.to(device).float())
            probs = torch.sigmoid(logits).cpu().numpy().ravel()
            train_probs.extend(probs)
            train_labels_arr.extend(labels.cpu().numpy().ravel())
            train_organoid_ids.extend(oids)
        for seqs, days_n, labels, _, oids in val_loader:
            logits, _ = model(seqs.to(device), days_n.to(device).float())
            probs = torch.sigmoid(logits).cpu().numpy().ravel()
            val_probs.extend(probs)
            val_labels.extend(labels.cpu().numpy().ravel())
            val_organoid_ids.extend(oids)
        for seqs, days_n, labels, _, oids in test_loader:
            logits, _ = model(seqs.to(device), days_n.to(device).float())
            probs = torch.sigmoid(logits).cpu().numpy().ravel()
            test_probs.extend(probs)
            test_labels.extend(labels.cpu().numpy().ravel())
            test_organoid_ids.extend(oids)
    train_probs, train_labels_arr = np.array(train_probs), np.array(train_labels_arr)
    val_probs, val_labels = np.array(val_probs), np.array(val_labels)
    test_probs, test_labels = np.array(test_probs), np.array(test_labels)
    best_thresh, val_metrics = find_best_threshold(val_probs, val_labels)
    test_metrics = metrics_at_threshold(test_probs, test_labels, best_thresh)
    train_at_05 = metrics_at_threshold(train_probs, train_labels_arr, 0.5)
    val_at_05 = metrics_at_threshold(val_probs, val_labels, 0.5)
    test_at_05 = metrics_at_threshold(test_probs, test_labels, 0.5)
    threshold_results = build_threshold_results(
        val_probs,
        val_labels,
        test_probs,
        test_labels,
        best_thresh,
        train_probs=train_probs,
        train_labels=train_labels_arr,
    )
    train_predictions = [
        {"organoid_id": oid, "label": int(l), "prob": float(p)}
        for oid, l, p in zip(train_organoid_ids, train_labels_arr, train_probs)
    ]
    val_predictions = [
        {"organoid_id": oid, "label": int(l), "prob": float(p)}
        for oid, l, p in zip(val_organoid_ids, val_labels, val_probs)
    ]
    test_predictions = [
        {"organoid_id": oid, "label": int(l), "prob": float(p)}
        for oid, l, p in zip(test_organoid_ids, test_labels, test_probs)
    ]
    save_result = {
        "model_type": "effnet_ts",
        "day": day,
        "best_val_acc": float(best_val_acc),
        "optimal_threshold": float(best_thresh),
        "val_at_optimal": val_metrics,
        "test_at_optimal": test_metrics,
        "train_at_0.5": train_at_05,
        "val_at_0.5": val_at_05,
        "test_at_0.5": test_at_05,
        "threshold_results": threshold_results,
        "train_history": train_history,
        "model_path": str(out_dir / "best_model.pth") if save_model else None,
        "train_predictions": train_predictions,
        "val_predictions": val_predictions,
        "test_predictions": test_predictions,
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(save_result, f, indent=2)
    return save_result


def run_effnet_tchange_accumulated(
    day,
    train_ids,
    val_ids,
    test_ids,
    series_metadata,
    data,
    device,
    base_dir,
    image_key="image_path",
    input_rgb_mask=False,
    in_channels=3,
    save_model=True,
):
    """Temporal-Change EfficientNet: explicitly captures morphological trajectory."""
    set_seed_tchange(SEED)
    out_dir = base_dir / "effnet_tchange" / f"day_{str(day).replace('.', '_')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    train_ids_f = filter_ids_with_frames_up_to_day(train_ids, series_metadata, day)
    val_ids_f = filter_ids_with_frames_up_to_day(val_ids, series_metadata, day)
    test_ids_f = filter_ids_with_frames_up_to_day(test_ids, series_metadata, day)
    if len(train_ids_f) == 0:
        print(f"No train organoids with frames <= day {day}, skipping.")
        return None
    print(
        f"[tchange] {len(train_ids_f)}/{len(train_ids)} train, {len(val_ids_f)}/{len(val_ids)} val, {len(test_ids_f)}/{len(test_ids)} test (frames <= {day})"
    )
    global_mean = compute_global_mean_from_ids(train_ids_f, series_metadata, data)
    np.save(out_dir / "global_mean.npy", global_mean)
    from torchvision.transforms import InterpolationMode

    train_tf = T.Compose(
        [
            T.Resize((384, 384), interpolation=InterpolationMode.BILINEAR),
            T.RandomHorizontalFlip(0.5),
            T.RandomVerticalFlip(0.5),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0, hue=0),
        ]
    )
    eval_tf = T.Compose(
        [T.Resize((384, 384), interpolation=InterpolationMode.BILINEAR)]
    )
    train_ds = OrganoidTimeSeriesDataset(
        train_ids_f,
        series_metadata,
        data,
        global_mean=global_mean,
        max_day=day,
        image_key=image_key,
        input_rgb_mask=input_rgb_mask,
        transform=train_tf,
    )
    val_ds = OrganoidTimeSeriesDataset(
        val_ids_f,
        series_metadata,
        data,
        global_mean=global_mean,
        max_day=day,
        image_key=image_key,
        input_rgb_mask=input_rgb_mask,
        transform=eval_tf,
    )
    test_ds = OrganoidTimeSeriesDataset(
        test_ids_f,
        series_metadata,
        data,
        global_mean=global_mean,
        max_day=day,
        image_key=image_key,
        input_rgb_mask=input_rgb_mask,
        transform=eval_tf,
    )
    g = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(
        train_ds,
        batch_size=EFFNET_TS_BATCH,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=collate_variable_length,
        pin_memory=(device.type == "cuda"),
        generator=g,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=EFFNET_TS_BATCH,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_variable_length,
        pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=EFFNET_TS_BATCH,
        shuffle=False,
        num_workers=NUM_WORKERS,
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
    n_total = len(train_labels)
    w_pos = n_total / (2.0 * max(n_good, 1))
    w_neg = n_total / (2.0 * max(n_bad, 1))
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    model = OrganoidCNN_TChange(attn_dropout=ATTN_DROPOUT, in_channels=in_channels).to(
        device
    )

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
        return optim.Adam(groups)

    optimizer = make_opt(0.0, LR_HEAD)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    best_val_acc, best_state, train_history, bad_epochs = -1, None, [], 0
    for epoch in range(1, MAX_EPOCHS + 1):
        if epoch == WARMUP_EPOCHS + 1:
            model.unfreeze_last_blocks()
            optimizer = make_opt(LR_CNN_UNFREEZE, LR_HEAD)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=5
            )
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for seqs, days_n, labels, weights, _ in tqdm(
            train_loader, desc=f"[tchange] Epoch {epoch}", leave=False
        ):
            seqs, days_n = seqs.to(device), days_n.to(device).float()
            labels, weights = labels.to(device).float(), weights.to(device).float()
            optimizer.zero_grad()
            logits, _ = model(seqs, days_n)
            loss_raw = criterion(logits, labels)
            cls_w = labels * w_pos + (1 - labels) * w_neg
            loss = (loss_raw * weights * cls_w).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            running_loss += loss.item() * labels.size(0)
            correct += (
                ((torch.sigmoid(logits) > 0.5).long() == labels.long()).sum().item()
            )
            total += labels.size(0)
        train_acc = correct / max(total, 1)
        train_loss = running_loss / max(total, 1)
        val_loss, val_acc, val_prec, val_rec, val_f1, _, _, _, _ = evaluate_tchange(
            model, val_loader, criterion, device
        )
        scheduler.step(val_loss)
        train_history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_precision": val_prec,
                "val_recall": val_rec,
                "val_f1": val_f1,
            }
        )
        print(
            f"  [tchange] Ep {epoch:02d} train_acc={train_acc:.3f} val_acc={val_acc:.3f} val_f1={val_f1:.3f}"
        )
        if val_acc > best_val_acc + 1e-4:
            best_val_acc, best_state = (
                val_acc,
                {k: v.cpu().clone() for k, v in model.state_dict().items()},
            )
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                break
    if best_state is None:
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state, strict=True)
    if save_model:
        torch.save(
            {"state_dict": best_state, "best_val_acc": float(best_val_acc)},
            out_dir / "best_model.pth",
        )
    train_eval_ds = OrganoidTimeSeriesDataset(
        train_ids_f,
        series_metadata,
        data,
        global_mean=global_mean,
        max_day=day,
        image_key=image_key,
        input_rgb_mask=input_rgb_mask,
        transform=eval_tf,
    )
    train_eval_loader = DataLoader(
        train_eval_ds,
        batch_size=EFFNET_TS_BATCH,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_variable_length,
        pin_memory=(device.type == "cuda"),
    )
    train_probs, train_labels_arr, train_organoid_ids = [], [], []
    val_probs, val_labels, val_organoid_ids = [], [], []
    test_probs, test_labels, test_organoid_ids = [], [], []
    model.eval()
    with torch.no_grad():
        for seqs, days_n, labels, _, oids in train_eval_loader:
            logits, _ = model(seqs.to(device), days_n.to(device).float())
            probs = torch.sigmoid(logits).cpu().numpy().ravel()
            train_probs.extend(probs)
            train_labels_arr.extend(labels.cpu().numpy().ravel())
            train_organoid_ids.extend(oids)
        for seqs, days_n, labels, _, oids in val_loader:
            logits, _ = model(seqs.to(device), days_n.to(device).float())
            probs = torch.sigmoid(logits).cpu().numpy().ravel()
            val_probs.extend(probs)
            val_labels.extend(labels.cpu().numpy().ravel())
            val_organoid_ids.extend(oids)
        for seqs, days_n, labels, _, oids in test_loader:
            logits, _ = model(seqs.to(device), days_n.to(device).float())
            probs = torch.sigmoid(logits).cpu().numpy().ravel()
            test_probs.extend(probs)
            test_labels.extend(labels.cpu().numpy().ravel())
            test_organoid_ids.extend(oids)
    train_probs, train_labels_arr = np.array(train_probs), np.array(train_labels_arr)
    val_probs, val_labels = np.array(val_probs), np.array(val_labels)
    test_probs, test_labels = np.array(test_probs), np.array(test_labels)
    best_thresh, val_metrics = find_best_threshold(val_probs, val_labels)
    test_metrics = metrics_at_threshold(test_probs, test_labels, best_thresh)
    train_at_05 = metrics_at_threshold(train_probs, train_labels_arr, 0.5)
    val_at_05 = metrics_at_threshold(val_probs, val_labels, 0.5)
    test_at_05 = metrics_at_threshold(test_probs, test_labels, 0.5)
    threshold_results = build_threshold_results(
        val_probs,
        val_labels,
        test_probs,
        test_labels,
        best_thresh,
        train_probs=train_probs,
        train_labels=train_labels_arr,
    )
    train_predictions = [
        {"organoid_id": oid, "label": int(l), "prob": float(p)}
        for oid, l, p in zip(train_organoid_ids, train_labels_arr, train_probs)
    ]
    val_predictions = [
        {"organoid_id": oid, "label": int(l), "prob": float(p)}
        for oid, l, p in zip(val_organoid_ids, val_labels, val_probs)
    ]
    test_predictions = [
        {"organoid_id": oid, "label": int(l), "prob": float(p)}
        for oid, l, p in zip(test_organoid_ids, test_labels, test_probs)
    ]
    save_result = {
        "model_type": "effnet_tchange",
        "day": day,
        "best_val_acc": float(best_val_acc),
        "optimal_threshold": float(best_thresh),
        "val_at_optimal": val_metrics,
        "test_at_optimal": test_metrics,
        "train_at_0.5": train_at_05,
        "val_at_0.5": val_at_05,
        "test_at_0.5": test_at_05,
        "threshold_results": threshold_results,
        "train_history": train_history,
        "model_path": str(out_dir / "best_model.pth") if save_model else None,
        "train_predictions": train_predictions,
        "val_predictions": val_predictions,
        "test_predictions": test_predictions,
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(save_result, f, indent=2)
    return save_result


# CSV columns for export (same as kfold metrics): primary metric = balanced_acc
RESULTS_CSV_COLUMNS = [
    "setup",
    "fold",
    "model",
    "day",
    "threshold",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "TNR",
    "TPR",
    "sensitivity",
    "specificity",
    "balanced_acc",
    "TN",
    "FP",
    "FN",
    "TP",
    "optimal_threshold",
]


def results_to_csv_rows(
    setup, fold, model_type, day, optimal_threshold, threshold_results, split="test"
):
    """Turn one results.json (with threshold_results) into CSV rows. split in ('test','val')."""
    rows = []
    for tr in threshold_results:
        th_val = tr["threshold_value"]
        th_type = tr["threshold"]
        m = tr[split]
        rows.append(
            {
                "setup": setup,
                "fold": fold,
                "model": model_type,
                "day": day,
                "threshold": th_val,
                "accuracy": m["accuracy"],
                "precision": m["precision"],
                "recall": m["recall"],
                "f1": m["f1"],
                "TNR": m["TNR"],
                "TPR": m["TPR"],
                "sensitivity": m["sensitivity"],
                "specificity": m["specificity"],
                "balanced_acc": m["balanced_acc"],
                "TN": m["TN"],
                "FP": m["FP"],
                "FN": m["FN"],
                "TP": m["TP"],
                "optimal_threshold": optimal_threshold,
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Per-day study: one of per_day, cnn_lstm, effnet_ts, effnet_tchange at one day"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=["per_day", "cnn_lstm", "effnet_ts", "effnet_tchange"],
    )
    parser.add_argument(
        "--day", type=float, required=True, help="Target day (e.g. 6, 8, 20.5, 30)"
    )
    parser.add_argument(
        "--input_mode",
        type=str,
        default="rgb",
        choices=["rgb", "overlay", "rgb_mask"],
        help="Input: rgb (default), overlay (RGB+outline), rgb_mask (4ch RGB+mask)",
    )
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Keep original results in per_day_study; new variants in separate dirs
    if args.input_mode == "overlay":
        base_dir = Path(__file__).parent / "per_day_study_overlay"
        image_key, use_rgb_mask, in_channels = "overlay_path", False, 3
    elif args.input_mode == "rgb_mask":
        base_dir = Path(__file__).parent / "per_day_study_rgb_mask"
        image_key, use_rgb_mask, in_channels = "img_path", True, 4
    else:
        base_dir = Path(__file__).parent / "per_day_study"
        image_key, use_rgb_mask, in_channels = "img_path", False, 3
    input_rgb_mask = use_rgb_mask
    split_dir = REPO_ROOT / "data_splits"
    train_ids, val_ids, test_ids, series_metadata, data = load_split_data(
        split_dir / "both_train_base.json",
        split_dir / "both_val_base.json",
        split_dir / "both_test_base.json",
    )
    print(
        f"Model: {args.model_type}, day: {args.day}, input_mode: {args.input_mode}, base_dir: {base_dir}"
    )
    print(f"  data: train={len(train_ids)} val={len(val_ids)} test={len(test_ids)}")
    if args.model_type == "per_day":
        run_per_day(
            args.day,
            train_ids,
            val_ids,
            test_ids,
            series_metadata,
            data,
            device,
            base_dir,
            image_key=image_key,
            use_rgb_mask=use_rgb_mask,
            in_channels=in_channels,
        )
    elif args.model_type == "cnn_lstm":
        run_cnn_lstm_accumulated(
            args.day,
            train_ids,
            val_ids,
            test_ids,
            series_metadata,
            data,
            device,
            base_dir,
            image_key=image_key,
            input_rgb_mask=input_rgb_mask,
            in_channels=in_channels,
        )
    elif args.model_type == "effnet_ts":
        run_effnet_ts_accumulated(
            args.day,
            train_ids,
            val_ids,
            test_ids,
            series_metadata,
            data,
            device,
            base_dir,
            image_key=image_key,
            input_rgb_mask=input_rgb_mask,
            in_channels=in_channels,
        )
    else:
        run_effnet_tchange_accumulated(
            args.day,
            train_ids,
            val_ids,
            test_ids,
            series_metadata,
            data,
            device,
            base_dir,
            image_key=image_key,
            input_rgb_mask=input_rgb_mask,
            in_channels=in_channels,
        )
    print("Done.")


if __name__ == "__main__":
    main()
