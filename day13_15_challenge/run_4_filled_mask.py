#!/usr/bin/env python3
"""
Task 4: Run effnet_ts and per_day for Days 13 and 15 only, with input = (gray, gray, filled_mask).
Uses GPU if available.
Run: python run_4_filled_mask.py
"""
import os
import sys
import json
from pathlib import Path

CHALLENGE_DIR = Path(__file__).resolve().parent
ROOT = CHALLENGE_DIR.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(CHALLENGE_DIR))

_env = ROOT / ".env"
if _env.exists():
    with open(_env) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ[k] = v
# Required by config.py when train_base_model is imported
_root = str(ROOT)
for k, v in [
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
]:
    os.environ.setdefault(k, v)

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from analysis.images.cnn_lstm.load_split_data import load_split_data
from analysis.images.cnn_lstm.train_base_model import (
    BaselineEfficientNet,
    set_seed as set_seed_base,
    TARGET_SIZE,
    BATCH_SIZE,
    NUM_WORKERS,
    MAX_EPOCHS,
    PATIENCE,
    LR,
    GRAD_CLIP,
)
from analysis.images.cnn_lstm.train_organoid_lstm_single_phase import collate_variable_length
from analysis.images.cnn_lstm.train_temporal_ablation_attn import (
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
from dataset_filled_mask import OrganoidTimeSeriesDatasetFilledMask, SingleDayFilledMaskDataset

SEED = 42
DAYS = [13.0, 15.0]


def filter_ids_with_frames_up_to_day(organoid_ids, series_metadata, max_day):
    return [oid for oid in organoid_ids if any(d <= max_day for d in series_metadata.get(oid, {}).get("days", []))]


def metrics_at_threshold(probs, labels, threshold):
    preds = (probs >= threshold).astype(int)
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    TN, FP, FN, TP = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    TNR = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    TPR = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
    balanced_acc = (TNR + TPR) / 2
    return {"balanced_acc": float(balanced_acc), "TNR": float(TNR), "TN": int(TN), "FP": int(FP), "FN": int(FN), "TP": int(TP), "accuracy": (TP + TN) / len(labels) if len(labels) > 0 else 0, "f1": float(f1)}


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


def run_effnet_ts_filled_mask(day, train_ids, val_ids, test_ids, series_metadata, data, device, base_dir):
    from torchvision.transforms import InterpolationMode
    set_seed_attn(SEED)
    out_dir = base_dir / "effnet_ts_filled_mask" / f"day_{str(day).replace('.', '_')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ids_f = filter_ids_with_frames_up_to_day(train_ids, series_metadata, day)
    val_ids_f = filter_ids_with_frames_up_to_day(val_ids, series_metadata, day)
    test_ids_f = filter_ids_with_frames_up_to_day(test_ids, series_metadata, day)
    if len(train_ids_f) == 0:
        print(f"No train organoids with frames <= day {day}, skipping effnet_ts filled_mask.")
        return None

    BILINEAR = InterpolationMode.BILINEAR
    train_tf = T.Compose([
        T.Resize((384, 384), interpolation=BILINEAR),
        T.RandomRotation(15, fill=128),
        T.RandomHorizontalFlip(0.5),
        T.RandomVerticalFlip(0.5),
        T.RandomResizedCrop(384, scale=(0.9, 1.0)),
        T.ColorJitter(0.2, 0.2, 0, 0),
    ])
    eval_tf = T.Compose([T.Resize((384, 384), interpolation=BILINEAR)])

    train_ds = OrganoidTimeSeriesDatasetFilledMask(train_ids_f, series_metadata, data, transform=train_tf, max_day=day)
    val_ds = OrganoidTimeSeriesDatasetFilledMask(val_ids_f, series_metadata, data, transform=eval_tf, max_day=day)
    test_ds = OrganoidTimeSeriesDatasetFilledMask(test_ids_f, series_metadata, data, transform=eval_tf, max_day=day)

    train_loader = DataLoader(train_ds, batch_size=EFFNET_TS_BATCH, shuffle=True, num_workers=EFFNET_NUM_WORKERS or 0, collate_fn=collate_variable_length, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=EFFNET_TS_BATCH, shuffle=False, num_workers=EFFNET_NUM_WORKERS or 0, collate_fn=collate_variable_length, pin_memory=(device.type == "cuda"))
    test_loader = DataLoader(test_ds, batch_size=EFFNET_TS_BATCH, shuffle=False, num_workers=EFFNET_NUM_WORKERS or 0, collate_fn=collate_variable_length, pin_memory=(device.type == "cuda"))

    train_labels = [1 if str(series_metadata[oid].get("label", "")).strip().lower() in ("good", "acceptable", "accepted") else 0 for oid in train_ids_f]
    n_good, n_bad = sum(train_labels), len(train_labels) - sum(train_labels)
    w_pos = max(n_bad, 1) / max(n_good, 1)
    w_neg = max(n_good, 1) / max(n_bad, 1)
    criterion = torch.nn.BCEWithLogitsLoss(reduction="none")
    model = OrganoidCNN_TAtt(attn_dropout=ATTN_DROPOUT, in_channels=3).to(device)

    def make_opt(lr_cnn, lr_head):
        params_cnn = [p for n, p in model.cnn.named_parameters() if p.requires_grad]
        params_head = [p for n, p in model.named_parameters() if not n.startswith("cnn.") and p.requires_grad]
        groups = []
        if params_cnn:
            groups.append({"params": params_cnn, "lr": lr_cnn})
        if params_head:
            groups.append({"params": params_head, "lr": lr_head})
        return torch.optim.Adam(groups)

    optimizer = make_opt(0.0, LR_HEAD)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    best_val_acc, best_state, bad_epochs = -1, None, 0

    for epoch in range(1, EFFNET_MAX_EPOCHS + 1):
        if epoch == WARMUP_EPOCHS + 1:
            model.unfreeze_last_blocks()
            optimizer = make_opt(LR_CNN_UNFREEZE, LR_HEAD)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
        model.train()
        for seqs, days_n, labels, weights, _ in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
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
        val_loss, val_acc, _, _, _, _, _, _, _ = evaluate_effnet_ts(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        if val_acc > best_val_acc + 1e-4:
            best_val_acc, best_state = val_acc, {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= EFFNET_PATIENCE:
                break

    if best_state is None:
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state, strict=True)
    torch.save({"state_dict": best_state, "best_val_acc": float(best_val_acc)}, out_dir / "best_model.pth")

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
    save_result = {"model_type": "effnet_ts_filled_mask", "day": day, "best_val_acc": float(best_val_acc), "optimal_threshold": float(best_thresh), "test_at_0.5": test_at_05}
    with open(out_dir / "results.json", "w") as f:
        json.dump(save_result, f, indent=2)
    print(f"Day {day} effnet_ts (filled_mask): test balanced_acc@0.5 = {test_at_05['balanced_acc']:.3f}")
    return save_result


def run_per_day_filled_mask(day, train_ids, val_ids, test_ids, series_metadata, data, device, base_dir):
    set_seed_base(SEED)
    out_dir = base_dir / "per_day_filled_mask" / f"day_{str(day).replace('.', '_')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    eval_tf = T.Compose([T.Resize(TARGET_SIZE)])
    train_tf = T.Compose([T.Resize(TARGET_SIZE), T.RandomHorizontalFlip(0.5), T.RandomVerticalFlip(0.5), T.ColorJitter(0.2, 0.2, 0.2, 0.1)])

    train_ds = SingleDayFilledMaskDataset(train_ids, series_metadata, data, day, transform=train_tf)
    val_ds = SingleDayFilledMaskDataset(val_ids, series_metadata, data, day, transform=eval_tf)
    test_ds = SingleDayFilledMaskDataset(test_ids, series_metadata, data, day, transform=eval_tf)
    if len(train_ds) == 0:
        print(f"No train samples for day {day}, skipping per_day filled_mask.")
        return None

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS or 0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS or 0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS or 0, pin_memory=True)

    train_labels = [s["label"] for s in train_ds.samples]
    n_good, n_bad = sum(train_labels), len(train_labels) - sum(train_labels)
    if n_good == 0: n_good = 1
    if n_bad == 0: n_bad = 1
    pos_weight = torch.tensor([n_bad / n_good], device=device)
    model = BaselineEfficientNet(in_channels=3).to(device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    best_val_acc, best_state, bad_epochs = -1, None, 0

    for epoch in range(1, MAX_EPOCHS + 1):
        if epoch == 4:
            model.unfreeze_backbone()
            optimizer = torch.optim.Adam(model.parameters(), lr=LR * 0.1)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
        model.train()
        for imgs, labels, _ in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
        model.eval()
        val_probs, val_labels_list = [], []
        with torch.no_grad():
            for imgs, labels, _ in val_loader:
                val_probs.extend(torch.sigmoid(model(imgs.to(device))).cpu().numpy().ravel())
                val_labels_list.extend(labels.numpy().ravel())
        val_probs = np.array(val_probs)
        val_labels_list = np.array(val_labels_list)
        val_acc = ((val_probs >= 0.5).astype(int) == val_labels_list).mean()
        scheduler.step(0.0)
        if val_acc > best_val_acc + 1e-4:
            best_val_acc, best_state = val_acc, {k: v.cpu() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                break

    if best_state is None:
        best_state = {k: v.cpu() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state, strict=True)
    torch.save({"state_dict": best_state, "best_val_acc": float(best_val_acc)}, out_dir / f"model_day_{day}.pth")

    test_probs, test_labels_list = [], []
    with torch.no_grad():
        for imgs, labels, _ in test_loader:
            test_probs.extend(torch.sigmoid(model(imgs.to(device))).cpu().numpy().ravel())
            test_labels_list.extend(labels.numpy().ravel())
    test_probs = np.array(test_probs)
    test_labels_list = np.array(test_labels_list)
    best_thresh, _ = find_best_threshold(val_probs, val_labels_list)
    test_at_05 = metrics_at_threshold(test_probs, test_labels_list, 0.5)
    save_result = {"model_type": "per_day_filled_mask", "day": day, "best_val_acc": float(best_val_acc), "optimal_threshold": float(best_thresh), "test_at_0.5": test_at_05}
    with open(out_dir / "results.json", "w") as f:
        json.dump(save_result, f, indent=2)
    print(f"Day {day} per_day (filled_mask): test balanced_acc@0.5 = {test_at_05['balanced_acc']:.3f}")
    return save_result


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Step 4: Filled-mask training using device: {device}", flush=True)
    split_dir = ROOT / "data_splits"
    train_ids, val_ids, test_ids, series_metadata, data = load_split_data(
        split_dir / "both_train_base.json",
        split_dir / "both_val_base.json",
        split_dir / "both_test_base.json",
    )
    base_dir = CHALLENGE_DIR / "runs_filled_mask"
    base_dir.mkdir(parents=True, exist_ok=True)
    for day in DAYS:
        run_effnet_ts_filled_mask(day, train_ids, val_ids, test_ids, series_metadata, data, device, base_dir)
        run_per_day_filled_mask(day, train_ids, val_ids, test_ids, series_metadata, data, device, base_dir)
    print("Done. Results under", base_dir)


if __name__ == "__main__":
    main()
