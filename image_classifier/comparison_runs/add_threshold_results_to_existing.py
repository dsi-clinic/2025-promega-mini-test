#!/usr/bin/env python3
"""
Add threshold_results (0.5, 0.6, 0.7, 0.8, 0.9 + optimal) to existing results.json via inference only.
No re-training. Use when results have val_at_optimal/test_at_optimal but no threshold_results.
"""

import os
import sys
import json
from pathlib import Path

COMPARISON_RUNS = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]
env_file = REPO_ROOT / ".env"
if env_file.exists():
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ[k] = v
for key, default in [
    ("BASE_PATH", "/net/projects2/promega/data-analysis"),
    ("OUTPUT_FOLDER", "/net/projects2/promega/data-analysis/output"),
    ("RAW_IMAGE_DATA", "/net/projects2/promega/data-analysis/raw-images"),
    (
        "IMAGE_VERIFICATION_FORM",
        "/net/projects2/promega/data-analysis/image-verification-form.json",
    ),
    ("PLOTS_FOLDER", "/net/projects2/promega/data-analysis/plots"),
    ("LOGS_FOLDER", "/net/projects2/promega/data-analysis/logs"),
    ("NPY_OUTPUTS", "/net/projects2/promega/data-analysis/npy-outputs"),
    ("PREDICTIONS_DIR", "/net/projects2/promega/data-analysis/predictions"),
    ("SURVEY_RESULTS", "/net/projects2/promega/data-analysis/survey-results"),
    ("MANUAL_MASKS_DIR", "/net/projects2/promega/data-analysis/manual-masks"),
    ("META_FILE", "/net/projects2/promega/data-analysis/metadata.json"),
    (
        "RAW_IMAGE_MAPPING_JSON",
        "/net/projects2/promega/data-analysis/image-mapping.json",
    ),
    ("TARGET_WIDTH", "512"),
    ("TARGET_HEIGHT", "384"),
    ("TRAIN_RESIZED_DIR", "/net/projects2/promega/data-analysis/train-resized"),
    ("TRAIN_MANUAL_MAPPING_DIR", "/net/projects2/promega/data-analysis/train-mapping"),
    (
        "TRAIN_MANUAL_PROCESSED_DIR",
        "/net/projects2/promega/data-analysis/train-processed",
    ),
    ("TRAIN_SPLITS_DIR", "/net/projects2/promega/data-analysis/train-splits"),
    ("INFER_RESIZED_DIR", "/net/projects2/promega/data-analysis/infer-resized"),
    (
        "INFER_MAPPING_TOTAL_JSON",
        "/net/projects2/promega/data-analysis/infer-mapping.json",
    ),
    (
        "MANUAL_THRESHOLD_MAPPING",
        "/net/projects2/promega/data-analysis/threshold-mapping.json",
    ),
]:
    os.environ.setdefault(key, default)

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(COMPARISON_RUNS))

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T

from image_classifier.cnn_lstm.load_split_data import load_split_data
from image_classifier.cnn_lstm.train_base_model import (
    SingleDayOrganoidDataset,
    BaselineEfficientNet,
    TARGET_SIZE,
)
from image_classifier.cnn_lstm.organoid_dataset import (
    OrganoidTimeSeriesDataset,
)
from image_classifier.cnn_lstm.train_organoid_lstm_single_phase import (
    collate_variable_length,
)
from image_classifier.cnn_lstm.train_temporal_ablation_attn import (
    OrganoidCNN_TAtt,
    BATCH_SIZE as EFFNET_TS_BATCH,
    NUM_WORKERS,
    ATTN_DROPOUT,
)
from run_per_day_study import (
    filter_ids_with_frames_up_to_day,
    find_best_threshold,
    build_threshold_results,
)


def add_threshold_results_per_day(
    out_dir, data, device, image_key="overlay_path", in_channels=3
):
    """Load checkpoint from out_dir, run val/test inference, return threshold_results."""
    with open(out_dir / "results.json") as f:
        data_json = json.load(f)
    day = float(data_json["day"])
    train_ids, val_ids, test_ids = data["train_ids"], data["val_ids"], data["test_ids"]
    series_metadata, data_dict = data["series_metadata"], data["data"]

    ckpt_path = out_dir / f"model_day_{day}.pth"
    if not ckpt_path.exists():
        ckpt_path = out_dir / f"model_day_{int(day)}.pth"
    if not ckpt_path.exists():
        alt = list(out_dir.glob("model_day_*.pth"))
        if not alt:
            return None
        ckpt_path = alt[0]
    eval_tf = T.Compose([T.Resize(TARGET_SIZE)])
    val_ds = SingleDayOrganoidDataset(
        val_ids,
        series_metadata,
        data_dict,
        day,
        transform=eval_tf,
        image_key=image_key,
        use_rgb_mask=False,
    )
    test_ds = SingleDayOrganoidDataset(
        test_ids,
        series_metadata,
        data_dict,
        day,
        transform=eval_tf,
        image_key=image_key,
        use_rgb_mask=False,
    )
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=0)
    model = BaselineEfficientNet(in_channels=in_channels).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    val_probs, val_labels = [], []
    test_probs, test_labels = [], []
    with torch.no_grad():
        for imgs, labels, _ in val_loader:
            logits = model(imgs.to(device))
            val_probs.extend(torch.sigmoid(logits).cpu().numpy().ravel())
            val_labels.extend(labels.numpy().ravel())
        for imgs, labels, _ in test_loader:
            logits = model(imgs.to(device))
            test_probs.extend(torch.sigmoid(logits).cpu().numpy().ravel())
            test_labels.extend(labels.numpy().ravel())
    val_probs = np.array(val_probs)
    val_labels = np.array(val_labels)
    test_probs = np.array(test_probs)
    test_labels = np.array(test_labels)
    best_thresh, _ = find_best_threshold(val_probs, val_labels)
    return build_threshold_results(
        val_probs, val_labels, test_probs, test_labels, best_thresh
    )


def add_threshold_results_effnet_ts(
    out_dir, data, device, image_key="overlay_path", in_channels=3
):
    """Load best_model.pth and global_mean, run val/test inference, return threshold_results."""
    with open(out_dir / "results.json") as f:
        data_json = json.load(f)
    day = float(data_json["day"])
    train_ids, val_ids, test_ids = data["train_ids"], data["val_ids"], data["test_ids"]
    series_metadata, data_dict = data["series_metadata"], data["data"]

    if (
        not (out_dir / "best_model.pth").exists()
        or not (out_dir / "global_mean.npy").exists()
    ):
        return None
    global_mean = np.load(out_dir / "global_mean.npy", allow_pickle=True)
    train_ids_f = filter_ids_with_frames_up_to_day(train_ids, series_metadata, day)
    val_ids_f = filter_ids_with_frames_up_to_day(val_ids, series_metadata, day)
    test_ids_f = filter_ids_with_frames_up_to_day(test_ids, series_metadata, day)
    val_ds = OrganoidTimeSeriesDataset(
        val_ids_f,
        series_metadata,
        data_dict,
        global_mean=global_mean,
        max_day=day,
        image_key=image_key,
        input_rgb_mask=False,
    )
    test_ds = OrganoidTimeSeriesDataset(
        test_ids_f,
        series_metadata,
        data_dict,
        global_mean=global_mean,
        max_day=day,
        image_key=image_key,
        input_rgb_mask=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=EFFNET_TS_BATCH,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_variable_length,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=EFFNET_TS_BATCH,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_variable_length,
    )
    model = OrganoidCNN_TAtt(attn_dropout=ATTN_DROPOUT, in_channels=in_channels).to(
        device
    )
    ckpt = torch.load(out_dir / "best_model.pth", map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    val_probs, val_labels = [], []
    test_probs, test_labels = [], []
    with torch.no_grad():
        for seqs, days_n, labels, _, _ in val_loader:
            logits, _ = model(seqs.to(device), days_n.to(device).float())
            val_probs.extend(torch.sigmoid(logits).cpu().numpy().ravel())
            val_labels.extend(labels.cpu().numpy().ravel())
        for seqs, days_n, labels, _, _ in test_loader:
            logits, _ = model(seqs.to(device), days_n.to(device).float())
            test_probs.extend(torch.sigmoid(logits).cpu().numpy().ravel())
            test_labels.extend(labels.cpu().numpy().ravel())
    val_probs = np.array(val_probs)
    val_labels = np.array(val_labels)
    test_probs = np.array(test_probs)
    test_labels = np.array(test_labels)
    best_thresh, _ = find_best_threshold(val_probs, val_labels)
    return build_threshold_results(
        val_probs, val_labels, test_probs, test_labels, best_thresh
    )


def main():
    base_dir = COMPARISON_RUNS / "per_day_study_overlay"
    if not base_dir.exists():
        print("per_day_study_overlay not found.")
        return
    split_dir = REPO_ROOT / "data_splits"
    train_ids, val_ids, test_ids, series_metadata, data_dict = load_split_data(
        split_dir / "both_train_base.json",
        split_dir / "both_val_base.json",
        split_dir / "both_test_base.json",
    )
    data = {
        "train_ids": train_ids,
        "val_ids": val_ids,
        "test_ids": test_ids,
        "series_metadata": series_metadata,
        "data": data_dict,
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_key, in_channels = "overlay_path", 3
    updated = 0
    for model_type, model_dir in [("per_day", "per_day"), ("effnet_ts", "effnet_ts")]:
        path = base_dir / model_dir
        if not path.exists():
            continue
        for day_dir in path.iterdir():
            if not day_dir.is_dir():
                continue
            res_file = day_dir / "results.json"
            if not res_file.exists():
                continue
            with open(res_file) as f:
                data_json = json.load(f)
            if "threshold_results" in data_json:
                continue
            print(f"Adding threshold_results: {model_type} {day_dir.name}")
            try:
                if model_type == "per_day":
                    tr = add_threshold_results_per_day(
                        day_dir,
                        data,
                        device,
                        image_key=image_key,
                        in_channels=in_channels,
                    )
                else:
                    tr = add_threshold_results_effnet_ts(
                        day_dir,
                        data,
                        device,
                        image_key=image_key,
                        in_channels=in_channels,
                    )
                if tr is None:
                    print("  Skip (no checkpoint or failed)")
                    continue
                data_json["threshold_results"] = tr
                with open(res_file, "w") as f:
                    json.dump(data_json, f, indent=2)
                updated += 1
            except Exception as e:
                print(f"  Error: {e}")
                import traceback

                traceback.print_exc()
    print(f"Updated {updated} results.json files.")


if __name__ == "__main__":
    main()
