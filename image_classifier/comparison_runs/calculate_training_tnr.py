#!/usr/bin/env python3
"""
Calculate training TNR for CNN-LSTM model from all_data trial.
"""

import sys
import os
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import confusion_matrix

# Load environment variables
REPO_ROOT = Path(__file__).resolve().parents[2]
env_file = REPO_ROOT / ".env"
if env_file.exists():
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                if "=" in line:
                    key, value = line.split("=", 1)
                    os.environ[key] = value

# Set required paths
os.environ.setdefault("BASE_PATH", "/net/projects2/promega/data-analysis")
os.environ.setdefault("OUTPUT_FOLDER", "/net/projects2/promega/data-analysis/output")
os.environ.setdefault(
    "RAW_IMAGE_DATA", "/net/projects2/promega/data-analysis/raw-images"
)
os.environ.setdefault(
    "IMAGE_VERIFICATION_FORM",
    "/net/projects2/promega/data-analysis/image-verification-form.json",
)
os.environ.setdefault("PLOTS_FOLDER", "/net/projects2/promega/data-analysis/plots")
os.environ.setdefault("LOGS_FOLDER", "/net/projects2/promega/data-analysis/logs")
os.environ.setdefault("NPY_OUTPUTS", "/net/projects2/promega/data-analysis/npy-outputs")
os.environ.setdefault(
    "PREDICTIONS_DIR", "/net/projects2/promega/data-analysis/predictions"
)
os.environ.setdefault(
    "SURVEY_RESULTS", "/net/projects2/promega/data-analysis/survey-results"
)
os.environ.setdefault(
    "MANUAL_MASKS_DIR", "/net/projects2/promega/data-analysis/manual-masks"
)
os.environ.setdefault("META_FILE", "/net/projects2/promega/data-analysis/metadata.json")
os.environ.setdefault(
    "RAW_IMAGE_MAPPING_JSON", "/net/projects2/promega/data-analysis/image-mapping.json"
)
os.environ.setdefault("TARGET_WIDTH", "512")
os.environ.setdefault("TARGET_HEIGHT", "384")
os.environ.setdefault(
    "TRAIN_RESIZED_DIR", "/net/projects2/promega/data-analysis/train-resized"
)
os.environ.setdefault(
    "TRAIN_MANUAL_MAPPING_DIR", "/net/projects2/promega/data-analysis/train-mapping"
)
os.environ.setdefault(
    "TRAIN_MANUAL_PROCESSED_DIR", "/net/projects2/promega/data-analysis/train-processed"
)
os.environ.setdefault(
    "TRAIN_SPLITS_DIR", "/net/projects2/promega/data-analysis/train-splits"
)
os.environ.setdefault(
    "INFER_RESIZED_DIR", "/net/projects2/promega/data-analysis/infer-resized"
)
os.environ.setdefault(
    "INFER_MAPPING_TOTAL_JSON",
    "/net/projects2/promega/data-analysis/infer-mapping.json",
)
os.environ.setdefault(
    "MANUAL_THRESHOLD_MAPPING",
    "/net/projects2/promega/data-analysis/threshold-mapping.json",
)

# Add project root to path
sys.path.insert(0, str(REPO_ROOT))

from image_classifier.cnn_lstm.load_split_data import load_split_data
from image_classifier.cnn_lstm.organoid_dataset import OrganoidTimeSeriesDataset
from image_classifier.cnn_lstm.organoid_model import OrganoidCNN_LSTM
from image_classifier.cnn_lstm.train_organoid_lstm_single_phase import (
    collate_variable_length,
)
from torch.utils.data import DataLoader


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    split_dir = REPO_ROOT / "data_splits"
    train_ids, val_ids, test_ids, series_metadata, data = load_split_data(
        split_dir / "both_train_base.json",
        split_dir / "both_val_base.json",
        split_dir / "both_test_base.json",
    )

    print(f"\nData: train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}")

    # Load model
    model_dir = Path(__file__).parent / "cnn_lstm_baseline"
    model_path = model_dir / "best_model.pth"
    global_mean = np.load(model_dir / "global_mean.npy")

    checkpoint = torch.load(model_path, map_location=device)
    model = OrganoidCNN_LSTM(num_classes=2, lstm_hidden=256, lstm_layers=2).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Create training dataloader
    train_dataset = OrganoidTimeSeriesDataset(
        train_ids, series_metadata, data, global_mean=global_mean
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_variable_length,
    )

    # Get predictions on training set
    print("\nGetting predictions on training set...")
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in train_loader:
            images, days_norm, labels, weights, ids = batch
            images = images.to(device)
            labels = labels.to(device).long()

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Calculate confusion matrix and TNR
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
    TN, FP, FN, TP = cm.ravel()

    TNR = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    TPR = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    accuracy = (TP + TN) / len(all_labels)

    print("\n" + "=" * 70)
    print("TRAINING SET METRICS (CNN-LSTM - All Data Trial)")
    print("=" * 70)
    print("\nConfusion Matrix:")
    print("                Predicted")
    print("              Bad    Good")
    print(f"Actual Bad   {TN:4d}   {FP:4d}")
    print(f"       Good  {FN:4d}   {TP:4d}")
    print("\nMetrics:")
    print(f"  TNR (Specificity): {TNR:.4f} ({TNR * 100:.1f}%)")
    print(f"  TPR (Sensitivity): {TPR:.4f} ({TPR * 100:.1f}%)")
    print(f"  Accuracy: {accuracy:.4f} ({accuracy * 100:.1f}%)")
    print(f"  Balanced Accuracy: {(TNR + TPR) / 2:.4f}")


if __name__ == "__main__":
    main()
