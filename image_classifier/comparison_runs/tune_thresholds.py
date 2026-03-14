#!/usr/bin/env python3
"""
Threshold tuning script for both CNN-LSTM and EfficientNet models.
Finds optimal threshold on validation set to maximize TNR.
"""

import sys
import os
from pathlib import Path
import numpy as np
import torch
import json
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

# Load environment variables from .env file
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

# Set required paths if not already set (fallback)
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
from image_classifier.cnn_lstm.organoid_dataset import (
    OrganoidTimeSeriesDataset,
)
from image_classifier.cnn_lstm.organoid_model import OrganoidCNN_LSTM
from image_classifier.cnn_lstm.train_organoid_lstm_single_phase import (
    collate_variable_length,
)
from image_classifier.cnn_lstm.train_base_model import SingleDayOrganoidDataset
from torch.utils.data import DataLoader


def get_cnn_lstm_probs(model, dataloader, device):
    """Get probability predictions from CNN-LSTM model."""
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            images, days_norm, labels, weights, ids = batch
            images = images.to(device)
            labels = labels.to(device).long()

            outputs = model(images)  # (B, 2) logits
            probs = torch.softmax(outputs, dim=1)  # (B, 2)
            prob_good = probs[:, 1].cpu().numpy()  # Probability of Good class

            all_probs.extend(prob_good)
            all_labels.extend(labels.cpu().numpy())

    return np.array(all_probs), np.array(all_labels)


def get_efficientnet_probs(model, dataloader, device):
    """Get probability predictions from EfficientNet model."""
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels, ids in dataloader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            logits = model(imgs)
            probs = torch.sigmoid(logits).cpu().numpy()  # Already probabilities

            all_probs.extend(probs)
            all_labels.extend(labels.cpu().numpy())

    return np.array(all_probs), np.array(all_labels)


def get_effnet_timeseries_probs(model, dataloader, device):
    """Get probability predictions from time-series EfficientNet (temporal attention)."""
    model.eval()
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for seqs, days_norm, labels, weights, ids in dataloader:
            seqs = seqs.to(device)
            days_norm = days_norm.to(device).float()
            logits, _ = model(seqs, days_norm)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.cpu().numpy())
    return np.array(all_probs), np.array(all_labels)


def calculate_metrics_at_threshold(probs, labels, threshold):
    """Calculate metrics at a given threshold."""
    preds = (probs >= threshold).astype(int)
    cm = confusion_matrix(labels, preds, labels=[0, 1])

    if cm.size == 4:
        TN, FP, FN, TP = cm.ravel()
    else:
        TN, FP, FN, TP = 0, 0, 0, 0

    TNR = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    TPR = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    accuracy = (TP + TN) / len(labels) if len(labels) > 0 else 0.0

    prec, rec, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )

    balanced_acc = (TNR + TPR) / 2

    return {
        "threshold": threshold,
        "TN": int(TN),
        "FP": int(FP),
        "FN": int(FN),
        "TP": int(TP),
        "TNR": TNR,
        "TPR": TPR,
        "accuracy": accuracy,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "balanced_acc": balanced_acc,
    }


def find_optimal_threshold(probs, labels, metric="tnr", min_tnr=0.0):
    """Find optimal threshold based on specified metric."""
    thresholds = np.linspace(0.1, 0.9, 81)  # 0.1 to 0.9, step 0.01
    best_metric = -1
    best_thresh = 0.5
    best_results = None
    all_results = []

    for thresh in thresholds:
        results = calculate_metrics_at_threshold(probs, labels, thresh)
        all_results.append(results)

        # Filter by minimum TNR if specified
        if results["TNR"] < min_tnr:
            continue

        # Choose metric to optimize
        if metric == "tnr":
            score = results["TNR"]
        elif metric == "balanced_acc":
            score = results["balanced_acc"]
        elif metric == "f1":
            score = results["f1"]
        elif metric == "tnr_f1":
            score = (results["TNR"] + results["f1"]) / 2
        else:
            score = results["TNR"]

        if score > best_metric:
            best_metric = score
            best_thresh = thresh
            best_results = results

    return best_thresh, best_results, all_results


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

    print(
        f"\nData loaded: train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}"
    )

    # ========== CNN-LSTM Threshold Tuning ==========
    print("\n" + "=" * 70)
    print("CNN-LSTM THRESHOLD TUNING")
    print("=" * 70)

    # Load CNN-LSTM model
    cnn_lstm_dir = Path(__file__).parent / "cnn_lstm_baseline"
    cnn_lstm_model_path = cnn_lstm_dir / "best_model.pth"

    if not cnn_lstm_model_path.exists():
        print(f"❌ CNN-LSTM model not found at {cnn_lstm_model_path}")
        cnn_lstm_results = None
    else:
        # Load model
        checkpoint = torch.load(cnn_lstm_model_path, map_location=device)
        global_mean = np.load(cnn_lstm_dir / "global_mean.npy")

        model = OrganoidCNN_LSTM(num_classes=2, lstm_hidden=256, lstm_layers=2).to(
            device
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        # Create validation dataloader
        val_dataset = OrganoidTimeSeriesDataset(
            val_ids, series_metadata, data, global_mean=global_mean
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=8,
            shuffle=False,
            num_workers=4,
            collate_fn=collate_variable_length,
        )

        # Get probabilities
        print("Getting predictions on validation set...")
        val_probs, val_labels = get_cnn_lstm_probs(model, val_loader, device)
        print(f"  Validation set: {len(val_probs)} samples")

        # Find optimal threshold
        print("\nSearching for optimal threshold...")
        best_thresh, best_results, all_results = find_optimal_threshold(
            val_probs, val_labels, metric="tnr_f1", min_tnr=0.5
        )

        print(f"\nOptimal threshold: {best_thresh:.3f}")
        print(f"  TNR: {best_results['TNR']:.4f} ({best_results['TNR'] * 100:.1f}%)")
        print(f"  TPR: {best_results['TPR']:.4f} ({best_results['TPR'] * 100:.1f}%)")
        print(f"  Accuracy: {best_results['accuracy']:.4f}")
        print(f"  F1: {best_results['f1']:.4f}")
        print(f"  Balanced Acc: {best_results['balanced_acc']:.4f}")
        print(
            f"  Confusion Matrix: TN={best_results['TN']}, FP={best_results['FP']}, FN={best_results['FN']}, TP={best_results['TP']}"
        )

        # Evaluate on test set with optimal threshold
        print("\nEvaluating test set with optimal threshold...")
        test_dataset = OrganoidTimeSeriesDataset(
            test_ids, series_metadata, data, global_mean=global_mean
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=8,
            shuffle=False,
            num_workers=4,
            collate_fn=collate_variable_length,
        )
        test_probs, test_labels = get_cnn_lstm_probs(model, test_loader, device)
        test_results = calculate_metrics_at_threshold(
            test_probs, test_labels, best_thresh
        )

        print(f"\nTest Set Results (threshold={best_thresh:.3f}):")
        print(f"  TNR: {test_results['TNR']:.4f} ({test_results['TNR'] * 100:.1f}%)")
        print(f"  TPR: {test_results['TPR']:.4f} ({test_results['TPR'] * 100:.1f}%)")
        print(f"  Accuracy: {test_results['accuracy']:.4f}")
        print(f"  F1: {test_results['f1']:.4f}")
        print(
            f"  Confusion Matrix: TN={test_results['TN']}, FP={test_results['FP']}, FN={test_results['FN']}, TP={test_results['TP']}"
        )

        cnn_lstm_results = {
            "optimal_threshold": float(best_thresh),
            "val_results": best_results,
            "test_results": test_results,
            "all_threshold_results": all_results,
        }

    # ========== EfficientNet Threshold Tuning ==========
    print("\n" + "=" * 70)
    print("EFFICIENTNET THRESHOLD TUNING")
    print("=" * 70)

    # Load EfficientNet model
    effnet_dir = Path(__file__).parent / "our_efficientnet_all_data" / "day_30"
    effnet_model_path = effnet_dir / "model_day_30.pth"

    if not effnet_model_path.exists():
        print(f"❌ EfficientNet model not found at {effnet_model_path}")
        effnet_results = None
    else:
        from image_classifier.cnn_lstm.train_base_model import BaselineEfficientNet

        # Load model
        model = BaselineEfficientNet().to(device)
        checkpoint = torch.load(effnet_model_path, map_location=device)
        # Checkpoint structure: {'state_dict': ..., 'target_day': ..., 'best_val_acc': ...}
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        else:
            model.load_state_dict(checkpoint)
        model.eval()

        # Create validation dataloader
        from torchvision import transforms as T

        eval_tf = T.Compose([T.Resize((384, 512))])
        val_dataset = SingleDayOrganoidDataset(
            val_ids, series_metadata, data, 30, transform=eval_tf
        )
        val_loader = DataLoader(
            val_dataset, batch_size=16, shuffle=False, num_workers=0
        )

        # Get probabilities
        print("Getting predictions on validation set...")
        val_probs, val_labels = get_efficientnet_probs(model, val_loader, device)
        print(f"  Validation set: {len(val_probs)} samples")

        # Find optimal threshold
        print("\nSearching for optimal threshold...")
        best_thresh, best_results, all_results = find_optimal_threshold(
            val_probs, val_labels, metric="tnr_f1", min_tnr=0.5
        )

        print(f"\nOptimal threshold: {best_thresh:.3f}")
        print(f"  TNR: {best_results['TNR']:.4f} ({best_results['TNR'] * 100:.1f}%)")
        print(f"  TPR: {best_results['TPR']:.4f} ({best_results['TPR'] * 100:.1f}%)")
        print(f"  Accuracy: {best_results['accuracy']:.4f}")
        print(f"  F1: {best_results['f1']:.4f}")
        print(f"  Balanced Acc: {best_results['balanced_acc']:.4f}")
        print(
            f"  Confusion Matrix: TN={best_results['TN']}, FP={best_results['FP']}, FN={best_results['FN']}, TP={best_results['TP']}"
        )

        # Evaluate on test set with optimal threshold
        print("\nEvaluating test set with optimal threshold...")
        test_dataset = SingleDayOrganoidDataset(
            test_ids, series_metadata, data, 30, transform=eval_tf
        )
        test_loader = DataLoader(
            test_dataset, batch_size=16, shuffle=False, num_workers=0
        )
        test_probs, test_labels = get_efficientnet_probs(model, test_loader, device)
        test_results = calculate_metrics_at_threshold(
            test_probs, test_labels, best_thresh
        )

        print(f"\nTest Set Results (threshold={best_thresh:.3f}):")
        print(f"  TNR: {test_results['TNR']:.4f} ({test_results['TNR'] * 100:.1f}%)")
        print(f"  TPR: {test_results['TPR']:.4f} ({test_results['TPR'] * 100:.1f}%)")
        print(f"  Accuracy: {test_results['accuracy']:.4f}")
        print(f"  F1: {test_results['f1']:.4f}")
        print(
            f"  Confusion Matrix: TN={test_results['TN']}, FP={test_results['FP']}, FN={test_results['FN']}, TP={test_results['TP']}"
        )

        effnet_results = {
            "optimal_threshold": float(best_thresh),
            "val_results": best_results,
            "test_results": test_results,
            "all_threshold_results": all_results,
        }

    # ========== EfficientNet Time-Series (Temporal Attention) Threshold Tuning ==========
    effnet_timeseries_results = None
    print("\n" + "=" * 70)
    print("EFFICIENTNET TIME-SERIES (Temporal Attention) THRESHOLD TUNING")
    print("=" * 70)

    effnet_ts_dir = Path(__file__).parent / "our_effnet_timeseries_all_data"
    effnet_ts_model_path = effnet_ts_dir / "best_model.pth"

    if not effnet_ts_model_path.exists():
        print(f"❌ EfficientNet time-series model not found at {effnet_ts_model_path}")
        effnet_timeseries_results = None
    else:
        from image_classifier.cnn_lstm.train_temporal_ablation_attn import (
            OrganoidCNN_TAtt,
        )

        checkpoint = torch.load(effnet_ts_model_path, map_location=device)
        global_mean_ts = np.load(effnet_ts_dir / "global_mean.npy")
        model = OrganoidCNN_TAtt(attn_dropout=0.4).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()

        val_dataset_ts = OrganoidTimeSeriesDataset(
            val_ids, series_metadata, data, global_mean=global_mean_ts
        )
        val_loader_ts = DataLoader(
            val_dataset_ts,
            batch_size=8,
            shuffle=False,
            num_workers=4,
            collate_fn=collate_variable_length,
        )
        print("Getting predictions on validation set...")
        val_probs, val_labels = get_effnet_timeseries_probs(
            model, val_loader_ts, device
        )
        print(f"  Validation set: {len(val_probs)} samples")

        best_thresh, best_results, all_results = find_optimal_threshold(
            val_probs, val_labels, metric="tnr_f1", min_tnr=0.5
        )
        print(f"\nOptimal threshold: {best_thresh:.3f}")
        print(f"  TNR: {best_results['TNR']:.4f} ({best_results['TNR'] * 100:.1f}%)")
        print(f"  TPR: {best_results['TPR']:.4f} ({best_results['TPR'] * 100:.1f}%)")
        print(
            f"  Accuracy: {best_results['accuracy']:.4f} | F1: {best_results['f1']:.4f}"
        )
        print(
            f"  Confusion Matrix: TN={best_results['TN']}, FP={best_results['FP']}, FN={best_results['FN']}, TP={best_results['TP']}"
        )

        test_dataset_ts = OrganoidTimeSeriesDataset(
            test_ids, series_metadata, data, global_mean=global_mean_ts
        )
        test_loader_ts = DataLoader(
            test_dataset_ts,
            batch_size=8,
            shuffle=False,
            num_workers=4,
            collate_fn=collate_variable_length,
        )
        test_probs, test_labels = get_effnet_timeseries_probs(
            model, test_loader_ts, device
        )
        test_results = calculate_metrics_at_threshold(
            test_probs, test_labels, best_thresh
        )
        print(f"\nTest Set Results (threshold={best_thresh:.3f}):")
        print(f"  TNR: {test_results['TNR']:.4f} ({test_results['TNR'] * 100:.1f}%)")
        print(f"  TPR: {test_results['TPR']:.4f} ({test_results['TPR'] * 100:.1f}%)")
        print(
            f"  Accuracy: {test_results['accuracy']:.4f} | F1: {test_results['f1']:.4f}"
        )
        print(
            f"  Confusion Matrix: TN={test_results['TN']}, FP={test_results['FP']}, FN={test_results['FN']}, TP={test_results['TP']}"
        )

        effnet_timeseries_results = {
            "optimal_threshold": float(best_thresh),
            "val_results": best_results,
            "test_results": test_results,
            "all_threshold_results": all_results,
        }

    # Save results
    output_file = Path(__file__).parent / "threshold_tuning_results.json"
    results = {
        "cnn_lstm": cnn_lstm_results,
        "efficientnet": effnet_results,
        "effnet_timeseries": effnet_timeseries_results,
    }
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print("=" * 70)
    if cnn_lstm_results:
        print("\nCNN-LSTM:")
        print(f"  Optimal threshold: {cnn_lstm_results['optimal_threshold']:.3f}")
        print(
            f"  Test TNR: {cnn_lstm_results['test_results']['TNR']:.4f} ({cnn_lstm_results['test_results']['TNR'] * 100:.1f}%)"
        )
        print(f"  Test F1: {cnn_lstm_results['test_results']['f1']:.4f}")
    if effnet_results:
        print("\nEfficientNet:")
        print(f"  Optimal threshold: {effnet_results['optimal_threshold']:.3f}")
        print(
            f"  Test TNR: {effnet_results['test_results']['TNR']:.4f} ({effnet_results['test_results']['TNR'] * 100:.1f}%)"
        )
        print(f"  Test F1: {effnet_results['test_results']['f1']:.4f}")
    if effnet_timeseries_results:
        print("\nEfficientNet Time-Series:")
        print(
            f"  Optimal threshold: {effnet_timeseries_results['optimal_threshold']:.3f}"
        )
        print(
            f"  Test TNR: {effnet_timeseries_results['test_results']['TNR']:.4f} ({effnet_timeseries_results['test_results']['TNR'] * 100:.1f}%)"
        )
        print(f"  Test F1: {effnet_timeseries_results['test_results']['f1']:.4f}")

    # --- TNR-focused summary (Bad class / specificity) ---
    print(f"\n{'=' * 70}")
    print("TNR (True Negative Rate) — Bad correctly predicted as Bad")
    print("=" * 70)
    if cnn_lstm_results:
        vr, tr = cnn_lstm_results["val_results"], cnn_lstm_results["test_results"]
        print(
            f"\n  CNN-LSTM:              Val TNR {vr['TNR'] * 100:.1f}%  →  Test TNR {tr['TNR'] * 100:.1f}%  (TN={tr['TN']}, FP={tr['FP']}, 9 Bad in test)"
        )
    if effnet_results:
        vr, tr = effnet_results["val_results"], effnet_results["test_results"]
        print(
            f"  EfficientNet (1-day):  Val TNR {vr['TNR'] * 100:.1f}%  →  Test TNR {tr['TNR'] * 100:.1f}%  (TN={tr['TN']}, FP={tr['FP']}, 9 Bad in test)"
        )
    if effnet_timeseries_results:
        vr, tr = (
            effnet_timeseries_results["val_results"],
            effnet_timeseries_results["test_results"],
        )
        print(
            f"  EfficientNet (TS):     Val TNR {vr['TNR'] * 100:.1f}%  →  Test TNR {tr['TNR'] * 100:.1f}%  (TN={tr['TN']}, FP={tr['FP']}, 9 Bad in test)"
        )
    print()

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
