#!/usr/bin/env python3
"""
Run CNN-LSTM with improved class weights and threshold-aware evaluation.
Changes:
1. Stronger class weights (5x penalty instead of 3.94x)
2. Threshold-aware evaluation (uses optimal threshold from tuning)
3. Early stopping based on balanced accuracy instead of just accuracy
"""

import sys
from pathlib import Path

# Add project root to path
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

# Import Amanda's ORIGINAL training functions (with modifications)
from image_classifier.cnn_lstm.organoid_dataset import (
    OrganoidTimeSeriesDataset,
    compute_global_mean_from_ids,
)
from image_classifier.cnn_lstm.organoid_model import OrganoidCNN_LSTM
from image_classifier.cnn_lstm.train_organoid_lstm_single_phase import (
    train_one_epoch,
    set_seed,
    collate_variable_length,
)

# Import our data loader
from image_classifier.cnn_lstm.load_split_data import load_split_data

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import json
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)
from tqdm import tqdm


def evaluate_with_threshold(model, dataloader, criterion, device, threshold=0.5):
    """Evaluate model with tunable threshold for binary classification."""
    model.eval()
    total_loss = 0
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            images, days_norm, labels, weights, ids = batch

            images = images.to(device)
            labels = labels.to(device).long()

            outputs = model(images)  # (B, 2) logits
            loss = criterion(outputs, labels)

            total_loss += loss.item()

            # Get probabilities for Good class (class 1)
            probs = torch.softmax(outputs, dim=1)
            prob_good = probs[:, 1].cpu().numpy()

            all_probs.extend(prob_good)
            all_labels.extend(labels.cpu().numpy())

    # Apply threshold
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    preds = (all_probs >= threshold).astype(int)

    # Calculate metrics
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, preds, average="binary", zero_division=0
    )

    # Calculate TNR and balanced accuracy
    cm = confusion_matrix(all_labels, preds, labels=[0, 1])
    if cm.size == 4:
        TN, FP, FN, TP = cm.ravel()
        TNR = TN / (TN + FP) if (TN + FP) > 0 else 0.0
        TPR = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        balanced_acc = (TNR + TPR) / 2
    else:
        TNR = 0.0
        TPR = 0.0
        balanced_acc = accuracy

    return (
        avg_loss,
        accuracy,
        precision,
        recall,
        f1,
        TNR,
        balanced_acc,
        preds,
        all_labels,
    )


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Train CNN-LSTM with improved class weights and threshold"
    )
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--lstm-hidden", type=int, default=256, help="LSTM hidden size")
    parser.add_argument(
        "--lstm-layers", type=int, default=2, help="Number of LSTM layers"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--weight-multiplier",
        type=float,
        default=1.27,
        help="Multiplier for class weights (1.27 gives ~5x penalty)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.460,
        help="Classification threshold (from tuning)",
    )
    args = parser.parse_args()

    # Set random seed
    set_seed(args.seed)
    print(f"Random seed set to: {args.seed}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Output directory
    output_dir = Path(__file__).parent / "cnn_lstm_improved"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving outputs to: {output_dir}")

    # Load data split
    print("\n" + "=" * 70)
    print("LOADING DATA SPLIT")
    print("=" * 70)

    split_dir = REPO_ROOT / "data_splits"
    train_ids, val_ids, test_ids, series_metadata, data = load_split_data(
        split_dir / "both_train_base.json",
        split_dir / "both_val_base.json",
        split_dir / "both_test_base.json",
    )

    print(f"\nSplits: train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}")

    # Compute global mean
    print("\nComputing global mean from training set...")
    global_mean = compute_global_mean_from_ids(train_ids, series_metadata, data)
    np.save(output_dir / "global_mean.npy", global_mean)
    print(f"Saved global mean to {output_dir / 'global_mean.npy'}")

    # Create datasets
    train_dataset = OrganoidTimeSeriesDataset(
        train_ids, series_metadata, data, global_mean=global_mean
    )
    val_dataset = OrganoidTimeSeriesDataset(
        val_ids, series_metadata, data, global_mean=global_mean
    )
    test_dataset = OrganoidTimeSeriesDataset(
        test_ids, series_metadata, data, global_mean=global_mean
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_variable_length,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_variable_length,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_variable_length,
    )

    # Calculate STRONGER class weights
    train_labels = []
    for org_id in train_ids:
        meta = series_metadata[org_id]
        label_str = str(meta.get("label", "")).strip().lower()
        label = 1 if label_str in ("good", "acceptable", "accepted") else 0
        train_labels.append(label)

    n_good = sum(train_labels)
    n_bad = len(train_labels) - n_good

    # Original formula: weight_for_0 = len(train_labels) / (2 * n_bad)
    # With multiplier: weight_for_0 = len(train_labels) / (2 * n_bad) * multiplier
    # To get ~5x penalty: multiplier = 5 / 3.94 ≈ 1.27
    weight_for_0 = (
        (len(train_labels) / (2 * n_bad)) * args.weight_multiplier if n_bad > 0 else 1.0
    )
    weight_for_1 = len(train_labels) / (2 * n_good) if n_good > 0 else 1.0
    class_weights = torch.FloatTensor([weight_for_0, weight_for_1]).to(device)

    penalty_ratio = weight_for_0 / weight_for_1
    print(f"\nClass weights: Bad={weight_for_0:.3f}, Good={weight_for_1:.3f}")
    print(f"Penalty ratio: {penalty_ratio:.3f}x (target: ~5x)")

    # Create model
    print("\n" + "=" * 70)
    print("CREATING MODEL")
    print("=" * 70)

    model = OrganoidCNN_LSTM(
        num_classes=2, lstm_hidden=args.lstm_hidden, lstm_layers=args.lstm_layers
    ).to(device)

    for param in model.cnn.parameters():
        param.requires_grad = True

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5
    )

    # Training loop with threshold-aware evaluation
    print("\n" + "=" * 70)
    print("TRAINING WITH IMPROVED CLASS WEIGHTS AND THRESHOLD-AWARE EVALUATION")
    print("=" * 70)
    print(f"Using threshold: {args.threshold:.3f} (from tuning)")
    print("Early stopping based on balanced accuracy (TNR + TPR) / 2")

    best_balanced_acc = 0
    best_val_acc = 0
    best_val_loss = float("inf")
    patience_counter = 0
    train_history = []

    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Use threshold-aware evaluation
        (
            val_loss,
            val_acc,
            val_prec,
            val_rec,
            val_f1,
            val_tnr,
            val_balanced_acc,
            _,
            _,
        ) = evaluate_with_threshold(
            model, val_loader, criterion, device, threshold=args.threshold
        )

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch + 1:2d}/{args.epochs}: Train {train_acc:.3f} | Val {val_acc:.3f} | "
            f"TNR {val_tnr:.3f} | Balanced {val_balanced_acc:.3f} | F1 {val_f1:.3f} | LR {current_lr:.6f}"
        )

        train_history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_tnr": val_tnr,
                "val_balanced_acc": val_balanced_acc,
                "val_precision": val_prec,
                "val_recall": val_rec,
                "val_f1": val_f1,
            }
        )

        # Early stopping based on balanced accuracy
        if val_balanced_acc > best_balanced_acc:
            best_balanced_acc = val_balanced_acc
            best_val_acc = val_acc
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                    "val_balanced_acc": val_balanced_acc,
                    "val_tnr": val_tnr,
                    "val_loss": val_loss,
                },
                output_dir / "best_model.pth",
            )
            print(
                f"  *** Best Balanced Acc: {val_balanced_acc:.4f} (TNR: {val_tnr:.4f})"
            )
        else:
            patience_counter += 1
            if patience_counter >= 10:
                print(f"  Early stopping at epoch {epoch + 1}")
                break

    print(f"\n✅ Training Complete! Final Best Balanced Acc: {best_balanced_acc:.4f}")

    # Final evaluation on test set with threshold
    print("\n" + "=" * 70)
    print("FINAL EVALUATION ON TEST SET (with threshold)")
    print("=" * 70)

    checkpoint = torch.load(output_dir / "best_model.pth")
    model.load_state_dict(checkpoint["model_state_dict"])

    (
        test_loss,
        test_acc,
        test_precision,
        test_recall,
        test_f1,
        test_tnr,
        test_balanced_acc,
        test_preds,
        test_labels,
    ) = evaluate_with_threshold(
        model, test_loader, criterion, device, threshold=args.threshold
    )

    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test TNR (Specificity): {test_tnr:.4f} ({test_tnr * 100:.1f}%)")
    print(f"Test TPR (Sensitivity): {test_recall:.4f} ({test_recall * 100:.1f}%)")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall: {test_recall:.4f}")
    print(f"Test F1: {test_f1:.4f}")
    print(f"Test Balanced Acc: {test_balanced_acc:.4f}")

    cm = confusion_matrix(test_labels, test_preds, labels=[0, 1])
    print("\nConfusion Matrix:")
    print("                Predicted")
    print("              Bad    Good")
    print(f"Actual Bad   {cm[0, 0]:4d}   {cm[0, 1]:4d}")
    print(f"       Good  {cm[1, 0]:4d}   {cm[1, 1]:4d}")

    # Save results
    results = {
        "args": vars(args),
        "best_val_acc": best_val_acc,
        "best_val_balanced_acc": best_balanced_acc,
        "best_val_loss": best_val_loss,
        "test_acc": test_acc,
        "test_tnr": test_tnr,
        "test_tpr": test_recall,
        "test_precision": test_precision,
        "test_recall": test_recall,
        "test_f1": test_f1,
        "test_balanced_acc": test_balanced_acc,
        "confusion_matrix": cm.tolist(),
        "train_history": train_history,
        "class_weights": {
            "bad": float(weight_for_0),
            "good": float(weight_for_1),
            "ratio": float(penalty_ratio),
        },
        "threshold_used": args.threshold,
    }

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_dir / 'results.json'}")
    print(f"Best model saved to {output_dir / 'best_model.pth'}")


if __name__ == "__main__":
    main()
