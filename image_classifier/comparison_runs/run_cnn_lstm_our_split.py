#!/usr/bin/env python3
"""
Run Amanda's ORIGINAL CNN-LSTM model using our own data split.
ONLY CHANGE: Data loading (uses our split instead of Amanda's metadata files)
EVERYTHING ELSE: Exactly as Amanda wrote it (model, training, hyperparameters)
"""

import sys
from pathlib import Path

# Add project root to path
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

# Import Amanda's ORIGINAL training functions (unchanged)
from image_classifier.cnn_lstm.organoid_dataset import (
    OrganoidTimeSeriesDataset,
    compute_global_mean_from_ids,
)
from image_classifier.cnn_lstm.organoid_model import OrganoidCNN_LSTM
from image_classifier.cnn_lstm.train_organoid_lstm_single_phase import (
    train_one_epoch,
    evaluate,
    set_seed,
    collate_variable_length,
)

# Import our data loader (ONLY change)
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


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Train CNN-LSTM with our data split (Amanda's original model)"
    )
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--lstm-hidden", type=int, default=256, help="LSTM hidden size")
    parser.add_argument(
        "--lstm-layers", type=int, default=2, help="Number of LSTM layers"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Set random seed (Amanda's original function)
    set_seed(args.seed)
    print(f"Random seed set to: {args.seed}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Output directory (using exclude-nothing data)
    output_dir = Path(__file__).parent / "amanda_cnn_lstm_all_data"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving outputs to: {output_dir}")

    # ========== ONLY CHANGE: Load our split data instead of Amanda's ==========
    print("\n" + "=" * 70)
    print("LOADING OUR DATA SPLIT (ONLY CHANGE FROM ORIGINAL)")
    print("=" * 70)

    # Use exclude-nothing version (includes stitched/presplit samples)
    split_dir = REPO_ROOT / "data_splits"
    train_ids, val_ids, test_ids, series_metadata, data = load_split_data(
        split_dir / "both_train_base.json",
        split_dir / "both_val_base.json",
        split_dir / "both_test_base.json",
    )
    # ========== END OF ONLY CHANGE ==========

    # Everything below is EXACTLY as Amanda wrote it
    print(f"\nSplits: train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}")

    # Compute global mean from training set (Amanda's original function)
    print("\nComputing global mean from training set...")
    global_mean = compute_global_mean_from_ids(train_ids, series_metadata, data)

    # Save
    np.save(output_dir / "global_mean.npy", global_mean)
    print(f"Saved global mean to {output_dir / 'global_mean.npy'}")

    # Create datasets (Amanda's original dataset class)
    train_dataset = OrganoidTimeSeriesDataset(
        train_ids, series_metadata, data, global_mean=global_mean
    )
    val_dataset = OrganoidTimeSeriesDataset(
        val_ids, series_metadata, data, global_mean=global_mean
    )
    test_dataset = OrganoidTimeSeriesDataset(
        test_ids, series_metadata, data, global_mean=global_mean
    )

    # Create dataloaders with custom collate function for variable-length sequences
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

    # Calculate class weights (Amanda's original logic)
    train_labels = []
    for org_id in train_ids:
        meta = series_metadata[org_id]
        label_str = str(meta.get("label", "")).strip().lower()
        label = 1 if label_str in ("good", "acceptable", "accepted") else 0
        train_labels.append(label)

    n_good = sum(train_labels)
    n_bad = len(train_labels) - n_good
    weight_for_0 = len(train_labels) / (2 * n_bad) if n_bad > 0 else 1.0
    weight_for_1 = len(train_labels) / (2 * n_good) if n_good > 0 else 1.0
    class_weights = torch.FloatTensor([weight_for_0, weight_for_1]).to(device)

    print(f"\nClass weights: Bad={weight_for_0:.3f}, Good={weight_for_1:.3f}")

    # Create model (Amanda's original model)
    print("\n" + "=" * 70)
    print("CREATING MODEL (AMANDA'S ORIGINAL)")
    print("=" * 70)

    model = OrganoidCNN_LSTM(
        num_classes=2, lstm_hidden=args.lstm_hidden, lstm_layers=args.lstm_layers
    ).to(device)

    # SINGLE-PHASE: Unfreeze all parameters (Amanda's original logic)
    for param in model.cnn.parameters():
        param.requires_grad = True

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print("✅ SINGLE-PHASE: All parameters trainable from start")

    # Loss and optimizer (Amanda's original settings)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5
    )

    # Training loop (Amanda's original training loop)
    print("\n" + "=" * 70)
    print("TRAINING (AMANDA'S ORIGINAL TRAINING LOOP)")
    print("=" * 70)

    best_val_acc = 0
    best_val_loss = float("inf")
    patience_counter = 0
    train_history = []

    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc, val_prec, val_rec, val_f1, _, _ = evaluate(
            model, val_loader, criterion, device
        )

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch + 1:2d}/{args.epochs}: Train {train_acc:.3f} | Val {val_acc:.3f} | F1 {val_f1:.3f} | LR {current_lr:.6f}"
        )

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
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                    "val_loss": val_loss,
                },
                output_dir / "best_model.pth",
            )
            print(f"  *** Best: {val_acc:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= 10:
                print(f"  Early stopping at epoch {epoch + 1}")
                break

    print(f"\n✅ Training Complete! Final Best Val Acc: {best_val_acc:.4f}")

    # Final evaluation (Amanda's original evaluation)
    print("\n" + "=" * 70)
    print("FINAL EVALUATION ON TEST SET")
    print("=" * 70)

    checkpoint = torch.load(output_dir / "best_model.pth")
    model.load_state_dict(checkpoint["model_state_dict"])

    (
        test_loss,
        test_acc,
        test_precision,
        test_recall,
        test_f1,
        test_preds,
        test_labels,
    ) = evaluate(model, test_loader, criterion, device)

    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall: {test_recall:.4f}")
    print(f"Test F1: {test_f1:.4f}")

    cm = confusion_matrix(test_labels, test_preds)
    print(f"\nConfusion Matrix:")
    print(f"                Predicted")
    print(f"              Bad    Good")
    print(f"Actual Bad   {cm[0, 0]:4d}   {cm[0, 1]:4d}")
    print(f"       Good  {cm[1, 0]:4d}   {cm[1, 1]:4d}")

    # Save results (Amanda's original format)
    results = {
        "args": vars(args),
        "best_val_acc": best_val_acc,
        "best_val_loss": best_val_loss,
        "test_acc": test_acc,
        "test_precision": test_precision,
        "test_recall": test_recall,
        "test_f1": test_f1,
        "confusion_matrix": cm.tolist(),
        "train_history": train_history,
    }

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_dir / 'results.json'}")
    print(f"Best model saved to {output_dir / 'best_model.pth'}")


if __name__ == "__main__":
    main()
