#!/usr/bin/env python3
"""Training-curve plots for the image-survey classifier."""

import matplotlib.pyplot as plt

METRICS_TO_COMBINE = [
    "loss", "val_loss", "accuracy", "val_accuracy", "auc",
    "val_auc", "precision", "val_precision", "recall", "val_recall",
]


def plot_model_metrics(history, history_fine_tune, out_dir):
    """Stitch phase-1 + phase-2 histories and save AUC + loss curves."""
    for key in METRICS_TO_COMBINE:
        if key in history.history and key in history_fine_tune.history:
            history.history[key].extend(history_fine_tune.history[key])
        elif key in history_fine_tune.history:
            history.history[key] = history_fine_tune.history[key]

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history.get("auc", []), label="Train AUC")
    plt.plot(history.history.get("val_auc", []), label="Validation AUC")
    plt.xlabel("Epoch")
    plt.ylabel("AUC Score")
    plt.legend()
    plt.title("Training and Validation AUC")
    plt.savefig(out_dir / "training_auc_final_model_with_augmentation.png")

    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Loss")
    plt.savefig(out_dir / "training_loss_final_model_with_augmentation.png")

    print("\nTraining history plots saved as "
          "'training_auc_final_model_with_augmentation.png' and "
          "'training_loss_final_model_with_augmentation.png'")
