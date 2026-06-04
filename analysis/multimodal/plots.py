#!/usr/bin/env python3
"""Per-day training curves + cross-day metric panels."""

import matplotlib.pyplot as plt


def plot_training_curves(history: dict, day: str, day_dir):
    """Two-panel loss + accuracy curve, saved as training_curves.png."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history["train_loss"])
    ax1.set_title(f"{day} - Loss")
    ax1.set_xlabel("Epoch")

    ax2.plot(history["train_acc"], label="Train")
    ax2.plot(history["val_acc"], label="Val")
    ax2.set_title(f"{day} - Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(day_dir / "training_curves.png", dpi=150)
    plt.close()


def plot_metrics_by_day(summary, output_dir):
    """3-panel: accuracy / F1 / ROC-AUC vs day."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(summary["Day_No"], summary["Test_Accuracy"], "o-")
    axes[0].set_title("Test Accuracy by Day")
    axes[0].set_xlabel("Day")
    axes[0].set_ylabel("Accuracy")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(summary["Day_No"], summary["Test_F1"], "o-", color="orange")
    axes[1].set_title("Test F1 by Day")
    axes[1].set_xlabel("Day")
    axes[1].set_ylabel("F1 Score")
    axes[1].grid(True, alpha=0.3)

    if summary["Test_ROC_AUC"].notna().any():
        axes[2].plot(summary["Day_No"], summary["Test_ROC_AUC"], "o-", color="green")
        axes[2].set_title("Test ROC-AUC by Day")
        axes[2].set_xlabel("Day")
        axes[2].set_ylabel("ROC-AUC")
        axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "metrics_by_day.png", dpi=150)
    plt.close()
