#!/usr/bin/env python3
"""Plotting helpers for the image quality classifier."""

import matplotlib.pyplot as plt


def plot_training_curve(history, model_dir):
    """Per-day train/val accuracy + loss plot."""
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history["train_acc"], label="Train")
    plt.plot(history["val_acc"], label="Val")
    plt.title("Accuracy")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history["train_loss"], label="Train")
    plt.plot(history["val_loss"], label="Val")
    plt.title("Loss")
    plt.legend()
    plt.tight_layout()
    out_path = model_dir / "training_curves.png"
    plt.savefig(out_path)
    plt.close()
    print(f"📈 Saved curves → {out_path}")


def plot_metric(metric_key, ylabel, title, filename, per_model_results,
                day_numbers, unique_day_nos, out_dir):
    """Per-day metric line plot, one line per backbone."""
    plt.figure(figsize=(9, 4))
    plotted_any = False
    for backbone_key, day_res in per_model_results.items():
        if not day_res:
            continue
        pairs = [
            (day_numbers[day], day_res[day].get(metric_key))
            for day in sorted(day_res.keys(), key=lambda d: day_numbers[d])
            if day_res[day].get(metric_key) is not None
        ]
        if not pairs:
            continue
        xs, ys = zip(*pairs)
        plt.plot(xs, ys, marker="o", label=backbone_key)
        plotted_any = True

    if plotted_any:
        plt.xlabel("Day")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.xticks(unique_day_nos)
        plt.ylim(0.0, 1.0)
        plt.legend()
        plt.tight_layout()
        out_path = out_dir / filename
        plt.savefig(out_path)
        print(f"📊 Saved {title.lower()} → {out_path}")
    plt.close()
