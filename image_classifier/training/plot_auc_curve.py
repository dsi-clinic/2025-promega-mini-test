#!/usr/bin/env python3
"""
Plot ROC AUC curve across different days
"""

import matplotlib.pyplot as plt
import json
from pathlib import Path


def extract_auc_data():
    """Extract day numbers and ROC AUC values from all metrics files"""
    base_dir = Path(
        "/net/scratch/jiaweizhang/2025-promega-mini-test/analysis/images/classifier/outputs_512x384_softlabels/soft_resnet"
    )

    days = []
    aucs = []

    # Get all day directories
    day_dirs = [d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith("Dy")]

    # Sort by day number (extract numeric part)
    def extract_day_number(day_name):
        # Remove 'Dy' prefix and convert to float to handle '20_5' -> 20.5
        day_str = day_name[2:].replace("_", ".")
        return float(day_str)

    for day_dir in sorted(day_dirs, key=lambda x: extract_day_number(x.name)):
        metrics_file = day_dir / "metrics_test.json"
        if metrics_file.exists():
            with open(metrics_file, "r") as f:
                data = json.load(f)
                days.append(data["day_no"])
                aucs.append(data["roc_auc"])
                print(
                    f"Day {data['day']}: day_no={data['day_no']}, roc_auc={data['roc_auc']:.4f}"
                )

    return days, aucs


def plot_auc_curve(days, aucs):
    """Create the ROC AUC curve plot"""
    plt.figure(figsize=(12, 8))

    # Convert day numbers to actual days (divide by 100)
    actual_days = [d / 100 for d in days]

    # Create the plot
    plt.plot(
        actual_days,
        aucs,
        "o-",
        linewidth=2,
        markersize=8,
        color="#2E86AB",
        markerfacecolor="#A23B72",
    )

    # Customize the plot
    plt.xlabel("Day", fontsize=14, fontweight="bold")
    plt.ylabel("ROC AUC", fontsize=14, fontweight="bold")
    plt.title(
        "ROC AUC Performance Across Different Days",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )

    # Add grid
    plt.grid(True, alpha=0.3, linestyle="--")

    # Set axis limits and ticks
    plt.xlim(min(actual_days) - 0.5, max(actual_days) + 0.5)
    plt.ylim(0, 1.05)

    # Add horizontal line at 0.5 (random classifier performance)
    plt.axhline(
        y=0.5,
        color="red",
        linestyle="--",
        alpha=0.7,
        label="Random Classifier (AUC=0.5)",
    )

    # Add value annotations on points
    for i, (day, auc) in enumerate(zip(actual_days, aucs)):
        plt.annotate(
            f"{auc:.3f}",
            (day, auc),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=10,
            fontweight="bold",
        )

    # Add legend
    plt.legend(fontsize=12)

    # Improve layout
    plt.tight_layout()

    # Save the plot
    output_path = "/net/scratch/jiaweizhang/2025-promega-mini-test/analysis/images/classifier/roc_auc_by_day.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nPlot saved to: {output_path}")

    # Show the plot
    plt.show()


def main():
    """Main function"""
    print("Extracting ROC AUC data from metrics files...")
    days, aucs = extract_auc_data()

    print(f"\nFound {len(days)} data points")
    print(f"Day range: {min(days) / 100:.1f} to {max(days) / 100:.1f}")
    print(f"AUC range: {min(aucs):.3f} to {max(aucs):.3f}")

    print("\nCreating ROC AUC curve plot...")
    plot_auc_curve(days, aucs)


if __name__ == "__main__":
    main()
