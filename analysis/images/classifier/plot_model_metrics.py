#!/usr/bin/env python3
import argparse, json, re
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ---------- Helpers ----------
DAY_PAT = re.compile(r"[Dd][Yy](\d+)")

def day_to_int(day_str: str) -> int:
    m = DAY_PAT.search(day_str)
    return int(m.group(1)) if m else -1

def load_metrics_for_model(out_root: Path, backbone_key: str):
    """
    Returns:
      days_sorted: list[int] day numbers
      per_day: dict[day_no] -> {"label": "DyXX",
                                "val_acc": float or None,
                                "test_acc": float or None,
                                "test_f1": float or None}
    """
    per_day = {}
    backbone_dir = out_root / backbone_key
    if not backbone_dir.exists():
        return [], {}

    for day_dir in backbone_dir.iterdir():
        if not day_dir.is_dir():
            continue
        day_label = day_dir.name
        dno = day_to_int(day_label)
        if dno < 0:
            continue

        val_path  = day_dir / "metrics_val.json"
        test_path = day_dir / "metrics_test.json"

        val_acc = None
        test_acc = None
        test_f1 = None

        if val_path.exists():
            try:
                with val_path.open() as f:
                    val = json.load(f)
                val_acc = float(val.get("accuracy", np.nan))
            except Exception:
                pass

        if test_path.exists():
            try:
                with test_path.open() as f:
                    tst = json.load(f)
                test_acc = float(tst.get("accuracy", np.nan))
                test_f1  = float(tst.get("f1", np.nan))
            except Exception:
                pass

        # keep only days that have at least one metric
        if any(x is not None and not np.isnan(x) for x in (val_acc, test_acc, test_f1)):
            per_day[dno] = {
                "label": day_label,
                "val_acc": val_acc,
                "test_acc": test_acc,
                "test_f1": test_f1,
            }

    days_sorted = sorted(per_day.keys())
    return days_sorted, per_day

def line(ax, xs, ys, label, color, marker="o"):
    ax.plot(xs, ys, marker=marker, label=label, color=color, linewidth=2, markersize=5)
    # annotate each point with value
    for x, y in zip(xs, ys):
        if y is None or np.isnan(y):
            continue
        ax.text(x, y + 0.01, f"{y*100:.1f}", ha="center", va="bottom",
                fontsize=8, color=color)


# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser(description="Plot per-model metrics across days.")
    parser.add_argument("--out-root", default="analysis/images/classifier/outputs_512x384",
                        help="Root output directory where metrics_* are saved")
    parser.add_argument("--models", nargs="*", default=["vit","resnet","efficientnet"],
                        help="Backbone keys (dirs under out-root) to include")
    parser.add_argument("--ylim", type=float, nargs=2, default=[0.0, 1.0],
                        help="Y-axis limits for metrics")
    parser.add_argument("--make-comparison", action="store_true",
                        help="Also produce a multi-model comparison figure for test accuracy and F1")
    parser.add_argument("--dpi", type=int, default=160)
    args = parser.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    sns.set_context("talk")
    sns.set_style("whitegrid")

    # color palette for the three lines (per model figure)
    metric_colors = {
        "test_acc": "#1f77b4",   # blue
        "val_acc":  "#ff7f0e",   # orange
        "test_f1":  "#2ca02c",   # green
    }

    # For optional comparison plots
    comp_acc = {}  # model -> (xs, ys)
    comp_f1  = {}  # model -> (xs, ys)

    for backbone_key in args.models:
        days_sorted, per_day = load_metrics_for_model(out_root, backbone_key)
        if not days_sorted:
            print(f"⚠ No metrics found for model '{backbone_key}' in {out_root}")
            continue

        xs = days_sorted
        # Build y-series aligned to xs
        y_val_acc  = [per_day[d]["val_acc"]  for d in xs]
        y_test_acc = [per_day[d]["test_acc"] for d in xs]
        y_test_f1  = [per_day[d]["test_f1"]  for d in xs]

        # Per-model figure
        fig, ax = plt.subplots(figsize=(10, 4.5), dpi=args.dpi)
        line(ax, xs, y_test_acc, "Test Accuracy", metric_colors["test_acc"])
        line(ax, xs, y_val_acc,  "Validation Accuracy", metric_colors["val_acc"])
        line(ax, xs, y_test_f1,  "Test F1", metric_colors["test_f1"])

        ax.set_title(f"{backbone_key}: metrics across days")
        ax.set_xlabel("Day")
        ax.set_ylabel("Score")
        ax.set_ylim(args.ylim[0], args.ylim[1])
        ax.set_xticks(xs)
        ax.legend(loc="best", frameon=True)
        fig.tight_layout()

        save_path = out_root / f"{backbone_key}_metrics_by_day.png"
        fig.savefig(save_path)
        plt.close(fig)
        print(f"📈 Saved → {save_path}")

        # prep for comparison
        comp_acc[backbone_key] = (xs, y_test_acc)
        comp_f1[backbone_key]  = (xs, y_test_f1)

    if args.make_comparison and (comp_acc or comp_f1):
        # Test Accuracy comparison
        if comp_acc:
            fig, ax = plt.subplots(figsize=(10, 4.5), dpi=args.dpi)
            palette = sns.color_palette("tab10", n_colors=len(comp_acc))
            for (i, (model, (xs, ys))) in enumerate(sorted(comp_acc.items())):
                ax.plot(xs, ys, marker="o", linewidth=2, markersize=5,
                        label=model, color=palette[i])
            ax.set_title("Test Accuracy across models")
            ax.set_xlabel("Day"); ax.set_ylabel("Accuracy"); ax.set_ylim(args.ylim[0], args.ylim[1])
            ax.legend(loc="best"); ax.set_xticks(sorted({x for xs,_ in comp_acc.values() for x in xs}))
            fig.tight_layout()
            p = Path(args.out_root) / "comparison_test_accuracy.png"
            fig.savefig(p); plt.close(fig)
            print(f"📊 Saved → {p}")

        # Test F1 comparison
        if comp_f1:
            fig, ax = plt.subplots(figsize=(10, 4.5), dpi=args.dpi)
            palette = sns.color_palette("tab10", n_colors=len(comp_f1))
            for (i, (model, (xs, ys))) in enumerate(sorted(comp_f1.items())):
                ax.plot(xs, ys, marker="o", linewidth=2, markersize=5,
                        label=model, color=palette[i])
            ax.set_title("Test F1 across models")
            ax.set_xlabel("Day"); ax.set_ylabel("F1"); ax.set_ylim(args.ylim[0], args.ylim[1])
            ax.legend(loc="best"); ax.set_xticks(sorted({x for xs,_ in comp_f1.values() for x in xs}))
            fig.tight_layout()
            p = Path(args.out_root) / "comparison_test_f1.png"
            fig.savefig(p); plt.close(fig)
            print(f"📊 Saved → {p}")

if __name__ == "__main__":
    main()
