#!/usr/bin/env python3
import argparse, json, re, csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

DAY_PAT = re.compile(r"[Dd][Yy](\d+)")

def day_to_int(day_str: str) -> int:
    m = DAY_PAT.search(day_str)
    return int(m.group(1)) if m else -1

def load_model_days(out_root: Path, model_key: str):
    """
    Load per-day TEST metrics for a given model.

    Returns:
      days_sorted: [int]
      recs: dict[int] -> {
          "label","test_acc","test_f1","test_num","actual_good","pred_good"
      }
    """
    recs = {}
    mdir = out_root / model_key
    if not mdir.exists():
        return [], {}

    for day_dir in mdir.iterdir():
        if not day_dir.is_dir():
            continue
        day_label = day_dir.name
        dno = day_to_int(day_label)
        if dno < 0:
            continue

        test_path = day_dir / "metrics_test.json"
        if not test_path.exists():
            continue

        try:
            with test_path.open() as f:
                t = json.load(f)
            recs[dno] = {
                "label": day_label,
                "test_acc": float(t.get("accuracy", np.nan)),
                "test_f1": float(t.get("f1", np.nan)),
                "test_num": int(t.get("test_n", 0)),
                "actual_good": int(t.get("actual_good", 0)),
                "pred_good": int(t.get("predicted_good", 0)),
            }
        except Exception:
            # skip malformed files
            continue

    days_sorted = sorted(k for k, v in recs.items() if not np.isnan(v["test_acc"]))
    return days_sorted, recs

def annotate_points(ax, xs, ys, color, y_offset=0.01):
    """Add small % labels (e.g., 93.3) above each point."""
    for x, y in zip(xs, ys):
        if y is None or np.isnan(y):
            continue
        ax.text(x, y + y_offset, f"{y*100:.1f}", ha="center", va="bottom",
                fontsize=8, color=color)

def write_table(csv_path: Path, rows):
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["Day No", "Num in Sample", "Actual Good", "Predicted Good"]
        )
        writer.writeheader()
        writer.writerows(rows)

def main():
    parser = argparse.ArgumentParser(description="Create one chart and one 4-col table per model.")
    parser.add_argument("--out-root", default="analysis/images/classifier/outputs_512x384",
                        help="Root directory with per-model day folders")
    parser.add_argument("--models", nargs="*", default=["vit","resnet","efficientnet"],
                        help="Model directory names under out-root")
    parser.add_argument("--dpi", type=int, default=160)
    parser.add_argument("--ylim", type=float, nargs=2, default=[0.0, 1.0],
                        help="Y-axis limits for accuracy plots")
    parser.add_argument("--quiet", action="store_true",
                        help="Don’t print the tables to stdout")
    args = parser.parse_args()

    out_root = Path(args.out_root)
    sns.set_context("talk")
    sns.set_style("whitegrid")

    for model in args.models:
        days_sorted, recs = load_model_days(out_root, model)
        if not days_sorted:
            print(f"⚠ No test metrics found for '{model}' in {out_root}")
            continue

        # ---- Table (per model)
        rows = [{
            "Day No": d,
            "Num in Sample": recs[d]["test_num"],
            "Actual Good": recs[d]["actual_good"],
            "Predicted Good": recs[d]["pred_good"],
        } for d in days_sorted]

        table_path = out_root / f"{model}_day_summary.csv"
        write_table(table_path, rows)
        print(f"🧾 [{model}] saved table → {table_path}")

        if not args.quiet:
            print(f"\n=== Summary Table (TEST) — {model} ===")
            print(f"{'Day No':>6} | {'Num in Sample':>13} | {'Actual Good':>11} | {'Predicted Good':>14}")
            print("-" * 54)
            for r in rows:
                print(f"{r['Day No']:>6} | {r['Num in Sample']:>13} | {r['Actual Good']:>11} | {r['Predicted Good']:>14}")
            print()

        # ---- Chart (per model): Test Accuracy vs Day, with tiny labels (e.g., 93.3)
        xs = days_sorted
        ys = [recs[d]["test_acc"] for d in xs]

        fig, ax = plt.subplots(figsize=(10, 4.5), dpi=args.dpi)
        ax.plot(xs, ys, marker="o", linewidth=2, markersize=5,
                color="#1f77b4", label="Test Accuracy")
        annotate_points(ax, xs, ys, color="#1f77b4", y_offset=0.01)
        ax.set_title(f"{model}: Test Accuracy by Day")
        ax.set_xlabel("Day")
        ax.set_ylabel("Accuracy")
        ax.set_xticks(xs)
        ax.set_ylim(args.ylim[0], args.ylim[1])
        ax.legend(loc="best", frameon=True)
        fig.tight_layout()

        chart_path = out_root / f"{model}_accuracy_by_day.png"
        fig.savefig(chart_path)
        plt.close(fig)
        print(f"📈 [{model}] saved chart → {chart_path}")

if __name__ == "__main__":
    main()
