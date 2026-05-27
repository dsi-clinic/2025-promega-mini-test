#!/usr/bin/env python3
"""
Aggregate Table 2 from per-day metrics_test.json files.
Reads balanced_accuracy from each backbone/day output directory.

Usage:
    python analysis/images/classifier/summarize_table2.py \
        --out_dir analysis/images/classifier/outputs_512x384_tony_dinov2_fixed_splits
"""

import json
import argparse
import csv
from pathlib import Path


def day_to_int(day_str: str) -> float:
    s = day_str.replace("Dy", "").replace("_", ".")
    try:
        return float(s)
    except ValueError:
        return -1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out_dir",
        default="analysis/images/classifier/outputs_512x384_tony_dinov2_fixed_splits",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    results = {}

    for backbone_dir in sorted(out_dir.iterdir()):
        if not backbone_dir.is_dir():
            continue
        backbone = backbone_dir.name
        day_results = {}
        for day_dir in sorted(backbone_dir.iterdir(), key=lambda p: day_to_int(p.name)):
            if not day_dir.is_dir():
                continue
            metrics_file = day_dir / "metrics_test.json"
            if not metrics_file.exists():
                continue
            with open(metrics_file) as f:
                m = json.load(f)
            bal_acc = m.get("balanced_accuracy")
            if bal_acc is None:
                continue
            day_results[day_dir.name] = bal_acc
        if day_results:
            results[backbone] = day_results

    if not results:
        print("No results found.")
        return

    print("\n=== Table 2: Backbone Comparison (Balanced Accuracy) ===\n")
    header = f"{'Model':<14} {'Avg Bal Acc':>11} {'Best Bal Acc':>12} {'Days':>5}"
    print(header)
    print("-" * len(header))

    csv_rows = []
    for backbone, day_results in sorted(results.items()):
        vals = list(day_results.values())
        avg = sum(vals) / len(vals)
        best = max(vals)
        csv_rows.append({
            "Model": backbone,
            "Avg Bal Acc": f"{avg*100:.1f}%",
            "Best Bal Acc": f"{best*100:.1f}%",
            "Days": len(vals),
        })
        print(f"{backbone:<14} {avg*100:>10.1f}% {best*100:>11.1f}% {len(vals):>5}")

    print("\nPer-day breakdown:")
    all_days = sorted(set(d for r in results.values() for d in r), key=day_to_int)
    print(f"{'Day':<10}", end="")
    for b in sorted(results):
        print(f"  {b:>14}", end="")
    print()
    for day in all_days:
        print(f"{day:<10}", end="")
        for b in sorted(results):
            val = results[b].get(day)
            print(f"  {val*100:>13.1f}%" if val is not None else f"  {'N/A':>13}", end="")
        print()

    csv_path = out_dir / "table2_balanced_accuracy.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Model", "Avg Bal Acc", "Best Bal Acc", "Days"])
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"\nSaved to {csv_path}")


if __name__ == "__main__":
    main()
