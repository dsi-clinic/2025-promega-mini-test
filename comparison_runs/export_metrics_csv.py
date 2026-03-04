#!/usr/bin/env python3
"""
Export results from per_day_study* (and optional kfold) dirs to one CSV.
Only per_day and effnet_ts (no cnn_lstm). Threshold study = these two only.
Columns: setup, fold, model, day, threshold, accuracy, precision, recall, f1,
TNR, TPR, sensitivity, specificity, balanced_acc, TN, FP, FN, TP, optimal_threshold.
Primary metric: balanced_acc = (Sensitivity + Specificity) / 2.
Uses test-set metrics by default; --split selects train/val/test/all.
Include threshold_results (0.5, 0.6, 0.7, 0.8, 0.9 + optimal) when present.
"""
import json
import csv
from pathlib import Path

COMPARISON_RUNS = Path(__file__).resolve().parent

# Single-split base dirs (fold=0)
SETUP_DIRS = [
    ("rgb", COMPARISON_RUNS / "per_day_study"),
    ("overlay", COMPARISON_RUNS / "per_day_study_overlay"),
    ("rgb_mask", COMPARISON_RUNS / "per_day_study_rgb_mask"),
]

CSV_COLUMNS = [
    "setup", "fold", "model", "day", "threshold",
    "accuracy", "precision", "recall", "f1",
    "TNR", "TPR", "sensitivity", "specificity", "balanced_acc",
    "TN", "FP", "FN", "TP",
    "optimal_threshold",
]

CSV_COLUMNS_WITH_SPLIT = [
    "setup", "split", "fold", "model", "day", "threshold",
    "accuracy", "precision", "recall", "f1",
    "TNR", "TPR", "sensitivity", "specificity", "balanced_acc",
    "TN", "FP", "FN", "TP",
    "optimal_threshold",
]


def results_to_rows(setup, fold, model_type, day, data, split="test"):
    """From one results.json (with or without threshold_results) yield CSV row dicts."""
    opt = float(data.get("optimal_threshold", 0.5))
    if "threshold_results" in data:
        for tr in data["threshold_results"]:
            if split not in tr:
                continue
            m = tr[split]
            yield {
                "setup": setup,
                "fold": fold,
                "model": model_type,
                "day": day,
                "threshold": tr["threshold_value"],
                "accuracy": m["accuracy"],
                "precision": m["precision"],
                "recall": m["recall"],
                "f1": m["f1"],
                "TNR": m["TNR"],
                "TPR": m["TPR"],
                "sensitivity": m["sensitivity"],
                "specificity": m["specificity"],
                "balanced_acc": m["balanced_acc"],
                "TN": m["TN"],
                "FP": m["FP"],
                "FN": m["FN"],
                "TP": m["TP"],
                "optimal_threshold": opt,
            }
    else:
        m = data.get(f"{split}_at_optimal") or data.get(f"{split}_at_0.5") or {}
        if not m:
            return
        yield {
            "setup": setup,
            "fold": fold,
            "model": model_type,
            "day": day,
            "threshold": opt,
            "accuracy": m.get("accuracy", ""),
            "precision": m.get("precision", ""),
            "recall": m.get("recall", ""),
            "f1": m.get("f1", ""),
            "TNR": m.get("TNR", ""),
            "TPR": m.get("TPR", ""),
            "sensitivity": m.get("sensitivity", m.get("TPR", "")),
            "specificity": m.get("specificity", m.get("TNR", "")),
            "balanced_acc": m.get("balanced_acc", ""),
            "TN": m.get("TN", ""),
            "FP": m.get("FP", ""),
            "FN": m.get("FN", ""),
            "TP": m.get("TP", ""),
            "optimal_threshold": opt,
        }


def _collect_from_base(base, setup, splits, fold=0):
    """Collect rows from one base directory for given splits."""
    rows = []
    for model_dir in ("per_day", "effnet_ts", "effnet_tchange"):
        path = base / model_dir
        if not path.exists():
            continue
        for day_dir in path.iterdir():
            if not day_dir.is_dir():
                continue
            res_file = day_dir / "results.json"
            if not res_file.exists():
                continue
            with open(res_file) as f:
                data = json.load(f)
            model_type = data.get("model_type", model_dir)
            day = data.get("day")
            if day is None:
                try:
                    day = float(day_dir.name.replace("day_", "").replace("_", "."))
                except Exception:
                    continue
            for split in splits:
                for row in results_to_rows(setup, fold=fold, model_type=model_type, day=day, data=data, split=split):
                    if len(splits) > 1:
                        row["split"] = split
                    rows.append(row)
    return rows


def collect_rows(include_kfold=False, setup_filter=None, overlay_base=None, splits=None):
    """Collect all rows from SETUP_DIRS and optionally per_day_study_kfold.
    setup_filter: if set (e.g. 'overlay'), only include that setup.
    overlay_base: if set (Path), use this as base for 'overlay' setup instead of SETUP_DIRS.
    splits: list of splits to collect (e.g. ['test'], ['train','val','test']).
    """
    if splits is None:
        splits = ["test"]
    rows = []
    for setup, base in SETUP_DIRS:
        if setup_filter is not None and setup != setup_filter:
            continue
        if setup == "overlay" and overlay_base is not None:
            base = Path(overlay_base)
        if not base.exists():
            continue
        rows.extend(_collect_from_base(base, setup, splits, fold=0))
    if include_kfold:
        kfold_base = COMPARISON_RUNS / "per_day_study_kfold"
        if kfold_base.exists():
            for setup_dir in kfold_base.iterdir():
                if not setup_dir.is_dir():
                    continue
                setup = setup_dir.name
                if setup_filter is not None and setup != setup_filter:
                    continue
                for fold_dir in setup_dir.iterdir():
                    if not fold_dir.is_dir() or not fold_dir.name.startswith("fold_"):
                        continue
                    try:
                        fold = int(fold_dir.name.split("_")[1])
                    except Exception:
                        continue
                    rows.extend(_collect_from_base(fold_dir, setup, splits, fold=fold))
    return rows


def main():
    import argparse
    p = argparse.ArgumentParser(description="Export per-day study results to CSV (columns: setup, fold, model, day, threshold, accuracy, ..., balanced_acc, ..., optimal_threshold)")
    p.add_argument("-o", "--output", default=None, help="Output CSV path (default: comparison_runs/metrics.csv)")
    p.add_argument("--kfold", action="store_true", help="Also include per_day_study_kfold if present")
    p.add_argument("--setup", default=None, choices=["rgb", "overlay", "rgb_mask"], help="Only export this setup (e.g. overlay)")
    p.add_argument("--overlay_dir", default=None, help="If set, read overlay results from this dir (for parallel regeneration runs)")
    p.add_argument("--split", default="test", choices=["test", "val", "train", "all"],
                   help="Which split(s) to export (default: test). 'all' exports train+val+test with a 'split' column.")
    args = p.parse_args()
    out = Path(args.output) if args.output else COMPARISON_RUNS / "metrics.csv"
    overlay_base = Path(args.overlay_dir) if args.overlay_dir else None
    splits = ["train", "val", "test"] if args.split == "all" else [args.split]
    rows = collect_rows(include_kfold=args.kfold, setup_filter=args.setup, overlay_base=overlay_base, splits=splits)
    if not rows:
        print("No results found.")
        return
    columns = CSV_COLUMNS_WITH_SPLIT if len(splits) > 1 else CSV_COLUMNS
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
