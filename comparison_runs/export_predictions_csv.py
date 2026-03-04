#!/usr/bin/env python3
"""
Export per-organoid predictions from per_day_study_overlay results to CSVs.

Reads results.json from each (model, day) under a base dir. For each model that has
train_predictions / val_predictions / test_predictions, writes one CSV per (model, day)
with columns: organoid_id, split, day, y_true, y_pred, y_score (no img_path).

Output files: predictions_{model}_Dy{day}.csv (e.g. predictions_effnet_ts_Dy28.csv).
If only one model is exported, filenames are predictions_Dy{day}.csv.

Usage:
  python export_predictions_csv.py --base_dir regeneration/run_5_all_samples/per_day_study_overlay
  python export_predictions_csv.py --base_dir regeneration/run_5_all_samples/per_day_study_overlay -o regeneration/run_5_all_samples/predictions --model effnet_ts
"""

from pathlib import Path
import csv
import json
import argparse


PREDICTION_COLUMNS = ["organoid_id", "split", "day", "y_true", "y_pred", "y_score"]


def day_to_dy(day):
    """Convert numeric day to Dy string: 28 -> Dy28, 20.5 -> Dy20_5."""
    d = float(day)
    if d == int(d):
        return f"Dy{int(d)}"
    return f"Dy{str(day).replace('.', '_')}"


def predictions_from_results(results_path, day, model_type):
    """Load results.json and yield dicts with organoid_id, split, day, y_true, y_pred, y_score."""
    if not results_path.exists():
        return
    with open(results_path) as f:
        data = json.load(f)
    day_str = day_to_dy(day)
    for split, key in [("train", "train_predictions"), ("val", "val_predictions"), ("test", "test_predictions")]:
        preds = data.get(key)
        if not preds:
            continue
        for p in preds:
            prob = float(p.get("prob", 0))
            label = int(p.get("label", 0))
            yield {
                "organoid_id": p.get("organoid_id", ""),
                "split": split,
                "day": day_str,
                "y_true": label,
                "y_pred": 1 if prob >= 0.5 else 0,
                "y_score": prob,
            }


def export_predictions(base_dir, output_dir=None, model_filter=None):
    """
    base_dir: path to per_day_study_overlay (contains per_day/, effnet_ts/, effnet_tchange/).
    output_dir: where to write CSVs; default base_dir.
    model_filter: optional list, e.g. ['effnet_ts']; if None, all models with predictions are used.
    """
    base_dir = Path(base_dir)
    output_dir = Path(output_dir) if output_dir else base_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    models_with_predictions = ("per_day", "effnet_ts", "effnet_tchange")
    if model_filter is not None:
        models_with_predictions = [m for m in models_with_predictions if m in model_filter]

    written = []
    for model_type in models_with_predictions:
        model_path = base_dir / model_type
        if not model_path.is_dir():
            continue
        for day_dir in sorted(model_path.iterdir(), key=lambda p: (p.name.replace("day_", "").replace("_", "."),)):
            if not day_dir.is_dir() or not day_dir.name.startswith("day_"):
                continue
            res_file = day_dir / "results.json"
            try:
                day = float(day_dir.name.replace("day_", "").replace("_", "."))
            except ValueError:
                continue
            rows = list(predictions_from_results(res_file, day, model_type))
            if not rows:
                continue
            day_str = day_to_dy(day)
            if len(models_with_predictions) == 1:
                out_name = f"predictions_{day_str}.csv"  # e.g. predictions_Dy28.csv
            else:
                out_name = f"predictions_{model_type}_{day_str}.csv"
            out_path = output_dir / out_name
            with open(out_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=PREDICTION_COLUMNS, extrasaction="ignore")
                w.writeheader()
                w.writerows(rows)
            written.append((out_path, len(rows)))
    return written


def main():
    p = argparse.ArgumentParser(description="Export per-organoid predictions to CSVs (organoid_id, split, day, y_true, y_pred, y_score)")
    p.add_argument("--base_dir", "-b", required=True, help="Path to per_day_study_overlay dir (e.g. regeneration/run_5_all_samples/per_day_study_overlay)")
    p.add_argument("-o", "--output_dir", default=None, help="Output directory for CSVs (default: base_dir)")
    p.add_argument("--model", default=None, choices=["per_day", "effnet_ts", "effnet_tchange"], help="Export only this model (default: all that have predictions)")
    args = p.parse_args()
    model_filter = [args.model] if args.model else None
    written = export_predictions(args.base_dir, output_dir=args.output_dir, model_filter=model_filter)
    if not written:
        print("No prediction data found (results.json must contain train/val/test_predictions).")
        return
    for path, n in written:
        print(f"Wrote {path} ({n} rows)")
    print(f"Done. {len(written)} file(s).")


if __name__ == "__main__":
    main()
