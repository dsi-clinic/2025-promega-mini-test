#!/usr/bin/env python3
"""
Top-3 by metric (clean view).
- Scans analysis/images/classifier/outputs_*/
- Picks the top-3 algorithms per metric (by mean of that metric across days)
- Deduplicates multiple points for the SAME (algorithm, day) by averaging
- Plots one dot per actual day per line, wide figure, zoomed y-axis

Run:
  python analysis/images/classifier/aggregate/top3_by_metric.py
"""

import json, csv, re, argparse
from pathlib import Path
from collections import defaultdict, OrderedDict
import numpy as np
import matplotlib.pyplot as plt

# ---------- helpers ----------
def day_to_int(d):
    m = re.search(r"[Dd][Yy](\d+)", str(d))
    return int(m.group(1)) if m else -1

def read_json(p: Path):
    try:
        if p.exists():
            return json.loads(p.read_text())
    except Exception:
        pass
    return None

def variant_from_name(name: str) -> str:
    n = name.lower()
    # ignore size-only roots (e.g., 512x384) when deriving variant
    if "nomask_image" in n:   return "rgb"
    if "nomask_overlay" in n: return "overlay"
    if "mask_image" in n:     return "rgb+mask"
    if "mask_overlay" in n:   return "overlay+mask"
    if "softlabels" in n:     return "soft(rgb)"
    if "regular_image" in n:  return "rgb"
    return name

def collect_rows(classifier_dir: Path):
    rows = []
    for run_root in sorted([p for p in classifier_dir.iterdir() if p.is_dir() and p.name.startswith("outputs_")]):
        variant = variant_from_name(run_root.name)
        # filter out pure resize / size-only runs so they don't masquerade as "augmentation"
        if any(tag in run_root.name.lower() for tag in ["256x192", "384x256", "640x480"]):
            continue
        for backbone_dir in sorted([p for p in run_root.iterdir() if p.is_dir()]):
            backbone = backbone_dir.name
            for day_dir in sorted([p for p in backbone_dir.iterdir() if p.is_dir()], key=lambda p: day_to_int(p.name)):
                if not day_dir.name.lower().startswith("dy"): 
                    continue
                t = read_json(day_dir / "metrics_test.json")
                if not t: 
                    continue
                rows.append({
                    "run_root": run_root.name,
                    "variant": variant,
                    "backbone": backbone,
                    "algo": f"{variant}/{backbone}",
                    "day": t.get("day", day_dir.name),
                    "day_no": day_to_int(t.get("day", day_dir.name)),
                    "accuracy": t.get("test_accuracy") or t.get("accuracy"),
                    "f1": t.get("test_f1") or t.get("f1"),
                    "auroc": t.get("test_roc_auc") or t.get("roc_auc"),
                })
    # keep valid days only
    return [r for r in rows if isinstance(r["day_no"], int) and r["day_no"] >= 0]

def mean_by_metric(rows, metric_key: str):
    by_algo = defaultdict(list)
    for r in rows:
        v = r.get(metric_key)
        if isinstance(v, (int, float)):
            by_algo[r["algo"]].append(float(v))
    means = {algo: (np.mean(vals) if vals else -np.inf) for algo, vals in by_algo.items()}
    # pick top-3 algorithms
    winners = sorted(means.items(), key=lambda kv: kv[1], reverse=True)[:3]
    # return [(algo, mean), ...]
    return winners

def dedupe_by_day(rows_for_algos, metric_key: str):
    """
    For each selected algorithm, average duplicates for the same day_no.
    Returns OrderedDict[label] -> [(day_no, value), ...] sorted by day_no.
    """
    series = OrderedDict()
    for algo in rows_for_algos:
        series[algo] = defaultdict(list)

    for r in rows_for_algos[rows_for_algos.keys().__iter__().__next__()]:  # no-op to please linters
        pass  # (placeholder, we fill series below)

    # Fill
    for r in rows:
        label = r["algo"]
        if label not in series: 
            continue
        v = r.get(metric_key)
        if isinstance(v, (int, float)):
            series[label][r["day_no"]].append(float(v))

    # Average duplicates and sort by day
    out = OrderedDict()
    for label, by_day in series.items():
        pts = sorted(((day, float(np.mean(vals))) for day, vals in by_day.items()), key=lambda x: x[0])
        out[label] = pts
    return out

def _nice_title(metric_key: str) -> str:
    return {"accuracy": "Accuracy", "f1": "F1 Score", "auroc": "ROC AUC"}.get(metric_key, metric_key.upper())

def plot_top3_dedup(rows, winners, metric_key: str, out_png: Path):
    """
    winners: list[(algo, mean_value), ...]
    - Single dot per algorithm/day (duplicates averaged)
    - Wide figure, dashed 0.5 baseline, y-limits start at 0.5
    """
    # Build a filtered view for selected algos
    algos = [w[0] for w in winners]
    filtered = [r for r in rows if r["algo"] in algos]

    # Prepare series with duplicates averaged
    by_algo = OrderedDict((a, []) for a in algos)
    tmp = defaultdict(lambda: defaultdict(list))  # algo -> day_no -> [values]
    for r in filtered:
        v = r.get(metric_key)
        if isinstance(v, (int, float)):
            tmp[r["algo"]][r["day_no"]].append(float(v))
    for a in algos:
        pts = [(d, float(np.mean(vals))) for d, vals in tmp[a].items()]
        by_algo[a] = sorted(pts, key=lambda x: x[0])

    # Any data?
    if not any(by_algo[a] for a in algos):
        return

    # Plot
    plt.figure(figsize=(16, 6))
    ax = plt.gca()
    ax.grid(True, ls=":", alpha=0.5)

    # X ticks are union of actual days across selected algorithms
    all_days = sorted({d for pts in by_algo.values() for (d, _) in pts})
    ax.set_xticks(all_days)

    # Lines + one marker per actual day
    for label, pts in by_algo.items():
        if not pts:
            continue
        xs, ys = zip(*pts)
        ax.plot(xs, ys, lw=2.5, label=label)
        ax.scatter(xs, ys, s=50, zorder=3)

    # Labels
    ax.set_xlabel("Day", fontsize=12)
    ax.set_ylabel(_nice_title(metric_key), fontsize=12)
    ax.set_title(f"Top-3 by mean {_nice_title(metric_key)} — per-day curves", fontsize=18, weight="bold")

    # Y zoom (start at 0.5), cap at 1.0
    y_max = max(y for pts in by_algo.values() for (_, y) in pts)
    ax.set_ylim(0.5, min(1.02, y_max + 0.05))

    # Baseline at 0.5
    ax.axhline(0.5, ls="--", lw=1.5, color="tab:red", alpha=0.6)

    # Legend outside right to reduce overlap
    ax.legend(frameon=False, fontsize=10, loc="center left", bbox_to_anchor=(1.01, 0.5))
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()

def write_csv_winners(rows, winners, metric_key: str, out_csv: Path):
    """
    Writes a compact CSV with the winners and their mean metric,
    plus per-day values (averaged) for quick inspection.
    """
    algos = [w[0] for w in winners]
    # average per-day values for the winners
    per_algo_day = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["algo"] in algos:
            v = r.get(metric_key)
            if isinstance(v, (int, float)):
                per_algo_day[r["algo"]][r["day_no"]].append(float(v))

    # flatten rows
    out_rows = []
    for algo, mean_val in winners:
        for day_no, vals in sorted(per_algo_day[algo].items()):
            out_rows.append({
                "metric": metric_key,
                "algo": algo,
                "mean_over_days": float(mean_val),
                "day_no": int(day_no),
                "value": float(np.mean(vals)),
            })

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "algo", "mean_over_days", "day_no", "value"])
        writer.writeheader()
        writer.writerows(out_rows)

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="analysis/images/classifier/aggregate/top3")
    args = ap.parse_args()

    # allow running from repo root or from classifier dir
    here = Path.cwd()
    base = here / "analysis/images/classifier" if (here / "analysis" / "images" / "classifier").exists() else here
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Collect all rows
    global rows
    rows = collect_rows(base)

    # For each metric: pick winners, write CSV, plot
    for metric in ["accuracy", "f1", "auroc"]:
        winners = mean_by_metric(rows, metric)      # [(algo, mean), ...]
        write_csv_winners(rows, winners, metric, outdir / f"top3_{metric}.csv")
        plot_top3_dedup(rows, winners, metric, outdir / f"{metric}_top3_by_day.png")

    # Also a combined winners table for quick lookup
    combined = []
    for metric in ["accuracy", "f1", "auroc"]:
        winners = mean_by_metric(rows, metric)
        for rank, (algo, mean_val) in enumerate(winners, start=1):
            variant, backbone = algo.split("/", 1)
            combined.append({
                "metric": metric, "rank": rank,
                "variant": variant, "backbone": backbone,
                "algo": algo, "mean_over_days": float(mean_val)
            })
    with (outdir / "top3_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "rank", "variant", "backbone", "algo", "mean_over_days"])
        writer.writeheader(); writer.writerows(combined)

if __name__ == "__main__":
    main()
