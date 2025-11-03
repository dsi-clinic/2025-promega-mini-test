#!/usr/bin/env python3
"""
Top-3 by metric (clean view).
- Scans analysis/images/classifier/outputs_*/
- Keeps ONLY real variants (image/overlay/mask/softlabels), skips size-only runs
- Picks the top-3 algorithms per metric (mean over days)
- Averages duplicates for the SAME (algorithm, day)
- Plots one dot per actual day per line, dotted lines, labels on every dot
- X-axis shows only days that exist; Y-axis fixed to [0, 1]

Run:
  python analysis/images/classifier/aggregate/top3_by_metric.py
"""

import json, csv, re, argparse
from pathlib import Path
from collections import defaultdict, OrderedDict
import numpy as np
import matplotlib.pyplot as plt

# ---------- helpers ----------
REAL_BASES = (
    "nomask_image",     # rgb
    "mask_image",       # rgb+mask
    "nomask_overlay",   # overlay
    "mask_overlay",     # overlay+mask
    "softlabels",       # soft(rgb)
)

SIZE_TAG_RE = re.compile(r"_(256x192|384x256|512x384|640x480)\b", re.IGNORECASE)

def _is_real_variant_folder(name: str) -> bool:
    """Keep only real variant families. Drop size-only folders."""
    n = name.lower()
    if SIZE_TAG_RE.search(n):
        return False
    return any(base in n for base in REAL_BASES)

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
    if "nomask_image"   in n: return "rgb"
    if "regular_image"  in n: return "rgb"       # alias
    if "mask_image"     in n: return "rgb+mask"
    if "nomask_overlay" in n: return "overlay"
    if "mask_overlay"   in n: return "overlay+mask"
    if "softlabels"     in n: return "soft(rgb)"
    return name

def collect_rows(classifier_dir: Path):
    rows = []
    for run_root in sorted([p for p in classifier_dir.iterdir() if p.is_dir() and p.name.startswith("outputs_")]):
        # keep only real variant families, drop size-only runs
        if not _is_real_variant_folder(run_root.name):
            continue
        variant = variant_from_name(run_root.name)
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
    return [r for r in rows if isinstance(r["day_no"], int) and r["day_no"] >= 0]

def mean_by_metric(rows, metric_key: str):
    by_algo = defaultdict(list)
    for r in rows:
        v = r.get(metric_key)
        if isinstance(v, (int, float)):
            by_algo[r["algo"]].append(float(v))
    means = {algo: (np.mean(vals) if vals else -np.inf) for algo, vals in by_algo.items()}
    winners = sorted(means.items(), key=lambda kv: kv[1], reverse=True)[:3]
    return winners  # [(algo, mean), ...]

def average_per_day(rows, algos, metric_key: str):
    """
    Returns OrderedDict[algo] -> list[(day_no, avg_value)] sorted by day_no.
    """
    out = OrderedDict((a, []) for a in algos)
    tmp = defaultdict(lambda: defaultdict(list))  # algo -> day_no -> [values]
    for r in rows:
        if r["algo"] not in out:
            continue
        v = r.get(metric_key)
        if isinstance(v, (int, float)):
            tmp[r["algo"]][r["day_no"]].append(float(v))
    for a in algos:
        pts = [(d, float(np.mean(vals))) for d, vals in tmp[a].items()]
        out[a] = sorted(pts, key=lambda x: x[0])
    return out

def _nice_title(metric_key: str) -> str:
    return {"accuracy": "Accuracy", "f1": "F1 Score", "auroc": "ROC AUC"}.get(metric_key, metric_key.upper())

def plot_top3_dedup(rows, winners, metric_key: str, out_png: Path):
    """
    winners: list[(algo, mean_value), ...]
    - Single dot per algorithm/day (duplicates averaged)
    - Dotted lines, labels on each dot
    - X ticks for union of existing days; Y fixed to [0, 1]
    """
    algos = [w[0] for w in winners]
    by_algo = average_per_day(rows, algos, metric_key)
    if not any(by_algo[a] for a in algos):
        return

    plt.figure(figsize=(18, 6))
    ax = plt.gca()
    ax.grid(True, ls=":", alpha=0.4)

    all_days = sorted({d for pts in by_algo.values() for (d, _) in pts})
    ax.set_xticks(all_days)
    ax.set_xlim(min(all_days) - 0.5, max(all_days) + 0.5)

    for label, pts in by_algo.items():
        if not pts:
            continue
        xs, ys = zip(*pts)
        ax.plot(xs, ys, lw=2.2, ls=":", marker="o", markersize=5, label=label)
        # per-point numeric labels
        for x, y in pts:
            ax.annotate(f"{y:.3f}", (x, y), xytext=(0, 8),
                        textcoords="offset points", ha="center", va="bottom", fontsize=9)

    ax.set_xlabel("Day", fontsize=12)
    ax.set_ylabel(_nice_title(metric_key), fontsize=12)
    ax.set_title(f"Top-3 by mean {_nice_title(metric_key)} — per-day curves", fontsize=18, weight="bold")

    ax.set_ylim(0.0, 1.0)  # show full metric range
    ax.legend(frameon=False, fontsize=10, loc="center left", bbox_to_anchor=(1.01, 0.5))

    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close()

def write_csv_winners(rows, winners, metric_key: str, out_csv: Path):
    algos = [w[0] for w in winners]
    per_algo_day = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["algo"] in algos:
            v = r.get(metric_key)
            if isinstance(v, (int, float)):
                per_algo_day[r["algo"]][r["day_no"]].append(float(v))
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

    here = Path.cwd()
    base = here / "analysis/images/classifier" if (here / "analysis" / "images" / "classifier").exists() else here
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows = collect_rows(base)

    for metric in ["accuracy", "f1", "auroc"]:
        winners = mean_by_metric(rows, metric)
        write_csv_winners(rows, winners, metric, outdir / f"top3_{metric}.csv")
        plot_top3_dedup(rows, winners, metric, outdir / f"{metric}_top3_by_day.png")

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
