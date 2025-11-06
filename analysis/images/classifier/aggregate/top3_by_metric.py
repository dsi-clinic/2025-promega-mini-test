#!/usr/bin/env python3
"""
Top-3 by metric (clean view, final).
- Scans analysis/images/classifier/outputs_*/
- Keeps only real variant runs (no size-only)
- Picks top-3 algorithms per metric
- Dedupe per (algorithm, day)
- Plots one dot per day, dotted lines, y=[0.4,1.0], no point labels
"""

import json, csv, re, argparse
from pathlib import Path
from collections import defaultdict, OrderedDict
import numpy as np
import matplotlib.pyplot as plt

REAL_BASES = (
    "nomask_image", "mask_image",
    "nomask_overlay", "mask_overlay",
    "softlabels",
)

NATURE_COLORS = [
    "#E41A1C",  
    "#1F77B4",  
    "#2CA02C",  
]

SIZE_TAG_RE = re.compile(r"_(256x192|384x256|512x384|640x480)\b", re.IGNORECASE)

def _is_real_variant_folder(name: str) -> bool:
    n = name.lower()
    if SIZE_TAG_RE.search(n): return False
    return any(base in n for base in REAL_BASES)

def day_to_int(d):
    m = re.search(r"[Dd][Yy](\d+)", str(d))
    return int(m.group(1)) if m else -1

def read_json(p: Path):
    try:
        if p.exists(): return json.loads(p.read_text())
    except Exception: pass
    return None

def variant_from_name(name: str) -> str:
    n = name.lower()
    if "nomask_image" in n or "regular_image" in n: return "rgb"
    if "mask_image" in n: return "rgb+mask"
    if "nomask_overlay" in n: return "overlay"
    if "mask_overlay" in n: return "overlay+mask"
    if "softlabels" in n: return "soft(rgb)"
    return name

def collect_rows(classifier_dir: Path):
    rows = []
    for run_root in sorted([p for p in classifier_dir.iterdir() if p.is_dir() and p.name.startswith("outputs_")]):
        if not _is_real_variant_folder(run_root.name): continue
        variant = variant_from_name(run_root.name)
        for backbone_dir in sorted([p for p in run_root.iterdir() if p.is_dir()]):
            backbone = backbone_dir.name
            for day_dir in sorted([p for p in backbone_dir.iterdir() if p.is_dir()], key=lambda p: day_to_int(p.name)):
                if not day_dir.name.lower().startswith("dy"): continue
                t = read_json(day_dir / "metrics_test.json")
                if not t: continue
                rows.append({
                    "algo": f"{variant}/{backbone}",
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
        if isinstance(v, (int, float)): by_algo[r["algo"]].append(float(v))
    means = {a: np.mean(v) for a, v in by_algo.items() if v}
    return sorted(means.items(), key=lambda kv: kv[1], reverse=True)[:3]

def average_per_day(rows, algos, metric_key: str):
    out = OrderedDict((a, []) for a in algos)
    tmp = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["algo"] in out and isinstance(r.get(metric_key), (int, float)):
            tmp[r["algo"]][r["day_no"]].append(float(r[metric_key]))
    for a in algos:
        pts = [(d, float(np.mean(v))) for d, v in tmp[a].items()]
        out[a] = sorted(pts, key=lambda x: x[0])
    return out

def _nice_title(k): return {"accuracy":"Accuracy","f1":"F1 Score","auroc":"ROC AUC"}.get(k,k.upper())

def plot_top3(rows, winners, metric_key: str, out_png: Path):
    # winners: [(algo, mean_value), ...]
    algos = [w[0] for w in winners]
    by_algo = average_per_day(rows, algos, metric_key)
    if not any(by_algo[a] for a in algos):
        return

    plt.figure(figsize=(12, 8))
    ax = plt.gca()
    ax.grid(True, linestyle=":", linewidth=0.8, color="0.85")

    all_days = sorted({d for pts in by_algo.values() for (d, _) in pts})
    ax.set_xticks(all_days)
    ax.set_xticklabels([f"Dy{d:02d}" for d in all_days])

    # solid lines with circular markers; custom nature palette
    for idx, (label, pts) in enumerate(by_algo.items()):
        if not pts:
            continue
        xs, ys = zip(*pts)
        color = NATURE_COLORS[idx % len(NATURE_COLORS)]
        ax.plot(
            xs, ys,
            lw=2.4, ls="-", marker="o", markersize=6,
            markerfacecolor="white", markeredgewidth=1.2,
            color=color, label=label,
        )

    ax.set_xlabel("Day", fontsize=12)
    ax.set_ylabel(_nice_title(metric_key), fontsize=12)
    ax.set_title("Predicting Dy30 Acceptability from Earlier-Day Features", fontsize=14)
    ax.set_ylim(0.4, 1.0)
    ax.set_yticks(np.linspace(0.4, 1.0, 7))

    ax.legend(loc="upper left", frameon=True, framealpha=0.9,
              facecolor="white", edgecolor="0.9", fontsize=10)

    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close()

def write_csv(rows,winners,metric_key,out_csv:Path):
    algos=[w[0] for w in winners]
    per_algo=defaultdict(lambda:defaultdict(list))
    for r in rows:
        if r["algo"] in algos and isinstance(r.get(metric_key),(int,float)):
            per_algo[r["algo"]][r["day_no"]].append(float(r[metric_key]))
    out=[]
    for algo,mean_val in winners:
        for d,v in sorted(per_algo[algo].items()):
            out.append({"metric":metric_key,"algo":algo,"mean_over_days":float(mean_val),
                        "day_no":int(d),"value":float(np.mean(v))})
    out_csv.parent.mkdir(parents=True,exist_ok=True)
    with out_csv.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=["metric","algo","mean_over_days","day_no","value"])
        w.writeheader(); w.writerows(out)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--outdir",default="analysis/images/classifier/aggregate/top3")
    args=ap.parse_args()
    base=Path.cwd()/ "analysis/images/classifier"
    outdir=Path(args.outdir); outdir.mkdir(parents=True,exist_ok=True)
    rows=collect_rows(base)
    for metric in ["accuracy","f1","auroc"]:
        winners=mean_by_metric(rows,metric)
        write_csv(rows,winners,metric,outdir/f"top3_{metric}.csv")
        plot_top3(rows,winners,metric,outdir/f"{metric}_top3_by_day.png")

if __name__=="__main__": main()
