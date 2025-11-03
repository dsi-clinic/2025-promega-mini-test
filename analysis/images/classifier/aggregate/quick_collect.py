#!/usr/bin/env python3
"""
Aggregate all classifier runs into three metric-specific summaries:
  - summary/accuracy/...
  - summary/auroc/...
  - summary/f1/...

For EACH metric, it writes:
  1) combined_summary.csv            (all rows)
  2) leaderboard_by_day.csv          (best run per day)
  3) earliest_reliable_day.csv       (>= threshold; AUROC: 0.80, Acc/F1: 0.75 default)
  4) best_by_day_<metric>.png        (bar, winner per day)
  5) heatmap_variant_<metric>.png    (Day × Variant, max per cell)
  6) heatmap_backbone_<metric>.png   (Day × Backbone, max per cell)
  7) early_late_summary.csv/.png     (means early vs late by variant/backbone)

Confusion matrices (optional):
  - Tries to derive TP/FP/FN/TN per (run_root, variant, backbone, day) by
    reading rows from misclassified CSVs if available. Results in:
      confusion_by_day.csv            (if data found)
    If not enough info → silently skips.

Usage:
  python analysis/images/classifier/aggregate/quick_collect.py
  # or
  python analysis/images/classifier/aggregate/quick_collect.py --outdir analysis/images/classifier/aggregate/summary
"""
import json, csv, re, argparse, math
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

# --------------------- helpers ---------------------
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

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p

def variant_from_name(name: str) -> str:
    n = name.lower()
    if "nomask_image"   in n: return "rgb"
    if "nomask_overlay" in n: return "overlay"
    if "mask_image"     in n: return "rgb+mask"
    if "mask_overlay"   in n: return "overlay+mask"
    if "softlabels"     in n: return "soft(rgb)"
    if "regular_image"  in n: return "rgb"
    return name

# ---------------- collection -----------------------
def collect_rows(classifier_dir: Path):
    rows = []
    for run_root in sorted([p for p in classifier_dir.iterdir() if p.is_dir() and p.name.startswith("outputs_")]):
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
                    "day": t.get("day", day_dir.name),
                    "day_no": day_to_int(t.get("day", day_dir.name)),
                    # allow both hard/soft naming
                    "test_accuracy":    t.get("test_accuracy", t.get("accuracy")),
                    "test_f1":          t.get("test_f1", t.get("f1")),
                    "test_roc_auc":     t.get("test_roc_auc", t.get("roc_auc")),
                    "test_pr_auc":      t.get("test_pr_auc", t.get("pr_auc")),
                    "test_n":           t.get("test_n"),
                    "actual_good":      t.get("actual_good"),
                    "predicted_good":   t.get("predicted_good"),
                })
    return [r for r in rows if isinstance(r["day_no"], int) and r["day_no"] >= 0]

# ------------- optional confusion matrix -----------
def try_confusion_from_miscsv(base_dir: Path):
    """
    Tries to compute TP/FP/FN/TN per (run_root, variant, backbone, day) by reading
    any CSVs in misclassifiedimages/*.csv that contain columns:
      organoid_key, y_true, y_pred, backbone, day, run_root (names may vary)
    This function is best-effort; if files/columns are missing it returns {}.
    """
    mis_dir = base_dir / "misclassifiedimages"
    if not mis_dir.exists():
        return {}
    # load all rows
    import glob
    tp = defaultdict(int); fp = defaultdict(int); fn = defaultdict(int); tn = defaultdict(int)
    keys_seen = set()
    for csv_path in glob.glob(str(mis_dir / "*.csv")):
        try:
            with open(csv_path, "r", newline="") as f:
                rd = csv.DictReader(f)
                for row in rd:
                    # flexible column names
                    b    = row.get("backbone") or row.get("model") or row.get("backbone_key")
                    day  = row.get("day") or row.get("day_id") or row.get("Dy")
                    rr   = row.get("run_root") or row.get("run") or row.get("run_name")
                    var  = row.get("variant") or row.get("input") or row.get("channels")
                    yt_s = row.get("y_true") or row.get("label") or row.get("true")
                    yp_s = row.get("y_pred") or row.get("pred") or row.get("pred_label")
                    if not (b and day and rr and yt_s is not None and yp_s is not None):
                        continue
                    # normalize
                    if not isinstance(day, str): day = str(day)
                    day_no = day_to_int(day)
                    if day_no < 0: continue
                    key = (rr, (var or ""), b, f"Dy{day_no}")
                    try:
                        yt = int(float(yt_s)); yp = int(float(yp_s))
                    except Exception:
                        continue
                    if yt == 1 and yp == 1: tp[key] += 1
                    elif yt == 0 and yp == 1: fp[key] += 1
                    elif yt == 1 and yp == 0: fn[key] += 1
                    elif yt == 0 and yp == 0: tn[key] += 1
                    keys_seen.add(key)
        except Exception:
            continue
    out = {}
    for k in keys_seen:
        out[k] = {"TP": tp[k], "FP": fp[k], "FN": fn[k], "TN": tn[k]}
    return out  # keys: (run_root, variant, backbone, day)

# ---------------- CSV/plot utils -------------------
def write_csv(rows, path: Path):
    if not rows: return
    keys = sorted({k for r in rows for k in r.keys()})
    ensure_dir(path.parent)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)

def build_leaderboard(rows, metric):
    by_day = defaultdict(list)
    for r in rows:
        v = r.get(metric)
        if isinstance(v, (int, float)): by_day[r["day"]].append(r)
    best = []
    for day, items in by_day.items():
        top = max(items, key=lambda x: x[metric])
        best.append({
            "day": day, "day_no": top["day_no"], "metric": metric,
            "best_score": float(top[metric]), "run_root": top["run_root"],
            "variant": top["variant"], "backbone": top["backbone"],
        })
    return sorted(best, key=lambda x: x["day_no"])

def earliest_reliable_day(rows, metric, threshold):
    groups = defaultdict(list)
    for r in rows:
        v = r.get(metric)
        if isinstance(v, (int, float)): groups[(r["run_root"], r["variant"], r["backbone"])].append(r)
    out = []
    for key, items in groups.items():
        items = sorted(items, key=lambda x: x["day_no"])
        first = next((it for it in items if it[metric] is not None and it[metric] >= threshold), None)
        out.append({
            "run_root": key[0], "variant": key[1], "backbone": key[2],
            "metric": metric, "threshold": threshold,
            "earliest_day": first["day"] if first else None,
            "earliest_day_no": first["day_no"] if first else None
        })
    return sorted(out, key=lambda r: (r["variant"], r["backbone"]))

def _sorted_days(rows, metric):
    return sorted(sorted({r["day_no"] for r in rows if isinstance(r.get(metric), (int, float))}))

def _bar_labels(ax, xs, ys):
    for x, y in zip(xs, ys):
        if y is None or (isinstance(y, float) and math.isnan(y)): continue
        ax.text(x, y + 0.01, f"{y:.2f}", ha="center", va="bottom", fontsize=8)

def plot_best_by_day(rows, metric, out_png: Path):
    by_day = defaultdict(list)
    for r in rows:
        if isinstance(r.get(metric), (int, float)):
            by_day[r["day_no"]].append(r)
    if not by_day: return
    days = sorted(by_day.keys())
    best_scores, labels = [], []
    for d in days:
        top = max(by_day[d], key=lambda x: x[metric])
        best_scores.append(top[metric])
        labels.append(f"{top['backbone']} · {top['variant']}")
    ensure_dir(out_png.parent)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(days, best_scores)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Day"); ax.set_ylabel(f"Best {metric}")
    ax.set_title(f"Winner per day ({metric})")
    _bar_labels(ax, days, best_scores)
    ax.set_xticks(days, [f"{d}\n{lab}" for d, lab in zip(days, labels)], rotation=0, fontsize=8)
    fig.tight_layout(); fig.savefig(out_png, dpi=150); plt.close(fig)

def _heatmap_matrix(rows, metric, group_key):
    valid = [r for r in rows if isinstance(r.get(metric), (int, float))]
    if not valid: return None, [], []
    days = _sorted_days(valid, metric)
    groups = sorted(sorted({r[group_key] for r in valid}))
    M = np.full((len(days), len(groups)), np.nan, dtype=float)
    bucket = defaultdict(list)
    for r in valid:
        bucket[(r["day_no"], r[group_key])].append(r[metric])
    for i, d in enumerate(days):
        for j, g in enumerate(groups):
            vals = bucket.get((d, g), [])
            if vals:
                M[i, j] = max(vals)
    return M, days, groups

def plot_heatmap(rows, metric, group_key, title, out_png):
    M, days, groups = _heatmap_matrix(rows, metric, group_key)
    if M is None: return
    ensure_dir(out_png.parent)
    fig, ax = plt.subplots(figsize=(1.4*len(groups)+2, 0.35*len(days)+2))
    im = ax.imshow(M, vmin=0.5, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(groups)), groups, rotation=45, ha="right")
    ax.set_yticks(range(len(days)), days)
    ax.set_xlabel(group_key.capitalize()); ax.set_ylabel("Day")
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax); cbar.set_label(metric)
    for i in range(len(days)):
        for j in range(len(groups)):
            v = M[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8)
    fig.tight_layout(); fig.savefig(out_png, dpi=150); plt.close(fig)

def early_late_summary(rows, metric, out_csv: Path, out_png: Path):
    early_set = {3,6,8,10}
    late_set  = {20,24,28,30}
    agg = defaultdict(lambda: {"early": [], "late": []})
    for r in rows:
        v = r.get(metric)
        if not isinstance(v, (int, float)): continue
        d = r["day_no"]
        key = (r["variant"], r["backbone"])
        if d in early_set: agg[key]["early"].append(v)
        if d in late_set:  agg[key]["late"].append(v)
    table = []
    for (variant, backbone), parts in agg.items():
        e = parts["early"]; l = parts["late"]
        e_mean = float(np.mean(e)) if e else None
        l_mean = float(np.mean(l)) if l else None
        delta  = (l_mean - e_mean) if (e_mean is not None and l_mean is not None) else None
        table.append({
            "variant": variant, "backbone": backbone,
            f"early_mean_{metric}": e_mean, f"late_mean_{metric}": l_mean,
            "n_early": len(e), "n_late": len(l),
            "delta_late_minus_early": delta
        })
    write_csv(table, out_csv)
    if not table: return
    labels = [f"{t['variant']} · {t['backbone']}" for t in table]
    e_vals = [t.get(f"early_mean_{metric}", np.nan) for t in table]
    l_vals = [t.get(f"late_mean_{metric}",  np.nan) for t in table]
    x = np.arange(len(labels)); width = 0.38
    fig, ax = plt.subplots(figsize=(max(10, 0.6*len(labels)+3), 4))
    ax.bar(x - width/2, e_vals, width, label="Early")
    ax.bar(x + width/2, l_vals, width, label="Late")
    ax.set_xticks(x, labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylim(0.0, 1.0); ax.set_ylabel(f"Mean {metric}")
    ax.set_title(f"Early vs Late (mean {metric})"); ax.legend()
    fig.tight_layout(); fig.savefig(out_png, dpi=150); plt.close(fig)

# ---------------- main per-metric pass --------------
def write_confusion_summary(base_dir: Path, rows, out_csv: Path):
    """Optional; only if we can infer from misclassified CSVs."""
    cm = try_confusion_from_miscsv(base_dir)
    if not cm: return
    index = {(r["run_root"], r["variant"], r["backbone"], r["day"]): r for r in rows}
    out = []
    for key, counts in cm.items():
        rr, var, back, day = key
        r = index.get((rr, var, back, day))
        if not r: continue
        total = sum(counts.values())
        acc = (counts["TP"] + counts["TN"]) / total if total > 0 else None
        out.append({
            "run_root": rr, "variant": var, "backbone": back, "day": day,
            "day_no": day_to_int(day), **counts, "acc_from_cm": acc, "n_from_cm": total
        })
    if out:
        write_csv(sorted(out, key=lambda x: (x["variant"], x["backbone"], x["day_no"])), out_csv)

def per_metric_pass(base_dir: Path, all_rows, metric: str, outdir: Path, threshold: float):
    ensure_dir(outdir)
    # CSVs
    write_csv(all_rows, outdir / "combined_summary.csv")
    write_csv(build_leaderboard(all_rows, metric), outdir / "leaderboard_by_day.csv")
    write_csv(earliest_reliable_day(all_rows, metric, threshold), outdir / "earliest_reliable_day.csv")
    # Plots
    plot_best_by_day(all_rows, metric, outdir / f"best_by_day_{metric}.png")
    plot_heatmap(all_rows, metric, "variant",  f"{metric} by day × variant (max per cell)",   outdir / f"heatmap_variant_{metric}.png")
    plot_heatmap(all_rows, metric, "backbone", f"{metric} by day × backbone (max per cell)", outdir / f"heatmap_backbone_{metric}.png")
    early_late_summary(all_rows, metric, outdir / "early_late_summary.csv", outdir / f"early_late_summary_{metric}.png")
    # Confusion (optional)
    write_confusion_summary(base_dir, all_rows, outdir / "confusion_by_day.csv")

# ---------------- CLI ------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="analysis/images/classifier/aggregate/summary")
    ap.add_argument("--acc-threshold", type=float, default=0.75)
    ap.add_argument("--f1-threshold",  type=float, default=0.75)
    ap.add_argument("--auc-threshold", type=float, default=0.80)
    args = ap.parse_args()

    # allow repo root or classifier dir
    here = Path.cwd()
    base = here / "analysis/images/classifier" if (here / "analysis" / "images" / "classifier").exists() else here

    rows = collect_rows(base)
    if not rows:
        print("No runs found under outputs_*.")
        return

    out_root = ensure_dir(Path(args.outdir))
    # Filter rows per metric and run the same reporting stack
    # ACCURACY
    acc_rows = [r for r in rows if isinstance(r.get("test_accuracy"), (int, float))]
    if acc_rows:
        per_metric_pass(base, acc_rows, "test_accuracy", ensure_dir(out_root / "accuracy"), args.acc_threshold)
    # AUROC
    auc_rows = [r for r in rows if isinstance(r.get("test_roc_auc"), (int, float))]
    if auc_rows:
        per_metric_pass(base, auc_rows, "test_roc_auc", ensure_dir(out_root / "auroc"), args.auc_threshold)
    # F1
    f1_rows = [r for r in rows if isinstance(r.get("test_f1"), (int, float))]
    if f1_rows:
        per_metric_pass(base, f1_rows, "test_f1", ensure_dir(out_root / "f1"), args.f1_threshold)

    print(f"✓ Wrote summaries to {out_root}")

if __name__ == "__main__":
    main()
