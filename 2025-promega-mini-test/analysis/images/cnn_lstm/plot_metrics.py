"""
Generate publication-style plots from one or more CNN-LSTM results.json files.

Outputs:
  - training_metrics_panel.png  (rows=models, cols=Loss/Acc/F1 when available)
  - confusion_matrices_panel.png (one CM per model)
  - metrics_summary.json         (optional, convenience)

Examples:
  python -m analysis.images.cnn_lstm.plot_metrics \
    --results /net/.../run_682522/results.json \
    --labels run_682522 \
    --outdir /home/your_name/.../plots

  python -m analysis.images.cnn_lstm.plot_metrics \
    --results A/results.json B/results.json \
    --labels "A" "B" \
    --outdir ./comparison
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ---- Matplotlib defaults (publication-ish) ----
plt.rcParams["font.size"] = 12
plt.rcParams["axes.labelsize"] = 13
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["xtick.labelsize"] = 11
plt.rcParams["ytick.labelsize"] = 11
plt.rcParams["legend.fontsize"] = 11
plt.rcParams["figure.dpi"] = 200


def load_json(p: Path) -> dict:
    with p.open() as f:
        return json.load(f)


def hist(series: dict, key: str) -> list[float]:
    out = []
    for ep in series.get("train_history", []):
        if key in ep and ep.get(key) is not None:
            out.append(ep.get(key))
    return out


def _have_any(results: dict, keys: list[str]) -> bool:
    return any(len(hist(r, k)) > 0 for r in results.values() for k in keys)


def _metric_cols(results: dict) -> list[str]:
    cols: list[str] = []
    if _have_any(results, ["train_loss", "val_loss"]):
        cols.append("loss")
    if _have_any(results, ["train_acc", "val_acc"]):
        cols.append("acc")
    if _have_any(results, ["train_f1", "val_f1", "val_f1_score"]):
        cols.append("f1")
    return cols


def plot_training_panel(results: dict, outdir: Path) -> Path | None:
    names = list(results.keys())
    cols = _metric_cols(results)
    if not cols:
        return None

    fig, axes = plt.subplots(
        len(names),
        len(cols),
        figsize=(5.0 * len(cols), 3.6 * len(names)),
        squeeze=False,
    )

    def plot_series(ax, y_train, y_val, ylabel, title):
            any_plotted = False
            if y_train:
            ax.plot(np.arange(1, len(y_train) + 1), y_train, linewidth=2, label="Train")
                any_plotted = True
            if y_val:
            ax.plot(np.arange(1, len(y_val) + 1), y_val, linestyle="--", linewidth=2, label="Val")
                any_plotted = True
            ax.set_xlabel("Epoch")
            ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight="bold")
            ax.grid(alpha=0.3)
            if any_plotted:
                ax.legend()

    for r, name in enumerate(names):
        res = results[name]
        tr_loss = hist(res, "train_loss")
        va_loss = hist(res, "val_loss")
        tr_acc = hist(res, "train_acc")
        va_acc = hist(res, "val_acc")
        tr_f1 = hist(res, "train_f1")
        va_f1 = hist(res, "val_f1") or hist(res, "val_f1_score")

        for c, group in enumerate(cols):
            ax = axes[r, c]
            if group == "loss":
                plot_series(ax, tr_loss, va_loss, "Loss", f"{name}: Loss")
            elif group == "acc":
                plot_series(ax, tr_acc, va_acc, "Accuracy", f"{name}: Accuracy")
                ax.set_ylim(0, 1.0)
            elif group == "f1":
                plot_series(ax, tr_f1, va_f1, "F1", f"{name}: F1")
                ax.set_ylim(0, 1.0)

    plt.tight_layout()
    out = outdir / "training_metrics_panel.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_confusion_panel(results: dict, outdir: Path) -> Path | None:
    names = list(results.keys())
    cms: list[np.ndarray | None] = []
    for n in names:
        cm = results[n].get("confusion_matrix")
        if cm is None:
            cms.append(None)
            continue
        arr = np.array(cm, dtype=float)
        if arr.ndim != 2:
            cms.append(None)
        else:
            cms.append(arr)

    if not any(cm is not None for cm in cms):
        return None

    fig, axes = plt.subplots(1, len(names), figsize=(4.6 * len(names), 4.2), squeeze=False)
    axes = axes[0]

    vmax = max((cm.max() for cm in cms if cm is not None), default=1.0)
    im = None
    for i, (name, cm) in enumerate(zip(names, cms)):
        ax = axes[i]
        if cm is None:
            ax.axis("off")
            ax.set_title(f"{name}\n(no confusion_matrix)")
            continue
        im = ax.imshow(cm, interpolation="nearest", vmin=0, vmax=vmax)
        for r in range(cm.shape[0]):
            for c in range(cm.shape[1]):
                txt = int(cm[r, c]) if float(cm[r, c]).is_integer() else f"{cm[r, c]:.1f}"
                ax.text(c, r, txt, ha="center", va="center", fontsize=11, color="white")
        ax.set_xticks(range(cm.shape[1]))
        ax.set_yticks(range(cm.shape[0]))
        if cm.shape == (2, 2):
            ax.set_xticklabels(["Pred 0", "Pred 1"])
            ax.set_yticklabels(["True 0", "True 1"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(name, fontweight="bold")

    if im is not None:
    cbar = fig.colorbar(im, ax=axes, fraction=0.022, pad=0.04)
    cbar.ax.set_ylabel("Count")

    plt.tight_layout()
    out = outdir / "confusion_matrices_panel.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def _safe_best(series: dict, key: str, fallback_hist_key: str | None = None):
    v = series.get(key)
    if v is not None:
        return v
    if fallback_hist_key:
        vals = hist(series, fallback_hist_key)
        if vals:
            try:
                return float(np.nanmax(vals))
            except Exception:
                return None
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="+", type=Path, required=True, help="One or more results.json paths")
    ap.add_argument("--labels", nargs="*", default=None, help="Optional labels (same count as --results)")
    ap.add_argument("--outdir", type=Path, required=True, help="Output directory for PNGs")
    args = ap.parse_args()

    outdir: Path = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    labels = args.labels if args.labels else [p.stem for p in args.results]
    if len(labels) != len(args.results):
        raise SystemExit("If provided, --labels must match the number of --results paths")

    results: dict[str, dict] = {}
    for label, path in zip(labels, args.results):
        if not path.exists():
            raise SystemExit(f"Missing results file: {path}")
        results[label] = load_json(path)

    train_png = plot_training_panel(results, outdir)
    cm_png = plot_confusion_panel(results, outdir)

    # Write a tiny summary to make comparisons easy
    summary = {}
    for name, r in results.items():
        summary[name] = {
            "best_val_acc": _safe_best(r, "best_val_acc", "val_acc"),
            "best_val_loss": _safe_best(r, "best_val_loss", "val_loss"),
            "best_val_f1": _safe_best(r, "best_val_f1", "val_f1"),
            "test_acc": r.get("test_acc"),
            "test_f1": r.get("test_f1"),
            "confusion_matrix": r.get("confusion_matrix"),
        }
    (outdir / "metrics_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"✅ Wrote plots to: {outdir}")
    if train_png:
        print(f"  - {train_png.name}")
    if cm_png:
        print(f"  - {cm_png.name}")
    print("  - metrics_summary.json")


if __name__ == "__main__":
    main()

