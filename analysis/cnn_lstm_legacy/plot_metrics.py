"""
Generate publication-quality panels from CNN-LSTM results:
1) Training metrics (per model): train/val Loss, Accuracy, F1 (if present)
2) Confusion matrices (per model)
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ---- Matplotlib defaults (publication-ish) ----
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 13
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['figure.dpi'] = 300

def load_json(p: Path):
    with open(p) as f:
        return json.load(f)

def hist(series: dict, key: str):
    return [ep.get(key) for ep in series.get("train_history", []) if key in ep]

def has_any(series: dict, keys):
    return any(hist(series, k) for k in keys)

def _nan_or_list(x):
    try:
        return list(x)
    except Exception:
        return []

def safe_metric_panel_cols(results_by_name):
    """Determine which metric groups exist in at least one model."""
    have_loss = any(has_any(r, ["train_loss", "val_loss"]) for r in results_by_name.values())
    have_acc  = any(has_any(r, ["train_acc", "val_acc"]) for r in results_by_name.values())
    have_f1   = any(has_any(r, ["train_f1", "val_f1", "val_f1_score"]) for r in results_by_name.values())
    cols = []
    if have_loss: cols.append("loss")
    if have_acc:  cols.append("acc")
    if have_f1:   cols.append("f1")
    return cols

def plot_training_panel(results_by_name: dict, output_dir: Path):
    """
    Rows = models; Cols = metric groups present across any model (Loss/Acc/F1)
    Each cell shows train (solid) and val (dashed) curves when available.
    """
    names = list(results_by_name.keys())
    cols = safe_metric_panel_cols(results_by_name)
    if not cols:
        print("⚠️ No training/validation series found. Skipping training panel.")
        return

    n_rows = len(names)
    n_cols = len(cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.0*n_cols, 3.6*n_rows), squeeze=False)

    for r, name in enumerate(names):
        res = results_by_name[name]
        # prefetch histories
        ep_train_loss = hist(res, "train_loss")
        ep_val_loss   = hist(res, "val_loss")
        ep_train_acc  = hist(res, "train_acc")
        ep_val_acc    = hist(res, "val_acc")
        ep_train_f1   = hist(res, "train_f1")
        ep_val_f1     = hist(res, "val_f1") or hist(res, "val_f1_score")  # tolerate alt key

        # epochs by available length for each plotted series
        def plot_series(ax, y_train, y_val, ylabel, title_suffix):
            any_plotted = False
            if y_train:
                ax.plot(np.arange(1, len(y_train)+1), y_train, linewidth=2, label="Train")
                any_plotted = True
            if y_val:
                ax.plot(np.arange(1, len(y_val)+1), y_val, linestyle="--", linewidth=2, label="Val")
                any_plotted = True
            ax.set_xlabel("Epoch")
            ax.set_ylabel(ylabel)
            ax.set_title(f"{name}: {title_suffix}", fontweight="bold")
            ax.grid(alpha=0.3)
            if any_plotted:
                ax.legend()

        for c, group in enumerate(cols):
            ax = axes[r, c]
            if group == "loss":
                plot_series(ax, ep_train_loss, ep_val_loss, "Loss", "Loss")
            elif group == "acc":
                plot_series(ax, ep_train_acc, ep_val_acc, "Accuracy", "Accuracy")
                ax.set_ylim(0, 1.0)
            elif group == "f1":
                plot_series(ax, ep_train_f1, ep_val_f1, "F1", "F1")
                ax.set_ylim(0, 1.0)

    plt.tight_layout()
    out = output_dir / "training_metrics_panel.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print(f"✓ Saved: {out}")
    plt.close()

def plot_confusion_panel(results_by_name: dict, output_dir: Path):
    """
    One confusion matrix per model in a single row panel.
    Expects 'confusion_matrix' shaped [[tn, fp],[fn, tp]] (binary).
    """
    names = list(results_by_name.keys())
    cms = []
    for name in names:
        cm = results_by_name[name].get("confusion_matrix")
        if cm is None:
            cms.append(None)
        else:
            cm = np.array(cm, dtype=float)
            if cm.ndim != 2:
                cms.append(None)
            else:
                cms.append(cm)

    if not any(cm is not None for cm in cms):
        print("⚠️ No confusion matrices found. Skipping CM panel.")
        return

    n = len(names)
    fig, axes = plt.subplots(1, n, figsize=(4.6*n, 4.2), squeeze=False)
    axes = axes[0]

    vmax = max((cm.max() for cm in cms if cm is not None), default=1.0)

    for i, (name, cm) in enumerate(zip(names, cms)):
        ax = axes[i]
        if cm is None:
            ax.axis("off")
            ax.set_title(f"{name}\n(no confusion_matrix)")
            continue

        im = ax.imshow(cm, interpolation="nearest", vmin=0, vmax=vmax)
        # annotate
        for r in range(cm.shape[0]):
            for c in range(cm.shape[1]):
                txt = int(cm[r, c]) if float(cm[r, c]).is_integer() else f"{cm[r,c]:.1f}"
                ax.text(c, r, txt, ha="center", va="center", fontsize=11, color="white")

        # try to be helpful with tick labels; default to binary
        ax.set_xticks(range(cm.shape[1])); ax.set_yticks(range(cm.shape[0]))
        if cm.shape == (2, 2):
            ax.set_xticklabels(["Pred 0", "Pred 1"])
            ax.set_yticklabels(["True 0", "True 1"])
        else:
            ax.set_xticklabels([f"P{j}" for j in range(cm.shape[1])])
            ax.set_yticklabels([f"T{i}" for i in range(cm.shape[0])])

        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(name, fontweight="bold")

    # single shared colorbar
    cbar = fig.colorbar(im, ax=axes, fraction=0.022, pad=0.04)
    cbar.ax.set_ylabel("Count")

    plt.tight_layout()
    out = output_dir / "confusion_matrices_panel.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print(f"✓ Saved: {out}")
    plt.close()

def safe_best(series, key, fallback_key=None):
    v = series.get(key, None)
    if v is not None:
        return v
    if fallback_key:
        vals = hist(series, fallback_key)
        if vals:
            try:
                return float(np.nanmax(vals))
            except Exception:
                pass
    return None

def fmt(x):
    return "N/A" if x is None else f"{x:.3f}"

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=Path,
                        required=True,
                        help="Directory containing results_*.json files")
    args = parser.parse_args()
    base_dir = args.base_dir
    paths = {
        "All-White": base_dir / 'results_allwhite.json',
        "Blur Only": base_dir / 'results_blur.json',
        "Clip+Blur": base_dir / 'results_clipblur.json',
    }

    output_dir = base_dir / 'comparison_plots'
    output_dir.mkdir(exist_ok=True, parents=True)

    results = {}
    for name, p in paths.items():
        if not p.exists():
            print(f"⚠️ Missing: {p}")
            continue
        try:
            results[name] = load_json(p)
            print(f"Loaded {name} ({p.name})")
        except Exception as e:
            print(f"❌ Failed to load {p}: {e}")

    if not results:
        print("❌ No results loaded. Exiting.")
        return

    # Training metrics panel
    plot_training_panel(results, output_dir)

    # Confusion matrices panel
    plot_confusion_panel(results, output_dir)

    # concise metrics summary
    print("\n=== FINAL METRICS (best val if available) ===")
    for name, r in results.items():
        best_val_acc  = safe_best(r, "best_val_acc",  "val_acc")
        best_val_f1   = safe_best(r, "best_val_f1",   "val_f1")
        best_val_loss = safe_best(r, "best_val_loss", "val_loss")
        test_acc = r.get("test_acc")
        test_f1  = r.get("test_f1")
        print(f"{name:<12} | Val Acc: {fmt(best_val_acc)} | Val F1: {fmt(best_val_f1)} | "
              f"Val Loss: {fmt(best_val_loss)} | Test Acc: {fmt(test_acc)} | Test F1: {fmt(test_f1)}")

    print(f"\n✅ Outputs in: {output_dir}")
    print("   - training_metrics_panel.png")
    print("   - confusion_matrices_panel.png")

if __name__ == "__main__":
    main()