#!/usr/bin/env python3
"""Plot per-day EfficientNet and CNN-LSTM temporal ablation results on one figure.

X-axis semantics differ between models:
  - Per-day EfficientNet: a separate classifier trained on that day's images only.
  - CNN-LSTM temporal: a single model trained on all days 3 through that day
    (temporal ablation — shows how much of the series is needed).

Both series use balanced accuracy for fair comparison. The CNN-LSTM JSON does not
store balanced_accuracy directly, so it is derived from test_false_positives,
test_false_negatives, test_acc, and test_recall (see _balanced_acc_from_entry).

Reads:
  - $ANALYSIS_OUTPUT_DIR/images/perday_results.json
  - outputs/cnn_lstm/temporal_ablation_attn/temporal_ablation_results.json

Outputs:
  - $ANALYSIS_OUTPUT_DIR/figures/image_model_comparison.png

Usage:
    make run ARGS="-m analysis.paper_2026_04.image_model_comparison_plot"
    make run ARGS="-m analysis.paper_2026_04.image_model_comparison_plot --cnn-lstm-path /path/to/results.json"
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

from pipeline.data_loader import ANALYSIS_OUTPUT_DIR, DAY_ORDER, FIGURE_DIR

# CNN-LSTM float max_day → canonical day label
_MAX_DAY_TO_LABEL = {
    8: "Dy08", 10: "Dy10", 13: "Dy13", 15: "Dy15",
    17: "Dy17", 20.5: "Dy20_5", 24: "Dy24", 28: "Dy28", 30: "Dy30",
}

# Canonical day label → numeric x-axis value
_LABEL_TO_NUM = {
    "Dy03": 3, "Dy06": 6, "Dy08": 8, "Dy10": 10, "Dy13": 13,
    "Dy15": 15, "Dy17": 17, "Dy20_5": 20.5, "Dy24": 24, "Dy28": 28, "Dy30": 30,
}

DEFAULT_CNN_LSTM_PATH = Path("outputs/cnn_lstm/temporal_ablation_attn/temporal_ablation_results.json")
DEFAULT_PERDAY_PATH = ANALYSIS_OUTPUT_DIR / "images" / "perday_results.json"


def _load(path):
    path = Path(path)
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _derive_test_class_sizes(cnn_lstm_list):
    """Return (n_pos, n_neg) for the CNN-LSTM test set.

    Uses the max-day entry, which has the best-converged model and therefore
    the most meaningful FP/FN counts. Derivation:
      N = (n_fp + n_fn) / (1 - test_acc)
      n_pos = n_fn / (1 - test_recall)    [requires test_recall < 1]
      n_neg = N - n_pos
    """
    ref = max(cnn_lstm_list, key=lambda r: r["max_day"])
    n_fp = len(ref.get("test_false_positives", []))
    n_fn = len(ref.get("test_false_negatives", []))
    acc = ref["test_acc"]
    recall = ref.get("test_recall", 0.0)

    if abs(1.0 - acc) < 1e-9 or (n_fp + n_fn) == 0:
        return None, None
    n_total = round((n_fp + n_fn) / (1.0 - acc))

    if abs(1.0 - recall) < 1e-9:
        return None, None
    n_pos = round(n_fn / (1.0 - recall))
    n_neg = n_total - n_pos
    return n_pos, n_neg


def _balanced_acc_from_entry(r, n_pos, n_neg):
    """Compute balanced accuracy for one CNN-LSTM result dict.

    sensitivity = TP / n_pos = test_recall  (already in the dict)
    specificity = TN / n_neg = (n_neg - n_fp) / n_neg
    balanced_accuracy = (sensitivity + specificity) / 2
    """
    n_fp = len(r.get("test_false_positives", []))
    sensitivity = r.get("test_recall", 0.0)
    tn = n_neg - n_fp
    specificity = max(tn, 0) / n_neg if n_neg > 0 else 0.0
    return round((sensitivity + specificity) / 2, 4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--perday-path", default=str(DEFAULT_PERDAY_PATH))
    parser.add_argument("--cnn-lstm-path", default=str(DEFAULT_CNN_LSTM_PATH))
    args = parser.parse_args()

    perday = _load(args.perday_path)
    cnn_lstm_list = _load(args.cnn_lstm_path)

    if perday is None:
        print(f"Missing per-day results: {args.perday_path}")
        return
    if cnn_lstm_list is None:
        print(f"Missing CNN-LSTM results: {args.cnn_lstm_path}")
        return

    n_pos, n_neg = _derive_test_class_sizes(cnn_lstm_list)
    if n_pos is None:
        print("Warning: could not derive test class sizes; falling back to test_acc for CNN-LSTM")
        use_balanced = False
    else:
        print(f"CNN-LSTM test set: n_pos={n_pos} (Not Acceptable), n_neg={n_neg} (Acceptable)")
        use_balanced = True

    # CNN-LSTM list → dict keyed by canonical day label
    cnn_by_day = {
        _MAX_DAY_TO_LABEL[r["max_day"]]: r
        for r in cnn_lstm_list
        if r["max_day"] in _MAX_DAY_TO_LABEL
    }

    days = [d for d in DAY_ORDER if d in perday or d in cnn_by_day]
    x_nums = [_LABEL_TO_NUM[d] for d in days]

    perday_ba = [perday[d]["balanced_accuracy"] if d in perday else None for d in days]
    if use_balanced:
        cnn_ba = [
            _balanced_acc_from_entry(cnn_by_day[d], n_pos, n_neg) if d in cnn_by_day else None
            for d in days
        ]
    else:
        cnn_ba = [cnn_by_day[d]["test_acc"] if d in cnn_by_day else None for d in days]

    # --- Style ---
    plt.rcParams.update({
        "font.size": 13, "axes.labelsize": 14, "axes.titlesize": 15,
        "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 12,
    })

    fig, ax = plt.subplots(figsize=(12, 6))

    pd_pairs = [(x_nums[i], v) for i, v in enumerate(perday_ba) if v is not None]
    ts_pairs = [(x_nums[i], v) for i, v in enumerate(cnn_ba)    if v is not None]
    pd_dict  = dict(pd_pairs)
    ts_dict  = dict(ts_pairs)

    # Per-Day: solid blue circles
    if pd_pairs:
        px, py = zip(*pd_pairs)
        ax.plot(px, py, "o-", color="#3a7bbf", linewidth=2.5, markersize=7,
                label="Per-Day", zorder=3)

    # Time Series: dashed red squares
    if ts_pairs:
        tx, ty = zip(*ts_pairs)
        ax.plot(tx, ty, "s--", color="#c0392b", linewidth=2.5, markersize=7,
                label="Time Series", zorder=3)

    # Shaded fill between overlapping region
    shared_x = sorted(set(pd_dict) & set(ts_dict))
    if shared_x:
        ax.fill_between(shared_x,
                        [pd_dict[x] for x in shared_x],
                        [ts_dict[x] for x in shared_x],
                        alpha=0.12, color="#3a7bbf")

    # Context-aware labels with manual offsets for crowded days.
    # The offsets are in screen points, not data units:
    #   (0, 12) means 12 points above the point
    #   (-18, 8) means left and slightly above
    #   (18, 8) means right and slightly above

    CLOSE = 0.05
    DELTA_MIN = 0.05

    # Manual label positions for crowded areas
    pd_label_offsets = {
        10: (0, -20, "center"),
        15: (-18, 10, "right"),
        17: (-18, 10, "right"),
        20.5: (0, -22, "center"),
        24: (0, -22, "center"),
        28: (-12, 12, "right"),
        30: (-18, -2, "right"),
    }

    ts_label_offsets = {
        8: (0, 12, "center"),
        10: (0, 12, "center"),
        13: (0, 12, "center"),
        15: (18, 10, "left"),
        17: (18, 10, "left"),
        20.5: (0, 14, "center"),
        24: (0, 14, "center"),
        28: (12, 14, "left"),
        30: (20, 10, "left"),
    }

    delta_offsets = {
        20.5: (12, 0),
        24: (12, 0),
        28: (0, -18),
        30: (0, -18),
    }

    for i, d in enumerate(days):
        x = x_nums[i]
        pd_y = perday_ba[i]
        ts_y = cnn_ba[i]

        both = pd_y is not None and ts_y is not None
        gap = abs(pd_y - ts_y) if both else None

        # Per-Day label
        if pd_y is not None:
            if x in pd_label_offsets:
                dx, dy, ha = pd_label_offsets[x]
            elif both and gap < CLOSE:
                dx, dy, ha = -16, 10, "right"
            elif both and pd_y >= ts_y:
                dx, dy, ha = 0, 12, "center"
            elif both:
                dx, dy, ha = 0, -20, "center"
            else:
                dx, dy, ha = 0, 12, "center"

            ax.annotate(
                f"{pd_y:.2f}",
                xy=(x, pd_y),
                xytext=(dx, dy),
                textcoords="offset points",
                ha=ha,
                va="center",
                fontsize=10,
                color="#3a7bbf",
                fontweight="bold",
            )

        # Time Series label
        if ts_y is not None:
            if x in ts_label_offsets:
                dx, dy, ha = ts_label_offsets[x]
            elif both and gap < CLOSE:
                dx, dy, ha = 16, 10, "left"
            elif both and ts_y >= pd_y:
                dx, dy, ha = 0, 12, "center"
            elif both:
                dx, dy, ha = 0, -20, "center"
            else:
                dx, dy, ha = 0, 12, "center"

            ax.annotate(
                f"{ts_y:.2f}",
                xy=(x, ts_y),
                xytext=(dx, dy),
                textcoords="offset points",
                ha=ha,
                va="center",
                fontsize=10,
                color="#c0392b",
                fontweight="bold",
            )

        # Delta annotation
        if both and gap >= DELTA_MIN:
            diff = pd_y - ts_y
            mid_y = (pd_y + ts_y) / 2
            sign = "+" if diff >= 0 else ""

            dx, dy = delta_offsets.get(x, (12, 0))

            ax.annotate(
                f"{sign}{diff:.2f}",
                xy=(x, mid_y),
                xytext=(dx, dy),
                textcoords="offset points",
                ha="left" if dx >= 0 else "right",
                va="center",
                fontsize=9,
                color="gray",
            )

    # Chance line
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1.5,
               label="Chance (0.50)", zorder=1)

    ax.set_xticks(x_nums)
    ax.set_xticklabels([str(x) for x in x_nums])
    ax.set_xlabel("Day")
    ax.set_ylabel("Balanced Accuracy (threshold = 0.5)")
    ax.set_title("Per-Day vs. Time Series")
    ax.set_ylim(0.4, 1.0)
    ax.legend(loc="upper left")
    ax.yaxis.grid(True, alpha=0.4)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURE_DIR / "image_model_comparison.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
