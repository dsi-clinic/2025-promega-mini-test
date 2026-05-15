#!/usr/bin/env python3
"""Plot per-day EfficientNet and CNN-LSTM temporal ablation results on one figure.

X-axis semantics differ between models:
  - Per-day EfficientNet: a separate classifier trained on that day's images only.
  - CNN-LSTM temporal: a single model trained on all days 3 through that day
    (temporal ablation — shows how much of the series is needed).

Reads:
  - $ANALYSIS_OUTPUT_DIR/images/perday_results.json
  - outputs/cnn_lstm/temporal_ablation_attn/temporal_ablation_results.json

Outputs:
  - $ANALYSIS_OUTPUT_DIR/figures/image_model_comparison.png

Usage:
    make run ARGS="-m analysis.paper_2026_04.image_model_comparison_plot"
    make run ARGS="-m analysis.paper_2026_04.image_model_comparison_plot --cnn-lstm-path /custom/path.json"
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

DEFAULT_CNN_LSTM_PATH = Path("outputs/cnn_lstm/temporal_ablation_attn/temporal_ablation_results.json")
DEFAULT_PERDAY_PATH = ANALYSIS_OUTPUT_DIR / "images" / "perday_results.json"


def _load(path):
    path = Path(path)
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _plot_valid(ax, xs, ys, **kwargs):
    pairs = [(x, y) for x, y in zip(xs, ys) if y is not None]
    if pairs:
        xi, yi = zip(*pairs)
        ax.plot(xi, yi, **kwargs)


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

    # CNN-LSTM list → dict keyed by canonical day label
    cnn_by_day = {
        _MAX_DAY_TO_LABEL[r["max_day"]]: r
        for r in cnn_lstm_list
        if r["max_day"] in _MAX_DAY_TO_LABEL
    }

    days = [d for d in DAY_ORDER if d in perday or d in cnn_by_day]
    xs = list(range(len(days)))

    perday_ba  = [perday[d]["balanced_accuracy"] if d in perday else None for d in days]
    cnn_acc    = [cnn_by_day[d]["test_acc"]       if d in cnn_by_day else None for d in days]
    cnn_f1     = [cnn_by_day[d]["test_f1"]        if d in cnn_by_day else None for d in days]
    cnn_auc    = [cnn_by_day[d].get("test_auc")   if d in cnn_by_day else None for d in days]

    fig, ax = plt.subplots(figsize=(12, 6))

    _plot_valid(ax, xs, perday_ba,
                marker="o", linewidth=2, color="#1f77b4",
                label="Per-day EfficientNet (balanced accuracy)")
    _plot_valid(ax, xs, cnn_acc,
                marker="s", linewidth=2, color="#d62728",
                label="CNN-LSTM temporal (test accuracy, series through that day)")
    _plot_valid(ax, xs, cnn_f1,
                marker="^", linewidth=2, color="#d62728", linestyle="--",
                label="CNN-LSTM temporal (test F1)")
    _plot_valid(ax, xs, cnn_auc,
                marker="D", linewidth=2, color="#9467bd", linestyle=":",
                label="CNN-LSTM temporal (AUC)")

    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, alpha=0.6, label="Chance (0.5)")

    ax.set_xticks(xs)
    ax.set_xticklabels(days, rotation=45)
    ax.set_xlabel("Day")
    ax.set_ylabel("Score")
    ax.set_title(
        "Image Model Comparison: Per-day EfficientNet vs CNN-LSTM Temporal Ablation\n"
        "(CNN-LSTM x-axis = last day included in time series)"
    )
    ax.set_ylim(0.0, 1.05)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURE_DIR / "image_model_comparison.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
