#!/usr/bin/env python3
"""2x2 publication panel of metabolite-based good/bad prediction.

Four subplots in one figure (for the paper). Rows = cohort
(strong-consensus, full); columns = feature set (raw vs. per-day winsorized).
Each subplot is the standard balanced-accuracy-by-day curve for LightGBM vs.
Logistic Regression, with the late stage (>= Dy20_5) shaded.

All curves are read from the already-computed result JSONs written by
``run.py`` (``analysis_output/metabolite_pred/results_<cohort>_<config>.json``)
-- nothing is re-trained here. The base config is fixed to ``nominal_delta``
(nominal amounts + day-over-day deltas, the headline metabolite analysis); the
raw/winsorized contrast is the ``_win`` twin of that config.

Run by path (package name starts with a digit):
    make run ARGS="analysis/2026_06_metabolite_pred/publication_panel.py"
    PYTHONPATH=. python analysis/2026_06_metabolite_pred/publication_panel.py
"""

import json
import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pipeline.data_loader import (
    ANALYSIS_OUTPUT_DIR,
    DAY_ORDER,
    FIGURE_DIR,
    get_day_int_floor,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = ANALYSIS_OUTPUT_DIR / "metabolite_pred"
BASE_CONFIG = "nominal_delta"

# Same colours/markers as the single-panel figures (run.py / metabolites_train.py)
# so the panel reads consistently with the rest of the paper. Blue/orange is the
# canonical colourblind-safe categorical pair.
_STYLE = {
    "LightGBM":            {"color": "#1f77b4", "marker": "o", "linestyle": "-"},
    "Logistic Regression": {"color": "#ff7f0e", "marker": "s", "linestyle": "--"},
}

# Rows = cohort, columns = raw vs winsorized. (row, col) -> (cohort, config, letter)
PANELS = {
    (0, 0): ("strong-consensus", BASE_CONFIG,          "A"),
    (0, 1): ("strong-consensus", f"{BASE_CONFIG}_win", "B"),
    (1, 0): ("full",             BASE_CONFIG,          "C"),
    (1, 1): ("full",             f"{BASE_CONFIG}_win", "D"),
}
_COHORT_LABEL = {"strong-consensus": "Strong consensus (n=198)", "full": "Full (n=248)"}
_FIELD_LABEL = {False: "raw concentration", True: "winsorized (1/99)"}
SHADE_FROM_DAY = 20  # Dy20_5 (floor 20)


def _load(cohort, config):
    path = RESULTS_DIR / f"results_{cohort}_{config}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found -- run `make run ARGS=\"analysis/2026_06_metabolite_pred/run.py\"` first."
        )
    with open(path) as f:
        return json.load(f)


def _plot_panel(ax, results, title, *, show_legend):
    series = {m: results[m] for m in ("LightGBM", "Logistic Regression") if results.get(m)}
    days = [d for d in DAY_ORDER if any(d in s for s in series.values())]
    x = list(range(len(days)))

    for label, s in series.items():
        st = _STYLE[label]
        ys = [s[d]["balanced_accuracy"] if d in s else None for d in days]
        valid = [(i, v) for i, v in zip(x, ys) if v is not None]
        if not valid:
            continue
        xs, vals = zip(*valid)
        ax.plot(xs, vals, marker=st["marker"], linestyle=st["linestyle"],
                color=st["color"], label=label, linewidth=2, markersize=6)

    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)  # chance
    late_idx = next((i for i, d in enumerate(days)
                     if (get_day_int_floor(d) or 0) >= SHADE_FROM_DAY), None)
    if late_idx is not None:
        ax.axvspan(late_idx, len(days) - 0.5, alpha=0.1, color="gray")

    ax.set_xticks(x)
    ax.set_xticklabels(days, rotation=45, ha="right", fontsize=8)
    ax.set_ylim(0.4, 1.0)
    ax.set_title(title, fontsize=11)
    ax.grid(True, alpha=0.3)
    if show_legend:
        ax.legend(fontsize=9, loc="upper left")


def main():
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True, sharey=True)
    for (r, c), (cohort, config, letter) in PANELS.items():
        ax = axes[r][c]
        results = _load(cohort, config)
        is_win = config.endswith("_win")
        title = f"({letter}) {_COHORT_LABEL[cohort]} — {_FIELD_LABEL[is_win]}"
        _plot_panel(ax, results, title, show_legend=(r == 0 and c == 0))
        if c == 0:
            ax.set_ylabel("Balanced accuracy")
        if r == 1:
            ax.set_xlabel("Day")

    fig.suptitle(
        "Metabolite-based good/bad prediction: balanced accuracy by day\n"
        "(LightGBM vs. Logistic Regression; nominal amounts + deltas; "
        "shaded region = late stage, ≥ Dy20_5)",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURE_DIR / "metabolite_pred_panel_2x2.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    logger.info("Saved %s", out)


if __name__ == "__main__":
    main()
