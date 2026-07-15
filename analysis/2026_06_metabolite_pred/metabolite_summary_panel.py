#!/usr/bin/env python3
"""Six-panel per-metabolite summary of concentration over each day.

One panel per metabolite (2x3 grid); each panel plots mean / median / min / max
across the IDOR sample, over ``DAY_ORDER``. Overall (no acceptance-class split).

Produces two figures, both reading values already stored in all_data.json (no
recomputation): the raw ``concentration_uM`` (``metabolite_summary_<cohort>.png``)
and the persisted per-day-winsorized ``concentration_uM_win``
(``metabolite_summary_<cohort>_win.png``).

Run by path (package name starts with a digit):
    make run ARGS="analysis/2026_06_metabolite_pred/metabolite_summary_panel.py"
    PYTHONPATH=. python analysis/2026_06_metabolite_pred/metabolite_summary_panel.py
"""

import logging
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pipeline.data_loader import (
    DAY_ORDER,
    FIGURE_DIR,
    REQUIRED_METABOLITES,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cohorts import ALL_DATA_PATH, build_cohort

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

STATS = ("mean", "median", "min", "max")
_STAT_FN = {"mean": np.mean, "median": np.median, "min": np.min, "max": np.max}
_STAT_STYLE = {
    "mean":   {"color": "#1f77b4", "lw": 2.0, "ls": "-"},
    "median": {"color": "#2ca02c", "lw": 2.0, "ls": "-"},
    "min":    {"color": "#d62728", "lw": 1.3, "ls": "--"},
    "max":    {"color": "#9467bd", "lw": 1.3, "ls": "--"},
}


def _per_day_values(ds, metabolite, field):
    """{day: np.array of ``field`` across organoids} for one metabolite."""
    out = {d: [] for d in DAY_ORDER}
    for _, info in ds.iter_organoids():
        for day, rec in info["records"].items():
            if day not in out:
                continue
            v = (rec.get("metabolite") or {}).get(metabolite, {}).get(field)
            if v is not None:
                out[day].append(v)
    return {d: np.array(vs, float) for d, vs in out.items()}


def main(cohort="full", field="concentration_uM"):
    suffix = "_win" if field.endswith("_win") else ""
    ds, counts = build_cohort(cohort, ALL_DATA_PATH)
    logger.info("Summary panel (%s) on cohort %s: %d organoids",
                field, cohort, len(ds.organoid_ids))

    mets = list(REQUIRED_METABOLITES)
    n = len(mets)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 3.6 * nrows), squeeze=False)
    x = np.arange(len(DAY_ORDER))

    for idx, met in enumerate(mets):
        ax = axes[idx // ncols][idx % ncols]
        per_day = _per_day_values(ds, met, field)
        series = {s: [] for s in STATS}
        for d in DAY_ORDER:
            vals = per_day[d]
            for s in STATS:
                series[s].append(_STAT_FN[s](vals) if vals.size else np.nan)
        for s in STATS:
            st = _STAT_STYLE[s]
            ax.plot(x, series[s], label=s, color=st["color"], lw=st["lw"], ls=st["ls"])
        ax.set_title(met)
        ax.set_xticks(x)
        ax.set_xticklabels(DAY_ORDER, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel(field)
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=8)

    # Hide any unused panels (6 mets fill a 2x3 grid exactly, but stay safe).
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    fig.suptitle(
        f"Metabolite {field} summary by day ({cohort}, n={len(ds.organoid_ids)})",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURE_DIR / f"metabolite_summary_{cohort}{suffix}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", out)


if __name__ == "__main__":
    # Both read values already stored in all_data.json (no recomputation):
    main(field="concentration_uM")       # -> metabolite_summary_<cohort>.png
    main(field="concentration_uM_win")   # -> metabolite_summary_<cohort>_win.png
