#!/usr/bin/env python3
"""Per-metabolite through-time summary, split good vs bad (Figure 2 variant).

The paper's Figure 2 shows each metabolite's concentration through time over the
whole sample. The plan asks to show the two quality groups **side by side**, so
this script splits the ``full`` cohort by its Dy30 label — Acceptable ("good")
vs Not Acceptable ("bad") — and, per metabolite per day, plots each group's mean
with a +/-1 SEM band, the two series distinguished by a colourblind-safe hue pair
(blue = Acceptable, orange = Not Acceptable) plus a legend.

Two figures are produced, both reading values already stored in all_data.json (no
recomputation; AGENTS.md rules 3 & 16): the raw ``concentration_uM``
(``metabolite_summary_goodbad_<cohort>.png``) and the persisted per-day-winsorized
``concentration_uM_win`` (``metabolite_summary_goodbad_<cohort>_win.png``).

Run by path (package name starts with a digit):
    make run ARGS="analysis/2026_06_metabolite_pred/metabolite_summary_goodbad.py"
    PYTHONPATH=. python analysis/2026_06_metabolite_pred/metabolite_summary_goodbad.py
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
    OrganoidDataset,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cohorts import ALL_DATA_PATH, build_cohort

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# The two quality groups. Colourblind-safe blue/orange pair (dataviz skill
# categorical slots 1 & 8; validated CVD-safe as a set). Acceptable = "good".
GROUPS = ("Acceptable", "Not Acceptable")
_GROUP_STYLE = {
    "Acceptable":     {"color": "#2a78d6", "label": "Acceptable (good)"},
    "Not Acceptable": {"color": "#eb6834", "label": "Not Acceptable (bad)"},
}


def split_ids_by_label(ds: OrganoidDataset) -> dict[str, list[str]]:
    """Partition the dataset's organoids into {label: sorted ids}.

    Asserts the split conserves the organoid count (AGENTS.md rule 11): the two
    groups are disjoint and together cover every organoid, with no unexpected
    label slipping in. Ids are sorted for deterministic iteration (rule 17).
    """
    labels = ds.organoid_labels()
    groups: dict[str, list[str]] = {g: [] for g in GROUPS}
    for oid in sorted(labels):
        label = labels[oid]
        if label not in groups:
            raise ValueError(f"unexpected label {label!r} for organoid {oid!r}")
        groups[label].append(oid)

    n_total = len(ds.organoid_ids)
    n_split = sum(len(v) for v in groups.values())
    assert n_split == n_total, f"split changed organoid count: {n_total} -> {n_split}"
    covered = set().union(*(set(v) for v in groups.values()))
    assert covered == set(ds.organoid_ids), "split does not cover the cohort exactly"
    for g in GROUPS:
        assert groups[g], f"group {g!r} is empty"
    return groups


def _per_day_values(
    ds: OrganoidDataset, ids: list[str], metabolite: str, field: str
) -> tuple[dict[str, np.ndarray], int]:
    """{day: values across ``ids``} for one metabolite, plus a missing-count.

    Reads ``rec['metabolite'][met][field]`` straight from all_data.json (rule 16).
    An organoid that lacks the metabolite/field on a given day is skipped and
    counted, so the drop is logged rather than swallowed (rule 15).
    """
    id_set = set(ids)
    out: dict[str, list[float]] = {d: [] for d in DAY_ORDER}
    missing = 0
    for oid, info in ds.iter_organoids():
        if oid not in id_set:
            continue
        for day, rec in info["records"].items():
            if day not in out:
                continue
            v = (rec.get("metabolite") or {}).get(metabolite, {}).get(field)
            if v is None:
                missing += 1
                continue
            out[day].append(float(v))
    return {d: np.array(vs, float) for d, vs in out.items()}, missing


def _group_series(
    per_day: dict[str, np.ndarray],
) -> tuple[list[float], list[float]]:
    """(mean, sem) per day over ``DAY_ORDER`` from per-day value arrays.

    NaN where a day has no observations. SEM = std / sqrt(n) (sample std, ddof=1
    when n>1), used as the +/-1 spread band.
    """
    means: list[float] = []
    sems: list[float] = []
    for d in DAY_ORDER:
        vals = per_day[d]
        if vals.size == 0:
            means.append(np.nan)
            sems.append(np.nan)
        else:
            means.append(float(np.mean(vals)))
            sem = (
                float(np.std(vals, ddof=1) / np.sqrt(vals.size))
                if vals.size > 1
                else 0.0
            )
            sems.append(sem)
    return means, sems


def main(cohort: str = "full", field: str = "concentration_uM") -> None:
    suffix = "_win" if field.endswith("_win") else ""
    ds, counts = build_cohort(cohort, ALL_DATA_PATH)
    groups = split_ids_by_label(ds)
    logger.info(
        "Good/bad summary (%s) on cohort %s: %d organoids  (%s)",
        field, cohort, len(ds.organoid_ids),
        ", ".join(f"{g}={len(groups[g])}" for g in GROUPS),
    )

    mets = list(REQUIRED_METABOLITES)
    n = len(mets)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5.0 * ncols, 3.6 * nrows), squeeze=False
    )
    x = np.arange(len(DAY_ORDER))

    total_missing = 0
    for idx, met in enumerate(mets):
        ax = axes[idx // ncols][idx % ncols]
        for g in GROUPS:
            per_day, missing = _per_day_values(ds, groups[g], met, field)
            total_missing += missing
            means, sems = _group_series(per_day)
            means_a = np.array(means, float)
            sems_a = np.array(sems, float)
            st = _GROUP_STYLE[g]
            ax.plot(
                x, means_a, color=st["color"], lw=2.0,
                marker="o", markersize=4,
                label=f"{st['label']} (n={len(groups[g])})",
            )
            ax.fill_between(
                x, means_a - sems_a, means_a + sems_a,
                color=st["color"], alpha=0.18, linewidth=0,
            )
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

    if total_missing:
        logger.warning(
            "Skipped %d missing (organoid, day) %s reads across all metabolites/groups",
            total_missing, field,
        )

    fig.suptitle(
        f"Metabolite {field} by day, good vs bad "
        f"({cohort}, n={len(ds.organoid_ids)}; mean +/-1 SEM)",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURE_DIR / f"metabolite_summary_goodbad_{cohort}{suffix}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", out)


if __name__ == "__main__":
    # Both read values already stored in all_data.json (no recomputation):
    main(field="concentration_uM")       # -> metabolite_summary_goodbad_<cohort>.png
    main(field="concentration_uM_win")   # -> metabolite_summary_goodbad_<cohort>_win.png
