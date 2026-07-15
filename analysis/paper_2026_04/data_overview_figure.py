#!/usr/bin/env python3
"""Data overview figure: organoid images, metabolite profiles, morphology, survey distribution, and classification framework.

Panels:
  A  Image strips for one Acceptable and one Not Acceptable organoid across all 11 days.
  B  Six metabolite concentrations over time for the same two example organoids.
  C  Segmentation-derived organoid area over time (morphology thread).
  D  Survey vote-split distribution across all Dy30-classified organoids.
  E  Classification framework: three modalities (image, metabolite, morphology) → binary label.

Organoids shown can be overridden via --acc-id / --nacc-id; defaults are the first
Acceptable / Not Acceptable in the canonical split with complete 11-day series.

Outputs:
  $ANALYSIS_OUTPUT_DIR/figures/data_overview.png

Usage:
  make run ARGS="-m analysis.paper_2026_04.data_overview_figure"
  make run ARGS="-m analysis.paper_2026_04.data_overview_figure --acc-id 'BA1 96_1 A1'"
"""

import argparse
import warnings
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image

from pipeline.data_loader import (
    ANALYSIS_OUTPUT_DIR,
    DAY_ORDER,
    FIGURE_DIR,
    LABEL_TO_INT,
    OrganoidDataset,
    _load_idor_organoid_ids,
    get_edge_fraction,
    get_mask_area_um2,
    get_survey_vote_counts,
    idor_ba1_ba2_filters,
    iter_organoid_records,
    filters_for_mode,
    HIGH_QUALITY_BATCHES,
    MIN_VOTES,
    LABEL_DAY,
    REQUIRED_METABOLITES,
)
from pipeline.splits import Splits

warnings.filterwarnings("ignore")

ALL_DATA_PATH = "data/all_data.json"

METABOLITE_COLORS = {
    "GlucoseGlo":   "#1f77b4",
    "GlutamateGlo": "#ff7f0e",
    "LactateGlo":   "#2ca02c",
    "PyruvateGlo":  "#d62728",
    "BCAAGlo":      "#9467bd",
    "MalateGlo":    "#8c564b",
}
METABOLITE_SHORT = {
    "GlucoseGlo":   "Glucose",
    "GlutamateGlo": "Glutamate",
    "LactateGlo":   "Lactate",
    "PyruvateGlo":  "Pyruvate",
    "BCAAGlo":      "BCAA",
    "MalateGlo":    "Malate",
}

ACCEPTABLE_COLOR = "#2196F3"   # blue
NOT_ACCEPTABLE_COLOR = "#F44336"  # red

MORPH_CSV_PATH = "data/normalized/CONC_data_organoides_residualized_final.csv"
MORPH_SHAPE_COLS = ["Circ._win", "AR_win", "Solidity_win", "Complexity_win"]

MORPH_COLORS = {
    "mask_area_um2": "#2ca02c",
    "edge_fraction": "#17becf",
    "Circ._win":     "#9467bd",
    "AR_win":        "#e377c2",
    "Solidity_win":  "#bcbd22",
    "Complexity_win":"#7f7f7f",
}
MORPH_SHORT = {
    "mask_area_um2": "Area (mm²)",
    "edge_fraction": "Edge frac.",
    "Circ._win":     "Circularity",
    "AR_win":        "Aspect ratio",
    "Solidity_win":  "Solidity",
    "Complexity_win":"Complexity",
}

# Mapping from DAY_ORDER strings to integer day numbers used in the CSV
_DAY_STR_TO_INT = {
    "Dy03": 3, "Dy06": 6, "Dy08": 8, "Dy10": 10, "Dy13": 13,
    "Dy15": 15, "Dy17": 17, "Dy20_5": 21, "Dy24": 24, "Dy28": 28, "Dy30": 30,
}

_MORPH_CSV_CACHE = None


def _load_morph_csv():
    global _MORPH_CSV_CACHE
    if _MORPH_CSV_CACHE is None:
        import pandas as pd
        p = Path(MORPH_CSV_PATH)
        _MORPH_CSV_CACHE = pd.read_csv(p) if p.exists() else pd.DataFrame()
    return _MORPH_CSV_CACHE


def _get_morph_csv_features(org_id: str, cols: list) -> dict:
    """Return {col: [value_per_day_in_DAY_ORDER]} from the shape CSV; None for missing days."""
    df = _load_morph_csv()
    if df.empty:
        return {c: [None] * len(DAY_ORDER) for c in cols}
    key = org_id.replace(" ", "_")
    sub = df[df["Organoid"] == key].set_index("Day")
    result = {}
    for c in cols:
        vals = []
        for day_str in DAY_ORDER:
            day_int = _DAY_STR_TO_INT[day_str]
            if day_int in sub.index and c in sub.columns:
                v = sub.loc[day_int, c]
                vals.append(float(v) if v == v else None)  # NaN → None
            else:
                vals.append(None)
        result[c] = vals
    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_example_organoids(ds, acc_id=None, nacc_id=None):
    """Return (acceptable_info, not_acceptable_info) from the filtered dataset.

    Each info dict is {org_id, label, records}. Uses provided IDs if given;
    otherwise selects the first organoid of each class with a complete 11-day
    series and all raw images accessible on disk.
    """
    acc = nacc = None
    for org_id, info in ds.iter_organoids():
        if not all(d in info["records"] for d in DAY_ORDER):
            continue
        images_ok = all(
            _raw_image_path(info["records"][d]) is not None
            for d in DAY_ORDER
        )
        if not images_ok:
            continue
        if info["label"] == "Acceptable" and acc is None:
            if acc_id is None or org_id == acc_id:
                acc = (org_id, info)
        elif info["label"] == "Not Acceptable" and nacc is None:
            if nacc_id is None or org_id == nacc_id:
                nacc = (org_id, info)
        if acc and nacc:
            break
    if acc is None or nacc is None:
        raise RuntimeError("Could not find example organoids with complete 11-day raw image series.")
    return acc, nacc


def _raw_image_path(record: dict):
    """Return the absolute path to the best-z raw TIF, or None if unavailable.

    Uses ``images.aspect_ratio.ar_raw_tif`` which is the absolute path to the
    Z0 (best-z) original microscopy file written by the pipeline.
    Falls back to ``images.img_path`` (resized 512×384 PNG) if the TIF is absent.
    """
    ar = (record.get("images") or {}).get("aspect_ratio") or {}
    tif = ar.get("ar_raw_tif")
    if tif and Path(tif).exists():
        return Path(tif)
    img = (record.get("images") or {}).get("img_path")
    if img and Path(img).exists():
        return Path(img)
    return None


def _load_image(path: Path) -> np.ndarray:
    """Load a microscopy image for display.

    Raw TIF files are 16-bit grayscale; apply 1st/99th-percentile normalization
    and convert to uint8 for rendering. PNG files (img_path) are already RGB uint8.
    """
    img = Image.open(path)
    arr = np.array(img)

    if arr.dtype == np.uint16 or str(img.mode).startswith("I"):
        # 16-bit grayscale: percentile-stretch then convert to uint8 RGB
        arr = arr.astype(np.float32)
        lo, hi = np.percentile(arr, 1), np.percentile(arr, 99)
        if hi > lo:
            arr = np.clip((arr - lo) / (hi - lo), 0, 1)
        else:
            arr = np.zeros_like(arr)
        arr = (arr * 255).astype(np.uint8)
        arr = np.stack([arr, arr, arr], axis=-1)  # grayscale → RGB
    elif arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)

    return arr


def _um_per_px(record: dict):
    """Return µm/px for the raw TIF (ar_orig_um_per_px), or None if unavailable."""
    ar = (record.get("images") or {}).get("aspect_ratio") or {}
    return ar.get("ar_orig_um_per_px")


def _add_scale_bar(ax, img_h: int, img_w: int, um_per_px: float, scale_um: int = 500):
    """Draw a white scale bar in the bottom-right corner of an image axes (data coords)."""
    bar_px = scale_um / um_per_px
    margin_x = img_w * 0.04
    margin_y = img_h * 0.07
    x1 = img_w - margin_x - bar_px
    x2 = img_w - margin_x
    y_bar = img_h - margin_y
    ax.plot([x1, x2], [y_bar, y_bar], color="white", linewidth=2.0,
            solid_capstyle="butt", zorder=5)


def _get_metabolites(info):
    """Return {metabolite: [conc_uM per day]} for an organoid."""
    result = {m: [] for m in REQUIRED_METABOLITES}
    for day in DAY_ORDER:
        rec = info["records"].get(day, {})
        mets = rec.get("metabolite", {})
        for m in REQUIRED_METABOLITES:
            val = (mets.get(m) or {}).get("concentration_uM")
            result[m].append(val)
    return result


def _get_morphology(info):
    """Return (areas_um2, edge_fracs) lists across DAY_ORDER."""
    areas, edges = [], []
    for day in DAY_ORDER:
        rec = info["records"].get(day, {})
        areas.append(get_mask_area_um2(rec))
        edges.append(get_edge_fraction(rec))
    return areas, edges


def _vote_distribution(all_data_path):
    """Return Counter of (n_acceptable, n_not_acceptable) vote tuples across all Dy30-classified organoids."""
    _col1, col2_pairs = _load_idor_organoid_ids()
    col2 = {oid for oid, _ in col2_pairs}
    orgs = {
        oid: recs
        for oid, recs, _batch in iter_organoid_records(all_data_path, batches=HIGH_QUALITY_BATCHES)
    }
    counts = Counter()
    for oid in col2:
        rec = orgs.get(oid, {}).get(LABEL_DAY)
        if rec is None:
            continue
        n_acc, n_total = get_survey_vote_counts(rec)
        n_nacc = n_total - n_acc
        counts[(n_acc, n_nacc)] += 1
    return counts


# ---------------------------------------------------------------------------
# Plotting sections
# ---------------------------------------------------------------------------

def _draw_image_strips(fig, outer_gs, acc_info, nacc_info, acc_id, nacc_id):
    """Panel A: two rows of 11 day-images each."""
    inner = gridspec.GridSpecFromSubplotSpec(2, len(DAY_ORDER), subplot_spec=outer_gs,
                                             wspace=0.03, hspace=0.05)
    row_configs = [
        (0, acc_info, acc_id, ACCEPTABLE_COLOR, "Acceptable"),
        (1, nacc_info, nacc_id, NOT_ACCEPTABLE_COLOR, "Not Acceptable"),
    ]
    for row_i, info, org_id, color, label_str in row_configs:
        for col_i, day in enumerate(DAY_ORDER):
            ax = fig.add_subplot(inner[row_i, col_i])
            path = _raw_image_path(info["records"][day])
            img = _load_image(path)
            ax.imshow(img, cmap="gray" if img.ndim == 2 else None)
            img_h, img_w = img.shape[:2]
            um_px = _um_per_px(info["records"][day])
            if um_px:
                _add_scale_bar(ax, img_h, img_w, um_px, scale_um=500)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(2.5)
            if col_i == 0:
                ax.set_ylabel(label_str, color=color, fontsize=8, fontweight="bold",
                              rotation=90, labelpad=4)
            if row_i == 0:
                day_label = day.replace("Dy", "Day ").replace("_5", ".5")
                ax.set_title(day_label, fontsize=6.5, pad=2)


def _draw_metabolites(ax, acc_info, nacc_info):
    """Panel B: metabolite concentrations over time."""
    x = range(len(DAY_ORDER))
    day_labels = [d.replace("Dy", "D").replace("_5", ".5") for d in DAY_ORDER]

    for met in REQUIRED_METABOLITES:
        color = METABOLITE_COLORS[met]
        short = METABOLITE_SHORT[met]
        acc_vals = _get_metabolites(acc_info)[met]
        nacc_vals = _get_metabolites(nacc_info)[met]
        # solid = acceptable, dashed = not acceptable
        ax.plot(x, acc_vals, color=color, linewidth=1.5, label=short, marker="o", markersize=3)
        ax.plot(x, nacc_vals, color=color, linewidth=1.5, linestyle="--", marker="s", markersize=3, alpha=0.7)

    ax.set_xticks(list(x))
    ax.set_xticklabels(day_labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Concentration (µM)", fontsize=8)
    ax.set_title("B  Metabolite Profiles", fontsize=9, fontweight="bold", loc="left")
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=6, ncol=2, loc="upper left", framealpha=0.7)

    # legend for line styles
    from matplotlib.lines import Line2D
    style_legend = [
        Line2D([0], [0], color="gray", linewidth=1.5, label="Acceptable"),
        Line2D([0], [0], color="gray", linewidth=1.5, linestyle="--", label="Not Acceptable"),
    ]
    ax.legend(handles=ax.get_legend_handles_labels()[0] + style_legend,
              labels=ax.get_legend_handles_labels()[1] + ["Acceptable", "Not Acceptable"],
              fontsize=5.5, ncol=2, loc="upper left", framealpha=0.8)


def _draw_morphology(ax, acc_info, nacc_info, acc_id, nacc_id):
    """Panel C: all morphology features over time (dual y-axis)."""
    x = list(range(len(DAY_ORDER)))
    day_labels = [d.replace("Dy", "D").replace("_5", ".5") for d in DAY_ORDER]

    acc_areas, acc_edges = _get_morphology(acc_info)
    nacc_areas, nacc_edges = _get_morphology(nacc_info)
    acc_shape = _get_morph_csv_features(acc_id, MORPH_SHAPE_COLS)
    nacc_shape = _get_morph_csv_features(nacc_id, MORPH_SHAPE_COLS)

    ax2 = ax.twinx()

    # Left axis: mask area in mm²
    c_area = MORPH_COLORS["mask_area_um2"]
    ax.plot(x, [a / 1e6 if a is not None else None for a in acc_areas],
            color=c_area, linewidth=2, marker="o", markersize=4,
            label=MORPH_SHORT["mask_area_um2"])
    ax.plot(x, [a / 1e6 if a is not None else None for a in nacc_areas],
            color=c_area, linewidth=2, marker="s", markersize=4, linestyle="--", alpha=0.7)
    ax.set_ylabel("Area (mm²)", fontsize=8, color=c_area)
    ax.tick_params(axis="y", labelcolor=c_area, labelsize=7)

    # Right axis: edge fraction + shape descriptors (0–1.5 range)
    right_features = [
        ("edge_fraction", acc_edges, nacc_edges),
    ] + [(col, acc_shape[col], nacc_shape[col]) for col in MORPH_SHAPE_COLS]

    for feat, acc_vals, nacc_vals in right_features:
        c = MORPH_COLORS[feat]
        ax2.plot(x, acc_vals, color=c, linewidth=1.5, marker="o", markersize=3,
                 label=MORPH_SHORT[feat])
        ax2.plot(x, nacc_vals, color=c, linewidth=1.5, marker="s", markersize=3,
                 linestyle="--", alpha=0.7)

    ax2.set_ylabel("Shape descriptors", fontsize=8)
    ax2.tick_params(axis="y", labelsize=7)
    ax2.set_ylim(-0.05, 1.6)

    ax.set_xticks(x)
    ax.set_xticklabels(day_labels, rotation=45, ha="right", fontsize=7)
    ax.set_title("C  Morphology Features", fontsize=9, fontweight="bold", loc="left")
    ax.grid(True, alpha=0.2)

    # Combined legend: left-axis feature + right-axis features, then line styles
    from matplotlib.lines import Line2D
    feat_handles = [
        Line2D([0], [0], color=MORPH_COLORS["mask_area_um2"], lw=2, label=MORPH_SHORT["mask_area_um2"]),
    ] + [
        Line2D([0], [0], color=MORPH_COLORS[f], lw=1.5, label=MORPH_SHORT[f])
        for f in ["edge_fraction"] + MORPH_SHAPE_COLS
    ]
    style_handles = [
        Line2D([0], [0], color="gray", lw=1.5, label="Acceptable"),
        Line2D([0], [0], color="gray", lw=1.5, linestyle="--", label="Not Acceptable"),
    ]
    ax.legend(handles=feat_handles + style_handles, fontsize=5.5, ncol=2,
              loc="upper left", framealpha=0.8)


def _draw_survey_distribution(ax, all_data_path):
    """Panel D: survey vote-split distribution across all Dy30-classified organoids."""
    counts = _vote_distribution(all_data_path)

    # Order vote splits from most acceptable to least
    vote_splits = sorted(counts.keys(), key=lambda k: (-k[0], k[1]))
    labels = [f"{a}-{n}" for a, n in vote_splits]
    values = [counts[k] for k in vote_splits]
    colors = [ACCEPTABLE_COLOR if a >= MIN_VOTES else
              (NOT_ACCEPTABLE_COLOR if n >= MIN_VOTES else "#9E9E9E")
              for a, n in vote_splits]

    ax.bar(range(len(labels)), values, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Organoid count", fontsize=8)
    ax.set_xlabel("Acc : NotAcc votes", fontsize=7)
    ax.set_title("D  Survey Vote Distribution (Dy30)", fontsize=9, fontweight="bold", loc="left")
    ax.grid(axis="y", alpha=0.2)

    legend_handles = [
        mpatches.Patch(color=ACCEPTABLE_COLOR, label="Consensus: Acceptable"),
        mpatches.Patch(color=NOT_ACCEPTABLE_COLOR, label="Consensus: Not Acceptable"),
        mpatches.Patch(color="#9E9E9E", label="No consensus (3-2)"),
    ]
    ax.legend(handles=legend_handles, fontsize=6.5, loc="upper right", framealpha=0.8)


def _stick_figure(ax, cx, cy, scale=0.30, vote_color=None):
    """Stick figure at (cx, cy) with left-facing arms and a colored vote dot above head."""
    c = "#444"
    r = scale * 0.22
    # Head
    ax.add_patch(mpatches.Circle((cx, cy + scale * 0.75), r, color=c, zorder=4))
    # Body
    ax.plot([cx, cx], [cy + scale * 0.53, cy - scale * 0.10],
            color=c, lw=1.5, zorder=4, solid_capstyle="round")
    # Arms: left arm extends toward screen, right arm relaxed
    ax.plot([cx - scale * 0.45, cx, cx + scale * 0.30],
            [cy + scale * 0.14, cy + scale * 0.28, cy + scale * 0.04],
            color=c, lw=1.5, zorder=4, solid_capstyle="round")
    # Legs
    ax.plot([cx, cx - scale * 0.28], [cy - scale * 0.10, cy - scale * 0.60],
            color=c, lw=1.5, zorder=4, solid_capstyle="round")
    ax.plot([cx, cx + scale * 0.28], [cy - scale * 0.10, cy - scale * 0.60],
            color=c, lw=1.5, zorder=4, solid_capstyle="round")
    # Vote dot above head
    if vote_color is not None:
        ax.add_patch(mpatches.Circle((cx, cy + scale * 1.22), r * 0.80,
                                     color=vote_color, zorder=5))


def _draw_survey_labeling(ax, acc_dy30_rec, nacc_dy30_rec):
    """Panel F: experts view Day 30 images and vote — schematic with stick figures."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis("off")
    ax.set_title("F  Survey Labeling Protocol", fontsize=9, fontweight="bold", loc="left")

    mw, mh = 1.20, 0.95
    mcx = 0.80
    fig_xs = [2.20 + i * 0.85 for i in range(5)]
    eye_dy = 0.30 * 0.75  # figure eye height above cy

    rows = [
        (acc_dy30_rec,  ACCEPTABLE_COLOR,    "Acceptable",    3.0),
        (nacc_dy30_rec, NOT_ACCEPTABLE_COLOR, "Not Acceptable", 1.4),
    ]

    for rec, color, label_str, yc in rows:
        # --- Monitor ---
        ax.add_patch(FancyBboxPatch(
            (mcx - mw / 2, yc - mh / 2), mw, mh,
            boxstyle="round,pad=0.04", lw=2.0,
            edgecolor="#444", facecolor="#111", zorder=2,
        ))
        path = _raw_image_path(rec)
        if path:
            mg = 0.05
            ax.imshow(_load_image(path),
                      extent=[mcx - mw/2 + mg, mcx + mw/2 - mg,
                               yc - mh/2 + mg, yc + mh/2 - mg],
                      aspect="auto", zorder=3)
        # Monitor stand
        ax.plot([mcx, mcx], [yc - mh/2, yc - mh/2 - 0.14],
                color="#555", lw=2.5, zorder=2)
        ax.plot([mcx - 0.18, mcx + 0.18], [yc - mh/2 - 0.14] * 2,
                color="#555", lw=2.5, zorder=2)
        ax.text(mcx, yc - mh/2 - 0.28, "Day 30",
                ha="center", va="top", fontsize=6, color="#555")

        # --- Gaze lines: monitor screen → each figure's eye level ---
        for fx in fig_xs:
            ax.plot([mcx + mw/2 + 0.03, fx - 0.06], [yc + eye_dy] * 2,
                    color="#ccc", lw=0.7, linestyle="--", zorder=1, alpha=0.8)

        # --- 5 stick figures with vote dots ---
        votes = (rec.get("label") or {}).get("regular_votes", {})
        n_acc_v  = votes.get("Acceptable", 0)
        n_nacc_v = votes.get("Not Acceptable", 0)
        vote_colors = [ACCEPTABLE_COLOR] * n_acc_v + [NOT_ACCEPTABLE_COLOR] * n_nacc_v

        for fx, vc in zip(fig_xs, vote_colors):
            _stick_figure(ax, fx, yc, scale=0.30, vote_color=vc)

        # --- Arrow: figures → consensus ---
        ax.annotate("", xy=(6.8, yc), xytext=(6.22, yc),
                    arrowprops=dict(arrowstyle="-|>", lw=1.3, color="#777"))

        # --- Consensus box ---
        ax.add_patch(FancyBboxPatch(
            (6.85, yc - 0.38), 2.90, 0.72,
            boxstyle="round,pad=0.08", lw=1.5,
            edgecolor=color, facecolor=color + "22", zorder=2,
        ))
        ax.text(8.30, yc, label_str, ha="center", va="center",
                fontsize=8, fontweight="bold", color=color)

    # --- Legend ---
    ax.add_patch(mpatches.Circle((0.22, 0.52), 0.09, color=ACCEPTABLE_COLOR, zorder=4))
    ax.text(0.38, 0.52, "Acceptable vote", va="center", fontsize=6.5,
            color=ACCEPTABLE_COLOR)
    ax.add_patch(mpatches.Circle((2.30, 0.52), 0.09, color=NOT_ACCEPTABLE_COLOR, zorder=4))
    ax.text(2.46, 0.52, "Not Acceptable vote", va="center", fontsize=6.5,
            color=NOT_ACCEPTABLE_COLOR)
    ax.text(7.80, 0.52, "≥ 4 / 5 votes → consensus",
            ha="center", va="center", fontsize=6.5, color="#555", style="italic")

    # --- Header ---
    ax.text(5.0, 3.90,
            "5 independent expert evaluators — Day 30 image only (no metabolite data)",
            ha="center", va="top", fontsize=7.5, color="#333", fontweight="bold")

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)


def _draw_framework(ax, combined=False):
    """Panel E: classification framework — independent (default) or combined-model variant."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis("off")
    title = "E  Combined Classification Framework" if combined else "E  Classification Framework"
    ax.set_title(title, fontsize=9, fontweight="bold", loc="left")

    def _box(x, y, w, h, label, sublabel, color, fontsize=8):
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                   boxstyle="round,pad=0.08", linewidth=1.5,
                                   edgecolor=color, facecolor=color + "22"))
        ax.text(x, y + 0.08, label, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color=color)
        ax.text(x, y - 0.22, sublabel, ha="center", va="center",
                fontsize=6, color="gray")

    def _arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", lw=1.5, color="#555555"))

    xs = [1.5, 5.0, 8.5]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    # Input boxes (shared by both variants)
    _box(xs[0], 2.55, 2.0, 0.75, "Images", "microscopy\nEfficientNet-B0", colors[0])
    _box(xs[1], 2.55, 2.2, 0.75, "Metabolites", "6 assays × 11 days\nLightGBM", colors[1])
    _box(xs[2], 2.55, 2.0, 0.75, "Morphology", "area, edge fraction\nLightGBM", colors[2])

    if not combined:
        # Three independent pipelines: input → output label each
        out_label = "Acceptable /\nNot Acceptable"
        for x, c in zip(xs, colors):
            _arrow(x, 2.17, x, 1.42)
            ax.add_patch(FancyBboxPatch((x - 1.0, 0.92), 2.0, 0.45,
                                       boxstyle="round,pad=0.08", linewidth=1.2,
                                       edgecolor="#444444", facecolor="#EEEEEE"))
            ax.text(x, 1.14, out_label, ha="center", va="center",
                    fontsize=6.5, fontweight="bold", color="#333333")
    else:
        # --- Section labels ---
        ax.text(0.18, 2.55, "Indep.\npipelines", ha="center", va="center",
                fontsize=6, color="#999", style="italic")
        ax.text(0.18, 0.82, "Combined\nmodel", ha="center", va="center",
                fontsize=6, color="#999", style="italic")

        # --- Independent outputs: arrow + compact label box for each modality ---
        for x, c in zip(xs, colors):
            _arrow(x, 2.175, x, 1.90)
            ax.add_patch(FancyBboxPatch((x - 0.92, 1.57), 1.84, 0.30,
                                       boxstyle="round,pad=0.05", linewidth=1.2,
                                       edgecolor="#555555", facecolor="#EEEEEE"))
            ax.text(x, 1.72, "Acc / Not Acc", ha="center", va="center",
                    fontsize=6, fontweight="bold", color="#333333")

        # --- Dashed divider between the two sections ---
        ax.plot([0.45, 9.55], [1.44, 1.44], color="#cccccc", lw=1.0, linestyle="--")

        # --- Curved arrows from each INPUT box down to the combined model box,
        #     curving around the independent output boxes ---
        ax.annotate("", xy=(4.45, 1.18), xytext=(xs[0], 2.175),
                    arrowprops=dict(arrowstyle="-|>", lw=1.2, color="#888888",
                                   connectionstyle="arc3,rad=0.20"))
        ax.annotate("", xy=(5.00, 1.18), xytext=(xs[1], 2.175),
                    arrowprops=dict(arrowstyle="-|>", lw=1.2, color="#888888"))
        ax.annotate("", xy=(5.55, 1.18), xytext=(xs[2], 2.175),
                    arrowprops=dict(arrowstyle="-|>", lw=1.2, color="#888888",
                                   connectionstyle="arc3,rad=-0.20"))

        # --- Combined classifier box ---
        _box(5.0, 0.96, 3.6, 0.42, "Combined Classifier",
             "concat features → LightGBM", "#555555", fontsize=7)

        # --- Arrow to final output ---
        _arrow(5.0, 0.75, 5.0, 0.48)

        # --- Combined output label ---
        ax.text(5.0, 0.24, "Acceptable  /  Not Acceptable",
                ha="center", va="center", fontsize=7.5, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.22", facecolor="#EEEEEE",
                          edgecolor="#444444", lw=1.5))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--acc-id", default=None,
                        help="Organoid ID for the Acceptable example (default: auto)")
    parser.add_argument("--nacc-id", default=None,
                        help="Organoid ID for the Not Acceptable example (default: auto)")
    parser.add_argument("--combined", action="store_true",
                        help="Show combined-model framework in Panel E instead of independent classifiers")
    args = parser.parse_args()

    ds = OrganoidDataset(
        ALL_DATA_PATH,
        splits=Splits.canonical(),
        filters=filters_for_mode("base"),
    )

    (acc_id, acc_info), (nacc_id, nacc_info) = _find_example_organoids(
        ds, acc_id=args.acc_id, nacc_id=args.nacc_id
    )
    print(f"Acceptable example:     {acc_id}")
    print(f"Not Acceptable example: {nacc_id}")

    # -----------------------------------------------------------------------
    # Figure layout
    # -----------------------------------------------------------------------
    fig = plt.figure(figsize=(22, 14))
    fig.suptitle("Organoid Quality Classification: Data Overview", fontsize=13,
                 fontweight="bold", y=0.99)

    # Three major rows
    outer = gridspec.GridSpec(3, 1, figure=fig, hspace=0.40,
                              height_ratios=[2.4, 2.2, 2.0])

    # Row 0: Panel A — image strips
    ax_img_label = fig.add_subplot(outer[0])
    ax_img_label.axis("off")
    ax_img_label.text(0.01, 0.98, "A  Organoid Images Across Development (11 timepoints)",
                      transform=ax_img_label.transAxes,
                      fontsize=9, fontweight="bold", va="top")
    ax_img_label.text(0.99, 0.01, "Scale bar: 500 µm",
                      transform=ax_img_label.transAxes,
                      fontsize=7, va="bottom", ha="right", color="#555555")
    _draw_image_strips(fig, outer[0], acc_info, nacc_info, acc_id, nacc_id)

    # Row 1: Panels B, C, D
    row1 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[1],
                                            wspace=0.38, width_ratios=[2, 1.2, 1.2])
    ax_met = fig.add_subplot(row1[0])
    ax_morph = fig.add_subplot(row1[1])
    ax_survey = fig.add_subplot(row1[2])

    _draw_metabolites(ax_met, acc_info, nacc_info)
    _draw_morphology(ax_morph, acc_info, nacc_info, acc_id, nacc_id)
    _draw_survey_distribution(ax_survey, ALL_DATA_PATH)

    # Row 2: Panel F (survey labeling) + Panel E (framework)
    row2 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[2],
                                            wspace=0.28, width_ratios=[1.2, 1])
    ax_survey_lbl = fig.add_subplot(row2[0])
    ax_fw = fig.add_subplot(row2[1])

    acc_dy30 = acc_info["records"].get("Dy30", {})
    nacc_dy30 = nacc_info["records"].get("Dy30", {})
    _draw_survey_labeling(ax_survey_lbl, acc_dy30, nacc_dy30)
    _draw_framework(ax_fw, combined=args.combined)

    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fname = "data_overview_combined.png" if args.combined else "data_overview.png"
    out_path = FIGURE_DIR / fname
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
