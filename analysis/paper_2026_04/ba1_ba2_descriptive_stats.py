#!/usr/bin/env python3
"""BA1 + BA2 dataset summary table for the paper methods section.

Stratifies counts, label distribution, and modality coverage across BA1, BA2,
and the combined pool. BA3 and BA4 are excluded (lower-quality batches per
IDOR/Promega assessment; BA3 has no metabolite/survey data, BA4 has only
sparse manual masks).

Iterates ``data/all_data.json`` directly rather than via ``OrganoidDataset``
because the loader filters out organoids without a Dy30 label, and we need
the unfiltered pool for unlabeled-organoid and modality-coverage counts.

Outputs:
  - Console: the same table that gets written to disk
  - $ANALYSIS_OUTPUT_DIR/figures/ba1_ba2_descriptive_stats.csv

Usage:
    make analysis-batch-summary
"""

import csv
from pathlib import Path

import pandas as pd

from pipeline.data_loader import (
    DAY_ORDER,
    FIGURE_DIR,
    HIGH_QUALITY_BATCHES,
    iter_organoid_records,
)

ALL_DATA_PATH = Path("data/all_data.json")
SPLITS_CSV = Path("data/2026_winter_student_splits.csv")
OUTPUT_PATH = FIGURE_DIR / "ba1_ba2_descriptive_stats.csv"

LABEL_DAY_CANONICAL = "Dy30"


def _load_splits(path: Path) -> dict:
    with open(path, newline="") as f:
        return {row["organoid_id"]: row["split"] for row in csv.DictReader(f)}


def _has_modality(records_by_day: dict, modality_check) -> bool:
    return any(modality_check(rec) for rec in records_by_day.values())


def _record_count(records_by_day: dict) -> int:
    return len(records_by_day)


def _has_metabolite(rec: dict) -> bool:
    return bool(rec.get("metabolite"))


def _has_survey(rec: dict) -> bool:
    return bool((rec.get("survey") or {}).get("evaluations"))


def _has_manual_mask(rec: dict) -> bool:
    return bool((rec.get("images") or {}).get("manual_mask_path"))


def _has_predicted_mask(rec: dict) -> bool:
    return bool((rec.get("images") or {}).get("mask_path"))


def _label_at_dy30(records_by_day: dict):
    rec = records_by_day.get(LABEL_DAY_CANONICAL)
    if rec is None:
        return None
    return (rec.get("label") or {}).get("value")


def _survey_votes_at_dy30(records_by_day: dict) -> int:
    rec = records_by_day.get(LABEL_DAY_CANONICAL)
    if rec is None:
        return 0
    return len((rec.get("survey") or {}).get("evaluations") or [])


def _plate_id(rec: dict) -> str:
    return (rec.get("plate") or {}).get("batch", "")


def _fmt_count_pct(count: int, total: int) -> str:
    if total == 0:
        return f"{count} (—)"
    return f"{count} ({100.0 * count / total:.1f}%)"


def compute_batch_stats(organoids: dict, splits: dict) -> dict:
    """Return a dict of statistics for one batch's organoid pool."""
    n_orgs = len(organoids)
    image_records = sum(_record_count(o["records_by_day"]) for o in organoids.values())

    plates = {_plate_id(r) for o in organoids.values() for r in o["records_by_day"].values()}
    plates.discard("")

    days = {d for o in organoids.values() for d in o["records_by_day"]}

    expected_days = set(DAY_ORDER)
    complete_series = sum(
        1 for o in organoids.values() if expected_days.issubset(o["records_by_day"].keys())
    )

    train = val = test = 0
    for org_id in organoids:
        s = splits.get(org_id)
        if s == "train":
            train += 1
        elif s == "val":
            val += 1
        elif s == "test":
            test += 1

    labels = [_label_at_dy30(o["records_by_day"]) for o in organoids.values()]
    n_acc = sum(1 for v in labels if v == "Acceptable")
    n_nacc = sum(1 for v in labels if v == "Not Acceptable")
    n_unlabeled = n_orgs - n_acc - n_nacc

    voted_orgs = [v for v in (_survey_votes_at_dy30(o["records_by_day"]) for o in organoids.values()) if v > 0]
    mean_votes = sum(voted_orgs) / len(voted_orgs) if voted_orgs else 0.0

    has_metabolite = sum(1 for o in organoids.values() if _has_modality(o["records_by_day"], _has_metabolite))
    has_survey = sum(1 for o in organoids.values() if _has_modality(o["records_by_day"], _has_survey))
    has_manual = sum(1 for o in organoids.values() if _has_modality(o["records_by_day"], _has_manual_mask))
    has_predicted = sum(1 for o in organoids.values() if _has_modality(o["records_by_day"], _has_predicted_mask))

    return {
        "Organoids (unique)": str(n_orgs),
        "Image records": str(image_records),
        "Plates": str(len(plates)),
        "Day timepoints": str(len(days)),
        "Train (organoids)": str(train),
        "Val (organoids)": str(val),
        "Test (organoids)": str(test),
        "Complete Dy03–Dy30 series": _fmt_count_pct(complete_series, n_orgs),
        "Acceptable (Dy30)": _fmt_count_pct(n_acc, n_orgs),
        "Not Acceptable (Dy30)": _fmt_count_pct(n_nacc, n_orgs),
        "No consensus / unlabeled": _fmt_count_pct(n_unlabeled, n_orgs),
        "Mean survey votes per organoid": f"{mean_votes:.2f}",
        "Has metabolite data": _fmt_count_pct(has_metabolite, n_orgs),
        "Has survey votes": _fmt_count_pct(has_survey, n_orgs),
        "Has manual mask": _fmt_count_pct(has_manual, n_orgs),
        "Has predicted mask": _fmt_count_pct(has_predicted, n_orgs),
    }


ROW_ORDER = [
    ("Counts & coverage", "Organoids (unique)"),
    ("Counts & coverage", "Image records"),
    ("Counts & coverage", "Plates"),
    ("Counts & coverage", "Day timepoints"),
    ("Counts & coverage", "Train (organoids)"),
    ("Counts & coverage", "Val (organoids)"),
    ("Counts & coverage", "Test (organoids)"),
    ("Counts & coverage", "Complete Dy03–Dy30 series"),
    ("Label distribution", "Acceptable (Dy30)"),
    ("Label distribution", "Not Acceptable (Dy30)"),
    ("Label distribution", "No consensus / unlabeled"),
    ("Label distribution", "Mean survey votes per organoid"),
    ("Modality coverage", "Has metabolite data"),
    ("Modality coverage", "Has survey votes"),
    ("Modality coverage", "Has manual mask"),
    ("Modality coverage", "Has predicted mask"),
]


def build_table(ba1_stats: dict, ba2_stats: dict, combined_stats: dict) -> pd.DataFrame:
    rows = []
    for category, stat in ROW_ORDER:
        rows.append({
            "Category": category,
            "Statistic": stat,
            "BA1": ba1_stats[stat],
            "BA2": ba2_stats[stat],
            "Combined": combined_stats[stat],
        })
    return pd.DataFrame(rows)


def main():
    splits = _load_splits(SPLITS_CSV)

    by_organoid = {
        org_id: {"batch": batch, "records_by_day": records}
        for org_id, records, batch in iter_organoid_records(
            ALL_DATA_PATH, batches=HIGH_QUALITY_BATCHES
        )
    }
    ba1 = {oid: o for oid, o in by_organoid.items() if o["batch"] == "BA1"}
    ba2 = {oid: o for oid, o in by_organoid.items() if o["batch"] == "BA2"}

    ba1_stats = compute_batch_stats(ba1, splits)
    ba2_stats = compute_batch_stats(ba2, splits)
    combined_stats = compute_batch_stats(by_organoid, splits)

    df = build_table(ba1_stats, ba2_stats, combined_stats)

    print("BA1 + BA2 Descriptive Statistics")
    print("=" * 80)
    print(df.to_string(index=False))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
