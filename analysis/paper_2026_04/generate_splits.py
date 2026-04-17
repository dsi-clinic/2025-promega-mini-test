#!/usr/bin/env python3
"""
One-time script to produce a splits CSV from all_data.json.

Applies the paper's default filters (BA1+BA2, complete metabolites, 4/5
consensus labels at Dy30), then performs stratified organoid-level splitting:

    72% train  /  8% val  /  20% test   (seed=42)

Output is a minimal CSV with just (organoid_id, split).  Everything else —
labels, features, filtering — is derived at runtime by data_loader.py.

Requires the project conda environment (see AGENTS.md):
    conda run --no-capture-output -n core_env \
        python -m analysis.generate_splits

Usage:
    python -m analysis.generate_splits                 # default
    python -m analysis.generate_splits --seed 42       # explicit seed
    python -m analysis.generate_splits --dry-run       # show counts, don't write
"""

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from sklearn.model_selection import train_test_split  # requires conda env

# ---------------------------------------------------------------------------
# Constants (must match data_loader.py)
# ---------------------------------------------------------------------------

REQUIRED_METABOLITES = ["GlucoseGlo", "GlutamateGlo", "LactateGlo", "PyruvateGlo"]
LABEL_DAY = "Dy30"
HIGH_QUALITY_BATCHES = ("BA1", "BA2")
MIN_VOTES = 4
TEST_SIZE = 0.20   # 20% test
VAL_SIZE = 0.10    # 10% of training → ~8% overall
DEFAULT_SEED = 42

# Paths (relative to repo root)
ALL_DATA_PATH = Path("data/all_data.json")
OUTPUT_PATH = Path("data/2026_winter_student_splits.csv")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_organoid_id(key: str) -> str:
    """'BA1 96_1 Dy30 A1' → 'BA1 96_1 A1'"""
    m = re.match(r"^(.*)\s+Dy\d+\s+(.*)$", key)
    return f"{m.group(1)} {m.group(2)}" if m else key


def get_batch(record: dict) -> Optional[str]:
    ba = record.get("BA", "")
    return ba.split()[0] if ba else None


def compute_majority_label(evaluations: list, min_votes: int = MIN_VOTES) -> Optional[str]:
    if not evaluations or len(evaluations) != 5:
        return None
    votes: Dict[str, int] = {}
    for e in evaluations:
        v = e.get("evaluation", "")
        if v:
            votes[v] = votes.get(v, 0) + 1
    for label in ("Acceptable", "Not Acceptable"):
        if votes.get(label, 0) >= min_votes:
            return label
    return None


def has_complete_metabolites(metabolites: Optional[dict]) -> bool:
    if not metabolites:
        return False
    for met in REQUIRED_METABOLITES:
        if met not in metabolites:
            return False
        if metabolites[met].get("concentration_uM") is None:
            return False
    return True


def has_valid_images(record: dict) -> bool:
    proc = record.get("processed")
    return bool(proc and "img_path" in proc and "mask_path" in proc)


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def collect_labeled_organoids(all_data: dict) -> Dict[str, str]:
    """Apply paper filters and return {organoid_id: label}.

    Filters:
      1. BA1 + BA2 only
      2. Dy30 survey with 4/5 consensus
      3. Complete required metabolites on every day
      4. Valid processed images on every day
    """
    # Pass 1: get Dy30 labels
    dy30_labels: Dict[str, str] = {}
    for key, rec in all_data.items():
        if rec.get("dayID") != LABEL_DAY:
            continue
        batch = get_batch(rec)
        if batch not in HIGH_QUALITY_BATCHES:
            continue
        survey = rec.get("survey")
        if not survey:
            continue
        label = compute_majority_label(survey.get("evaluations", []))
        if label is None:
            continue
        org_id = extract_organoid_id(key)
        dy30_labels[org_id] = label

    # Pass 2: group all records by organoid
    org_records: Dict[str, Dict[str, dict]] = {}
    for key, rec in all_data.items():
        org_id = extract_organoid_id(key)
        if org_id not in dy30_labels:
            continue
        batch = get_batch(rec)
        if batch not in HIGH_QUALITY_BATCHES:
            continue
        org_records.setdefault(org_id, {})[rec.get("dayID", "")] = rec

    # Pass 3: filter organoids
    # - Days with no metabolite data (e.g. Dy20) are skipped for metabolite check
    # - All days must have valid processed images
    result: Dict[str, str] = {}
    for org_id, records in org_records.items():
        all_ok = True
        has_any_met_day = False
        for day_id, rec in records.items():
            if not has_valid_images(rec):
                all_ok = False
                break
            mets = rec.get("metabolites")
            if mets:
                has_any_met_day = True
                if not has_complete_metabolites(mets):
                    all_ok = False
                    break
        if all_ok and has_any_met_day:
            result[org_id] = dy30_labels[org_id]

    return result


def stratified_split(
    org_labels: Dict[str, str],
    seed: int = DEFAULT_SEED,
    test_size: float = TEST_SIZE,
    val_size: float = VAL_SIZE,
) -> Dict[str, str]:
    """Stratified train/val/test split → {organoid_id: split_name}."""
    ids = sorted(org_labels.keys())  # sort for determinism
    labels = [org_labels[i] for i in ids]

    # 80/20 split
    train_val_ids, test_ids = train_test_split(
        ids, test_size=test_size, stratify=labels, random_state=seed
    )

    # Within 80%: 90/10 → overall 72/8
    tv_labels = [org_labels[i] for i in train_val_ids]
    train_ids, val_ids = train_test_split(
        train_val_ids, test_size=val_size, stratify=tv_labels, random_state=seed
    )

    splits = {}
    for i in train_ids:
        splits[i] = "train"
    for i in val_ids:
        splits[i] = "val"
    for i in test_ids:
        splits[i] = "test"
    return splits


def write_csv(splits: Dict[str, str], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Sort for deterministic output
    rows = sorted(splits.items())
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["organoid_id", "split"])
        for org_id, split in rows:
            writer.writerow([org_id, split])


def print_summary(org_labels: Dict[str, str], splits: Dict[str, str]):
    total = len(splits)
    label_counts = Counter(org_labels.values())
    print(f"Total organoids passing filters: {total}")
    print(f"  Acceptable:     {label_counts.get('Acceptable', 0)}")
    print(f"  Not Acceptable: {label_counts.get('Not Acceptable', 0)}")
    print()

    for split_name in ("train", "val", "test"):
        ids_in_split = [i for i, s in splits.items() if s == split_name]
        split_labels = Counter(org_labels[i] for i in ids_in_split)
        n = len(ids_in_split)
        pct = 100 * n / total if total else 0
        print(
            f"  {split_name:5s}: {n:3d} ({pct:5.1f}%)  "
            f"Acc={split_labels.get('Acceptable', 0)}  "
            f"NAcc={split_labels.get('Not Acceptable', 0)}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--all-data", type=Path, default=ALL_DATA_PATH,
                        help="Path to all_data.json")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH,
                        help="Output CSV path")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="Random seed for splitting")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print summary without writing CSV")
    args = parser.parse_args()

    print(f"Loading {args.all_data} ...")
    with open(args.all_data) as f:
        all_data = json.load(f)
    print(f"Loaded {len(all_data)} records")

    org_labels = collect_labeled_organoids(all_data)
    print(f"Organoids passing filters: {len(org_labels)}")

    splits = stratified_split(org_labels, seed=args.seed)
    print()
    print_summary(org_labels, splits)

    if args.dry_run:
        print("\n[dry-run] No file written.")
    else:
        write_csv(splits, args.output)
        print(f"\nWrote {args.output}  ({len(splits)} organoids)")


if __name__ == "__main__":
    main()
