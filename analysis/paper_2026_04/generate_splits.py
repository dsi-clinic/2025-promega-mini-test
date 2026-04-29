#!/usr/bin/env python3
"""
One-time script to produce a splits CSV from all_data.json.

All filtering and label derivation comes from ``pipeline.data_loader.OrganoidDataset``
with the paper-default filter preset (BA1+BA2, complete metabolites, valid images,
4/5 vote consensus at Dy30). This script just performs the stratified
organoid-level split and writes the CSV.

    72% train  /  8% val  /  20% test   (seed=42)

Output: ``data/2026_winter_student_splits.csv`` with columns ``organoid_id, split``.

Usage:
    make run ARGS="-m analysis.paper_2026_04.generate_splits"            # default
    make run ARGS="-m analysis.paper_2026_04.generate_splits --dry-run"  # preview
    make run ARGS="-m analysis.paper_2026_04.generate_splits --seed 7"   # custom seed
"""

import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import Dict

from sklearn.model_selection import train_test_split

from pipeline.data_loader import OrganoidDataset

ALL_DATA_PATH = Path("data/all_data.json")
OUTPUT_PATH = Path("data/2026_winter_student_splits.csv")
TEST_SIZE = 0.20   # 20% test
VAL_SIZE = 0.10    # 10% of train+val → ~8% overall
DEFAULT_SEED = 42


def collect_labeled_organoids(all_data_path: Path) -> Dict[str, str]:
    """Build {organoid_id: label} using the paper-default filters via OrganoidDataset.

    OrganoidDataset normally requires either a splits CSV or split ratios; we
    pass dummy ratios since we only care about the filtered + labeled set.
    """
    ds = OrganoidDataset(
        str(all_data_path),
        split_ratios={"_unused": 1.0},  # no real splits needed
        split_seed=0,
    )
    return {org_id: info["label"] for org_id, info in ds.iter_organoids()}


def stratified_split(
    org_labels: Dict[str, str],
    seed: int = DEFAULT_SEED,
    test_size: float = TEST_SIZE,
    val_size: float = VAL_SIZE,
) -> Dict[str, str]:
    """Stratified train/val/test split → {organoid_id: split_name}."""
    ids = sorted(org_labels.keys())
    labels = [org_labels[i] for i in ids]

    train_val_ids, test_ids = train_test_split(
        ids, test_size=test_size, stratify=labels, random_state=seed
    )
    tv_labels = [org_labels[i] for i in train_val_ids]
    train_ids, val_ids = train_test_split(
        train_val_ids, test_size=val_size, stratify=tv_labels, random_state=seed
    )

    return (
        {i: "train" for i in train_ids}
        | {i: "val" for i in val_ids}
        | {i: "test" for i in test_ids}
    )


def write_csv(splits: Dict[str, str], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["organoid_id", "split"])
        for org_id, split in sorted(splits.items()):
            writer.writerow([org_id, split])


def print_summary(org_labels: Dict[str, str], splits: Dict[str, str]) -> None:
    total = len(splits)
    label_counts = Counter(org_labels.values())
    print(f"Total organoids passing filters: {total}")
    for label, count in label_counts.items():
        print(f"  {label}: {count}")
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--all-data", type=Path, default=ALL_DATA_PATH,
                        help="Path to all_data.json")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH,
                        help="Output CSV path")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="Random seed for splitting")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print summary without writing CSV")
    args = parser.parse_args()

    print(f"Loading {args.all_data} via OrganoidDataset (paper-default filters) ...")
    org_labels = collect_labeled_organoids(args.all_data)
    print(f"Organoids passing filters: {len(org_labels)}\n")

    splits = stratified_split(org_labels, seed=args.seed)
    print_summary(org_labels, splits)

    if args.dry_run:
        print("\n[dry-run] No file written.")
    else:
        write_csv(splits, args.output)
        print(f"\nWrote {args.output}  ({len(splits)} organoids)")


if __name__ == "__main__":
    main()
