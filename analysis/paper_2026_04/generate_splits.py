#!/usr/bin/env python3
"""
One-time script to produce a stratified train/val/test splits CSV from all_data.json.

All filtering and label derivation comes from ``pipeline.data_loader.OrganoidDataset``
with the paper-default filter preset (BA1+BA2, complete metabolites, valid images,
4/5 vote consensus at Dy30). Stratified split is delegated to
``pipeline.splits.Splits.stratified_random``.

    72% train  /  8% val  /  20% test   (seed=42)

Default output: ``data/splits/canonical_2026_winter.csv`` — note this is the
frozen reference file, so the script will refuse to overwrite it unless
``--overwrite`` is passed. For regenerations, point ``--output`` at a new path
(see ``data/splits/README.md``).

Usage:
    make run ARGS="-m analysis.paper_2026_04.generate_splits --dry-run"
    make run ARGS="-m analysis.paper_2026_04.generate_splits --output data/splits/regen.csv"
    make run ARGS="-m analysis.paper_2026_04.generate_splits --output data/splits/regen.csv --seed 7"
"""

import argparse
from collections import Counter
from pathlib import Path

from pipeline.data_loader import OrganoidDataset
from pipeline.splits import CANONICAL_PATH, Splits

ALL_DATA_PATH = Path("data/all_data.json")
DEFAULT_OUTPUT = CANONICAL_PATH
RATIOS = {"train": 0.72, "val": 0.08, "test": 0.20}
DEFAULT_SEED = 42


def collect_labeled_organoids(all_data_path: Path) -> dict[str, str]:
    """Build {organoid_id: label} via OrganoidDataset's paper-default filters."""
    ds = OrganoidDataset(str(all_data_path))
    return ds.organoid_labels()


def print_summary(org_labels: dict[str, str], splits: Splits) -> None:
    total = len(splits)
    label_counts = Counter(org_labels.values())
    print(f"Total organoids passing filters: {total}")
    for label, count in label_counts.items():
        print(f"  {label}: {count}")
    print()

    for split_name in ("train", "val", "test"):
        ids_in_split = [oid for oid, s in splits.mapping.items() if s == split_name]
        split_labels = Counter(org_labels[oid] for oid in ids_in_split)
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
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="Output CSV path (default: data/splits/canonical_2026_winter.csv)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="Random seed for splitting")
    parser.add_argument("--name", type=str, default=None,
                        help="Name to embed in the Splits object (defaults to output filename stem)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print summary without writing CSV")
    parser.add_argument("--overwrite", action="store_true",
                        help="Allow overwriting the frozen canonical CSV path")
    args = parser.parse_args()

    if args.output.resolve() == CANONICAL_PATH.resolve() and not args.overwrite and not args.dry_run:
        parser.error(
            f"Refusing to overwrite frozen canonical splits at {CANONICAL_PATH}. "
            "Pass --overwrite to force, or --output <path> to write elsewhere."
        )

    print(f"Loading {args.all_data} via OrganoidDataset (paper-default filters) ...")
    org_labels = collect_labeled_organoids(args.all_data)
    print(f"Organoids passing filters: {len(org_labels)}\n")

    splits = Splits.stratified_random(
        org_labels,
        ratios=RATIOS,
        seed=args.seed,
        name=args.name or args.output.stem,
    )
    print_summary(org_labels, splits)

    if args.dry_run:
        print("\n[dry-run] No file written.")
    else:
        splits.to_csv(args.output)
        print(f"\nWrote {args.output}  ({len(splits)} organoids)")


if __name__ == "__main__":
    main()
