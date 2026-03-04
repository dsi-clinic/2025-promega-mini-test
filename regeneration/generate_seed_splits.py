#!/usr/bin/env python3
"""
Generate train/val/test splits for a given seed.

Reuses the exact collection and splitting logic from split_data_reproducible.py,
but writes the output to a custom directory instead of the default data_splits/.

Usage:
    python regeneration/generate_seed_splits.py --seed 7 --output_dir regeneration/seed_rotation_splits/s7
    # produces: regeneration/seed_rotation_splits/s7/data_splits/both_{train,val,test}_base.json
"""
import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent / "2025-promega-mini-test"
sys.path.insert(0, str(REPO))

from split_data_reproducible import (
    collect_organoid_data,
    split_by_organoid,
    print_statistics,
    ALL_DATA_JSON,
)


def main():
    p = argparse.ArgumentParser(description="Generate train/val/test splits for a given seed")
    p.add_argument("--seed", type=int, required=True, help="Random seed for split generation")
    p.add_argument("--output_dir", type=str, required=True,
                   help="Base dir; splits written to <output_dir>/data_splits/")
    args = p.parse_args()

    splits_dir = Path(args.output_dir) / "data_splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    all_data_path = REPO / ALL_DATA_JSON
    print(f"Loading {all_data_path} ...")
    with open(all_data_path) as f:
        all_data = json.load(f)

    organoid_data = collect_organoid_data(all_data, batches=["BA1", "BA2"], require_metabolites=True)
    print(f"Collected {len(organoid_data)} organoids (BA1+BA2, both image & metabolite)")

    train_data, val_data, test_data = split_by_organoid(organoid_data, random_seed=args.seed)

    print(f"\nSeed {args.seed} split:")
    print_statistics(train_data, "Train")
    print_statistics(val_data, "Val")
    print_statistics(test_data, "Test")

    for tag, d in [("train", train_data), ("val", val_data), ("test", test_data)]:
        out = splits_dir / f"both_{tag}_base.json"
        with open(out, "w") as f:
            json.dump(d, f, indent=2)
        print(f"  Wrote {len(d)} organoids -> {out}")

    print(f"\nDone. Splits for seed {args.seed} saved to {splits_dir}")


if __name__ == "__main__":
    main()
