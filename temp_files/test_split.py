# This script splits a manual-processed mapping into train/val/test sets.

import json
import random
import argparse
from pathlib import Path

def main(mapping_path: Path, train_frac: float, val_frac: float):
    if not mapping_path.exists():
        raise FileNotFoundError(f"Mapping not found: {mapping_path}")

    # load the full manual-processed mapping
    with open(mapping_path, 'r') as f:
        full_map = json.load(f)

    # shuffle the keys
    keys = list(full_map.keys())
    random.seed(42)
    random.shuffle(keys)

    n = len(keys)
    n_train = int(train_frac * n)
    n_val   = int(val_frac   * n)

    splits = {
        "train": keys[:n_train],
        "val":   keys[n_train:n_train + n_val],
        "test":  keys[n_train + n_val:],
    }

    # make the split/ folder next to the mapping
    split_dir = mapping_path.parent / "split"
    split_dir.mkdir(exist_ok=True)

    base = mapping_path.stem  # e.g. image_mapping_day30_manual_processed_256x192
    for name, klist in splits.items():
        out_map = {k: full_map[k] for k in klist}
        out_file = split_dir / f"{base}_{name}.json"
        with open(out_file, 'w') as f:
            json.dump(out_map, f, indent=2)
        print(f"Wrote {len(klist)} entries → {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split a manual-processed mapping into train/val/test"
    )
    parser.add_argument(
        "--mapping", type=Path, required=True,
        help="Path to your processed mapping JSON"
    )
    parser.add_argument(
        "--train_frac", type=float, default=0.8,
        help="Fraction of data to use for training"
    )
    parser.add_argument(
        "--val_frac", type=float, default=0.1,
        help="Fraction of data to use for validation"
    )
    args = parser.parse_args()
    main(args.mapping, args.train_frac, args.val_frac)
