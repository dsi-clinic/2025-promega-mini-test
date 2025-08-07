# This script splits a manual-processed mapping into train/val/test sets.

import json
import random
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paths import MAPPING_PROCESSED_TOTAL

EARLY_DAYS = {"Dy03", "Dy06", "Dy08"}
LATE_DAYS  = {"Dy17", "Dy20", "Dy21", "Dy24", "Dy28", "Dy30"}

def split_and_save(full_map, out_prefix, split_dir, train_frac, val_frac):
    keys = list(full_map.keys())
    random.seed(42)
    random.shuffle(keys)

    n = len(keys)
    n_train = int(train_frac * n)
    n_val = int(val_frac * n)

    splits = {
        "train": keys[:n_train],
        "val": keys[n_train:n_train + n_val],
        "test": keys[n_train + n_val:],
    }

    for name, klist in splits.items():
        out_map = {k: full_map[k] for k in klist}
        out_file = split_dir / f"{out_prefix}_{name}.json"
        with open(out_file, 'w') as f:
            json.dump(out_map, f, indent=2)
        print(f"Wrote {len(klist)} entries → {out_file}")

def main(mapping_path: Path, train_frac: float, val_frac: float, split_days: bool):
    if not mapping_path.exists():
        raise FileNotFoundError(f"Mapping not found: {mapping_path}")

    with open(mapping_path, 'r') as f:
        full_map = json.load(f)

    # Standard split of full mapping
    split_dir = mapping_path.parent / "split"
    split_dir.mkdir(exist_ok=True)

    base = mapping_path.stem
    print(f"\n[Full Mapping] Splitting: {base}")
    split_and_save(full_map, base, split_dir, train_frac, val_frac)

    if split_days:
        # Optional: additional filtered splits
        early_map = {k: v for k, v in full_map.items() if v.get("dayID") in EARLY_DAYS}
        late_map  = {k: v for k, v in full_map.items() if v.get("dayID") in LATE_DAYS}

        print(f"\n[Early Days] Splitting {len(early_map)} entries")
        split_and_save(early_map, "mapping_days038", split_dir, train_frac, val_frac)

        print(f"\n[Late Days] Splitting {len(late_map)} entries")
        split_and_save(late_map, "mapping_days2430", split_dir, train_frac, val_frac)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mapping", type=Path, default=MAPPING_PROCESSED_TOTAL,
        help="Path to your processed mapping JSON"
    )
    parser.add_argument("--train_frac", type=float, default=0.8)
    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument(
        "--split_days", action="store_true",
        help="Also split early (03/06/08) and late (17–30) days separately"
    )
    args = parser.parse_args()
    main(args.mapping, args.train_frac, args.val_frac, args.split_days)
