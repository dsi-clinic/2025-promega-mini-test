# This script splits a manual-processed mapping into train/val/test sets.

import json
import random
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paths import MAPPING_PROCESSED_TOTAL

EARLY_DAYS = {"Dy03", "Dy06", "Dy08", "Dy10"}
LATE_DAYS  = {"Dy13", "Dy15", "Dy17", "Dy20", "Dy21", "Dy24", "Dy28", "Dy30"}

# Load mapping from env default
mapping_path = MAPPING_PROCESSED_TOTAL
with open(mapping_path, 'r') as f:
    full_map = json.load(f)

# Define bad entries as (BA, dayID, wellID)
bad_entries = {
    ("BA1 96_1", "Dy30", "A4"),
    ("BA1 96_1", "Dy17", "A7"),
    ("BA1 96_1", "Dy17", "A8"),
    ("BA2 96_1", "Dy30", "C1"),
    ("BA2 96_1", "Dy30", "D10"),
    ("BA2 96_1", "Dy30", "E8"),
    ("BA2 96_1", "Dy28", "D10"),
    ("BA2 96_1", "Dy28", "E7"),
    ("BA2 96_1", "Dy21", "E7"),
    ("BA2 96_2", "Dy30", "A7"),
    ("BA2 96_2", "Dy30", "B8"),
    ("BA2 96_2", "Dy30", "D5"),   # fixed "DD5" to "D5"
    ("BA2 96_2", "Dy30", "D7"),
    ("BA2 96_2", "Dy30", "D12"),
    ("BA2 96_2", "Dy30", "E3"),
    ("BA2 96_2", "Dy28", "B3"),
    ("BA2 96_2", "Dy28", "B8"),
    ("BA2 96_2", "Dy28", "D5"),
    ("BA2 96_2", "Dy28", "E3"),
    ("BA2 96_2", "Dy28", "E12"),
    ("BA2 96_2", "Dy08", "D5"),
}

# Filter them out
before_count = len(full_map)
full_map = {
    k: v for k, v in full_map.items()
    if (v.get("BA"), v.get("dayID"), v.get("wellID")) not in bad_entries
}
after_count = len(full_map)
print(f"Removed {before_count - after_count} bad entries; {after_count} remain.")

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

def main(train_frac: float, val_frac: float, split_days: bool):
    split_dir = mapping_path.parent / "split"
    split_dir.mkdir(exist_ok=True)

    base = mapping_path.stem
    print(f"\n[Full Mapping] Splitting: {base}")
    split_and_save(full_map, base, split_dir, train_frac, val_frac)

    if split_days:
        early_map = {k: v for k, v in full_map.items() if v.get("dayID") in EARLY_DAYS}
        late_map  = {k: v for k, v in full_map.items() if v.get("dayID") in LATE_DAYS}

        print(f"\n[Early Days] Splitting {len(early_map)} entries")
        split_and_save(early_map, "mapping_days0310", split_dir, train_frac, val_frac)

        print(f"\n[Late Days] Splitting {len(late_map)} entries")
        split_and_save(late_map, "mapping_days1330", split_dir, train_frac, val_frac)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_frac", type=float, default=0.8)
    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument("--split_days", action="store_true")
    args = parser.parse_args()
    main(args.train_frac, args.val_frac, args.split_days)
