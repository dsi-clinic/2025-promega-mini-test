#!/usr/bin/env python3
# Split the *training* mapping into train/val/test (and optional early/late)

import json, random, argparse, sys
from pathlib import Path

# --- locate repo root (has paths.py + .env) ---
HERE = Path(__file__).resolve()
for p in HERE.parents:
    if (p / "config.py").exists() and (p / ".env").exists():
        sys.path.insert(0, str(p))
        break
else:
    raise RuntimeError("Could not locate repo root containing config.py and .env")

# --- canonical paths ---
from config import (
    TRAIN_MANUAL_PROCESSED_DIR,  # where resize wrote images/masks
    TRAIN_SPLITS_DIR,            # where we want the split JSONs
    TARGET_SUFFIX,               # e.g. "512x384"
)

# Default training mapping produced by resize_img_masks.py
DEFAULT_TRAIN_MAPPING = TRAIN_MANUAL_PROCESSED_DIR / f"mapping_processed_total_{TARGET_SUFFIX}.json"

EARLY_DAYS = {"Dy03", "Dy06", "Dy08", "Dy10"}
LATE_DAYS  = {"Dy13", "Dy15", "Dy17", "Dy20", "Dy21", "Dy24", "Dy28", "Dy30"}

# Known-bad entries to drop (BA, dayID, wellID)
BAD_ENTRIES = {
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
    ("BA2 96_2", "Dy30", "D5"),
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

def load_and_filter(mapping_path: Path):
    with open(mapping_path, "r") as f:
        full_map = json.load(f)
    before = len(full_map)
    full_map = {
        k: v for k, v in full_map.items()
        if (v.get("BA"), v.get("dayID"), v.get("wellID")) not in BAD_ENTRIES
    }
    after = len(full_map)
    print(f"Removed {before - after} bad entries; {after} remain.")
    return full_map

def split_and_save(full_map, out_prefix: str, split_dir: Path, train_frac: float, val_frac: float):
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

    split_dir.mkdir(parents=True, exist_ok=True)
    for name, klist in splits.items():
        out_map = {k: full_map[k] for k in klist}
        out_file = split_dir / f"{out_prefix}_{name}.json"
        with open(out_file, "w") as f:
            json.dump(out_map, f, indent=2)
        print(f"Wrote {len(klist)} entries → {out_file}")

def main(mapping_override: str | None, train_frac: float, val_frac: float, split_days: bool):
    mapping_path = Path(mapping_override) if mapping_override else DEFAULT_TRAIN_MAPPING
    print(f"Using training mapping: {mapping_path}")
    if not mapping_path.exists():
        raise FileNotFoundError(f"Mapping not found: {mapping_path}")

    full_map = load_and_filter(mapping_path)
    split_dir = TRAIN_SPLITS_DIR

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
    ap = argparse.ArgumentParser()
    ap.add_argument("--mapping", default=None, help="Optional path to training mapping JSON")
    ap.add_argument("--train_frac", type=float, default=0.8)
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--split_days", action="store_true")
    args = ap.parse_args()
    main(args.mapping, args.train_frac, args.val_frac, args.split_days)
