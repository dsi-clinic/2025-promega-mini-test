#!/usr/bin/env python3
from __future__ import annotations
import json
import subprocess
from pathlib import Path
from tqdm import tqdm

from config import ALL_DATA_JSON, OUTPUT_FOLDER
from file_utils.common.organoid_patterns import OrganoidNormalizer

# --------------- Config ---------------
RCLONE_REMOTE = "cloudflare-r2-admin"
RCLONE_PATH = "image-data/images-for-masks/"
OUTPUT_JSON = OUTPUT_FOLDER / "all_data_with_good_flags.json"
# --------------------------------------

def get_good_keys_from_cloudflare() -> set[str]:
    print("Fetching file list from Cloudflare R2...")
    
    result = subprocess.run(
        ["rclone", "lsf", "-R", f"{RCLONE_REMOTE}:{RCLONE_PATH}"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    
    if result.returncode != 0:
        raise RuntimeError(f"rclone failed: {result.stderr}")
    
    all_image_paths = result.stdout.strip().splitlines()
    print(f"Found {len(all_image_paths)} image keys")
    
    good_keys = set()
    failed_parses = []
    
    for path in tqdm(all_image_paths, desc="🧩 Parsing keys"):
        name = Path(path).name
        try:
            # Use normalize_key_with_suffix to preserve split information
            key = OrganoidNormalizer.normalize_key_with_suffix(name)
            good_keys.add(key)
        except ValueError as e:
            failed_parses.append(f"{name}: {e}")
            continue  # skip if filename can't be parsed
    
    if failed_parses:
        print(f"Failed to parse {len(failed_parses)} filenames:")
        for failure in failed_parses[:10]:  # Show first 10 failures
            print(f"    {failure}")
        if len(failed_parses) > 10:
            print(f"    ... and {len(failed_parses) - 10} more")
    
    return good_keys

def main():
    good_keys = get_good_keys_from_cloudflare()
    
    print(f"Loading mapping from: {ALL_DATA_JSON}")
    with open(ALL_DATA_JSON) as f:
        mapping = json.load(f)
    
    print(f"Sample good keys: {sorted(list(good_keys))[:5]}")
    print(f"Sample mapping keys: {sorted(list(mapping.keys()))[:5]}")
    
    modified = 0
    missing_keys = []
    
    for key, val in mapping.items():
        if key in good_keys:
            val["good_image"] = True
            modified += 1
        else:
            val["good_image"] = False
            # Only track missing keys for entries that should have images
            # (i.e., not parent entries with split_children)
            if not val.get("split_children"):
                missing_keys.append(key)
    
    print(f"🧪 Marked {modified} entries as good images")
    
    if missing_keys:
        print(f"Found {len(missing_keys)} entries without matching images:")
        for missing in missing_keys[:10]:  # Show first 10 missing
            print(f"    {missing}")
        if len(missing_keys) > 10:
            print(f"    ... and {len(missing_keys) - 10} more")
    
    # Calculate some stats
    total_entries = len(mapping)
    parent_entries = sum(1 for v in mapping.values() if v.get("split_children"))
    split_entries = sum(1 for k in mapping.keys() if " split_" in k)
    regular_entries = total_entries - parent_entries - split_entries
    
    print(f"\nStatistics:")
    print(f"    Total entries: {total_entries}")
    print(f"    Parent entries (with splits): {parent_entries}")
    print(f"    Split entries: {split_entries}")
    print(f"    Regular entries (no splits): {regular_entries}")
    print(f"    Entries with good images: {modified}")
    print(f"    Coverage: {modified/total_entries*100:.1f}%")
    
    with open(OUTPUT_JSON, "w") as f:
        json.dump(mapping, f, indent=2)
    
    print(f"Saved updated mapping to: {OUTPUT_JSON}")

if __name__ == "__main__":
    main()