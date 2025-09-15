#!/usr/bin/env python3

from __future__ import annotations
import json
import subprocess
from pathlib import Path
from tqdm import tqdm

from config import ALL_DATA_JSON, OUTPUT_FOLDER
from file_utils.common.organoid_patterns import norm_key

# --------------- Config ---------------
RCLONE_REMOTE = "cloudflare-r2-admin"
RCLONE_PATH = "image-data/images-for-masks/"
OUTPUT_JSON = OUTPUT_FOLDER / "all_data_with_good_flags.json"
# --------------------------------------


def get_good_keys_from_cloudflare() -> set[str]:
    print("📡 Fetching file list from Cloudflare R2...")

    result = subprocess.run(
        ["rclone", "lsf", "-R", f"{RCLONE_REMOTE}:{RCLONE_PATH}"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )

    if result.returncode != 0:
        raise RuntimeError(f"rclone failed: {result.stderr}")

    all_image_paths = result.stdout.strip().splitlines()
    print(f"✅ Found {len(all_image_paths)} image keys")

    good_keys = set()
    for path in tqdm(all_image_paths, desc="🧩 Parsing keys"):
        name = Path(path).name
        try:
            key = norm_key(name)
            good_keys.add(key)
        except ValueError:
            continue  # skip if filename can't be parsed

    return good_keys


def main():
    good_keys = get_good_keys_from_cloudflare()

    print(f"📂 Loading mapping from: {ALL_DATA_JSON}")
    with open(ALL_DATA_JSON) as f:
        mapping = json.load(f)

    modified = 0
    for key, val in mapping.items():
        if key in good_keys:
            val["good_image"] = True
            modified += 1
        else:
            val["good_image"] = False

    print(f"🧪 Marked {modified} entries as good images")

    with open(OUTPUT_JSON, "w") as f:
        json.dump(mapping, f, indent=2)

    print(f"✅ Saved updated mapping to: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
