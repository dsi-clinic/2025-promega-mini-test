# analysis/images/mask_edge_fraction.py
# from __future__ import annotations
import argparse
import json
import logging
from pathlib import Path

import numpy as np
from skimage.io import imread
from tqdm import tqdm

logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(format='%(asctime)s,%(msecs)d %(module)s:%(lineno)d %(levelname)s %(message)s',
                    datefmt='%Y-%m-%dT%H:%M:%S',
                    level=logging.INFO)

EXPECTED_RECORDS_NUM = 5168

def create_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute edge fraction for masks")
    parser.add_argument(
        "--image-mapping-json",
         type=Path,
          required=True,
           help="Resized image mapping JSON file, e.g. 'image_map_resized_512x384.json'"
    )
    return parser

def load_json(p: Path):
    with open(p, "r") as f:
        return json.load(f)

def load_mask(mask_path: Path) -> np.ndarray | None:
    try:
        arr = imread(mask_path)
        if arr.ndim == 3:
            arr = arr[..., 0]  # take first channel if RGB
        return (arr > 0).astype(np.uint8)
    except Exception as e:
        print(f"Failed to load mask {mask_path}: {e}")
        return None

def edge_fraction(mask: np.ndarray) -> float:
    h, w = mask.shape
    if h == 0 or w == 0:
        return 0.0
    border = np.zeros_like(mask, dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    border_total = border.sum()
    if border_total == 0:
        return 0.0
    return float((mask.astype(bool) & border).sum() / border_total)

def save_json(p: Path, data: dict):
    with open(p, "w") as f:
        json.dump(data, f, indent=2)

def main():
    args = create_args().parse_args()
    for key, value in vars(args).items():
        logging.info(f"  {key}: {value}")

    mapping = load_json(args.image_mapping_json)
    logging.info(f"Found {len(mapping.get('entries', {}))} records in: %s", args.image_mapping_json)

    processed = 0
    failed = 0
    no_mask = 0
    for key, entry in tqdm(mapping.get('entries', {}).items(), desc="Computing edge_fraction"):
        mask_path_str: str = entry["predicted_mask_path"]

        if mask_path_str:
            if Path(mask_path_str).exists():
                # Load and process mask
                mask = load_mask(Path(mask_path_str))
            else:
                mask = None
                no_mask += 1

            if mask is not None:
                entry["edge_fraction"] = edge_fraction(mask)
                processed += 1
            else:
                entry["edge_fraction"] = None
                failed += 1
        else:
            no_mask += 1

    logging.info(f"Successfully processed {processed} masks")
    if no_mask > 0:
        logging.info(f"No mask found for {no_mask} masks")
    if failed > 0:
        logging.info(f"Failed to process {failed} masks")

    assert processed + failed + no_mask == EXPECTED_RECORDS_NUM, f"Expected {EXPECTED_RECORDS_NUM} records, got {processed + failed + no_mask}"

    if processed > 0 or failed > 0:
        save_json(args.image_mapping_json, mapping)
        logging.info(f"Updated mapping JSON saved in-place to: {args.image_mapping_json}")

if __name__ == "__main__":
    main()