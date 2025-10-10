# analysis/images/mask_edge_fraction.py
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from skimage.io import imread
from tqdm import tqdm

from config import ALL_DATA_JSON, OUTPUT_FOLDER

def load_json(p: Path):
    with open(p, "r") as f:
        return json.load(f)

def save_json(p: Path, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(obj, f, indent=2)

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

def insert_after(d: dict, after_key: str, new_key: str, new_val):
    """
    Rebuilds a dict so that `new_key` appears right after `after_key` if present.
    If `after_key` isn't present, appends at the end.
    """
    out = {}
    inserted = False
    for k, v in d.items():
        out[k] = v
        if not inserted and k == after_key:
            out[new_key] = new_val
            inserted = True
    if not inserted:
        out[new_key] = new_val
    return out

def has_mask_data(entry: dict) -> bool:
    proc = entry.get("processed")
    return isinstance(proc, dict) and "mask_path" in proc


def main():
    # Use local all_data.json instead of remote path
    inp = out = "all_data.json"

    data = load_json(inp)

    # Filter entries that need processing
    entries_to_process = []
    parent_entries = 0
    
    for key, entry in data.items():
        if not (isinstance(key, str) and key.upper().startswith("BA")):
            continue
            
        if entry.get("split_children"):
            # Parent entry with splits - skip
            parent_entries += 1
            continue
            
        if has_mask_data(entry):
            entries_to_process.append((key, entry))

    print(f"Found {len(entries_to_process)} entries with masks to process")
    print(f"Skipping {parent_entries} parent entries (no direct masks)")
    
    processed = 0
    failed = 0
    
    for key, entry in tqdm(entries_to_process, desc="Computing edge_fraction"):
        mask_path = entry["processed"]["mask_path"]

        
        # Load and process mask
        mask = load_mask(Path(mask_path))
        if mask is not None:
            ef = edge_fraction(mask)
            processed += 1
        else:
            ef = None
            failed += 1
        
        # Insert edge_fraction after blank_area_frac if it exists, otherwise at end
        data[key] = insert_after(entry, "blank_area_frac", "edge_fraction", ef)

    print(f"Successfully processed {processed} masks")
    if failed > 0:
        print(f"Failed to process {failed} masks")

    # Always write a backup
    backup_path = Path(str(inp) + ".bak")
    save_json(backup_path, data)
    save_json(Path(out), data)
    
    print(f"Updated data saved to: {out}")
    print(f"Backup saved to: {backup_path}")


if __name__ == "__main__":
    main()