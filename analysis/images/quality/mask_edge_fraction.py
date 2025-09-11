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
    except Exception:
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

def main():
    from config import ALL_DATA_JSON
    inp = out = ALL_DATA_JSON

    data = load_json(inp)

    changed = 0
    for key, entry in tqdm(list(data.items()), desc="computing edge_fraction"):
        if not (isinstance(key, str) and key.upper().startswith("BA")):
            continue
        info_512 = entry.get("512x384")
        mask_path = info_512.get("mask_path") if isinstance(info_512, dict) else None
        if mask_path:
            m = load_mask(Path(mask_path))
            ef = edge_fraction(m) if m is not None else None
        else:
            ef = None
        data[key] = insert_after(entry, "blank_area_frac", "edge_fraction", ef)
        changed += 1

    # always write a backup
    save_json(Path(str(inp) + ".bak"), data)
    save_json(out, data)
    print(f"Updated {changed} entries with edge_fraction → {out} (backup alongside)")


if __name__ == "__main__":
    main()
