#!/usr/bin/env python3
import json
import argparse
from pathlib import Path
import cv2
import numpy as np
import re
from collections import defaultdict

ROWS = list("ABCDEFGH")
COLS = [str(i) for i in range(1, 13)]

def well_to_rc(well):
    # "A1" -> (0,0), "H12" -> (7,11)
    m = re.match(r'^([A-Ha-h])\s*([1-9]|1[0-2])$', str(well).strip())
    if not m:
        return None
    r = ROWS.index(m.group(1).upper())
    c = int(m.group(2)) - 1
    return r, c

def draw_mask_outline_on_image(img, mask, color=(0, 0, 255), thickness=2):
    # mask expected uint8 {0,1} or {0,255}
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    mask = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img, contours, -1, color, thickness)
    return img

def safe_read_image(path):
    im = cv2.imread(str(path), cv2.IMREAD_COLOR)
    return im

def safe_read_mask(path, target_hw=None):
    mk = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mk is None:
        return None
    if target_hw is not None and (mk.shape[0] != target_hw[0] or mk.shape[1] != target_hw[1]):
        mk = cv2.resize(mk, (target_hw[1], target_hw[0]), interpolation=cv2.INTER_NEAREST)
    return mk

def sanitize(s):
    return re.sub(r'[^A-Za-z0-9._-]+', '_', s)

def main():
    ap = argparse.ArgumentParser(description="Make plate mosaics with mask-outline overlays.")
    ap.add_argument("--json", required=True, help="Path to all_data.json")
    ap.add_argument("--size-key", default="512x384",
                    help="Which processed size block to use if present (e.g., 512x384, 256x192).")
    ap.add_argument("--outdir", default="/net/projects2/promega/data-analysis/npy_outputs",
                    help="Output directory for mosaics (PNG).")
    ap.add_argument("--outline-thickness", type=int, default=1, help="Contour outline thickness.")
    ap.add_argument("--label", action="store_true", help="Draw well labels on tiles.")
    args = ap.parse_args()

    json_path = Path(args.json)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load JSON (Python's json accepts NaN -> float('nan'))
    data = json.loads(json_path.read_text())

    # Group entries by (BA, dayID)
    groups = defaultdict(dict)  # {(BA, dayID): {wellID: info}}
    for img_id, info in data.items():
        BA = info.get("BA")
        day = info.get("dayID")
        well = info.get("wellID")
        if not (BA and day and well):
            continue
        groups[(BA, day)][well] = info

    # Build mosaics per (BA, dayID)
    for (BA, day), wells in groups.items():
        # Determine image size from the first available entry
        tile_h = tile_w = None

        # Prepare an empty tile to fill gaps
        def make_blank_tile():
            if tile_h is None or tile_w is None:
                # Fallback size if no image found yet
                return np.zeros((192, 256, 3), dtype=np.uint8)
            return np.zeros((tile_h, tile_w, 3), dtype=np.uint8)

        # First pass: discover tile size
        for wellID in wells:
            info = wells[wellID]
            # prefer processed block if present
            block = info.get(args.size_key, {})
            img_path = block.get("img_path") or info.get("Best Z Filename")
            if img_path and Path(img_path).exists():
                img = safe_read_image(img_path)
                if img is not None:
                    tile_h, tile_w = img.shape[:2]
                    break

        if tile_h is None:
            # No images readable; skip this plate-day
            print(f"[WARN] No readable images for {BA} {day}. Skipping.")
            continue

        # Create canvas: 8 rows x 12 cols
        H, W = 8 * tile_h, 12 * tile_w
        canvas = np.zeros((H, W, 3), dtype=np.uint8)

        # Put each well into canvas
        for r_idx, row in enumerate(ROWS):
            for c_idx, col in enumerate(COLS):
                wellID = f"{row}{col}"
                y0, y1 = r_idx * tile_h, (r_idx + 1) * tile_h
                x0, x1 = c_idx * tile_w, (c_idx + 1) * tile_w

                info = wells.get(wellID)
                if info is None:
                    canvas[y0:y1, x0:x1] = make_blank_tile()
                    continue

                block = info.get(args.size_key, {})
                img_path = block.get("img_path") or info.get("Best Z Filename")
                mask_path = block.get("mask_path") or info.get("MT Mask Path")

                tile = safe_read_image(img_path) if img_path and Path(img_path).exists() else make_blank_tile()
                if tile is None:
                    tile = make_blank_tile()
                # Ensure tile size matches
                if tile.shape[0] != tile_h or tile.shape[1] != tile_w:
                    tile = cv2.resize(tile, (tile_w, tile_h), interpolation=cv2.INTER_LINEAR)

                # Overlay outline if mask exists
                if mask_path and Path(mask_path).exists():
                    mk = safe_read_mask(mask_path, target_hw=(tile_h, tile_w))
                    if mk is not None:
                        tile = draw_mask_outline_on_image(tile, mk, color=(255, 0, 255), thickness=args.outline_thickness)

                # Optional label
                if args.label:
                    cv2.putText(tile, wellID, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

                canvas[y0:y1, x0:x1] = tile

        # Save
        ba_s = sanitize(BA)
        day_s = sanitize(day)
        out_png = outdir / f"{ba_s}_{day_s}_plate_overlay.png"
        cv2.imwrite(str(out_png), canvas)
        print(f"[OK] Saved {out_png}")

if __name__ == "__main__":
    main()
