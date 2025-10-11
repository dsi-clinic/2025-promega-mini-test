#!/usr/bin/env python3
import json, argparse, re
from pathlib import Path
import cv2, numpy as np
from file_utils.common.organoid_patterns import OrganoidPatterns
from collections import defaultdict

ROWS = list("ABCDEFGH")
COLS = [str(i) for i in range(1, 13)]

def sanitize(s): return re.sub(r'[^A-Za-z0-9._-]+', '_', str(s))

def well_to_rc(well):
    m = OrganoidPatterns.WELL_FLEXIBLE.match(str(well).strip())
    return (ROWS.index(m.group(1).upper()), int(m.group(2))-1) if m else None

def safe_read_image(p):
    return cv2.imread(str(p), cv2.IMREAD_COLOR) if p and Path(p).exists() else None

def safe_read_mask(p, target_hw=None):
    if not p or not Path(p).exists(): return None
    mk = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if mk is None: return None
    if target_hw and (mk.shape[0] != target_hw[0] or mk.shape[1] != target_hw[1]):
        mk = cv2.resize(mk, (target_hw[1], target_hw[0]), interpolation=cv2.INTER_NEAREST)
    return mk

def draw_outline(img, mask, thickness=2):
    # OpenCV is BGR: pure red
    mask = ((mask > 0).astype(np.uint8)) * 255
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img, cnts, -1, (0, 0, 255), int(thickness))
    return img

def main():
    ap = argparse.ArgumentParser(description="Plate overlays from MT-mask JSON (saves *_MT.png).")
    ap.add_argument("--json", required=True, help="Path to JSON with 'Best Z Filename' and 'MT Mask Path'.")
    ap.add_argument("--outdir", default="/net/projects2/promega/data-analysis/npy_outputs", help="Output dir.")
    ap.add_argument("--outline-thickness", type=int, default=2)
    ap.add_argument("--label", action="store_true", help="Draw well labels.")
    ap.add_argument("--suffix", default="_MT", help="Filename suffix (default: _MT).")
    args = ap.parse_args()

    data = json.loads(Path(args.json).read_text())
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # Group by (BA, dayID)
    groups = defaultdict(dict)
    for k, info in data.items():
        BA, day, well = info.get("BA"), info.get("dayID"), info.get("wellID")
        if not (BA and day and well): continue
        groups[(BA, day)][well] = info

    for (BA, day), wells in groups.items():
        # Determine tile size
        tile_h = tile_w = None
        for w in wells.values():
            img = safe_read_image(w.get("Best Z Filename"))
            if img is not None:
                tile_h, tile_w = img.shape[:2]
                break
        if tile_h is None:
            print(f"[WARN] No readable images for {BA} {day}. Skipping.")
            continue

        H, W = 8*tile_h, 12*tile_w
        canvas = np.zeros((H, W, 3), dtype=np.uint8)

        for r_i, row in enumerate(ROWS):
            for c_i, col in enumerate(COLS):
                wellID = f"{row}{col}"
                y0, y1 = r_i*tile_h, (r_i+1)*tile_h
                x0, x1 = c_i*tile_w, (c_i+1)*tile_w

                info = wells.get(wellID)
                if not info:
                    canvas[y0:y1, x0:x1] = 0
                    continue

                img = safe_read_image(info.get("Best Z Filename"))
                if img is None:
                    canvas[y0:y1, x0:x1] = 0
                    continue
                if img.shape[:2] != (tile_h, tile_w):
                    img = cv2.resize(img, (tile_w, tile_h), interpolation=cv2.INTER_LINEAR)

                mpath = info.get("MT Mask Path")
                mk = safe_read_mask(mpath, (tile_h, tile_w)) if mpath else None
                if mk is not None:
                    img = draw_outline(img, mk, thickness=args.outline_thickness)

                if args.label:
                    cv2.putText(img, wellID, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

                canvas[y0:y1, x0:x1] = img

        out_png = outdir / f"{sanitize(BA)}_{sanitize(day)}_plate_overlay{args.suffix}.png"
        cv2.imwrite(str(out_png), canvas)
        print(f"[OK] Saved {out_png}")

if __name__ == "__main__":
    main()
