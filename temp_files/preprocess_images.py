# create_batch_mapping.py
import json
from pathlib import Path
import cv2
import argparse
import numpy as np
import shutil

# ---------------- configuration ----------------
TARGET_SIZE        = (256, 192)                # network input (w, h)
INTERPOLATION      = cv2.INTER_LINEAR

# >>>>>  physical‑scale parameters  <<<<<
TARGET_UM_PER_PX   = 2.46                      # common scale
SIZE_TO_SCALE = {                              # (raw w, h) : µm / px   ← update!
    (1128, 832):  2.46,                        # cytation
    (1920, 1440): 2.25,                        # keyence
    (2048, 1536): 2.24                         # EVOS
}
# ------------------------------------------------

ORIGINAL_MAPPING = Path("/net/projects2/promega/data-analysis/output/image_mapping.json")
OUTPUT_DIR       = Path("/net/projects2/promega/data-analysis/output/processed_dataset_256x192")

# -------------- helper functions ----------------
def norm(s: str) -> str:
    return s.lower().replace(' ', '') if isinstance(s, str) else ''

def ba_match(json_ba: str, batch_id: str) -> bool:
    return norm(json_ba).startswith(norm(batch_id))

def resample_to_physical(img: np.ndarray):
    """Rescale so that 1 px ≈ TARGET_UM_PER_PX. Returns img_rs, original_um_per_px."""
    h, w = img.shape[:2]
    um_px = SIZE_TO_SCALE.get((w, h))
    if um_px is None:
        print(f"Unknown raw size {w}×{h}; leaving scale unchanged")
        return img, None
    factor = um_px / TARGET_UM_PER_PX          # >1 → down‑sample
    if abs(factor - 1.0) < 0.05:
        return img, um_px                      # already near target
    new_w, new_h = int(w / factor), int(h / factor)
    img_rs = cv2.resize(img, (new_w, new_h), INTERPOLATION)
    return img_rs, um_px

# -------------- main functions ------------------
def process_batch(batch_num: int, day_num: int = 30):
    with ORIGINAL_MAPPING.open() as f:
        mapping = json.load(f)

    if batch_num == 2:        # BA2 splits
        create_mapping(mapping, "BA2 96_1", day_num)
        create_mapping(mapping, "BA2 96_2", day_num)
    else:
        create_mapping(mapping, f"BA{batch_num}", day_num)

def create_mapping(mapping: dict, batch_id: str, day_num: int):
    day_id = f"Dy{day_num:02d}"
    safe_batch = batch_id.replace(' ', '_')
    output_dir = OUTPUT_DIR / f"{safe_batch}_{day_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_json = output_dir / f"image_mapping_{safe_batch}_{day_id}_processed.json"
    if output_json.exists():
        print(f"Mapping exists: {output_json}")
        return

    matches = [
        (mid, info['Best Z Filename'])
        for mid, info in mapping.items()
        if norm(info.get('dayID')) == norm(day_id) and ba_match(info.get('BA', ''), batch_id)
    ]
    print(f"BA/day filter found {len(matches)} candidates.")

    new_mapping = {}
    for img_id, path in matches:
        img_path = Path(path)
        if not img_path.exists():
            print(f"Skipped (missing): {img_path}")
            continue

        img_raw = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img_raw is None:
            print(f"Skipped (unreadable): {img_path}")
            continue

        # ---- physical rescale ----
        img_scaled, orig_um_px = resample_to_physical(img_raw)

        # compute final_um_per_px directly from the ORIGINAL raw width
        raw_w = img_raw.shape[1]
        resize_factor = raw_w / TARGET_SIZE[0]
        final_um_px   = (orig_um_px or TARGET_UM_PER_PX) * resize_factor

        # ---- final resize & save ----
        img_final = cv2.resize(img_scaled, TARGET_SIZE, INTERPOLATION)
        out_path  = output_dir / f"{img_id.replace(' ', '_')}.png"
        cv2.imwrite(str(out_path), img_final)

        # ---- record mapping ----
        new_mapping[img_id] = {
            "img_path": str(out_path),
            "orig_um_per_px": orig_um_px,
            "final_um_per_px": final_um_px
        }

    with output_json.open('w') as f:
        json.dump(new_mapping, f, indent=2)

    print(f"\nCreated mapping for {batch_id} with {len(new_mapping)} images")
    print(f"Images saved to: {output_dir}")
    print(f"Mapping saved to: {output_json}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--batches',
        type=lambda s: [int(x) for x in s.split(',')],
        required=True,
        help='Comma-separated batch numbers, e.g. 1,2,3'
    )
    parser.add_argument(
        '--days',
        type=lambda s: [int(x) for x in s.split(',')],
        required=True,
        help='Comma-separated day numbers, e.g. 3,6,8'
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Delete existing processed folders before rebuilding'
    )
    args = parser.parse_args()

    for batch in args.batches:
        for day in args.days:
            print(f"\nProcessing Batch {batch}, Day {day}")
            if args.overwrite:
                if batch == 2:
                    for sub in ("96_1", "96_2"):
                        folder = OUTPUT_DIR / f"BA2_{sub}_Dy{day}"
                        if folder.exists():
                            shutil.rmtree(folder)
                            print(f"  Removed old folder {folder}")
                else:
                    folder = OUTPUT_DIR / f"BA{batch}_Dy{day}"
                    if folder.exists():
                        shutil.rmtree(folder)
                        print(f"  Removed old folder {folder}")
            process_batch(batch, day)

    print("\nAll done.")