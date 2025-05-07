# create_batch_mapping.py
import json
from pathlib import Path
import cv2
import argparse
import numpy as np
import shutil

# --------------- configuration -----------------
TARGET_SIZE     = (256, 192)          # (w, h) for your network
INTERPOLATION   = cv2.INTER_LINEAR
TARGET_UM_PER_PX = 1.687               # desired µm-per-network-pixel

ORIGINAL_MAPPING = Path("/net/projects2/promega/data-analysis/output/image_mapping.json")
OUTPUT_DIR       = Path("/net/projects2/promega/data-analysis/output/processed_dataset_256x192")

# --------------- helpers -----------------------
def norm(s: str) -> str:
    return s.lower().replace(' ', '') if isinstance(s, str) else ''

def ba_match(json_ba: str, batch_id: str) -> bool:
    return norm(json_ba).startswith(norm(batch_id))

# --------------- main functions ----------------
def process_batch(batch_num: int, day_num: int = 30):
    with ORIGINAL_MAPPING.open() as f:
        mapping = json.load(f)

    if batch_num == 2:
        create_mapping(mapping, "BA2 96_1", day_num)
        create_mapping(mapping, "BA2 96_2", day_num)
    else:
        create_mapping(mapping, f"BA{batch_num}", day_num)

def create_mapping(mapping: dict, batch_id: str, day_num: int):
    day_id     = f"Dy{day_num:02d}"
    safe_batch = batch_id.replace(' ', '_')
    output_dir = OUTPUT_DIR / f"{safe_batch}_{day_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_json = output_dir / f"image_mapping_{safe_batch}_{day_id}_processed.json"
    if output_json.exists():
        print(f"Mapping exists: {output_json}")
        return

    # collect (id, filename, orig_um_px)
    matches = [
        (img_id, info['Best Z Filename'], info.get('um_per_px'))
        for img_id, info in mapping.items()
        if norm(info.get('dayID')) == norm(day_id)
        and ba_match(info.get('BA', ''), batch_id)
    ]
    print(f"BA/day filter found {len(matches)} candidates.")
    for img_id, path, _ in matches[:3]:
        print("   ", img_id, "→", path)

    new_mapping = {}
    for img_id, path, orig_um_px in matches:
        img_path = Path(path)
        if not img_path.exists():
            print(f"Skipped (missing): {img_path}")
            continue

        img_raw = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img_raw is None:
            print(f"Skipped (unreadable): {img_path}")
            continue

        # ---- physical rescale to TARGET_UM_PER_PX ----
        if orig_um_px is None:
            print(f"no um_per_px for {img_id}, skipping physical rescale")
            img_scaled = img_raw
            final_um_px = None
        else:
            factor = orig_um_px / TARGET_UM_PER_PX
            if abs(factor - 1.0) < 0.01:
                img_scaled = img_raw
            else:
                new_w = int(img_raw.shape[1]  / factor)
                new_h = int(img_raw.shape[0]  / factor)
                img_scaled = cv2.resize(img_raw, (new_w, new_h), interpolation=INTERPOLATION)

            # after down/up-sampling, everything is at TARGET_UM_PER_PX
            # now compute final_um_px after the final 256×192 resize:
            scale1 = img_scaled.shape[1] / img_raw.shape[1]
            scale2 = TARGET_SIZE[0]     / img_scaled.shape[1]
            final_um_px = orig_um_px / (scale1 * scale2)

        # ---- network input resize ----
        img_final = cv2.resize(img_scaled, TARGET_SIZE, INTERPOLATION)

        # ---- save ----
        out_path = output_dir / f"{img_id.replace(' ', '_')}.png"
        cv2.imwrite(str(out_path), img_final)

        # ---- record in new mapping ----
        new_mapping[img_id] = {
            "img_path"       : str(out_path),
            "orig_um_per_px" : orig_um_px,
            "final_um_per_px": final_um_px
        }

    # write JSON
    with output_json.open('w') as f:
        json.dump(new_mapping, f, indent=2)

    print(f"\nCreated mapping for {batch_id} with {len(new_mapping)} images")
    print(f"Images saved to: {output_dir}")
    print(f"Mapping saved to: {output_json}")

# --------------- CLI ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batches',
                        type=lambda s: [int(x) for x in s.split(',')],
                        required=True,
                        help='Comma-separated batch numbers, e.g. 1,2,3')
    parser.add_argument('--days',
                        type=lambda s: [int(x) for x in s.split(',')],
                        required=True,
                        help='Comma-separated day numbers, e.g. 3,6,8')
    parser.add_argument('--overwrite',
                        action='store_true',
                        help='Remove existing processed folders first')
    args = parser.parse_args()

    for batch in args.batches:
        for day in args.days:
            print(f"\nProcessing Batch {batch}, Day {day}")
            if args.overwrite:
                sub_ids = ["96_1","96_2"] if batch==2 else [None]
                for sub in sub_ids:
                    fname = f"BA{batch}_{sub}_Dy{day:02d}" if sub else f"BA{batch}_Dy{day:02d}"
                    folder = OUTPUT_DIR / fname
                    if folder.exists():
                        import shutil
                        shutil.rmtree(folder)
                        print(f"  Removed old folder {folder}")
            process_batch(batch, day)

    print("\nAll done.")
