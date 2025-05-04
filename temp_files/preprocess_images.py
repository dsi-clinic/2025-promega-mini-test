# create_batch_mapping.py
import json
from pathlib import Path
import cv2
import argparse

# --------------- configuration -----------------
TARGET_SIZE   = (256, 192)          # (w, h)
INTERPOLATION = cv2.INTER_LINEAR

ORIGINAL_MAPPING = Path("/net/projects2/promega/data-analysis/output/image_mapping.json")
OUTPUT_DIR       = Path("/net/projects2/promega/data-analysis/output/processed_dataset_256x192")

# --------------- helpers -----------------------
def norm(s: str) -> str:
    """Lower‑case and strip spaces for tolerant comparison."""
    return s.lower().replace(' ', '') if isinstance(s, str) else ''

# --------------- main functions ----------------
def process_batch(batch_num: int, day_num: int = 30):
    """Create mapping(s) for one batch and day."""
    with ORIGINAL_MAPPING.open() as f:
        mapping = json.load(f)

    if batch_num == 2:                       # BA2 has two sub‑IDs
        create_mapping(mapping, "BA2 96_1", day_num)
        create_mapping(mapping, "BA2 96_2", day_num)
    else:
        create_mapping(mapping, f"BA{batch_num}", day_num)

def ba_match(json_ba: str, batch_id: str) -> bool:
    return norm(json_ba).startswith(norm(batch_id))

def create_mapping(mapping: dict, batch_id: str, day_num: int):
    day_id     = f"Dy{day_num}"
    safe_batch = batch_id.replace(' ', '_')
    output_dir = OUTPUT_DIR / f"{safe_batch}_{day_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_json = output_dir / f"image_mapping_{safe_batch}_{day_id}_processed.json"
    if output_json.exists():
        print(f"Mapping exists: {output_json}")
        return

    # ---------- find all matching entries once ----------
    matches = [
        (mid, info['Best Z Filename'])
        for mid, info in mapping.items()
        if norm(info.get('dayID')) == norm(day_id)
        and ba_match(info.get('BA', ''), batch_id)
    ]
    print(f"BA/day filter found {len(matches)} candidates.")
    for mid, path in matches[:3]:
        print("   ", mid, "→", path)

    # ---------- process each match ----------
    new_mapping = {}
    for img_id, path in matches:
        img_path = Path(path)
        if not img_path.exists():
            print(f"Skipped: Image not found {img_path}")
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Skipped: OpenCV couldn’t read {img_path}")
            continue

        resized  = cv2.resize(img, TARGET_SIZE, interpolation=INTERPOLATION)
        out_path = output_dir / f"{img_id.replace(' ', '_')}.png"
        cv2.imwrite(str(out_path), resized)
        new_mapping[img_id] = {'img_path': str(out_path)}

    # ---------- write new mapping ----------
    with output_json.open('w') as f:
        json.dump(new_mapping, f, indent=2)

    print(f"\nCreated mapping for {batch_id} with {len(new_mapping)} images")
    print(f"Images saved to: {output_dir}")
    print(f"Mapping saved to: {output_json}")

# --------------- CLI ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, required=True, help='Batch number to process')
    parser.add_argument('--day',   type=int, default=30,    help='Day number (default: 30)')
    args = parser.parse_args()

    print(f"\nCreating mapping for Batch {args.batch}, Day {args.day}")
    process_batch(args.batch, args.day)
    print("\nDone.")