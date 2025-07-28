import json
from pathlib import Path
from PIL import Image

# Path to your JSON file
DATA_JSON = Path("data/preprocessed/majority/Dy30.json")

# Load image paths from JSON
records = json.loads(DATA_JSON.read_text())

print(f"\nLoaded {len(records)} image records.\n")

# Iterate and print each image size
for i, record in enumerate(records):
    img_path = Path(record["img_path"])
    if not img_path.exists():
        print(f"[{i:03}] MISSING: {img_path}")
        continue

    try:
        with Image.open(img_path) as img:
            w, h = img.size
            print(f"[{i:03}] {img_path.name} → {w}×{h}")
    except Exception as e:
        print(f"[{i:03}] ERROR: {img_path} — {e}")
