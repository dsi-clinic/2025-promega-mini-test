import json
import random
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode, RandomResizedCrop

# ---------------- CONFIG ----------------
DATA_JSON = Path("data/preprocessed/majority/Dy30.json")
OUT_DIR = Path("zoom_vis_dy30")
N_ZOOMS_PER_IMAGE = 5
ZOOM_SCALE = (0.5, 0.8)
NUM_IMAGES_TO_PROCESS = 10
TARGET_SIZE = (256, 192)  # (W, H)
ASPECT_RATIO = 192 / 256
SEED = 42
# ----------------------------------------

random.seed(SEED)
OUT_DIR.mkdir(parents=True, exist_ok=True)


def get_zoomed_versions(img_pil, n=5, scale=(0.5, 0.8), out_size=(256, 192)):
    zoomed = []
    w_orig, h_orig = img_pil.size
    img_area = w_orig * h_orig

    for _ in range(n):
        i, j, h, w = RandomResizedCrop.get_params(
            img_pil,
            scale=scale,
            ratio=(ASPECT_RATIO, ASPECT_RATIO),  # ✅ Enforce 4:3 crop
        )
        crop = TF.resized_crop(
            img_pil, i, j, h, w, size=out_size, interpolation=InterpolationMode.BILINEAR
        )
        crop_area = h * w
        scale_ratio = crop_area / img_area
        zoomed.append((crop, scale_ratio))  # return both image and scale
    return zoomed


def visualize_and_save(original, zooms_with_scale, save_path):
    cols = len(zooms_with_scale) + 1
    plt.figure(figsize=(3 * cols, 3))
    plt.subplot(1, cols, 1)
    plt.imshow(original)
    plt.title("Original")
    plt.axis("off")

    for i, (img, scale_ratio) in enumerate(zooms_with_scale):
        plt.subplot(1, cols, i + 2)
        plt.imshow(img)
        scale_percent = int(scale_ratio * 100)
        plt.title(f"{scale_percent}%")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main():
    records = json.loads(DATA_JSON.read_text())
    print(f"Loaded {len(records)} records from Dy30")

    random.shuffle(records)
    selected = records[:NUM_IMAGES_TO_PROCESS]

    for record in selected:
        img_path = Path(record["img_path"])
        if not img_path.exists():
            print(f"Image not found: {img_path}")
            continue

        stem = img_path.stem
        out_img_dir = OUT_DIR / stem
        out_img_dir.mkdir(parents=True, exist_ok=True)

        img = Image.open(img_path).convert("RGB")

        zooms = get_zoomed_versions(
            img,
            n=N_ZOOMS_PER_IMAGE,
            scale=ZOOM_SCALE,
            out_size=TARGET_SIZE,  # ✅ Resize to 256x192
        )

        img.save(out_img_dir / f"{stem}_original.jpg")

        for crop_img, scale_ratio in zooms:
            scale_percent = int(scale_ratio * 100)
            crop_img.save(out_img_dir / f"{stem}_zoom_{scale_percent}.jpg")

        visualize_and_save(img, zooms, out_img_dir / f"{stem}_comparison.jpg")
        print(f"Processed {img_path.name}")


if __name__ == "__main__":
    main()
