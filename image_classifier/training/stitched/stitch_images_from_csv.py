#!/usr/bin/env python3

import csv
import math
import os
from pathlib import Path
from typing import List, Tuple

from PIL import Image, ImageDraw, ImageFont


def read_csv_records(csv_path: Path) -> List[dict]:
    records = []
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(row)
    return records


def group_paths_by_label(
    records: List[dict],
) -> Tuple[List[Tuple[str, Path]], List[Tuple[str, Path]]]:
    accepted, not_accepted = [], []
    for r in records:
        img_name = r.get("image") or os.path.basename(r.get("img_path", ""))
        img_path = r.get("img_path")
        label_str = (r.get("gt_label_str") or "").strip()
        if not img_path:
            continue
        path = Path(img_path)
        item = (img_name, path)
        if label_str.lower().startswith("accepted"):
            accepted.append(item)
        else:
            # treat anything else as Not Accepted
            not_accepted.append(item)
    return accepted, not_accepted


def safe_load_image(image_path: Path) -> Image.Image:
    try:
        return Image.open(image_path).convert("RGB")
    except Exception:
        # Create a red placeholder if image cannot be loaded
        placeholder = Image.new("RGB", (512, 384), color=(200, 0, 0))
        draw = ImageDraw.Draw(placeholder)
        draw.text((10, 10), "MISSING", fill=(255, 255, 255))
        return placeholder


def draw_captioned_tile(
    img: Image.Image,
    caption: str,
    font: ImageFont.ImageFont,
    caption_height: int,
    padding: int = 4,
) -> Image.Image:
    width, height = img.size
    tile = Image.new("RGB", (width, height + caption_height), color=(255, 255, 255))
    tile.paste(img, (0, 0))
    draw = ImageDraw.Draw(tile)
    text_w, text_h = draw.textlength(caption, font=font), font.size
    x = max(0, (width - int(text_w)) // 2)
    y = height + max(0, (caption_height - text_h) // 2)
    draw.text((x, y), caption, font=font, fill=(0, 0, 0))
    return tile


def stitch_grid(
    items: List[Tuple[str, Path]],
    out_path: Path,
    cols: int,
    rows: int,
    caption_height: int,
    spacing: int,
) -> None:
    if not items:
        return

    # Load first image to get dimensions
    first_img = safe_load_image(items[0][1])
    tile_w, img_h = first_img.size
    font = ImageFont.load_default()

    # Build tiles
    tiles: List[Image.Image] = []
    for name, p in items:
        img = safe_load_image(p)
        # Ensure uniform size by resizing to first image dims
        if img.size != first_img.size:
            img = img.resize((tile_w, img_h))
        tile = draw_captioned_tile(img, name, font, caption_height)
        tiles.append(tile)

    tile_h = img_h + caption_height
    grid_w = cols * tile_w + (cols - 1) * spacing
    grid_h = rows * tile_h + (rows - 1) * spacing
    canvas = Image.new("RGB", (grid_w, grid_h), color=(245, 245, 245))

    for idx, tile in enumerate(tiles):
        r = idx // cols
        c = idx % cols
        x = c * (tile_w + spacing)
        y = r * (tile_h + spacing)
        canvas.paste(tile, (x, y))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def chunk(
    items: List[Tuple[str, Path]], max_items: int
) -> List[List[Tuple[str, Path]]]:
    return [items[i : i + max_items] for i in range(0, len(items), max_items)]


def stitch_category(
    items: List[Tuple[str, Path]],
    out_dir: Path,
    title: str,
    cols: int,
    rows: int,
    caption_height: int,
    spacing: int,
) -> List[Path]:
    out_paths = []
    max_items = cols * rows
    for part_idx, group in enumerate(chunk(items, max_items), start=1):
        # Dynamically set rows for the last (possibly partial) page
        num_rows = min(rows, max(1, math.ceil(len(group) / float(cols))))
        out_name = f"stitched_{title}_{part_idx:03d}.png"
        out_path = out_dir / out_name
        stitch_grid(group, out_path, cols, num_rows, caption_height, spacing)
        out_paths.append(out_path)

    return out_paths


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Stitch images by label from CSV with captions."
    )
    parser.add_argument(
        "--csv",
        dest="csv_path",
        required=True,
        help="Path to aggregated_misclassified_by_all_models.csv",
    )
    parser.add_argument(
        "--outdir",
        dest="out_dir",
        required=False,
        default="analysis/images/classifier/stitched",
        help="Output directory for stitched images",
    )
    parser.add_argument("--cols", type=int, default=8, help="Number of columns in grid")
    parser.add_argument("--rows", type=int, default=10, help="Number of rows in grid")
    parser.add_argument(
        "--caption-height",
        type=int,
        default=18,
        help="Caption height in pixels under each tile",
    )
    parser.add_argument(
        "--spacing", type=int, default=6, help="Spacing in pixels between tiles"
    )
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    out_dir = Path(args.out_dir)

    records = read_csv_records(csv_path)
    accepted, not_accepted = group_paths_by_label(records)

    print(f"Found {len(accepted)} Accepted and {len(not_accepted)} Not Accepted images")

    accepted_outs = stitch_category(
        accepted,
        out_dir,
        "Accepted",
        args.cols,
        args.rows,
        args.caption_height,
        args.spacing,
    )
    not_accepted_outs = stitch_category(
        not_accepted,
        out_dir,
        "NotAccepted",
        args.cols,
        args.rows,
        args.caption_height,
        args.spacing,
    )

    for p in accepted_outs:
        print(f"Saved {p}")
    for p in not_accepted_outs:
        print(f"Saved {p}")


if __name__ == "__main__":
    main()
