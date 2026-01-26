# from __future__ import annotations

import argparse
import dataclasses
import json
import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import cv2
import numpy as np
import tifffile  # type: ignore
import tqdm


logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(format='%(asctime)s,%(msecs)d %(module)s:%(lineno)d %(levelname)s %(message)s',
                    datefmt='%Y-%m-%dT%H:%M:%S',
                    level=logging.INFO)


def pad_to_square_image(img: np.ndarray, target: int) -> np.ndarray:
    """Pad to square with edge padding (matches your old idea)."""
    h, w = img.shape[:2]
    if h > target or w > target:
        # Crop only if oversized; keep deterministic.
        img = img[: min(h, target), : min(w, target)]
        h, w = img.shape[:2]

    pad_h = target - h
    pad_w = target - w
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    if img.ndim == 2:
        return np.pad(img, ((top, bottom), (left, right)), mode="edge")
    return np.pad(img, ((top, bottom), (left, right), (0, 0)), mode="edge")


def pad_to_square_mask(mask: np.ndarray, target: int) -> np.ndarray:
    """Pad to square with zeros."""
    h, w = mask.shape[:2]
    if h > target or w > target:
        mask = mask[: min(h, target), : min(w, target)]
        h, w = mask.shape[:2]

    pad_h = target - h
    pad_w = target - w
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    return np.pad(mask, ((top, bottom), (left, right)), mode="constant", constant_values=0)


def safe_stem(main_id: str) -> str:
    # Match your prior convention (spaces to underscores). Keep conservative.
    s = main_id.strip().replace(" ", "_")
    for ch in ["/", "\\", ":", ";"]:
        s = s.replace(ch, "_")
    return s


def read_raw_shape(tif_path: Path) -> Tuple[int, int]:
    """Return (orig_h, orig_w) from the actual TIFF on disk."""
    with tifffile.TiffFile(str(tif_path)) as tf:
        page = tf.pages[0]
        shape = page.shape

    # Normalize shape to H,W
    if len(shape) == 2:
        orig_h, orig_w = shape
    elif len(shape) == 3:
        # (H,W,C) or (Z,H,W)
        if shape[-1] in (3, 4):  # RGB/RGBA
            orig_h, orig_w = shape[0], shape[1]
        else:
            orig_h, orig_w = shape[1], shape[2]
    elif len(shape) >= 4:
        orig_h, orig_w = shape[-3], shape[-2]
    else:
        raise RuntimeError(f"Unexpected TIFF shape {shape} for {tif_path}")

    return int(orig_h), int(orig_w)

def read_raw_image_as_bgr(tif_path: Path) -> np.ndarray:
    """
    Read raw TIFF pixels and return a BGR uint8 image for OpenCV.
    - If grayscale: converts to 3-channel BGR.
    - If uint16: scales to uint8.
    """
    img = tifffile.imread(str(tif_path))

    # If it's a stack, take the first plane (best-Z should be 2D anyway)
    if img.ndim >= 3 and img.shape[0] not in (3, 4) and img.shape[-1] not in (3, 4):
        img = img[0]

    # Normalize to uint8
    if img.dtype == np.uint16:
        img8 = (img.astype(np.float32) / 65535.0 * 255.0).clip(0, 255).astype(np.uint8)
    elif img.dtype != np.uint8:
        img8 = np.clip(img, 0, 255).astype(np.uint8)
    else:
        img8 = img

    # Convert to 3-channel BGR
    if img8.ndim == 2:
        img8 = cv2.cvtColor(img8, cv2.COLOR_GRAY2BGR)
    elif img8.ndim == 3 and img8.shape[-1] == 3:
        # tifffile gives RGB; OpenCV expects BGR
        img8 = img8[:, :, ::-1]
    elif img8.ndim == 3 and img8.shape[-1] == 4:
        img8 = img8[:, :, :3][:, :, ::-1]

    return img8

@dataclasses.dataclass
class Config:
    image_mapping_json: Path = dataclasses.field(metadata={
        "help": "Processed image mapping JSON (std_512x384). This file will be UPDATED in-place."
    })
    raw_images_dir: Path = dataclasses.field(metadata={
        "help": "Base directory containing raw TIFF images referenced by 'Best Z Filename'."
    })
    out_images_dir: Path = dataclasses.field(metadata={
        "help": "Output dir for images (writes to out-images-dir/images)."
    })
    out_masks_dir: Path = dataclasses.field(metadata={
        "help": "Output dir for masks (writes to out-masks-dir/masks)."
    })
    target_um_per_px: float = dataclasses.field(default=9.0, metadata={
        "help": "Target um per pixel."
    })
    target_size: int = dataclasses.field(default=575, metadata={
        "help": "Target size."
    })
    overwrite: bool = dataclasses.field(default=False, metadata={
        "help": "Overwrite existing files."
    })
    require_mask: bool = dataclasses.field(default=False, metadata={
        "help": "Require mask."
    })
    smoke: int = dataclasses.field(default=0, metadata={
        "help": "Limit to N records for quick test."
    })

    def __post_init__(self):
        if not self.image_mapping_json.exists():
            raise ValueError(f"Mapping JSON does not exist: {self.image_mapping_json}")
        if not self.raw_images_dir.exists():
            raise ValueError(f"Raw images directory does not exist: {self.raw_images_dir}")
        if not self.out_images_dir.exists():
            self.out_images_dir.mkdir(parents=True, exist_ok=True)
        if not self.out_masks_dir.exists():
            self.out_masks_dir.mkdir(parents=True, exist_ok=True)

def get_args():
    arg_parser = create_args()
    args = arg_parser.parse_args()
    cfg = Config(**vars(args))
    return cfg


def create_args() -> argparse.ArgumentParser:
    """Create an ArgumentParser from the Config dataclass."""
    parser = argparse.ArgumentParser(description="Run image classifier on organoid images")

    for field in dataclasses.fields(Config):
        # Build argument flag and help message
        flags = [f"--{field.name.replace('_', '-')}"]
        kwargs = {"help": field.metadata.get("help", "")}
        # REQUIRED vs optional
        if field.default is not dataclasses.MISSING:
            kwargs["default"] = field.default
        else:
            kwargs["required"] = True


        # Determine argument type
        if field.type == bool:
            kwargs["action"] = "store_true" if field.default is False else "store_false"
        else:
            kwargs["type"] = field.type

        parser.add_argument(*flags, **kwargs)

    return parser


def main() -> None:
    args = get_args()
    for key, value in vars(args).items():
        logging.info("%s: %s", key, value)

    mapping = json.loads(args.image_mapping_json.read_text())
    entries: Dict[str, Dict[str, Any]] = mapping.get("entries", {})
    if not entries:
        raise RuntimeError("No entries found in mapping JSON.")

    # NEW: raw TIFF base folder comes from CLI, not mapping JSON
    raw_base = args.raw_images_dir


    record_ids = list(entries.keys())
    if args.smoke and args.smoke > 0:
        record_ids = record_ids[: args.smoke]

    processed = 0
    skipped_no_mask = 0
    failed = 0

    for rid in tqdm.tqdm(record_ids, desc="Resize aspect ratio processing"):
        e = entries[rid]
        try:
            main_id = e.get("verification", {}).get("main_id") or rid

            # --- raw tif for TRUE shape ---
            raw_rel = e.get("Best Z Filename")
            if not raw_rel:
                raise KeyError("Missing Best Z Filename in entry.")
            raw_path = raw_base / str(raw_rel)
            if not raw_path.exists():
                raise FileNotFoundError(str(raw_path))

            orig_h, orig_w = read_raw_shape(raw_path)

            # --- um_per_px (assume you fixed this upstream; we trust JSON here) ---
            orig_um = e.get("um_per_px")
            if isinstance(orig_um, (list, tuple)) and len(orig_um) > 0:
                orig_um = orig_um[0]
            if orig_um is None:
                raise KeyError("Missing um_per_px in entry (needed for physical scaling).")
            orig_um = float(orig_um)

            # --- std mask (optional) ---
            mask_path_str = e.get("predicted_mask_path")

            if args.require_mask and not mask_path_str:
                skipped_no_mask += 1
                continue

            std_mask = None
            if mask_path_str:
                mp = Path(mask_path_str)
                if mp.exists():
                    std_mask = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
                else:
                    if args.require_mask:
                        raise FileNotFoundError(str(mp))
                    # else: leave std_mask as None

            # --- raw image pixels (rigorous) ---
            raw_img = read_raw_image_as_bgr(raw_path)
            if raw_img is None:
                raise RuntimeError(f"Failed to read raw tif pixels: {raw_path}")

            # Ensure raw_img matches orig_h/orig_w (it should, but be safe)
            raw_img = cv2.resize(raw_img, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)


            # 2) Physical-scale resize to target um/px
            scale = orig_um / args.target_um_per_px
            scaled_h = int(round(orig_h * scale))
            scaled_w = int(round(orig_w * scale))
            if scaled_h <= 0 or scaled_w <= 0:
                raise RuntimeError(f"Bad scaled dims: ({scaled_h},{scaled_w}) from scale={scale}")

            img_scaled = cv2.resize(
                raw_img,
                (scaled_w, scaled_h),
                interpolation=cv2.INTER_LINEAR,
            )


            # 3) Pad to square
            img_final = pad_to_square_image(img_scaled, args.target_size)

            # Masks: unstretch -> scale -> pad (nearest)
            mask_final = None
            if std_mask is not None:
                mask_unstretched = cv2.resize(
                    std_mask,
                    (orig_w, orig_h),
                    interpolation=cv2.INTER_NEAREST,
                )
                mask_scaled = cv2.resize(
                    mask_unstretched,
                    (scaled_w, scaled_h),
                    interpolation=cv2.INTER_NEAREST,
                )
                mask_final = pad_to_square_mask(mask_scaled, args.target_size)

            # Write outputs
            stem = safe_stem(str(main_id))
            out_img = args.out_images_dir / f"{stem}.png"
            out_msk = args.out_masks_dir / f"{stem}.png"

            if args.overwrite or not out_img.exists():
                ok = cv2.imwrite(str(out_img), img_final)
                if not ok:
                    raise RuntimeError(f"Failed to write image: {out_img}")

            if mask_final is not None:
                if args.overwrite or not out_msk.exists():
                    ok = cv2.imwrite(str(out_msk), mask_final)
                    if not ok:
                        raise RuntimeError(f"Failed to write mask: {out_msk}")

            entry = {
                "aspect_ratio": {
                    "ar_raw_tif": str(raw_path),
                    "ar_image": str(out_img.relative_to(args.out_images_dir)),
                    "ar_mask": str(out_msk.relative_to(args.out_masks_dir)) if mask_final is not None else None,
                    "ar_orig_um_per_px": orig_um,
                    "ar_target_um_per_px": args.target_um_per_px,
                    "ar_scale_factor": scale,
                    "ar_scaled_h": scaled_h,
                    "ar_scaled_w": scaled_w,
                    "ar_target_size": args.target_size,
                }
            }
            e.update(entry)

        except Exception:
            failed += 1
            logging.exception("Failed record_id=%s", rid)
            continue

    mapping["aspect_ratio"] = {
        "directory_meta": {
            "_source_mapping": str(args.image_mapping_json),
            "_raw_base_folder": str(raw_base),
            "_ar_images_base_folder": str(args.out_images_dir),
            "_ar_masks_base_folder": str(args.out_masks_dir),
        },
        "params": {
            "target_um_per_px": args.target_um_per_px,
            "target_size": args.target_size,
            "image_interpolation": "INTER_LINEAR",
            "mask_interpolation": "INTER_NEAREST",
            "source_pixels": "raw_tif_pixels",
        },
        "stats": {
            "seen": len(record_ids),
            "processed": processed,
            "skipped_no_mask": skipped_no_mask,
            "failed": failed,
        }
    }

    new_json = Path(args.image_mapping_json.parent / (args.image_mapping_json.stem + "_ar.json"))
    new_json.write_text(json.dumps(mapping, indent=2))
    logging.info("Wrote AR mapping: %s", new_json.name)
    logging.info("Done. processed=%d skipped_no_mask=%d failed=%d", processed, skipped_no_mask, failed)


if __name__ == "__main__":
    main()