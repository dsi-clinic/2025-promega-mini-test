from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import cv2
import numpy as np
import tifffile  # type: ignore


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
LOG = logging.getLogger("aspect_ratio_resize")


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


@dataclass
class Args:
    image_mapping_json: Path
    raw_images_dir: Path  # NEW
    out_images_dir: Path
    out_masks_dir: Path
    out_mapping_json: Path
    target_um_per_px: float
    target_size: int
    overwrite: bool
    require_mask: bool
    smoke: int


def parse_args() -> Args:
    p = argparse.ArgumentParser(
        description="Post-inference aspect-ratio conserved resize + physical-scale normalize + pad-to-square."
    )
    p.add_argument("--image-mapping-json", type=Path, required=True)

    # NEW: explicit raw TIFF base folder
    p.add_argument(
        "--raw-images-dir",
        type=Path,
        required=True,
        help="Base directory containing raw TIFF images referenced by 'Best Z Filename'.",
    )

    p.add_argument("--out-images-dir", type=Path, required=True)
    p.add_argument("--out-masks-dir", type=Path, required=True)
    p.add_argument("--out-mapping-json", type=Path, required=True)

    p.add_argument("--target-um-per-px", type=float, default=9.0)
    p.add_argument("--target-size", type=int, default=575)

    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--require-mask", action="store_true")
    p.add_argument("--smoke", type=int, default=0)

    a = p.parse_args()
    return Args(
        image_mapping_json=a.image_mapping_json,
        raw_images_dir=a.raw_images_dir,  # NEW
        out_images_dir=a.out_images_dir,
        out_masks_dir=a.out_masks_dir,
        out_mapping_json=a.out_mapping_json,
        target_um_per_px=float(a.target_um_per_px),
        target_size=int(a.target_size),
        overwrite=bool(a.overwrite),
        require_mask=bool(a.require_mask),
        smoke=int(a.smoke),
    )


def main() -> None:
    args = parse_args()

    mapping = json.loads(args.image_mapping_json.read_text())
    entries: Dict[str, Dict[str, Any]] = mapping.get("entries", {})
    if not entries:
        raise RuntimeError("No entries found in mapping JSON.")

    # NEW: raw TIFF base folder comes from CLI, not mapping JSON
    raw_base = args.raw_images_dir
    if not raw_base.exists():
        raise RuntimeError(f"--raw-images-dir missing/invalid: {raw_base}")

    processed_base = Path(mapping.get("_processed_base_folder", args.image_mapping_json.parent))
    if not processed_base.exists():
        raise RuntimeError(f"Processed base folder missing/invalid: {processed_base}")

    args.out_images_dir.mkdir(parents=True, exist_ok=True)
    args.out_masks_dir.mkdir(parents=True, exist_ok=True)

    record_ids = list(entries.keys())
    if args.smoke and args.smoke > 0:
        record_ids = record_ids[: args.smoke]

    out_entries: Dict[str, Any] = {}
    processed = 0
    skipped_no_mask = 0
    failed = 0

    for rid in record_ids:
        e = entries[rid]
        try:
            main_id = e.get("main_id") or rid

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

            # --- std image (stretched 512x384) ---
            rel_proc = e.get("processed_image")
            if not rel_proc:
                raise KeyError("Missing processed_image in entry.")
            std_img_path = processed_base / str(rel_proc)
            if not std_img_path.exists():
                raise FileNotFoundError(str(std_img_path))

            std_img = cv2.imread(str(std_img_path), cv2.IMREAD_COLOR)
            if std_img is None:
                raise RuntimeError(f"cv2 failed to read std image: {std_img_path}")

            # --- std mask (optional but usually required) ---
            mask_path_str = e.get("mask_path")
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

            # 1) Unstretch std image back to raw shape
            img_unstretched = cv2.resize(
                std_img,
                (orig_w, orig_h),
                interpolation=cv2.INTER_LINEAR,
            )

            # 2) Physical-scale resize to target um/px
            scale = orig_um / args.target_um_per_px
            scaled_h = int(round(orig_h * scale))
            scaled_w = int(round(orig_w * scale))
            if scaled_h <= 0 or scaled_w <= 0:
                raise RuntimeError(f"Bad scaled dims: ({scaled_h},{scaled_w}) from scale={scale}")

            img_scaled = cv2.resize(
                img_unstretched,
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

            out_entries[rid] = {
                "main_id": main_id,
                "raw_tif": str(raw_path),
                "std_image": str(std_img_path),
                "std_mask": str(mask_path_str) if mask_path_str else None,
                "ar_image": str(out_img.relative_to(args.out_images_dir)),
                "ar_mask": str(out_msk.relative_to(args.out_masks_dir)) if mask_final is not None else None,
                "orig_h": orig_h,
                "orig_w": orig_w,
                "orig_um_per_px": orig_um,
                "target_um_per_px": args.target_um_per_px,
                "scale_factor": scale,
                "scaled_h": scaled_h,
                "scaled_w": scaled_w,
                "target_size": args.target_size,
            }
            processed += 1

        except Exception:
            failed += 1
            LOG.exception("Failed record_id=%s", rid)
            continue

    out = {
        "_source_mapping": str(args.image_mapping_json),
        "_raw_base_folder": str(raw_base),
        "_std_processed_base_folder": str(processed_base),
        "_ar_images_base_folder": str(args.out_images_dir),
        "_ar_masks_base_folder": str(args.out_masks_dir),
        "params": {
            "target_um_per_px": args.target_um_per_px,
            "target_size": args.target_size,
            "image_interpolation": "INTER_LINEAR",
            "mask_interpolation": "INTER_NEAREST",
            "source_pixels": "std_512x384_then_unstretch_to_raw_shape",
        },
        "stats": {
            "seen": len(record_ids),
            "processed": processed,
            "skipped_no_mask": skipped_no_mask,
            "failed": failed,
        },
        "entries": out_entries,
    }

    args.out_mapping_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_mapping_json.write_text(json.dumps(out, indent=2))
    LOG.info("Wrote AR mapping: %s", args.out_mapping_json)
    LOG.info("Done. processed=%d skipped_no_mask=%d failed=%d", processed, skipped_no_mask, failed)


if __name__ == "__main__":
    main()