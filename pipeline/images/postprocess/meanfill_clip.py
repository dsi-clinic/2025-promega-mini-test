#!/usr/bin/env python3
"""
meanfill_clip.py

Generate mean-filled "clipped" images using a mask, matching the training view.

What it does:
- Reads a mapping JSON (std or AR).
- Resolves (image, mask) paths per entry (auto-detects common fields).
- Computes a SINGLE RGB mean vector (default: from BACKGROUND pixels where mask<=127).
- Writes mean-filled images to --out-images-dir.
- Writes a NEW mapping JSON to --out-mapping-json (never edits input mapping).

Key semantics:
- Masks are assumed: organoid/foreground = white (255), background = black (0)
  (If your masks are inverted, pass --mask-foreground black.)
- Mean-fill operation:
    filled = img * keep_mask + mean * (1 - keep_mask)
  where keep_mask is the (possibly blurred) foreground mask normalized to [0,1].

Example (compute mean from background pixels):
python -m pipeline.images.postprocess.meanfill_clip \
  --mapping-json /path/to/image_mapping_ar.json \
  --compute-mean \
  --mean-region background \
  --out-images-dir /path/to/meanfill_clip/images \
  --out-mapping-json /path/to/image_mapping_ar_meanfill.json \
  --require-mask \
  --overwrite
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
from skimage.io import imread, imsave
from tqdm import tqdm

logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(format='%(asctime)s,%(msecs)d %(module)s:%(lineno)d %(levelname)s %(message)s',
                    datefmt='%Y-%m-%dT%H:%M:%S',
                    level=logging.INFO)


# -------------------------
# CLI
# -------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate mean-filled clipped images from a mapping JSON (writes new images + new mapping JSON)."
    )

    p.add_argument("--image-mapping-json",
        type=Path,
        required=True,
        help="Path to the image mapping JSON file.",
    )
    p.add_argument(
    "--source",
    choices=["auto", "std", "ar"],
    default="auto",
    help="Which image/mask source to use. auto=prefer AR if present; std=use processed_image(+predicted_mask); ar=use aspect_ratio.ar_*.",
    )

    p.add_argument(
    "--out-mapping-json",
    type=Path,
    default=None,
    help="Where to write the updated mapping JSON. If omitted, writes next to input with _meanfill suffix.",
    )

    # Provide ONE of:
    p.add_argument("--global-mean-npy",
        type=Path,
        default=None,
        help="Optional. Precomputed RGB mean as .npy shape (3,) in [0,1] (or [0,255] if --loaded-mean-scale=255).",
    )
    p.add_argument("--compute-mean",
        action="store_true",
        help="Compute RGB mean from images referenced by the mapping (recommended for standalone usage).",
    )
    p.add_argument("--mean-region",
        choices=["background", "foreground", "all"],
        default="background",
        help="When --compute-mean: pixels to use for mean. background=mask<=127, foreground=mask>127, all=entire image.",
    )
    p.add_argument("--mean-sample",
        type=int,
        default=0,
        help="When --compute-mean: if >0, sample N entries (faster). 0 = use all entries.",
    )
    p.add_argument("--save-computed-mean",
        action="store_true",
        help="When --compute-mean: save computed mean as global_mean.npy next to out-mapping-json.",
    )
    p.add_argument("--loaded-mean-scale",
        type=float,
        default=1.0,
        help="Only used when --global-mean-npy is provided. 1 if mean is in [0,1], 255 if in [0,255].",
    )

    p.add_argument("--out-images-dir",
        type=Path,
        required=True,
        help="Output directory for mean-filled images.",
    )
    p.add_argument("--images-base",
        type=Path,
        default=None,
        help="Base folder to resolve relative image paths in the mapping (e.g. .../images).",
    )
    p.add_argument("--masks-base",
        type=Path,
        default=None,
        help="Base folder to resolve relative mask paths in the mapping (e.g. .../masks).",
    )

    p.add_argument("--blur-kernel",
        type=int,
        nargs=2,
        default=(5, 5),
        help="Gaussian blur kernel, e.g. 5 5",
    )
    p.add_argument("--dilate-kernel",
        type=int,
        nargs=2,
        default=(5, 5),
        help="Ellipse kernel size, e.g. 5 5",
    )
    p.add_argument("--dilate-iterations",
        type=int,
        default=1,
        help="Number of dilation iterations.",
    )

    p.add_argument("--mask-foreground",
        choices=["white", "black"],
        default="white",
        help="How the mask encodes foreground. If 'white', organoid=255 background=0. If 'black', invert it.",
    )

    p.add_argument("--overwrite",
        action="store_true",
        help="Overwrite existing output files.",
    )
    p.add_argument("--require-mask",
        action="store_true",
        help="Require mask to be present for processing.",
    )
    p.add_argument("--smoke",
        type=int,
        default=0,
        help="Process only first N entries (debug).",
    )

    # Optional: force which mapping fields to use
    p.add_argument("--image-field",
        type=str,
        default="",
        help="Optional: entry field for image path.",
    )
    p.add_argument("--mask-field",
        type=str,
        default="",
        help="Optional: entry field for mask path.",
    )

    args = p.parse_args()

    if args.global_mean_npy is None and not args.compute_mean:
        raise SystemExit("ERROR: Provide --global-mean-npy OR pass --compute-mean.")

    return args


# -------------------------
# Helpers
# -------------------------
def to_rgb(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return np.stack([img] * 3, axis=-1)
    if img.ndim == 3 and img.shape[2] == 4:
        return img[:, :, :3]
    return img


def safe_stem(s: str) -> str:
    s2 = s.strip().replace(" ", "_")
    for ch in ["/", "\\", ":", ";", "(", ")", "[", "]"]:
        s2 = s2.replace(ch, "_")
    return s2


def pick_paths(
    entry: Dict[str, Any],
    processed_base: Path,
    ar_images_base: Optional[Path],
    ar_masks_base: Optional[Path],
    images_base: Optional[Path],
    masks_base: Optional[Path],
    image_field_override: str,
    mask_field_override: str,
    source: str = "auto",
) -> Tuple[Optional[Path], Optional[Path], str, str]:
    """
    Returns: (img_path, mask_path, used_img_field, used_mask_field)
    """
    ar = entry.get("aspect_ratio") or {}
    if isinstance(ar, dict):
        if "ar_image" in ar and not entry.get("ar_image"):
            entry = dict(entry)
            entry["ar_image"] = ar.get("ar_image")
        if "ar_mask" in ar and not entry.get("ar_mask"):
            if "entry" not in locals():
                entry = dict(entry)
            entry["ar_mask"] = ar.get("ar_mask")

    has_ar = bool(entry.get("ar_image") or entry.get("ar_image_abs"))
    # Image candidates
    image_candidates = []
    if image_field_override:
        image_candidates.append(image_field_override)

    if source == "ar":
        image_candidates += ["ar_image_abs", "ar_image"]
    elif source == "std":
        image_candidates += ["processed_image_abs", "processed_image", "std_image"]
    else:  # auto
        if has_ar:
            image_candidates += ["ar_image_abs", "ar_image"]
        image_candidates += ["processed_image_abs", "processed_image", "std_image"]
        if not has_ar:
            image_candidates += ["ar_image_abs", "ar_image"]
            
    img_path: Optional[Path] = None
    used_img_field = ""
    for k in image_candidates:
        v = entry.get(k)
        if not v:
            continue
        p = Path(str(v))
        if not p.is_absolute():
            if images_base is not None:
                p = images_base / p
            elif k.startswith("ar_") and ar_images_base is not None:
                p = ar_images_base / p
            else:
                p = processed_base / p
        img_path = p
        used_img_field = k
        break

    # Mask keys - prioritize AR masks if we're using AR images
    mask_candidates = []
    if mask_field_override:
        mask_candidates.append(mask_field_override)

    if source == "ar":
        mask_candidates += ["ar_mask_abs", "ar_mask"]
    elif source == "std":
        mask_candidates += ["predicted_mask_path", "mask_path", "processed_mask", "std_mask"]
    else:  # auto
        # choose masks based on the image we actually selected
        if used_img_field.startswith("ar_"):
            mask_candidates += ["ar_mask_abs", "ar_mask"]

        mask_candidates += ["predicted_mask_path", "mask_path", "processed_mask", "std_mask"]

        # optional fallback at the very end if std masks missing
        if not used_img_field.startswith("ar_") and has_ar:
            mask_candidates += ["ar_mask_abs", "ar_mask"]

    mask_candidates += [
        "mask_path",
        "predicted_mask_path",
        "processed_mask",
        "std_mask",  # Put std_mask last to avoid mismatched dimensions
    ]

    mask_path: Optional[Path] = None
    used_mask_field = ""
    for k in mask_candidates:
        v = entry.get(k)
        if not v:
            continue
        p = Path(str(v))
        if not p.is_absolute():
            if masks_base is not None:
                p = masks_base / p
            elif k.startswith("ar_") and ar_masks_base is not None:
                p = ar_masks_base / p
            elif k.startswith("ar_") and ar_images_base is not None:
                # Fallback: try to derive masks base from images base
                # e.g., /path/to/images/resized_575_square -> /path/to/masks/resized_575_square
                masks_dir = ar_images_base.parent.parent / "masks" / ar_images_base.name
                p = masks_dir / p
            else:
                p = processed_base / p

        mask_path = p
        used_mask_field = k
        break

    return img_path, mask_path, used_img_field, used_mask_field


def load_global_mean_rgb01(path: Path, loaded_scale: float) -> np.ndarray:
    gm = np.load(path)
    gm = np.array(gm, dtype=np.float32).reshape(-1)
    if gm.size != 3:
        raise ValueError(f"global_mean.npy must have shape (3,), got {gm.shape} from {path}")
    gm = gm / float(loaded_scale)
    gm = np.clip(gm, 0.0, 1.0)
    return gm


def apply_mean_fill(
    img_rgb_u8: np.ndarray,
    mask_u8: np.ndarray,
    mean_rgb01: np.ndarray,
    blur_kernel: Tuple[int, int],
    dilate_kernel: Tuple[int, int],
    dilate_iterations: int,
    mask_foreground: str,
) -> np.ndarray:
    """
    img_rgb_u8: HxWx3 uint8
    mask_u8: HxW uint8 (0..255), foreground typically 255
    mean_rgb01: (3,) float32 in [0,1]
    returns float32 image in [0,1]
    """
    if mask_u8.ndim == 3:
        mask_u8 = mask_u8[:, :, 0]
    mask_u8 = mask_u8.astype(np.uint8)

    # Ensure foreground is white for "keep" mask
    if mask_foreground == "black":
        mask_u8 = 255 - mask_u8

    # Dilate + blur for softer edges (same spirit as your old script)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, tuple(dilate_kernel))
    if dilate_iterations > 0:
        mask_u8 = cv2.dilate(mask_u8, kernel, iterations=int(dilate_iterations))
    if blur_kernel[0] > 0 and blur_kernel[1] > 0:
        mask_u8 = cv2.GaussianBlur(mask_u8, tuple(blur_kernel), 0)

    keep = mask_u8.astype(np.float32) / 255.0  # 1=keep image, 0=fill mean

    img01 = img_rgb_u8.astype(np.float32) / 255.0
    mean = mean_rgb01[None, None, :]

    filled01 = img01 * keep[:, :, None] + mean * (1.0 - keep[:, :, None])
    return np.clip(filled01, 0.0, 1.0)


def compute_global_mean_rgb01(
    entries: Dict[str, Dict[str, Any]],
    processed_base: Path,
    ar_images_base: Optional[Path],
    ar_masks_base: Optional[Path],
    images_base: Optional[Path],
    masks_base: Optional[Path],
    image_field_override: str,
    mask_field_override: str,
    mask_foreground: str,
    mean_region: str,
    sample_n: int,
    source: str = "auto") -> Tuple[np.ndarray, int]:
    """
    Compute ONE RGB mean vector in [0,1].

    mean_region:
      - background: mask<=127 pixels (after optional inversion to ensure foreground is white)
      - foreground: mask>127 pixels
      - all: entire image

    Returns: (mean_rgb01, used_count)
    """
    rids = list(entries.keys())
    if sample_n and sample_n > 0 and len(rids) > sample_n:
        rids = random.sample(rids, sample_n)

    per_image_means = []
    used = 0
    skipped = 0

    for rid in tqdm(rids, desc=f"Computing global mean ({mean_region})", ncols=100, mininterval=0.5):
        e = entries[rid]

        img_path, mask_path, _, _ = pick_paths(
            e,
            processed_base=processed_base,
            ar_images_base=ar_images_base,
            ar_masks_base=ar_masks_base,
            images_base=images_base,
            masks_base=masks_base,
            image_field_override=image_field_override,
            mask_field_override=mask_field_override,
            source=source,
        )


        if img_path is None or not img_path.exists():
            skipped += 1
            continue

        img = to_rgb(imread(str(img_path))).astype(np.float32)
        if img.ndim != 3 or img.shape[2] != 3:
            skipped += 1
            continue

        if mean_region == "all":
            per_image_means.append(img.reshape(-1, 3).mean(axis=0))
            used += 1
            continue

        # background/foreground mean needs mask
        if mask_path is None or not mask_path.exists():
            skipped += 1
            continue

        mask = imread(str(mask_path))
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        mask = mask.astype(np.uint8)

        # Ensure foreground is white before selecting background/foreground regions
        if mask_foreground == "black":
            mask = 255 - mask

        if mean_region == "foreground":
            sel = mask > 127
        else:  # background
            sel = mask <= 127

        if sel.sum() == 0:
            skipped += 1
            continue

        pixels = img[sel]  # Nx3
        per_image_means.append(pixels.mean(axis=0))
        used += 1

    logging.info("Mean computation: used=%d skipped=%d (sample_n=%d)", used, skipped, sample_n)

    if used == 0:
        raise RuntimeError("Could not compute global mean: no valid images/masks contributed.")

    mean_u8 = np.mean(np.stack(per_image_means, axis=0), axis=0)  # RGB in 0..255-ish
    mean01 = np.clip(mean_u8 / 255.0, 0.0, 1.0).astype(np.float32)
    logging.info("Computed global mean RGB (0..1): %s", mean01)
    return mean01, used


# -------------------------
# Main
# -------------------------
def main() -> None:
    args = parse_args()
    for key, value in args.__dict__.items():
        logging.info(f"{key}: {value}")
    args.out_images_dir.mkdir(parents=True, exist_ok=True)

    mapping = json.loads(args.image_mapping_json.read_text())
    entries: Dict[str, Dict[str, Any]] = mapping.get("entries", {})
    if not isinstance(entries, dict) or not entries:
        raise RuntimeError("image-mapping-json missing or empty 'entries' dict")

    processed_base = Path(mapping.get("_processed_base_folder", args.image_mapping_json.parent))
    ar_images_base = None
    if "_ar_images_base_folder" in mapping:
        try:
            cand = Path(mapping["_ar_images_base_folder"])
            ar_images_base = cand if cand.exists() else None
        except Exception:
            ar_images_base = None

    ar_masks_base = None
    if "_ar_masks_base_folder" in mapping:
        try:
            cand = Path(mapping["_ar_masks_base_folder"])
            ar_masks_base = cand if cand.exists() else None
        except Exception:
            ar_masks_base = None

    # Decide mean
    computed_used = 0
    mean_source = ""
    if args.global_mean_npy is not None:
        mean_rgb01 = load_global_mean_rgb01(args.global_mean_npy, args.loaded_mean_scale)
        mean_source = f"loaded:{args.global_mean_npy}"
    else:
        mean_rgb01, computed_used = compute_global_mean_rgb01(
            entries=entries,
            processed_base=processed_base,
            ar_images_base=ar_images_base,
            ar_masks_base=ar_masks_base,
            images_base=args.images_base,
            masks_base=args.masks_base,
            image_field_override=args.image_field.strip(),
            mask_field_override=args.mask_field.strip(),
            mask_foreground=args.mask_foreground,
            mean_region=args.mean_region,
            sample_n=int(args.mean_sample),
            source=args.source,
        )

        mean_source = f"computed:{args.mean_region}"
        if args.save_computed_mean:
            mean_path = args.out_images_dir / "global_mean.npy"
            np.save(mean_path, mean_rgb01)
            logging.info("Saved computed mean to: %s", mean_path)

    # Process entries
    record_ids = list(entries.keys())
    if args.smoke and args.smoke > 0:
        record_ids = record_ids[: args.smoke]

    processed = 0
    failed = 0
    skipped_no_mask = 0
    skipped_missing_files = 0
    skipped_exists = 0

    out_key = (
    "clipped_meanfill_ar" if args.source == "ar"
    else "clipped_meanfill_std" if args.source == "std"
    else "clipped_meanfill_auto"
    )

    for rid in tqdm(record_ids, desc="Mean-fill clip", ncols=100, mininterval=0.5):
        e = entries[rid]
        try:
            main_id = e.get("main_id") or rid
            stem = safe_stem(str(main_id))
            out_img = args.out_images_dir / f"{stem}_{out_key}_filled.png"


            img_path, mask_path, used_img_field, used_mask_field = pick_paths(
                e,
                processed_base=processed_base,
                ar_images_base=ar_images_base,
                ar_masks_base=ar_masks_base,
                images_base=args.images_base,
                masks_base=args.masks_base,
                image_field_override=args.image_field.strip(),
                mask_field_override=args.mask_field.strip(),
                source=args.source,
            )


            if mask_path is None:
                if args.require_mask:
                    skipped_no_mask += 1
                    continue
                skipped_no_mask += 1
                continue

            if img_path is None or not img_path.exists() or not mask_path.exists():
                skipped_missing_files += 1
                continue

            if out_img.exists() and not args.overwrite:
                skipped_exists += 1
                e[out_key] = {
                    "cm_image_abs": str(out_img),
                    "cm_image": str(out_img.relative_to(args.out_images_dir)),
                    "cm_source_image_field": used_img_field,
                    "cm_source_mask_field": used_mask_field,
                }

                continue

            img = to_rgb(imread(str(img_path))).astype(np.uint8)
            mask = imread(str(mask_path))
            if mask.ndim == 3:
                mask = mask[:, :, 0]
            mask_u8 = mask.astype(np.uint8)

            filled01 = apply_mean_fill(
                img_rgb_u8=img,
                mask_u8=mask_u8,
                mean_rgb01=mean_rgb01,
                blur_kernel=tuple(args.blur_kernel),
                dilate_kernel=tuple(args.dilate_kernel),
                dilate_iterations=int(args.dilate_iterations),
                mask_foreground=args.mask_foreground,
            )

            imsave(str(out_img), (filled01 * 255.0).astype(np.uint8), check_contrast=False)

            e[out_key] = {
                "cm_image_abs": str(out_img),
                "cm_image": str(out_img.relative_to(args.out_images_dir)),
                "cm_source_image_abs": str(img_path),
                "cm_source_mask_abs": str(mask_path),
                "cm_source_image_field": used_img_field,
                "cm_source_mask_field": used_mask_field,
            }

            processed += 1

        except Exception:
            failed += 1
            logging.exception("Failed record_id=%s", rid)

    mapping[out_key] = {
        "directory_meta": {
            "_source_mapping": str(args.image_mapping_json),
            "_processed_base_folder": str(processed_base),
            "_ar_images_base_folder": str(ar_images_base) if ar_images_base is not None else None,
            "_clipped_meanfill_images_base_folder": str(args.out_images_dir),
        },
        "params": {
            "mean_source": mean_source,
            "global_mean_rgb01": [float(x) for x in mean_rgb01],
            "computed_mean": bool(args.compute_mean) and args.global_mean_npy is None,
            "computed_mean_region": args.mean_region if args.global_mean_npy is None else None,
            "computed_mean_sample": int(args.mean_sample) if args.global_mean_npy is None else None,
            "computed_mean_used_images": int(computed_used) if args.global_mean_npy is None else None,
            "save_computed_mean": bool(args.save_computed_mean),
            "loaded_mean_scale": float(args.loaded_mean_scale) if args.global_mean_npy is not None else None,
            "blur_kernel": list(args.blur_kernel),
            "dilate_kernel": list(args.dilate_kernel),
            "dilate_iterations": int(args.dilate_iterations),
            "mask_foreground": args.mask_foreground,
            "require_mask": bool(args.require_mask),
            "image_field_override": args.image_field,
            "mask_field_override": args.mask_field,
            "source": args.source,
        },
        "stats": {
            "seen": len(record_ids),
            "processed": processed,
            "skipped_exists": skipped_exists,
            "skipped_no_mask": skipped_no_mask,
            "skipped_missing_files": skipped_missing_files,
            "failed": failed,
        }
    }

   
    new_json = args.out_mapping_json
    if new_json is None:
        new_json = args.image_mapping_json.parent / (args.image_mapping_json.stem + "_meanfill.json")

    new_json = Path(new_json)
    new_json.write_text(json.dumps(mapping, indent=2))
    logging.info("Wrote mean-fill mapping: %s", new_json)
    logging.info(
        "Done. processed=%d skipped_exists=%d skipped_no_mask=%d skipped_missing_files=%d failed=%d",
        processed,
        skipped_exists,
        skipped_no_mask,
        skipped_missing_files,
        failed,
    )


if __name__ == "__main__":
    main()