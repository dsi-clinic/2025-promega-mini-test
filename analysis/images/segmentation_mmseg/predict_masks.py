"""
Run MMSeg inference and append mask paths into the *same* processed image mapping JSON.

This matches the old workflow:
- Choose a model (early or late)
- Optionally filter which days to run
- Write masks to an output folder
- Append mask_path into the JSON

Input JSON expected shape:
{
  "_processed_base_folder": "/path/to/std_512x384/images",
  "entries": {
    "<record_id>": {
      "processed_image": "BA1_...tif",
      "main_id": "BA1 96_1 Dy03 A1",
      "dayID": "Dy03",
      ...
    },
    ...
  }
}
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Set, Tuple

import numpy as np

import torch  # type: ignore
import cv2  # type: ignore

try:
    from mmseg.apis import init_model, inference_model  # type: ignore
    from mmengine.model.utils import revert_sync_batchnorm  # type: ignore
except Exception as e:
    raise SystemExit(
        "Failed to import MMSeg. Activate your mmseg env.\n"
        f"Import error: {e}"
    )

# Internal model presets (same pattern you used before)
#from analysis.images.segmentation_mmseg.mmseg_paths import EARLY_MODEL, LATE_MODEL  # type: ignore


logging.basicConfig(
    format="%(asctime)s,%(msecs)d %(levelname)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    level=logging.INFO,
)

# model paths, early days 3-10, late 13-30
EARLY_MODEL = {
    "config": Path(
        "/net/projects2/promega/data_reorg/data/masks/trained_models/512x384/october_early/early/vis_data/config.py"
    ),
    "checkpoint": Path(
        "/net/projects2/promega/data_reorg/data/masks/trained_models/512x384/october_early/iter_1000.pth"
    ),
}

LATE_MODEL = {
    "config": Path(
        "/net/projects2/promega/data_reorg/data/masks/trained_models/512x384/october_late/late/vis_data/config.py"
    ),
    "checkpoint": Path(
        "/net/projects2/promega/data_reorg/data/masks/trained_models/512x384/october_late/iter_1000.pth"
    ),
}

def safe_stem(s: str) -> str:
    # keep it simple and stable
    return (
        s.replace(" ", "_")
         .replace("/", "_")
         .replace("\\", "_")
         .replace(":", "_")
    )


def parse_days(days_arg: Optional[str]) -> Optional[Set[str]]:
    if not days_arg:
        return None
    days = {d.strip() for d in days_arg.split(",") if d.strip()}
    return days or None


def pick_model_paths(model_type: str, cfg_override: Optional[Path], ckpt_override: Optional[Path]) -> Tuple[Path, Path]:
    # Explicit overrides always win if both provided
    if cfg_override is not None and ckpt_override is not None:
        return cfg_override, ckpt_override

    model_info = EARLY_MODEL if model_type == "early" else LATE_MODEL
    cfg_path = Path(model_info["config"])
    ckpt_path = Path(model_info["checkpoint"])
    return cfg_path, ckpt_path


def load_mapping(mapping_json: Path) -> Tuple[Path, Dict[str, Dict[str, Any]], Dict[str, Any]]:
    data = json.loads(mapping_json.read_text())

    processed_base = data.get("_processed_base_folder")
    if not processed_base:
        raise ValueError(f"Mapping JSON missing _processed_base_folder: {mapping_json}")

    entries = data.get("entries")
    if not isinstance(entries, dict):
        raise ValueError(f"Mapping JSON missing dict entries: {mapping_json}")

    return Path(processed_base), entries, data


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Predict masks with MMSeg and append mask_path into mapping JSON")

    p.add_argument(
        "--image-mapping-json",
        type=Path,
        required=True,
        help="Processed image mapping JSON (std_512x384). This file will be UPDATED in-place.",
    )

    p.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Output dir for masks (writes to out-dir/masks).",
    )

    p.add_argument(
        "--model-type",
        choices=["early", "late"],
        required=True,
        help="Which internal model preset to use.",
    )

    p.add_argument(
        "--days",
        type=str,
        default=None,
        help="Comma-separated dayIDs to run (e.g. Dy03,Dy06,Dy08,Dy10). If omitted, runs all days.",
    )

    p.add_argument("--overwrite", action="store_true", help="Overwrite existing mask files.")
    p.add_argument("--dry-run", action="store_true", help="No inference; just validate + report counts.")
    p.add_argument("--smoke", type=int, default=None, help="Limit to N records for quick test.")

    p.add_argument("--config", type=Path, default=None, help="Override MMSeg config path (optional).")
    p.add_argument("--checkpoint", type=Path, default=None, help="Override MMSeg checkpoint path (optional).")

    p.add_argument("--write-collage", action="store_true", help="Write inference_collage.png for quick QC.")
    p.add_argument("--collage-n", type=int, default=10, help="Number of samples in collage if enabled.")
    p.add_argument("--seed", type=int, default=0)

    # mask file format
    p.add_argument("--mask-ext", choices=["png", "tif"], default="png")
    p.add_argument("--mask-suffix", type=str, default="_mask", help="Suffix appended to mask filename stem.")

    return p.parse_args()


def main() -> None:
    args = get_args()
    allowed_days = parse_days(args.days)

    logging.info("image_mapping_json: %s", args.image_mapping_json)
    logging.info("out_dir: %s", args.out_dir)
    logging.info("model_type: %s", args.model_type)
    logging.info("days filter: %s", sorted(allowed_days) if allowed_days else None)
    logging.info("overwrite: %s", args.overwrite)
    logging.info("dry_run: %s", args.dry_run)
    logging.info("smoke: %s", args.smoke)

    processed_base, entries, full_json = load_mapping(args.image_mapping_json)
    logging.info("processed_base: %s", processed_base)
    logging.info("loaded entries: %d", len(entries))

    cfg_path, ckpt_path = pick_model_paths(args.model_type, args.config, args.checkpoint)
    if not cfg_path.exists():
        raise SystemExit(f"Missing config: {cfg_path}")
    if not ckpt_path.exists():
        raise SystemExit(f"Missing checkpoint: {ckpt_path}")

    out_dir = args.out_dir
    masks_dir = out_dir

    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = None
    if not args.dry_run:
        model = init_model(str(cfg_path), str(ckpt_path), device=device)
        if device == "cpu":
            model = revert_sync_batchnorm(model)
        logging.info("Model loaded on %s", device)
    else:
        logging.info("Dry-run: not loading model (would use %s)", device)

    # Determine iteration list
    record_ids = list(entries.keys())
    if args.smoke is not None and args.smoke > 0:
        record_ids = record_ids[: args.smoke]

    # Collage sampling (deterministic)
    rng = np.random.default_rng(args.seed)
    sample_ids = set()
    if args.write_collage and record_ids:
        n = min(args.collage_n, len(record_ids))
        sample_ids = set(rng.choice(record_ids, size=n, replace=False).tolist())

    collage_rows = []

    processed = 0
    skipped_day = 0
    skipped_exists = 0
    failed = 0

    for record_id in record_ids:
        entry = entries[record_id]
        day = entry.get("dayID")

        if allowed_days is not None and day not in allowed_days:
            skipped_day += 1
            continue

        try:
            rel = entry.get("processed_image")
            if not rel:
                raise KeyError("Entry missing processed_image")

            img_path = processed_base / str(rel)
            if not img_path.exists():
                raise FileNotFoundError(str(img_path))

            main_id = entry.get("main_id") or record_id
            mask_stem = safe_stem(main_id) + args.mask_suffix
            mask_path = masks_dir / f"{mask_stem}.{args.mask_ext}"

            if args.dry_run:
                # Just pretend we wrote it; do not modify JSON
                processed += 1
                continue

            if mask_path.exists() and not args.overwrite:
                skipped_exists += 1
                # Still append mask path into JSON if you want it “filled”
                entry["mask_path"] = str(mask_path)
                entry["mask_model"] = args.model_type
                continue

            # Run inference
            result = inference_model(model, str(img_path))
            pred = result.pred_sem_seg.data.squeeze().cpu().numpy().astype(np.uint8)

            # Save mask
            if args.mask_ext == "png":
                # scale to 0/255 for visualization like older scripts
                out = (pred * 255).astype(np.uint8)
                ok = cv2.imwrite(str(mask_path), out)
                if not ok:
                    raise RuntimeError(f"cv2.imwrite failed: {mask_path}")
            else:
                # tif, save 0/1 mask
                import tifffile  # type: ignore
                tifffile.imwrite(str(mask_path), pred.astype(np.uint8))

            # Append into the SAME JSON entry (your requested behavior)
            entry["mask_path"] = str(mask_path)
            entry["mask_model"] = args.model_type

            processed += 1

            # Optional collage row
            if args.write_collage and record_id in sample_ids:
                img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
                if img is not None:
                    mask_vis = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                    if mask_vis is not None:
                        mask_vis = cv2.cvtColor(mask_vis, cv2.COLOR_GRAY2BGR)
                        # ensure same size
                        if img.shape[:2] != mask_vis.shape[:2]:
                            mask_vis = cv2.resize(mask_vis, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
                        collage_rows.append(np.hstack([img, mask_vis]))

        except Exception:
            failed += 1
            logging.exception("Failed record_id=%s", record_id)
            continue

    if args.dry_run:
        logging.info("Dry-run complete.")
        logging.info("Would process: %d", processed)
        logging.info("Skipped by day: %d", skipped_day)
        logging.info("Would fail: %d", failed)
        return

    # Write back the updated mapping JSON (in place)
    args.image_mapping_json.write_text(json.dumps(full_json, indent=2))
    logging.info("Updated mapping JSON in-place: %s", args.image_mapping_json)

    # Write collage
    if args.write_collage and collage_rows:
        collage_path = out_dir / "inference_collage.png"
        cv2.imwrite(str(collage_path), np.vstack(collage_rows))
        logging.info("Wrote collage: %s", collage_path)

    logging.info("Done. processed=%d skipped_day=%d skipped_exists=%d failed=%d",
                 processed, skipped_day, skipped_exists, failed)


if __name__ == "__main__":
    main()

