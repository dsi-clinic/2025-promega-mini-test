"""
Run MMSeg inference and append mask paths into the *same* processed image mapping JSON.

This matches the old workflow:
- Choose a model (early or late)
- Optionally filter which days to run
- Write masks to an output folder
- Append mask_path into the JSON

Input JSON expected shape:
{
  "_base_folder": "/path/to/std_512x384/images",
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

# from __future__ import annotations

import argparse
import dataclasses
import datetime
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Set, Tuple, get_args, get_origin

import cv2  # type: ignore
import numpy as np
#import tifffile  # type: ignore
import torch  # type: ignore
from tqdm import tqdm

try:
    from mmseg.apis import init_model, inference_model  # type: ignore
    from mmengine.model.utils import revert_sync_batchnorm  # type: ignore
except Exception as e:
    raise SystemExit(
        "Failed to import MMSeg. Activate your mmseg env.\n"
        f"Import error: {e}"
    )


logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(format='%(asctime)s,%(msecs)d %(module)s:%(lineno)d %(levelname)s %(message)s',
                    datefmt='%Y-%m-%dT%H:%M:%S',
                    level=logging.INFO)


EXPECTED_RECORDS_NUM = 5168

@dataclasses.dataclass
class Config:
    image_mapping_json: Path = dataclasses.field(metadata={
        "help": "Processed image mapping JSON (std_512x384). This file will be UPDATED in-place."
    })
    out_dir: Path = dataclasses.field(metadata={
        "help": "Output dir for masks (writes to out-dir/masks)."
    })
    config: Path = dataclasses.field(metadata={
        "help": "MMSeg config path."
    })
    checkpoint: Path = dataclasses.field(metadata={
        "help": "MMSeg checkpoint path."
    })
    model_type: str = dataclasses.field(metadata={
        "help": "Which internal model preset to use.",
        "choices": ["early", "late"] # model paths, early days 3-10, late 13-30
    })
    days: Optional[Set[str]] = dataclasses.field(default=None, metadata={
        "help": "Comma-separated dayIDs to run (e.g. Dy03,Dy06,Dy08,Dy10). If omitted, runs all days."
    })
    overwrite: bool = dataclasses.field(default=False, metadata={
        "help": "Overwrite existing mask files."
    })
    dry_run: bool = dataclasses.field(default=False, metadata={
        "help": "No inference; just validate + report counts."
    })
    smoke: Optional[int] = dataclasses.field(default=None, metadata={
        "help": "Limit to N records for quick test."
    })
    write_collage: bool = dataclasses.field(default=False, metadata={
        "help": "Write inference_collage.png for quick QC."
    })
    collage_n: int = dataclasses.field(default=10, metadata={
        "help": "Number of samples in collage if enabled."
    })
    seed: int = dataclasses.field(default=1, metadata={
        "help": "Random seed for reproducibility."
    })
    mask_ext: str = dataclasses.field(default="png", metadata={
        "help": "Mask file extension.",
        "choices": ["png", "tif"]
    })
    mask_suffix: str = dataclasses.field(default="_mask", metadata={
        "help": "Suffix appended to mask filename stem."
    })

    def __post_init__(self):
        if self.model_type not in ["early", "late"]:
            raise ValueError(f"Invalid model type: {self.model_type}")
        if self.mask_ext not in ["png", "tif"]:
            raise ValueError(f"Invalid mask extension: {self.mask_ext}")

        if not self.out_dir.exists():
            self.out_dir.mkdir(parents=True, exist_ok=True)

        if not self.image_mapping_json.exists():
            raise ValueError(f"Mapping JSON does not exist: {self.image_mapping_json}")
        if not self.config.exists():
            raise ValueError(f"Config does not exist: {self.config}")
        if not self.checkpoint.exists():
            raise ValueError(f"Checkpoint does not exist: {self.checkpoint}")


def get_args():
    arg_parser = create_args()
    args = arg_parser.parse_args()
    args_dict = vars(args)
    # Convert days from string to Optional[Set[str]]
    if 'days' in args_dict:
        args_dict['days'] = parse_days(args_dict['days'])  # Handles None correctly
    cfg = Config(**args_dict)
    return cfg


def create_args() -> argparse.ArgumentParser:
    """Create an ArgumentParser from the Config dataclass."""
    parser = argparse.ArgumentParser(description="Run image classifier on organoid images")

    for field in dataclasses.fields(Config):
        # Build argument flag and help message
        flags = [f"--{field.name.replace('_', '-')}"]
        kwargs = {
            "help": field.metadata.get("help", ""),
            "default": field.default
        }

        # Determine argument type
        if field.type == bool:
            kwargs["action"] = "store_true" if field.default is False else "store_false"
        elif field.name == "days":
            # Special handling for days: Optional[Set[str]] - store as string, parse_days will convert it
            kwargs["type"] = str
        elif field.type == str and field.metadata.get("choices"):
            kwargs["choices"] = field.metadata.get("choices")
        else:
            kwargs["type"] = field.type

        parser.add_argument(*flags, **kwargs)

    return parser


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


def load_mapping(mapping_json: Path) -> Tuple[Path, Dict[str, Dict[str, Any]], Dict[str, Any]]:
    data = json.loads(mapping_json.read_text())

    processed_base = data.get("_processed_base_folder")
    if not processed_base:
        raise ValueError(f"Mapping JSON missing _processed_base_folder: {mapping_json}")

    entries = data.get("entries")
    if not isinstance(entries, dict):
        raise ValueError(f"Mapping JSON missing dict entries: {mapping_json}")

    return Path(processed_base), entries, data


def main() -> None:
    start = datetime.datetime.now()
    args = get_args()
    for key, value in vars(args).items():
        logging.info("%s: %s", key, value)

    processed_base, entries, full_json = load_mapping(args.image_mapping_json)
    logging.info("processed_base: %s", processed_base)
    logging.info("loaded entries: %d", len(entries))

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = None
    if not args.dry_run:
        model = init_model(str(args.config), str(args.checkpoint), device=device)
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

    for record_id in tqdm(record_ids, desc="Predicting masks"):
        entry = entries[record_id]
        day = entry.get("dayID")

        if args.days is not None and day not in args.days:
            skipped_day += 1
            continue

        try:
            img_path_str = entry.get("processed_image")
            if not img_path_str:
                raise KeyError("Entry missing processed_image")

            img_path = Path(img_path_str)
            if not img_path.exists():
                raise FileNotFoundError(str(img_path))

            main_id = entry.get("main_id") or record_id
            mask_stem = safe_stem(main_id) + args.mask_suffix
            mask_path = args.out_dir / f"{mask_stem}.{args.mask_ext}"

            if args.dry_run:
                # Just pretend we wrote it; do not modify JSON
                processed += 1
                continue

            if mask_path.exists() and not args.overwrite:
                skipped_exists += 1
                # Still append mask path into JSON if you want it “filled”
                entry["predicted_mask_path"] = str(mask_path)
                entry["predicted_mask_model"] = args.model_type
                processed += 1  # count toward expected total for final check
                continue

            # Run inference
            result = inference_model(model, img_path_str)
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
                tifffile.imwrite(str(mask_path), pred.astype(np.uint8))

            # Append into the SAME JSON entry (your requested behavior)
            entry["predicted_mask_path"] = str(mask_path)
            entry["predicted_mask_model"] = args.model_type

            processed += 1

            # Optional collage row
            if args.write_collage and record_id in sample_ids:
                img = cv2.imread(img_path_str, cv2.IMREAD_COLOR)
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

    # Assert completed records number
    # if days filter used, processed will be smaller and that's fine
    if args.days is None and processed != EXPECTED_RECORDS_NUM:
        raise ValueError(f"Expected {EXPECTED_RECORDS_NUM} records, got {processed}")


    # Write back the updated mapping JSON (to a new file)
    new_json = Path(args.image_mapping_json.parent / (args.image_mapping_json.stem + "_predicted.json"))
    new_json.write_text(json.dumps(full_json, indent=2))
    logging.info("Updated mapping JSON: %s", new_json)

    # Write collage
    if args.write_collage and collage_rows:
        collage_path = args.out_dir / "inference_collage.png"
        cv2.imwrite(str(collage_path), np.vstack(collage_rows))
        logging.info("Wrote collage: %s", collage_path)

    logging.info("Done. processed=%d skipped_day=%d skipped_exists=%d failed=%d",
                 processed, skipped_day, skipped_exists, failed)
    end = datetime.datetime.now()
    logging.info("Elapsed time: %s", end - start)


if __name__ == "__main__":
    main()
