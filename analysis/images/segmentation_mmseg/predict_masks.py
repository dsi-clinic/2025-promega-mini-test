# predict_new_batch.py  (trimmed to show key edits)

import torch
from pathlib import Path
import cv2, numpy as np, warnings, json, random, argparse, sys

HERE = Path(__file__).resolve()
for p in HERE.parents:
    if (p / "config.py").exists() and (p / ".env").exists():
        sys.path.insert(0, str(p))  # repo root
        break
else:
    raise RuntimeError("Could not locate repo root containing config.py and .env")

# Root paths (canonical dirs)
from config import INFER_AUTO_PROCESSED_DIR  # where the per-day mapping JSONs live

# mmseg-specific model locations / output base
from analysis.images.segmentation_mmseg.mmseg_paths import (
    EARLY_MODEL, LATE_MODEL, OUTPUT_MASKS_BASE_DIR,
)


warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

try:
    from mmseg.apis import init_model, inference_model
    from mmengine.model.utils import revert_sync_batchnorm
    print("MMSegmentation API imported successfully.")
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

NUM_SAMPLES_FOR_COLLAGE = 10

def get_mapping_paths(batch_number, day_number=30):
    """Find the JSON(s) created by the infer-prep step (auto_processed)."""
    day_str = f"{day_number:02d}"
    base = Path(INFER_AUTO_PROCESSED_DIR)

    paths = []
    if batch_number == 2:
        for part in ["96_1", "96_2"]:
            batch_str = f"ba{batch_number}{part}_Dy{day_str}"
            paths.append(base / batch_str / f"image_mapping_{batch_str}_processed.json")
    else:
        batch_str = f"ba{batch_number}96_1_Dy{day_str}"
        paths.append(base / batch_str / f"image_mapping_{batch_str}_processed.json")
    return paths


def run_inference(batch_number, day_number=30, model_type="early", overwrite=False, dry_run=False, smoke=None):
    day_str = f"{day_number:02d}"
    mapping_paths = get_mapping_paths(batch_number, day_number)

    model_info = EARLY_MODEL if model_type == "early" else LATE_MODEL
    cfg_path = Path(model_info["config"])
    ckpt_path = Path(model_info["checkpoint"])

    # Validate model files up front
    if not cfg_path.exists():
        raise SystemExit(f"Missing config: {cfg_path}")
    if not ckpt_path.exists():
        raise SystemExit(f"Missing checkpoint: {ckpt_path}")

    # Init model once
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = None if dry_run else init_model(str(cfg_path), str(ckpt_path), device=device)
    if not dry_run and device == 'cpu':
        model = revert_sync_batchnorm(model)
    print(f"Model {'validated' if dry_run else 'loaded'} on {device}")

    total_processed = total_failed = 0

    for json_mapping_path in mapping_paths:
        print(f"Checking for file: {json_mapping_path}  ->  {json_mapping_path.exists()}")
        if not json_mapping_path.exists():
            print(f"Warning: Preprocessed JSON not found at {json_mapping_path}")
            continue

        # Determine output dir
        if batch_number == 2:
            part = "96_1" if "96_1" in str(json_mapping_path) else "96_2"
            output_dir = Path(OUTPUT_MASKS_BASE_DIR) / f"batch{batch_number}_{part}" / f"day{day_str}"
        else:
            output_dir = Path(OUTPUT_MASKS_BASE_DIR) / f"batch{batch_number}" / f"day{day_str}"
        masks_dir = output_dir / "predicted_masks"
        if not dry_run:
            masks_dir.mkdir(parents=True, exist_ok=True)

        collage_path = output_dir / (
            f"inference_collage_batch{batch_number}"
            f"{('_part'+part) if batch_number==2 else ''}_day{day_str}.png"
        )

        with open(json_mapping_path, 'r') as f:
            batch_mapping = json.load(f)
        print(f"Loaded {len(batch_mapping)} entries from {json_mapping_path.name}")

        # Pick sample ids once per mapping
        all_ids = list(batch_mapping)
        sample_ids = set(random.sample(all_ids, min(NUM_SAMPLES_FOR_COLLAGE, len(all_ids)))) if all_ids else set()

        processed = failed = 0
        collage_pairs, img_h, img_w = [], None, None
        max_items = smoke if (smoke is not None and smoke > 0) else len(batch_mapping)
        iter_items = list(batch_mapping.items())[:max_items]

        for img_id, img_info in iter_items:
            img_path = Path(img_info['img_path'])
            if not img_path.exists():
                failed += 1
                continue

            if dry_run:
                # Just pretend success; record intended output
                mask_path = masks_dir / f"{img_path.stem}_predmask.png"
                batch_mapping[img_id]['mask_path'] = str(mask_path)
                processed += 1
                continue

            try:
                result = inference_model(model, str(img_path))
                pred_mask = (result.pred_sem_seg.data.squeeze().cpu().numpy() * 255).astype(np.uint8)

                mask_path = masks_dir / f"{img_path.stem}_predmask.png"
                if not overwrite and mask_path.exists():
                    processed += 1
                    batch_mapping[img_id]['mask_path'] = str(mask_path)
                    continue

                cv2.imwrite(str(mask_path), pred_mask)
                processed += 1
                batch_mapping[img_id]['mask_path'] = str(mask_path)

                if img_id in sample_ids:
                    img = cv2.imread(str(img_path))
                    if img is None:
                        continue
                    if img_h is None:
                        img_h, img_w = img.shape[:2]
                    img_resized = cv2.resize(img, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
                    mask_vis   = cv2.resize(pred_mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
                    mask_vis   = cv2.cvtColor(mask_vis, cv2.COLOR_GRAY2BGR)
                    collage_pairs.append(np.hstack((img_resized, mask_vis)))

            except Exception:
                failed += 1

        if not dry_run and collage_pairs:
            cv2.imwrite(str(collage_path), np.vstack(collage_pairs))
            print(f"Collage saved to {collage_path}")

        print(f"Finished part. Processed: {processed}, Failed: {failed}")
        total_processed += processed
        total_failed   += failed

        if not dry_run:
            with open(json_mapping_path, 'w') as f:
                json.dump(batch_mapping, f, indent=2)
            print(f"Updated mapping with mask paths: {json_mapping_path}")

    print(f"\nTotal processed: {total_processed}, Total failed: {total_failed}")
    return total_processed

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_type', choices=['early', 'late'], required=True)
    ap.add_argument('--batches', type=lambda s: [int(x) for x in s.split(',')], required=True)
    ap.add_argument('--days',    type=lambda s: [int(x) for x in s.split(',')], required=True)
    ap.add_argument('--overwrite', action='store_true')
    ap.add_argument('--dry-run', action='store_true', help='Validate configs/paths without running inference')
    ap.add_argument('--smoke', type=int, default=None, help='Limit to N images per mapping for a quick test')
    args = ap.parse_args()

    for batch in args.batches:
        for day in args.days:
            print(f"\n{'='*40}\nProcessing Batch {batch}, Day {day}\n{'='*40}")
            run_inference(batch, day,
                          model_type=args.model_type,
                          overwrite=args.overwrite,
                          dry_run=args.dry_run,
                          smoke=args.smoke)
    print("\nAll done.")
