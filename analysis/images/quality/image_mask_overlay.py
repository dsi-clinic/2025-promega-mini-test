#!/usr/bin/env python3
import json
from pathlib import Path
import re
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image
from collections import Counter

# ------ repo config ------
try:
    from config import INFER_AUTO_PROCESSED_DIR
except Exception:
    raise RuntimeError("config.INFER_AUTO_PROCESSED_DIR is required")

# ------ helpers ------

def load_json(p: Path):
    with p.open("r") as f:
        return json.load(f)

def save_json(p: Path, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        json.dump(obj, f, indent=2)

def read_image_bgr(p: Path) -> np.ndarray | None:
    """Read as BGR (OpenCV) with PIL fallback."""
    img = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if img is None:
        try:
            img = np.array(Image.open(p).convert("RGB"))[:, :, ::-1]  # RGB->BGR
        except Exception:
            return None
    return img

def read_mask_bin(p: Path) -> np.ndarray | None:
    """Read mask as binary uint8 (0/1), with PIL fallback."""
    m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if m is None:
        try:
            m = np.array(Image.open(p).convert("L"))
        except Exception:
            return None
    return (m > 0).astype(np.uint8)


def ensure_gray_binary(mask: np.ndarray) -> np.ndarray | None:
    if mask is None:
        return None
    if mask.ndim == 3:
        mask = mask[..., 0]
    return (mask > 0).astype(np.uint8)

def derive_overlay_path(mask_path_str: str) -> Path:
    """
    predictions/<batch>/<day>/image_mask_overlays/<basename>_overlay.png
    mirroring 'predicted_masks' placement.
    """
    mp = Path(mask_path_str)
    mask_dir = mp.parent
    day_dir = mask_dir.parent               # e.g., .../day28
    overlays_dir = day_dir / "image_mask_overlays"
    stem = mp.stem  # e.g., BA2_96_1_Dy28_B9_predmask
    out_stem = re.sub(r"_predmask$", "", stem, flags=re.IGNORECASE) + "_overlay"
    return overlays_dir / f"{out_stem}.png"

def draw_outline_overlay(img_bgr: np.ndarray, mask_bin: np.ndarray, color=(0,255,0), thickness=2) -> np.ndarray:
    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out = img_bgr.copy()
    if contours:
        cv2.drawContours(out, contours, contourIdx=-1, color=color, thickness=thickness, lineType=cv2.LINE_AA)
    return out

def process_mapping_json(mapping_path: Path, overwrite: bool = False, strict: bool = False, sample_limit: int | None = 10) -> dict:
    mapping = load_json(mapping_path)

    updated = False
    made = 0
    skipped = 0
    missing = 0

    # strict accounting
    pairs_total = 0
    overlays_on_disk = 0
    overlay_recorded = 0

    # detailed reasons
    reason_counts = {
        "no_json_path": 0,
        "missing_img_file": 0,
        "missing_mask_file": 0,
        "decode_img": 0,
        "decode_mask": 0,
        "write_fail": 0,
    }

    missing_pairs = []          # (key, reason)
    overlays_missing_list = []  # (key, expected_overlay_path)

    for k, rec in mapping.items():
        if not isinstance(rec, dict):
            continue

        img_path = rec.get("img_path")
        mask_path = rec.get("mask_path")

        # 1) missing paths in JSON
        if not img_path or not mask_path:
            missing += 1
            reason_counts["no_json_path"] += 1
            missing_pairs.append((k, "no img_path or mask_path in JSON"))
            continue

        img_p = Path(img_path)
        mask_p = Path(mask_path)

        # 2) files missing on disk
        miss_img = not img_p.exists()
        miss_msk = not mask_p.exists()
        if miss_img or miss_msk:
            missing += 1
            if miss_img: reason_counts["missing_img_file"] += 1
            if miss_msk: reason_counts["missing_mask_file"] += 1
            reason = "missing file(s): "
            if miss_img: reason += f"[img:{img_p}] "
            if miss_msk: reason += f"[mask:{mask_p}] "
            missing_pairs.append((k, reason.strip()))
            continue

        # 3) decode
        img = read_image_bgr(img_p)
        mask_bin = read_mask_bin(mask_p)
        if img is None or mask_bin is None:
            missing += 1
            if img is None: reason_counts["decode_img"] += 1
            if mask_bin is None: reason_counts["decode_mask"] += 1
            missing_pairs.append((k, f"decode failed: img({img is None}), mask({mask_bin is None})"))
            continue

        # valid pair
        pairs_total += 1
        out_path = derive_overlay_path(mask_path)

        if out_path.exists() and not overwrite:
            if rec.get("overlay_path") != str(out_path):
                rec["overlay_path"] = str(out_path)
                updated = True
            skipped += 1
        else:
            overlay = draw_outline_overlay(img, mask_bin, color=(0,255,0), thickness=2)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            ok = cv2.imwrite(str(out_path), overlay)
            if not ok:
                missing += 1
                reason_counts["write_fail"] += 1
                missing_pairs.append((k, f"write failed: {out_path}"))
                continue
            rec["overlay_path"] = str(out_path)
            updated = True
            made += 1

        if out_path.exists():
            overlays_on_disk += 1
        else:
            overlays_missing_list.append((k, str(out_path)))
        if rec.get("overlay_path") == str(out_path):
            overlay_recorded += 1

    if updated:
        backup = mapping_path.with_suffix(mapping_path.suffix + ".bak")
        save_json(backup, mapping)
        save_json(mapping_path, mapping)

    # strict check per-file
    if strict:
        assert overlays_on_disk == pairs_total, (
            f"[STRICT] {mapping_path.name}: overlays_on_disk({overlays_on_disk}) != pairs_total({pairs_total})"
        )

    # sampling control for summaries vs. full report
    if sample_limit is None:
        miss_out = missing_pairs
        ovl_out  = overlays_missing_list
    else:
        miss_out = missing_pairs[:sample_limit]
        ovl_out  = overlays_missing_list[:sample_limit]

    return {
        "file": str(mapping_path),
        "made": made,
        "skipped_existing": skipped,
        "missing_or_failed": missing,
        "updated_json": updated,
        "pairs_total": pairs_total,
        "overlays_on_disk": overlays_on_disk,
        "overlay_recorded": overlay_recorded,
        "reason_counts": reason_counts,           # <<— needed in main
        "missing_pairs": miss_out,
        "overlays_missing": ovl_out,
    }

# ------ main ------

def main(overwrite=False, strict=False):
    root = Path(INFER_AUTO_PROCESSED_DIR)
    if not root.exists():
        raise SystemExit(f"INFER_AUTO_PROCESSED_DIR does not exist: {root}")

    mapping_files = list(root.rglob("image_mapping*_processed.json"))
    print(f"Found {len(mapping_files)} processed mapping JSONs under {root}")
    totals_reasons = Counter()
    totals = {
        "made": 0,
        "skipped_existing": 0,
        "missing_or_failed": 0,
        "files_updated": 0,
        "pairs_total": 0,
        "overlays_on_disk": 0,
        "overlay_recorded": 0,
    }

    any_missing_samples = []
    any_overlay_missing_samples = []

    for mp in tqdm(mapping_files, desc="Overlays"):
        stats = process_mapping_json(mp, overwrite=overwrite, strict=strict)
        totals["made"] += stats["made"]
        totals["skipped_existing"] += stats["skipped_existing"]
        totals["missing_or_failed"] += stats["missing_or_failed"]
        totals["pairs_total"] += stats["pairs_total"]
        totals["overlays_on_disk"] += stats["overlays_on_disk"]
        totals["overlay_recorded"] += stats["overlay_recorded"]
        if stats["updated_json"]:
            totals["files_updated"] += 1
        totals_reasons.update(stats["reason_counts"])

        if stats["missing_pairs"]:
            any_missing_samples.extend([(stats["file"], *x) for x in stats["missing_pairs"]])
        if stats["overlays_missing"]:
            any_overlay_missing_samples.extend([(stats["file"], *x) for x in stats["overlays_missing"]])

    print("\nSummary")
    print(f"  overlays created        : {totals['made']}")
    print(f"  overlays already exist  : {totals['skipped_existing']}")
    print(f"  missing/failed pairs    : {totals['missing_or_failed']}")
    print(f"  mapping files updated   : {totals['files_updated']}")
    print(f"  valid (img+mask) pairs  : {totals['pairs_total']}")
    print(f"  overlays on disk        : {totals['overlays_on_disk']}")
    print(f"  overlay_path recorded   : {totals['overlay_recorded']}")
    print("\nBreakdown of missing/failed pairs:")
    for k, v in totals_reasons.items():
        print(f"  {k:>18}: {v}")
    # after the summary prints
    report_path = Path("analysis/images/quality/overlay_decode_failures.json")
    full_report = []
    for mp in tqdm(mapping_files, desc="Collect failures"):
        s = process_mapping_json(mp, overwrite=False, strict=False, sample_limit=None)
        full_report.extend([(s["file"],) + t for t in s.get("missing_pairs", [])])
    with report_path.open("w") as f:
        json.dump(full_report, f, indent=2)
    print(f"Wrote failure report to: {report_path}")



    # global strict check: overlays_on_disk == pairs_total
    if strict:
        assert totals["overlays_on_disk"] == totals["pairs_total"], (
            f"[STRICT] GLOBAL: overlays_on_disk({totals['overlays_on_disk']}) != pairs_total({totals['pairs_total']})"
        )
    else:
        if totals["overlays_on_disk"] != totals["pairs_total"]:
            print("\n Mismatch detected (non-strict mode). Examples:")
            for f, key, why in any_missing_samples[:5]:
                print(f"  - missing pair in {f}: {key} — {why}")
            for f, key, exp in any_overlay_missing_samples[:5]:
                print(f"  - overlay missing in {f}: {key} expected {exp}")
    

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(
        description="Build outline overlays for all processed image/mask pairs and update mapping JSONs with overlay_path."
    )
    ap.add_argument("--overwrite", action="store_true", help="Rebuild overlays even if they already exist.")
    ap.add_argument("--strict", action="store_true", help="Assert overlays_on_disk == valid (img+mask) pairs.")
    args = ap.parse_args()
    main(overwrite=args.overwrite, strict=args.strict)