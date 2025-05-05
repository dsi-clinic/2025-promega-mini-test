# ----------------------------------------
# quick_mask_metrics.py
# ----------------------------------------
import json, csv, sys
from pathlib import Path
import numpy as np
import cv2
from skimage.measure import label, regionprops_table
import pandas as pd

def measure_mask(mask_path, px_um):
    """Return a dict of props (largest object) in micron units."""
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise IOError(f"Cannot read {mask_path}")
    binmask = mask > 127            # threshold 0/255 → bool
    lab = label(binmask)
    if lab.max() == 0:               # no object
        return None
    props = regionprops_table(
        lab,
        properties=("area", "perimeter", "major_axis_length",
                    "minor_axis_length", "eccentricity")
    )
    # keep the largest region (index of max area)
    idx = int(np.argmax(props["area"]))
    μm_px = px_um                     # microns per network pixel
    out = {
        "area_μm2"      : props["area"][idx]            * (μm_px**2),
        "perimeter_μm"  : props["perimeter"][idx]       * μm_px,
        "major_ax_μm"   : props["major_axis_length"][idx]* μm_px,
        "minor_ax_μm"   : props["minor_axis_length"][idx]* μm_px,
        "eccentricity"  : float(props["eccentricity"][idx]),
    }
    return out

def process_mapping(json_path):
    rows = []
    with open(json_path) as f:
        mapping = json.load(f)
    for img_id, info in mapping.items():
        mask_path = Path(info["mask_path"])
        if not mask_path.exists():
            print("  ⚠️  missing mask:", mask_path.name)
            continue
        px_um = info["final_um_per_px"]       # microns per *network* pixel
        metrics = measure_mask(mask_path, px_um)
        if metrics:
            metrics.update({"img_id": img_id,
                            "batch_day": mask_path.parts[-4]})  # e.g. batch2_96_1/day03
            rows.append(metrics)
    return rows


# ------------- CLI: glob every mapping under a root -------------
"""
Usage:
python quick_mask_metrics.py /net/projects2/promega/data-analysis/output/processed_dataset_256x192
"""
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python quick_mask_metrics.py <root_processed_dataset_dir>")
        sys.exit(1)

    root = Path(sys.argv[1])
    if not root.is_dir():
        print("Given path is not a directory")
        sys.exit(1)

    json_paths = sorted(root.rglob("image_mapping_*_processed.json"))
    if not json_paths:
        print("No mapping JSONs found under", root)
        sys.exit(0)

    all_rows = []
    for jp in json_paths:
        print("»", jp.relative_to(root))
        all_rows += process_mapping(jp)

    df = pd.DataFrame(all_rows)
    if df.empty:
        print("No masks found!")
        sys.exit(0)

    # quick summary
    print("\n── summary (largest object per image) ──")
    print(df.describe().loc[["mean","std","min","max"]][
            ["area_μm2","major_ax_μm","eccentricity"]])

    out_csv = root / "mask_metrics.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nsaved full table →  {out_csv}")
