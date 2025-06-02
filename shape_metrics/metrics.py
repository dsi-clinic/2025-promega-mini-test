import json
import math
import cv2
import numpy as np
import pandas as pd

from pathlib import Path
from skimage import measure, morphology
from scipy.stats import entropy as shannon_entropy
from skimage.measure import find_contours
from pyefd import elliptic_fourier_descriptors, reconstruct_contour
from concurrent.futures import ProcessPoolExecutor


def classic_shape_metrics(mask_bin: np.ndarray, μm_per_px: float):
    regions = measure.regionprops(mask_bin.astype(int))
    if not regions:
        return None
    props = max(regions, key=lambda r: r.area)

    area_px = props.area
    peri_px = props.perimeter
    major = props.major_axis_length
    minor = props.minor_axis_length

    hull = morphology.convex_hull_image(mask_bin)
    peri_ch = measure.perimeter(hull)
    area_ch = hull.sum()

    return {
        "area_um2": area_px * μm_per_px**2,
        "perimeter_um": peri_px * μm_per_px,
        "major_um": major * μm_per_px,
        "minor_um": minor * μm_per_px,
        "circularity": 4 * math.pi * area_px / (peri_px**2),
        "solidity": area_px / area_ch,
        "convexity": peri_ch / peri_px,
        "aspect_ratio": major / minor,
    }


def efa_timbre_metrics(mask_bin: np.ndarray, μm_per_px: float, n_harmonics: int = 10):
    contours = find_contours(mask_bin.astype(int), level=0.5)
    if not contours:
        return None
    contour = max(contours, key=len)
    xy = np.fliplr(contour)

    coeffs = elliptic_fourier_descriptors(xy, order=n_harmonics, normalize=True)

    orig_area = mask_bin.sum() * (μm_per_px**2)
    xor_curve = np.zeros(n_harmonics + 1, float)
    xor_curve[0] = orig_area

    for k in range(1, n_harmonics + 1):
        rec_xy = reconstruct_contour(coeffs[:k], locus=(0, 0), num_points=xy.shape[0])
        pts = np.round(rec_xy).astype(np.int32)

        ui = np.zeros(mask_bin.shape, np.uint8)
        cv2.fillPoly(ui, [pts], 1)
        filled = ui.astype(bool)

        xor_curve[k] = np.logical_xor(mask_bin, filled).sum() * (μm_per_px**2)

    marginal = xor_curve[:-1] - xor_curve[1:]
    cumulative = xor_curve[2:].sum()

    amps = np.sqrt((coeffs**2).sum(axis=1))
    p = amps / (amps.sum() + 1e-12)
    E = shannon_entropy(p)

    return {
        "xor_curve": xor_curve.tolist(),
        "marginal_diff": marginal.tolist(),
        "cumulative_diff": float(cumulative),
        "entropy": float(E),
    }


def process_entry(entry):
    jp_path, img_id, info = entry
    try:
        mp_x = info["final_um_per_px_x"]
        mp_y = info["final_um_per_px_y"]
        μm_per_px = (mp_x + mp_y) / 2

        mask = cv2.imread(info["mask_path"], cv2.IMREAD_GRAYSCALE) > 127
        if not mask.sum():
            return None

        classic = classic_shape_metrics(mask, μm_per_px)
        if classic is None:
            return None

        efa = efa_timbre_metrics(mask, μm_per_px)

        return {
            "img_id": img_id,
            "batch_day": jp_path.parts[-2],
            **classic,
            **efa,
        }
    except Exception as e:
        print(f"❌ Error in {img_id}: {e}")
        return None


def main():
    root = Path("/net/projects2/promega/data-analysis/output/processed_dataset_256x192")
    out_csv = root / "morphology_timbre_metrics.csv"

    json_paths = list(root.rglob("image_mapping_*_processed.json"))
    print(f"Found {len(json_paths)} JSON files.")

    all_tasks = []
    for jp in json_paths:
        mapping = json.loads(jp.read_text())
        print(f"Preparing {jp.name} with {len(mapping)} entries...")
        all_tasks.extend([(jp, img_id, info) for img_id, info in mapping.items()])

    with ProcessPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(process_entry, all_tasks))

    valid = [r for r in results if r is not None]
    print(f"✔ Done: {len(valid)} valid entries.")

    df = pd.DataFrame(valid)
    df.to_csv(out_csv, index=False)
    print(f"📁 Saved → {out_csv}")


if __name__ == "__main__":
    main()
