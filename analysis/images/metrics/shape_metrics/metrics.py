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

import re
from pathlib import Path

def parse_batch_day_from_key(key: str):
    s = re.sub(r'[_\-]+', ' ', key).strip()  # normalize separators
    # batch: BA3, BA 3, Batch3, batch 3
    m_b = re.search(r'\b(?:BA|Batch)\s*(\d+)\b', s, flags=re.I)
    # day: Dy15, Dy 15, Day15, Day 15
    m_d = re.search(r'\b(?:Dy|Day)\s*(\d+)\b', s, flags=re.I)
    batch = int(m_b.group(1)) if m_b else None
    day   = int(m_d.group(1)) if m_d else None
    return batch, day

def infer_batch_day(key: str, info: dict):
    batch, day = parse_batch_day_from_key(key)
    if batch is not None and day is not None:
        return batch, day
    # fall back to mask_path like .../batch3/day15/...
    mp = str(info.get("mask_path", ""))
    m = re.search(r'/batch(\d+)/day(\d+)/', mp, flags=re.I)
    if m:
        return int(m.group(1)), int(m.group(2))
    # last resort: try img_path
    ip = str(info.get("img_path", ""))
    m2 = re.search(r'(?:BA|Batch)\s*(\d+).*?(?:Dy|Day)\s*(\d+)', ip, flags=re.I)
    if m2:
        return int(m2.group(1)), int(m2.group(2))
    return None, None

def classic_shape_metrics(mask_bin: np.ndarray, μm_per_px: float):
    if mask_bin.ndim != 2 or mask_bin.size == 0:
        return None
    regions = measure.regionprops(mask_bin.astype(np.uint8), cache=True)
    if not regions:
        return None
    props = max(regions, key=lambda r: r.area)

    area_px  = max(float(props.area), 1e-6)
    peri_px  = max(float(props.perimeter), 1e-6)
    major    = float(props.major_axis_length)
    minor    = max(float(props.minor_axis_length), 1e-6)

    hull     = morphology.convex_hull_image(mask_bin)
    peri_ch  = max(float(measure.perimeter(hull)), 1e-6)
    area_ch  = max(float(hull.sum()), 1e-6)

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
    contours = find_contours(mask_bin.astype(np.uint8), level=0.5)
    if not contours:
        return None
    contour = max(contours, key=len)
    if len(contour) > 20000:   # skip pathological shapes
        return None

    xy = np.fliplr(contour)
    coeffs = elliptic_fourier_descriptors(xy, order=n_harmonics, normalize=True)
    if not np.isfinite(coeffs).all():
        return None

    H, W = mask_bin.shape
    orig_area = float(mask_bin.sum()) * (μm_per_px**2)
    xor_curve = np.zeros(n_harmonics + 1, float)
    xor_curve[0] = orig_area

    for k in range(1, n_harmonics + 1):
        rec_xy = reconstruct_contour(coeffs[:k], locus=(0, 0), num_points=xy.shape[0])
        pts = np.round(rec_xy).astype(np.int32)
        # clamp to image bounds for cv2.fillPoly
        pts[:, 0] = np.clip(pts[:, 0], 0, W - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, H - 1)

        ui = np.zeros((H, W), np.uint8)
        cv2.fillPoly(ui, [pts], 1)
        filled = ui.astype(bool)

        xor_curve[k] = float(np.logical_xor(mask_bin, filled).sum()) * (μm_per_px**2)

    marginal   = xor_curve[:-1] - xor_curve[1:]
    cumulative = float(xor_curve[2:].sum())

    amps = np.sqrt((coeffs**2).sum(axis=1))
    p = amps / (amps.sum() + 1e-12)
    E = float(shannon_entropy(p))

    return {"xor_curve": xor_curve.tolist(),
            "marginal_diff": marginal.tolist(),
            "cumulative_diff": cumulative,
            "entropy": E}


def process_entry(entry):
    jp_path, img_id, info = entry
    try:
        mp_x = info["final_um_per_px_x"]
        mp_y = info["final_um_per_px_y"]
        μm_per_px = (mp_x + mp_y) / 2

        img = cv2.imread(info["mask_path"], cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Missing mask: {info['mask_path']}")
            return None
        mask = img > 127
        if not mask.any():
            return None

        classic = classic_shape_metrics(mask, μm_per_px)
        if classic is None:
            return None

        efa = efa_timbre_metrics(mask, μm_per_px)

        batch, day = infer_batch_day(img_id, info)
        return {
            "img_id": img_id,
            "batch": batch,
            "day": day,
            "batch_day": (f"batch{batch}_day{day}"
                  if (batch is not None and day is not None) else None),
            **classic, **efa,
        }
    except Exception as e:
        print(f"❌ Error in {img_id}: {e}")
        return None


def main():
    root = Path("/net/projects2/promega/data-analysis/output/processed_dataset_512x384/auto_processed")
    out_csv = root / "morphology_timbre_metrics.csv"

    json_paths = list(root.rglob("image_mapping_*_processed.json"))
    print(f"Found {len(json_paths)} JSON files.")

    all_tasks = []
    for jp in json_paths:
        mapping = json.loads(jp.read_text())
        print(f"Preparing {jp.name} with {len(mapping)} entries...")
        all_tasks.extend([(jp, img_id, info) for img_id, info in mapping.items()])

    with ProcessPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(process_entry, all_tasks, chunksize=64))


    valid = [r for r in results if r is not None]
    print(f"✔ Done: {len(valid)} valid entries.")

    df = pd.DataFrame(valid)
    # Make batch/day nullable integers (won’t crash on missing)
    df["batch"] = pd.to_numeric(df["batch"], errors="coerce").astype("Int64")
    df["day"]   = pd.to_numeric(df["day"],   errors="coerce").astype("Int64")

    # OPTIONAL: if some rows missed batch/day, recover from batch_day text
    m = df["batch_day"].astype("string").str.extract(r"(?i)(?:ba|batch)\s*(\d+).*?(?:dy|day)\s*(\d+)")
    b = pd.to_numeric(m[0], errors="coerce")
    d = pd.to_numeric(m[1], errors="coerce")

    need_b = df["batch"].isna()
    need_d = df["day"].isna()
    df.loc[need_b, "batch"] = b[need_b].astype("Int64")
    df.loc[need_d, "day"]   = d[need_d].astype("Int64")

    df.to_csv(out_csv, index=False)
    print(f"📁 Saved → {out_csv}")


if __name__ == "__main__":
    import os, multiprocessing as mp
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    try:
        cv2.setNumThreads(1)
    except Exception:
        pass
    mp.set_start_method("spawn", force=True)   # don’t fork OpenMP libs
    main()
