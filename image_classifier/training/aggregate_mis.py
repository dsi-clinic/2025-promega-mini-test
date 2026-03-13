#!/usr/bin/env python3
import argparse
import json
import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

# --------- Config / Defaults ----------
DEFAULT_INPUT_DIR = Path("misclassifiedimages")
DEFAULT_SEARCH_ROOT = Path(".")  # where we'll recursively search for Dy*.json
INTERSECTION_OUT = "aggregated_misclassified_by_all_models.csv"
DETAILS_OUT = "aggregated_misclassified_details.csv"

LABEL_MAP = {"Accepted": 1, "Not Accepted": 0}
MODELS = ["efficientnet", "resnet", "vit"]
IMG_COL_CANDS = ["img_path", "image_path", "path", "image"]
# -------------------------------------


def parse_dy_token(name: str) -> Optional[Tuple[str, float]]:
    """
    Extract day from names like:
      Dy13_...     -> token='13'     -> dy=13.0
      Dy20_5_...   -> token='20_5'   -> dy=20.5
    """
    m = re.search(r"Dy(\d+(?:_\d+)?)", name)
    if not m:
        return None
    token = m.group(1)
    return token, float(token.replace("_", "."))


def standardize_img_col(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    lower = {c.lower(): c for c in df.columns}
    for cand in IMG_COL_CANDS:
        if cand.lower() in lower:
            col = lower[cand.lower()]
            if col != "img_path":
                df = df.rename(columns={col: "img_path"})
            return df
    return None


def read_csv(path: Path) -> Optional[pd.DataFrame]:
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"[warn] Could not read {path}: {e}", file=sys.stderr)
        return None


def looks_like_training_gt_json(p: Path) -> bool:
    """
    Quick sniff test: file must be JSON list and contain dicts with
    'img_path' and 'label' (Accepted/Not Accepted).
    """
    try:
        data = json.loads(p.read_text())
        if not isinstance(data, list) or not data:
            return False
        sample = data[0]
        if not isinstance(sample, dict):
            return False
        return ("img_path" in sample) and ("label" in sample)
    except Exception:
        return False


def discover_gt_jsons(search_roots: List[Path]) -> List[Path]:
    """
    Recursively find Dy*.json files that match the expected schema.
    Prefer paths that include '/preprocessed/' and '/majority/' like your training setup.
    """
    candidates: List[Path] = []
    for root in search_roots:
        for p in root.rglob("Dy*.json"):
            if looks_like_training_gt_json(p):
                candidates.append(p)

    if not candidates:
        return []

    # Rank by preference: ones that look like the training dir get higher score
    def score(p: Path) -> int:
        s = 0
        sp = str(p.as_posix()).lower()
        if "/preprocessed/" in sp:
            s += 2
        if "/majority/" in sp:
            s += 2
        if re.search(r"/(512x384|384x512)/", sp):
            s += 1
        return s

    candidates.sort(key=lambda p: (score(p), p.as_posix()))
    # We’ll keep ALL, just ordered (no need to filter down)
    return candidates


def load_global_gt_from_jsons(json_paths: List[Path]) -> pd.DataFrame:
    rows = []
    for jp in json_paths:
        try:
            data = json.loads(jp.read_text())
        except Exception as e:
            print(f"[warn] Skipping {jp}: {e}", file=sys.stderr)
            continue

        for r in data:
            ip = r.get("img_path")
            ls = r.get("label")
            if ip is None or ls not in LABEL_MAP:
                continue
            rows.append({"img_path": ip, "gt_label_str": ls, "gt_label": LABEL_MAP[ls]})

    if not rows:
        sys.exit(
            "[error] Found Dy*.json files but none had usable (img_path, label) records."
        )
    gt = pd.DataFrame(rows).drop_duplicates("img_path")
    print(
        f"[info] Loaded global GT from {len(json_paths)} JSON file(s): {len(gt)} unique images"
    )
    return gt


def main(input_dir: Path, search_root: Path):
    # 1) Discover and load GT JSONs automatically
    json_paths = discover_gt_jsons([search_root])
    if not json_paths:
        sys.exit(
            f"[error] No Dy*.json found under {search_root.resolve()} "
            f"(recursively). If they live elsewhere, pass --gt-search-root PATH"
        )

    gt_df = load_global_gt_from_jsons(json_paths)

    # 2) Build INTERSECTION: files misclassified by all models
    by_all_files = sorted(
        input_dir.glob("Dy*_misclassified_by_all_models.csv"),
        key=lambda p: (parse_dy_token(p.name) or ("", -1.0))[1],
    )

    inter_frames = []
    union_frames = []  # collect per-model (img_path, dy) for the union

    for f in by_all_files:
        parsed = parse_dy_token(f.name)
        if not parsed:
            print(f"[warn] Could not parse day from {f.name}", file=sys.stderr)
            continue
        dy_token, dy_val = parsed

        base = read_csv(f)
        if base is None:
            continue
        base = standardize_img_col(base)
        if base is None:
            print(f"[warn] {f} missing image path column", file=sys.stderr)
            continue

        base = base[["img_path"]].copy()
        base["dy"] = dy_val
        inter_frames.append(base)

        # UNION sources: any model's misclassified list for this day
        for m in ["efficientnet", "resnet", "vit"]:
            mfile = input_dir / f"Dy{dy_token}_misclassified_{m}.csv"
            mdf = read_csv(mfile)
            if mdf is None:
                continue
            mdf = standardize_img_col(mdf)
            if mdf is None:
                print(f"[warn] {mfile} missing image path column", file=sys.stderr)
                continue
            tmp = mdf[["img_path"]].copy()
            tmp["dy"] = dy_val
            union_frames.append(tmp)

    # 3) Write INTERSECTION (keep only 4 columns)
    if inter_frames:
        inter_df = pd.concat(inter_frames, ignore_index=True).drop_duplicates()
        inter_df = inter_df.merge(gt_df, on="img_path", how="left")
        inter_df = inter_df[["img_path", "dy", "gt_label_str", "gt_label"]]
        inter_df.to_csv(INTERSECTION_OUT, index=False)
        print(f"[ok] wrote {INTERSECTION_OUT} ({len(inter_df)} rows)")
    else:
        print(
            "[info] No Dy*_misclassified_by_all_models.csv files found; intersection not created."
        )

    # 4) Write UNION (any model misclassified) — also only 4 columns
    if union_frames:
        union_base = pd.concat(union_frames, ignore_index=True).drop_duplicates()
        union_df = union_base.merge(gt_df, on="img_path", how="left")
        union_df = union_df[["img_path", "dy", "gt_label_str", "gt_label"]]
        union_df.to_csv(DETAILS_OUT, index=False)
        print(f"[ok] wrote {DETAILS_OUT} ({len(union_df)} rows)")
    else:
        print("[info] No per-model misclassified files found; union not created.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Auto-discover GT from Dy*.json (recursive), support fractional days (Dy20_5), and aggregate."
    )
    ap.add_argument(
        "-i",
        "--input-dir",
        default=str(DEFAULT_INPUT_DIR),
        help=f"Folder with Dy*_misclassified_*.csv (default: {DEFAULT_INPUT_DIR})",
    )
    ap.add_argument(
        "-r",
        "--gt-search-root",
        default=str(DEFAULT_SEARCH_ROOT),
        help="Root directory to recursively search for Dy*.json GT files (default: current directory)",
    )
    args = ap.parse_args()
    main(Path(args.input_dir), Path(args.gt_search_root))
