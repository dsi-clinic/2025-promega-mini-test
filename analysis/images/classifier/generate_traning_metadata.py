#!/usr/bin/env python3
"""
data_preprocessing.py — all_data.json with Dy30 stats + label propagation

This version:
  • Uses ONLY `day_num` (ignores `dayID` entirely).
  • Merges day_num 20 and 21 into 20.5 (both in stats and outputs).
  • Handles two image variants ("512x384" and "256x192") and writes to:
        data/preprocessed/512x384/{mode}/{day}.json
        data/preprocessed/256x192/{mode}/{day}.json
  • Run from repo root: default --all is all_data.json.
  • Natural day sorting is numeric (supports decimals like 20.5).

Usage (from repo root):
    python scripts/data_preprocessing.py --majority_threshold 4
    # optional:
    # python scripts/data_preprocessing.py --all all_data.json

Inputs (example record shape):
{
  "BA1 96_1 Dy03 A1": {
    "day_num": 3,           # used
    "mdl_day": 3.0,         # ignored for day selection (kept for reference)
    "BA": "BA1 96_1",
    "wellID": "A1",
    "Best Z Filename": "...",
    "512x384": {"img_path":"...", "mask_path":"..."},
    "256x192": {"img_path":"...", "mask_path":"..."},
    "survey": { "evaluations": [ ... 5 items ... ] }   # present on day_num == 30
  }
}

Outputs:
  • data/preprocessed/<variant>/complete/{DAY}.json   (propagated ONLY if day 30 had 5/5)
  • data/preprocessed/<variant>/majority/{DAY}.json   (ONLY labeled entries; unlabeled omitted)
  • data/preprocessed/unmatched/unmatched_cases.csv

Notes:
  • Stats are variant-agnostic (count a record if it has at least one valid variant).
  • day filename token is file-safe: e.g., 20.5 → "20_5.json".
"""

import json
import csv
import argparse
import sys
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

# ----------- Default Paths (run from repo root) -----------
ALL_JSON    = Path("all_data.json")
OUT_ROOT    = Path("analysis/images/classifier/data/preprocessed")
UNMATCH_CSV = OUT_ROOT / "unmatched" / "unmatched_cases.csv"
# ----------------------------------------------------------

# ----------------- Helpers -----------------

def is_nan(x: Any) -> bool:
    return isinstance(x, float) and math.isnan(x)

def s(x: Any) -> str:
    return "" if x is None or is_nan(x) else str(x)

def norm(sval: Any) -> str:
    return s(sval).strip().upper()

def label_from_votes(votes: List[str], mode: str = "majority", majority_threshold: int = 4) -> Optional[str]:
    """Return 'Accepted' / 'Not Accepted' / None based on 5 votes."""
    if len(votes) != 5:
        return None
    cnt = Counter((v or "").strip().lower() for v in votes)
    acc = cnt.get("acceptable", 0)
    nacc = cnt.get("not acceptable", 0)

    if mode == "complete":
        if acc == 5: return "Accepted"
        if nacc == 5: return "Not Accepted"
        return None

    if acc >= majority_threshold and nacc < majority_threshold:
        return "Accepted"
    if nacc >= majority_threshold and acc < majority_threshold:
        return "Not Accepted"
    return None

def get_votes(rec: Dict[str, Any]) -> List[str]:
    survey = rec.get("survey")
    if not isinstance(survey, dict):
        return []
    evs = survey.get("evaluations")
    if not isinstance(evs, list) or len(evs) != 5:
        return []
    votes = []
    for e in evs:
        if isinstance(e, dict):
            votes.append(s(e.get("evaluation")))
    return votes

# ---- day_num helpers ----
def extract_day_num(rec: Dict[str, Any]) -> Optional[float]:
    """
    Read day_num from record.
    Returns float (with 20/21 merged to 20.5) or None if unavailable/invalid.
    """
    dn = rec.get("day_num", None)
    try:
        x = float(dn)
    except (TypeError, ValueError):
        return None
    # Merge rule: 20 and 21 -> 20.5
    if math.isfinite(x) and (abs(x - 20.0) < 1e-9 or abs(x - 21.0) < 1e-9):
        x = 20.5
    return x if math.isfinite(x) else None

def day_sort_key(day_str: str) -> Tuple[float, str]:
    """
    Sorts days numerically. day_str should be something like '3', '20.5', etc.
    """
    try:
        return (float(day_str), day_str)
    except Exception:
        return (10**9, day_str)

def day_to_str(day: float) -> str:
    """Human/readable key like '3' or '20.5' (no trailing .0)."""
    day = round(day, 4)
    if abs(day - int(day)) < 1e-9:
        return str(int(day))
    return str(day).rstrip('0').rstrip('.')

def day_to_file_token(day: float) -> str:
    """File-safe token: replace '.' with '_'."""
    return day_to_str(day).replace(".", "_")

# ---- Variant utilities ----
VARIANTS = ("512x384", "256x192")

def variant_payload(rec: Dict[str, Any], variant: str) -> Optional[Dict[str, Any]]:
    block = rec.get(variant)
    if not isinstance(block, dict):
        return None
    img_path = s(block.get("img_path"))
    mask_path = s(block.get("mask_path"))
    if not img_path or not mask_path:
        return None
    return {"img_path": img_path, "mask_path": mask_path}

# ------------- Stats Computation -------------

def compute_stats(
    all_data: Dict[str, Dict[str, Any]],
    dy30_labels: Dict[Tuple[str, str], Dict[str, Optional[str]]],
    majority_threshold: int
) -> Dict[str, Any]:
    """
    Build a comprehensive stats dictionary and return it.
    Stats are variant-agnostic: a record counts as 'usable' if it has BA/well/day_num
    and at least one valid variant with img+mask.
    """

    # Day 30 (dy30) coverage (by day_num==30)
    dy30_total = 0
    dy30_with_valid_survey = 0
    dy30_label_dist_majority = Counter()  # Accepted / Not Accepted
    dy30_label_dist_complete = Counter()

    for _, rec in all_data.items():
        if not isinstance(rec, dict): continue
        dn = extract_day_num(rec)
        if dn is None: continue
        if abs(dn - 30.0) >= 1e-9:  # only true day 30
            continue
        dy30_total += 1
        votes = get_votes(rec)
        if len(votes) == 5:
            dy30_with_valid_survey += 1
            maj = label_from_votes(votes, mode="majority", majority_threshold=majority_threshold)
            com = label_from_votes(votes, mode="complete")
            if maj: dy30_label_dist_majority[maj] += 1
            if com: dy30_label_dist_complete[com] += 1

    # Per-day breakdown (variant-agnostic)
    per_day_total = Counter()
    per_day_with_match = Counter()        # has day30 majority label for that BA/well
    per_day_without_match = Counter()
    per_day_label_dist = defaultdict(Counter)  # day_str -> {Accepted, Not Accepted}
    unmatched_reason_counts = Counter()

    for image_id, rec in all_data.items():
        if not isinstance(rec, dict):
            unmatched_reason_counts["record_not_dict"] += 1
            continue

        day_val = extract_day_num(rec)
        ba = norm(rec.get("BA"))
        well = norm(rec.get("wellID"))

        has_variant = any(variant_payload(rec, v) for v in VARIANTS)

        if day_val is None:
            unmatched_reason_counts["missing_or_invalid_day_num"] += 1
            continue
        if not ba or not well:
            unmatched_reason_counts["missing_BA_or_wellID"] += 1
            continue
        if not has_variant:
            unmatched_reason_counts["no_valid_variant_img_or_mask"] += 1
            continue

        day_key = day_to_str(day_val)
        per_day_total[day_key] += 1

        inherited = dy30_labels.get((ba, well))
        maj_label = inherited.get("majority") if inherited else None
        if maj_label:
            per_day_with_match[day_key] += 1
            per_day_label_dist[day_key][maj_label] += 1
        else:
            per_day_without_match[day_key] += 1

    # Fully matched vs partially matched days
    fully_matched_days = []
    partially_matched_days = []
    for day, total in per_day_total.items():
        if per_day_with_match[day] == total:
            fully_matched_days.append(day)
        else:
            partially_matched_days.append(day)

    stats = {
        "dy30": {
            "total_entries": dy30_total,
            "with_valid_5_votes": dy30_with_valid_survey,
            "majority_label_distribution": dict(dy30_label_dist_majority),
            "complete_label_distribution": dict(dy30_label_dist_complete),
        },
        "per_day": {
            "totals": dict(per_day_total),
            "with_dy30_match": dict(per_day_with_match),
            "without_dy30_match": dict(per_day_without_match),
            "majority_label_distribution_by_day": {k: dict(v) for k, v in per_day_label_dist.items()},
            "fully_matched_days": sorted(fully_matched_days, key=day_sort_key),
            "partially_matched_days": sorted(partially_matched_days, key=day_sort_key),
        },
        "unmatched_reason_counts": dict(unmatched_reason_counts),
    }
    return stats

def print_stats_report(stats: Dict[str, Any]) -> None:
    """Pretty-print the stats report to stdout."""
    dy30 = stats["dy30"]
    per_day = stats["per_day"]
    reasons = stats["unmatched_reason_counts"]

    print("\n" + "📊" * 3 + "  STATS REPORT  " + "📊" * 3)
    print(f"Day 30 entries total: {dy30['total_entries']}")
    print(f"Day 30 with valid 5/5 survey: {dy30['with_valid_5_votes']}")
    print(f"Day 30 majority label distribution: {dy30['majority_label_distribution']}")
    print(f"Day 30 complete (5/5) distribution: {dy30['complete_label_distribution']}")

    print("\nPer-day breakdown (by day_num):")
    all_days = sorted(set(per_day["totals"].keys()), key=day_sort_key)
    for day in all_days:
        total = per_day["totals"].get(day, 0)
        with_m = per_day["with_dy30_match"].get(day, 0)
        without_m = per_day["without_dy30_match"].get(day, 0)
        dist = per_day["majority_label_distribution_by_day"].get(day, {})
        pct = f"{(with_m/total*100):.1f}%" if total else "n/a"
        print(f"  Day {day}: total={total}, matched={with_m} ({pct}), unmatched={without_m}, labels={dist}")

    print("\nDays fully matched:", per_day["fully_matched_days"])
    print("Days partially matched:", per_day["partially_matched_days"])

    if reasons:
        print("\nUnmatched / skipped reasons:")
        for k, v in reasons.items():
            print(f"  - {k}: {v}")
    print("—" * 60 + "\n")

# ------------- Main Pipeline -------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", default=ALL_JSON, help="Path to combined all_data.json")
    parser.add_argument("--outdir", default=OUT_ROOT, help="Where to save outputs")
    parser.add_argument("--majority_threshold", type=int, default=4,
                        help="Threshold for majority agreement (3 or 4)")
    args = parser.parse_args()

    # ---- Load unified JSON ----
    try:
        all_data = json.loads(Path(args.all).read_text(encoding="utf-8"))
    except FileNotFoundError:
        print(f"❌ Could not find all_data.json at: {args.all}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"❌ all_data.json is not valid JSON: {e}", file=sys.stderr)
        sys.exit(1)

    if not isinstance(all_data, dict):
        print("❌ Expected all_data.json to be a dict mapping image_id → record.", file=sys.stderr)
        sys.exit(2)

    # ---------- Pass 1: build Day30 label map ----------
    # key: (BA.upper(), wellID.upper()) → {"majority": str|None, "complete": str|None}
    dy30_labels: Dict[Tuple[str, str], Dict[str, Optional[str]]] = {}

    for _, rec in all_data.items():
        if not isinstance(rec, dict):
            continue
        dn = extract_day_num(rec)
        if dn is None or abs(dn - 30.0) >= 1e-9:
            continue

        ba = norm(rec.get("BA"))
        well = norm(rec.get("wellID"))
        if not ba or not well:
            continue

        votes = get_votes(rec)
        maj = label_from_votes(votes, mode="majority", majority_threshold=args.majority_threshold)
        comp = label_from_votes(votes, mode="complete")
        dy30_labels[(ba, well)] = {"majority": maj, "complete": comp}

    # ---------- Stats phase (pre-output) ----------
    stats = compute_stats(all_data, dy30_labels, args.majority_threshold)
    print_stats_report(stats)

    # ---------- Pass 2: emit per-variant datasets with propagation ----------
    # datasets_by_variant[variant][(mode, day_str)] -> List[dict]
    datasets_by_variant: Dict[str, Dict[Tuple[str, str], List[Dict[str, Any]]]] = {
        v: defaultdict(list) for v in VARIANTS
    }
    unmatched_rows: List[Dict[str, str]] = []

    for image_id, rec in all_data.items():
        if not isinstance(rec, dict):
            unmatched_rows.append({"image_id": s(image_id), "reason": "record_not_dict"})
            continue

        day_val = extract_day_num(rec)
        ba       = norm(rec.get("BA"))
        well     = norm(rec.get("wellID"))
        best_zfn = s(rec.get("Best Z Filename"))

        if day_val is None:
            unmatched_rows.append({"image_id": s(image_id), "reason": "missing_or_invalid_day_num"})
            continue
        if not ba or not well:
            unmatched_rows.append({"image_id": s(image_id), "reason": "missing_BA_or_wellID"})
            continue

        # Determine labels (variant-agnostic) via day 30 map
        dn_raw = rec.get("day_num", None)
        try:
            dn_raw_f = float(dn_raw)
        except (TypeError, ValueError):
            dn_raw_f = None

        if dn_raw_f is not None and abs(dn_raw_f - 30.0) < 1e-9:
            votes = get_votes(rec)
            maj_label = label_from_votes(votes, mode="majority", majority_threshold=args.majority_threshold)
            comp_label = label_from_votes(votes, mode="complete")
        else:
            inherited = dy30_labels.get((ba, well), {})
            maj_label = inherited.get("majority")
            comp_label = inherited.get("complete")

        day_key = day_to_str(day_val)
        day_file = day_to_file_token(day_val)

        # For each available variant, push records to that variant's dataset
        any_variant_present = False
        for variant in VARIANTS:
            payload = variant_payload(rec, variant)
            if not payload:
                unmatched_rows.append({"image_id": s(image_id), "reason": f"{variant}_missing_img_or_mask"})
                continue

            any_variant_present = True

            base_record = {
                "id": s(image_id).strip().upper(),
                "metadata_key": s(image_id),
                "variant": variant,
                "day_num": float(day_val),
                "img_path": payload["img_path"],
                "mask_path": payload["mask_path"],
                "Best Z Filename": best_zfn,
            }

            if comp_label:
                rec_c = dict(base_record)
                rec_c["label"] = comp_label
                datasets_by_variant[variant][("complete", day_key)].append(rec_c)

            if maj_label:
                rec_m = dict(base_record)
                rec_m["label"] = maj_label
                datasets_by_variant[variant][("majority", day_key)].append(rec_m)
            #else:
                #unmatched_rows.append({"image_id": s(image_id), "reason": "no_day30_label"})

        if not any_variant_present:
            unmatched_rows.append({"image_id": s(image_id), "reason": "no_valid_variant_img_or_mask"})

    # ---------- Save ----------
    out_root = Path(args.outdir)
    for variant, datasets in datasets_by_variant.items():
        for (mode, day_key), records in datasets.items():
            day_file = day_key.replace(".", "_")
            out_path = out_root / variant / mode / f"Dy{day_file}.json"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
            print(f"✅ Saved {len(records)} records → {out_path}")

    if unmatched_rows:
        UNMATCH_CSV.parent.mkdir(parents=True, exist_ok=True)
        with UNMATCH_CSV.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["image_id", "reason"])
            writer.writeheader()
            writer.writerows(unmatched_rows)
        print(f"⚠  Issues logged: {len(unmatched_rows)} → {UNMATCH_CSV}")
    else:
        print("🎉 All entries processed successfully!")

if __name__ == "__main__":
    main()
