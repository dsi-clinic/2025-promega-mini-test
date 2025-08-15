#!/usr/bin/env python3
"""
data_preprocessing.py — all_data.json with Dy30 stats + label propagation

Usage:
    python scripts/data_preprocessing.py --majority_threshold 4
    # optional:
    # python scripts/data_preprocessing.py --all ../../../all_data.json

Inputs:
  • ../../../all_data.json: dict[image_id] -> {
        "dayID": "Dy28",
        "BA": "BA1 96_1",
        "wellID": "A2",
        "img_path": "...",
        "mask_path": "...",
        "Best Z Filename": "...",
        "survey": { "evaluations": [ ... 5 items ... ], ... }   # present for Dy30
        ...
    }

Outputs:
  • data/preprocessed/complete/{day}.json   (propagated ONLY if Dy30 had 5/5)
  • data/preprocessed/majority/{day}.json   (ONLY labeled entries; unlabeled are omitted)
  • data/preprocessed/unmatched/unmatched_cases.csv

The script prints a comprehensive stats report before writing outputs.
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

# ----------- Default Paths -----------
ALL_JSON    = Path("../../../all_data.json")
OUT_ROOT    = Path("data/preprocessed")
UNMATCH_CSV = OUT_ROOT / "unmatched" / "unmatched_cases.csv"
# -------------------------------------

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

def day_sort_key(day_id: str) -> Tuple[int, str]:
    """
    Sorts days like Dy3, Dy05, Dy28, Dy30 numerically if possible.
    Returns (num, original_upper) for stable ordering.
    """
    d = norm(day_id)
    m = re.search(r"DY(\d+)", d)
    return (int(m.group(1)) if m else 10**9, d)

# ------------- Stats Computation -------------

def compute_stats(
    all_data: Dict[str, Dict[str, Any]],
    dy30_labels: Dict[Tuple[str, str], Dict[str, Optional[str]]],
    majority_threshold: int
) -> Dict[str, Any]:
    """
    Build a comprehensive stats dictionary and return it.
    Also returns per-day tallies and label distributions.
    """

    # Dy30 coverage
    dy30_total = 0
    dy30_with_valid_survey = 0
    dy30_label_dist_majority = Counter()  # Accepted / Not Accepted
    dy30_label_dist_complete = Counter()

    for image_id, rec in all_data.items():
        if not isinstance(rec, dict): continue
        if norm(rec.get("dayID")) != "DY30": continue
        dy30_total += 1
        votes = get_votes(rec)
        if len(votes) == 5:
            dy30_with_valid_survey += 1
            maj = label_from_votes(votes, mode="majority", majority_threshold=majority_threshold)
            com = label_from_votes(votes, mode="complete")
            if maj: dy30_label_dist_majority[maj] += 1
            if com: dy30_label_dist_complete[com] += 1

    # Per-day breakdown
    per_day_total = Counter()
    per_day_with_match = Counter()        # has Dy30 majority label for that BA/well
    per_day_without_match = Counter()
    per_day_label_dist = defaultdict(Counter)  # day -> {Accepted, Not Accepted}
    unmatched_reason_counts = Counter()

    for image_id, rec in all_data.items():
        if not isinstance(rec, dict):
            unmatched_reason_counts["record_not_dict"] += 1
            continue

        day_id = s(rec.get("dayID"))
        ba = norm(rec.get("BA"))
        well = norm(rec.get("wellID"))
        img_path = s(rec.get("img_path"))
        mask_path = s(rec.get("mask_path"))

        # Count totals only for usable (has day/BA/well/img/mask) entries
        if not day_id or not ba or not well or not img_path or not mask_path:
            if not day_id: unmatched_reason_counts["missing_dayID"] += 1
            elif not ba or not well: unmatched_reason_counts["missing_BA_or_wellID"] += 1
            elif not img_path or not mask_path: unmatched_reason_counts["missing_img_or_mask_path"] += 1
            continue

        per_day_total[day_id] += 1

        # Dy30 match availability (majority label propagation availability)
        inherited = dy30_labels.get((ba, well))
        maj_label = inherited.get("majority") if inherited else None
        if maj_label:
            per_day_with_match[day_id] += 1
            per_day_label_dist[day_id][maj_label] += 1
        else:
            per_day_without_match[day_id] += 1

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
    print(f"Dy30 entries total: {dy30['total_entries']}")
    print(f"Dy30 with valid 5/5 survey: {dy30['with_valid_5_votes']}")
    print(f"Dy30 majority label distribution: {dy30['majority_label_distribution']}")
    print(f"Dy30 complete (5/5) distribution: {dy30['complete_label_distribution']}")

    print("\nPer-day breakdown:")
    # Sort days naturally by number
    all_days = sorted(set(per_day["totals"].keys()), key=day_sort_key)
    for day in all_days:
        total = per_day["totals"].get(day, 0)
        with_m = per_day["with_dy30_match"].get(day, 0)
        without_m = per_day["without_dy30_match"].get(day, 0)
        dist = per_day["majority_label_distribution_by_day"].get(day, {})
        pct = f"{(with_m/total*100):.1f}%" if total else "n/a"
        print(f"  {day}: total={total}, matched={with_m} ({pct}), unmatched={without_m}, labels={dist}")

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

    # ---------- Pass 1: build Dy30 label map ----------
    # key: (BA.upper(), wellID.upper()) → {"majority": str|None, "complete": str|None}
    dy30_labels: Dict[Tuple[str, str], Dict[str, Optional[str]]] = {}

    for image_id, rec in all_data.items():
        if not isinstance(rec, dict):
            continue
        if norm(rec.get("dayID")) != "DY30":
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

    # ---------- Pass 2: emit per-day datasets with propagation ----------
    datasets = defaultdict(list)   # (mode, dayID) -> List[dict]
    unmatched_rows: List[Dict[str, str]] = []

    for image_id, rec in all_data.items():
        if not isinstance(rec, dict):
            unmatched_rows.append({"image_id": s(image_id), "reason": "record_not_dict"})
            continue

        day_id   = s(rec.get("dayID"))
        ba       = norm(rec.get("BA"))
        well     = norm(rec.get("wellID"))
        img_path = s(rec.get("img_path"))
        mask_path= s(rec.get("mask_path"))
        best_zfn = s(rec.get("Best Z Filename"))

        if not day_id:
            unmatched_rows.append({"image_id": s(image_id), "reason": "missing_dayID"})
            continue
        if not ba or not well:
            unmatched_rows.append({"image_id": s(image_id), "reason": "missing_BA_or_wellID"})
            continue
        if not img_path or not mask_path:
            unmatched_rows.append({"image_id": s(image_id), "reason": "missing_img_or_mask_path"})
            continue

        # Determine labels:
        maj_label: Optional[str] = None
        comp_label: Optional[str] = None

        if norm(day_id) == "DY30":
            # compute from its own survey
            votes = get_votes(rec)
            maj_label = label_from_votes(votes, mode="majority", majority_threshold=args.majority_threshold)
            comp_label = label_from_votes(votes, mode="complete")
        else:
            # inherit from Dy30 map via (BA, well)
            inherited = dy30_labels.get((ba, well), {})
            maj_label = inherited.get("majority")
            comp_label = inherited.get("complete")

        # Base record
        base_record = {
            "id": s(image_id).strip().upper(),
            "metadata_key": s(image_id),
            "img_path": img_path,
            "mask_path": mask_path,
            "Best Z Filename": best_zfn,
        }

        # complete: only if 5/5 (from Dy30 or this record if Dy30)
        if comp_label:
            rec_c = dict(base_record)
            rec_c["label"] = comp_label
            datasets[("complete", day_id)].append(rec_c)

        # majority: ONLY emit if we have a propagated/own label; otherwise skip & log
        if maj_label:
            rec_m = dict(base_record)
            rec_m["label"] = maj_label
            datasets[("majority", day_id)].append(rec_m)
        else:
            unmatched_rows.append({"image_id": s(image_id), "reason": "no_Dy30_label"})

    # ---------- Save ----------
    out_root = Path(args.outdir)
    for (mode, day), records in datasets.items():
        out_path = out_root / mode / f"{day}.json"
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
