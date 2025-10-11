#!/usr/bin/env python3
"""
data_preprocessing.py

Usage:
    python scripts/data_preprocessing.py --majority_threshold 4

This script processes:
  • organoid_surveys_aggregated.json (contains evaluations)
  • final_combined_metadata.json (contains img/mask paths)

Outputs:
  • Per-day JSONs in:
      - data/preprocessed/complete/  (5/5 agreement)
      - data/preprocessed/majority/  (≥ majority_threshold)
  • CSV for unmatched survey cases:
      - data/preprocessed/unmatched/unmatched_cases.csv
"""

import json, re, csv, argparse
from collections import Counter, defaultdict
from pathlib import Path

# ----------- Default Paths -----------
SURVEY_JSON   = Path("data/raw/organoid_surveys_aggregated.json")
METADATA_JSON = Path("data/raw/final_combined_metadata.json")
OUT_ROOT      = Path("data/preprocessed")
UNMATCH_CSV   = OUT_ROOT / "unmatched" / "unmatched_cases.csv"
# -------------------------------------

def label_from_votes(votes, mode="majority", majority_threshold=4):
    """Returns 'Accepted', 'Not Accepted', or None based on votes."""
    if len(votes) != 5:
        raise ValueError(f"Expected 5 votes, got {len(votes)}")
    
    cnt = Counter(v.strip().lower() for v in votes)
    acc, nacc = cnt["acceptable"], cnt["not acceptable"]

    if mode == "complete":
        if acc == 5: return "Accepted"
        if nacc == 5: return "Not Accepted"
        return None
    
    if acc >= majority_threshold and nacc < majority_threshold:
        return "Accepted"
    if nacc >= majority_threshold and acc < majority_threshold:
        return "Not Accepted"
    return None

def extract_ba_and_well(img_id: str):
    """
    From 'Ba2 96_2 Dy30 D6' extract:
      → full BA = 'BA2 96_2'
      → short BA = 'BA2'
      → well = 'D6'
    """
    parts = img_id.strip().split()
    if len(parts) >= 4:
        ba_full = f"{parts[0].upper()} {parts[1].upper()}"    # e.g., 'BA2 96_2'
        ba_short = parts[0].upper()                            # e.g., 'BA2'
        well = parts[-1].upper()                               # 'D6'
        return (ba_full, ba_short, well)
    return None, None, None

def normalize_image_id(img_id: str) -> str:
    """Uppercase entire image ID."""
    return img_id.strip().upper()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--survey", default=SURVEY_JSON, help="Path to aggregated survey votes")
    parser.add_argument("--metadata", default=METADATA_JSON, help="Path to combined metadata JSON")
    parser.add_argument("--outdir", default=OUT_ROOT, help="Where to save outputs")
    parser.add_argument("--majority_threshold", type=int, default=4,
                        help="Threshold for majority agreement (3 or 4)")
    args = parser.parse_args()

    survey = json.loads(Path(args.survey).read_text())
    metadata = json.loads(Path(args.metadata).read_text())

    # Uppercase relevant metadata fields
    for v in metadata.values():
        if "BA" in v:
            v["BA"] = v["BA"].strip().upper()
        if "wellID" in v:
            v["wellID"] = v["wellID"].strip().upper()

    # Create lookup: (BA, wellID) → list of metadata keys
    metadata_by_ba_well = defaultdict(list)
    for key, meta in metadata.items():
        ba_full = meta.get("BA", "").strip().upper()
        ba_short = ba_full.split()[0] if ba_full else ""
        well = meta.get("wellID", "").strip().upper()
        if ba_full and well:
            metadata_by_ba_well[(ba_full, well)].append(key)
        if ba_short and well:
            metadata_by_ba_well[(ba_short, well)].append(key)

    datasets = defaultdict(list)     # (mode, dayID) → list of labeled records
    unmatched = []                   # List of unmatched survey entries
    seen_metadata_keys = set()       # To avoid duplication based on metadata key

    for obj_key, obj in survey.items():
        evals = obj.get("evaluations", [])
        if len(evals) != 5:
            continue  # Skip malformed or incomplete entries

        raw_id = evals[0]["image_id"]
        ba_full, ba_short, well = extract_ba_and_well(raw_id)

        if not ba_short or not well:
            unmatched.append({"survey_object": obj_key, "image_id": raw_id})
            continue

        matched_keys = metadata_by_ba_well.get((ba_full, well)) or metadata_by_ba_well.get((ba_short, well))
        if not matched_keys:
            unmatched.append({"survey_object": obj_key, "image_id": raw_id})
            continue

        votes = [e["evaluation"] for e in evals]
        lbl_complete = label_from_votes(votes, mode="complete")
        lbl_majority = label_from_votes(votes, mode="majority", majority_threshold=args.majority_threshold)
        norm_id = normalize_image_id(raw_id)

        for meta_key in matched_keys:
            if meta_key in seen_metadata_keys:
                continue
            seen_metadata_keys.add(meta_key)

            meta_rec = metadata[meta_key]
            day_id = meta_rec["dayID"]

            record = {
                "id": norm_id,
                "metadata_key": meta_key,
                "img_path": meta_rec["img_path"],
                "mask_path": meta_rec["mask_path"],
                "Best Z Filename": meta_rec.get("Best Z Filename", ""),
            }

            if lbl_complete:
                record["label"] = lbl_complete
                datasets[("complete", day_id)].append(record)

            if lbl_majority:
                record["label"] = lbl_majority
                datasets[("majority", day_id)].append(record)

    # Save per-day JSON files
    for (mode, day), records in datasets.items():
        out_path = Path(args.outdir) / mode / f"{day}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(records, indent=2))
        print(f"✅ Saved {len(records)} records → {out_path}")

    # Save unmatched entries
    if unmatched:
        UNMATCH_CSV.parent.mkdir(parents=True, exist_ok=True)
        with UNMATCH_CSV.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=unmatched[0].keys())
            writer.writeheader()
            writer.writerows(unmatched)
        print(f"⚠  Unmatched entries: {len(unmatched)} → {UNMATCH_CSV}")

        print("\n🔍 UNMATCHED ENTRIES:")
        for entry in unmatched:
            print(f" • survey_object: {entry['survey_object']}  →  image_id: {entry['image_id']}")
    else:
        print("🎉 All survey entries matched successfully!")

if __name__ == "__main__":
    main()
