#!/usr/bin/env python3
"""Verify the IDOR-supplied organoid list against ``data/all_data.json``.

Thin CLI wrapper around ``pipeline.data_loader.verify_idor_list`` — the
assertions live there so they're available to filter constructors that want
to gate on data validity at runtime (e.g. ``idor_ba1_ba2_filters(verify=...)``).

Partner statement (paraphrased) being verified:
  - Column 1 = all 266 organoids evaluated; "Dy" and "stitched" tokens stripped.
  - Column 2 = organoids that reached Day 30 AND were classified in the survey.
    5 reached Day 30 but were used as the introductory part of the survey
    and kept out of column 2.
  - Split organoids are excluded from column 1.
  - Images with edge_fraction > 0.05 were excluded from feature extraction
    (concentrated on Dy24 / Dy28 / Dy30) — checked via the loader's edge
    accessor, not asserted (it's a downstream-usage policy, not a CSV claim).

Usage:
    make analysis-verify-idor
"""

from collections import Counter

from pipeline.data_loader import (
    HIGH_QUALITY_BATCHES,
    _load_idor_organoid_ids,
    get_edge_fraction,
    is_stitched_record,
    iter_organoid_records,
    verify_idor_list,
)

EDGE_FRACTION_THRESHOLD = 0.05
LABEL_DAY = "Dy30"


def main():
    summary = verify_idor_list(verbose=True)

    # ---- supplementary, informational reports (not strict assertions) ----

    # Edge-fraction by-day distribution — partner said >0.05 was excluded
    # (mainly Dy24/28/30).
    by_day_high: Counter = Counter()
    by_day_total: Counter = Counter()
    for _, records, _ in iter_organoid_records(
        "data/all_data.json", batches=HIGH_QUALITY_BATCHES
    ):
        for day, rec in records.items():
            ef = get_edge_fraction(rec)
            if ef is None:
                continue
            by_day_total[day] += 1
            if ef > EDGE_FRACTION_THRESHOLD:
                by_day_high[day] += 1

    total_populated = sum(by_day_total.values())
    total_high = sum(by_day_high.values())
    print(
        f"\nEdge-fraction policy ({EDGE_FRACTION_THRESHOLD} threshold):"
        f"\n  records with edge_fraction populated: {total_populated}"
        f"\n  records with edge_fraction > {EDGE_FRACTION_THRESHOLD}: {total_high}"
    )
    if total_high:
        print("  by day (high-edge counts):")
        for day in sorted(by_day_high):
            print(f"    {day}: {by_day_high[day]}")

    # Stitched count in col2 — partner permits stitched (only split is excluded).
    _, col2_pairs = _load_idor_organoid_ids()
    organoids = {
        oid: records
        for oid, records, _ in iter_organoid_records(
            "data/all_data.json", batches=HIGH_QUALITY_BATCHES
        )
    }
    stitched = sum(
        1 for oid, _ in col2_pairs
        if is_stitched_record(organoids[oid][LABEL_DAY])
    )
    print(
        f"\nCol2 stitched-at-Dy30 organoids: {stitched}/{summary['col2_count']} "
        f"(stitched is permitted; only split was eliminated)"
    )

    print("\nAll claims verified.")


if __name__ == "__main__":
    main()
