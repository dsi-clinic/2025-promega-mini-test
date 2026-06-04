#!/usr/bin/env python3
"""Basic counting EDA for the IDOR sample (BA1 + BA2).

Recreates the headline organoid counts through ``pipeline.data_loader`` (no raw
``json.load``; see AGENTS.md rule #3).

Table 1 — vote-split distribution over the IDOR organoids classified at Dy30
(col2, N=248): each organoid grouped by its Dy30 survey tally
``Acceptable-NotAcceptable`` (e.g. ``5-0``, ``4-1``), with counts and percents.
Uses the *regular-image* vote bucket only (``get_survey_vote_counts``) — the
bucket that decides the consensus label — so tallies cap at 5 and align with
the good/bad consensus in Table 2. The implied majority consensus is shown per
row. (The inverted-image re-show pass, which would push totals to 10, is the
``get_complete_survey_vote_counts`` view and is intentionally excluded here.)

Table 2 — the cohort cascade that situates those 248 within BA1+BA2:

    BA1+BA2 total                       (iter_organoid_records, unfiltered)
      └─ IDOR evaluated   (col1)        (verify_idor_list)
           └─ IDOR classified at Dy30 (col2)   ← the 248
                └─ with a good/bad consensus label   (OrganoidDataset)
                     ├─ good  (Acceptable)
                     └─ bad   (Not Acceptable)

Outputs:
  - Console: both tables
  - $ANALYSIS_OUTPUT_DIR/figures/idor_eda_vote_splits.csv
  - $ANALYSIS_OUTPUT_DIR/figures/idor_eda_counts.csv

Usage (the package name starts with a digit, so it is run by path, not ``-m``):
    make run ARGS="analysis/2026_07_EDA/eda.py"
    # or directly:
    PYTHONPATH=. python analysis/2026_07_EDA/eda.py
"""

from collections import Counter
from pathlib import Path

import pandas as pd

from pipeline.data_loader import (
    FIGURE_DIR,
    HIGH_QUALITY_BATCHES,
    LABEL_DAY,
    MIN_VOTES,
    OrganoidDataset,
    _load_idor_organoid_ids,
    get_survey_vote_counts,
    idor_ba1_ba2_filters,
    iter_organoid_records,
    verify_idor_list,
)

ALL_DATA_PATH = Path("data/all_data.json")
OUTPUT_DIR = FIGURE_DIR


def _col2_records(all_data_path: Path):
    """Yield Dy30 records for the IDOR col2 set (classified at Dy30, N=248).

    col2 is the IDOR partner's list of organoids with an assigned Dy30 main_id.
    We pull the Dy30 record for each from the unfiltered BA1+BA2 pool.
    """
    _col1, col2_pairs = _load_idor_organoid_ids()
    col2 = {oid for oid, _ in col2_pairs}
    orgs = {
        oid: recs
        for oid, recs, _batch in iter_organoid_records(
            all_data_path, batches=HIGH_QUALITY_BATCHES
        )
    }
    for oid in col2:
        rec = orgs.get(oid, {}).get(LABEL_DAY)
        if rec is not None:
            yield oid, rec


def _vote_split_table(all_data_path: Path) -> pd.DataFrame:
    """Vote-split distribution: organoids grouped by their Dy30 vote tally.

    Uses the public ``get_survey_vote_counts`` accessor (regular-image bucket
    only, capped at 5); the not-acceptable count is ``total - acceptable``.
    Sorted by total votes, then by acceptable count descending. A trailing Total
    row sums to col2.
    """
    splits = Counter()
    for _oid, rec in _col2_records(all_data_path):
        n_acc, n_total = get_survey_vote_counts(rec)
        n_nacc = n_total - n_acc
        splits[(n_acc, n_nacc)] += 1

    n = sum(splits.values())
    rows = []
    for (acc, nacc), count in sorted(
        splits.items(), key=lambda kv: (kv[0][0] + kv[0][1], -kv[0][0])
    ):
        total = acc + nacc
        # Consensus follows the canonical merge rule (compute_survey_majority):
        # a label needs at least MIN_VOTES in the regular bucket; anything short
        # of that is "no consensus" (this is why 3-2 / 2-3 land in the 50
        # reviewed-without-consensus organoids of Table 2), not a bare majority.
        if total == 0:
            consensus = "no votes"
        elif acc >= MIN_VOTES:
            consensus = "Acceptable"
        elif nacc >= MIN_VOTES:
            consensus = "Not Acceptable"
        else:
            consensus = "no consensus"
        rows.append({
            "split (acc-nacc)": f"{acc}-{nacc}",
            "votes": total,
            "consensus": consensus,
            "count": count,
            "pct": round(100 * count / n, 1),
        })
    rows.append({
        "split (acc-nacc)": "Total",
        "votes": "",
        "consensus": "",
        "count": n,
        "pct": 100.0,
    })
    return pd.DataFrame(rows)


def _cohort_cascade(all_data_path: Path) -> pd.DataFrame:
    """Reconciled BA1+BA2 → IDOR → classified → consensus cascade."""
    ba1_ba2_total = sum(
        1 for _ in iter_organoid_records(all_data_path, batches=HIGH_QUALITY_BATCHES)
    )
    idor = verify_idor_list(all_data_path=str(all_data_path))
    ds = OrganoidDataset(str(all_data_path), filters=idor_ba1_ba2_filters())
    labels = Counter(ds.organoid_labels().values())
    good = labels.get("Acceptable", 0)
    bad = labels.get("Not Acceptable", 0)

    rows = [
        ("BA1+BA2 organoids (total)", ba1_ba2_total),
        ("IDOR evaluated (col1)", idor["col1_count"]),
        ("IDOR classified at Dy30 (col2)", idor["col2_count"]),
        ("  with good/bad consensus label", good + bad),
        ("    good (Acceptable)", good),
        ("    bad (Not Acceptable)", bad),
        ("  reviewed, no consensus", idor["no_consensus_count"]),
        ("col1 never reached Dy30", len(idor["no_dy30_organoids"])),
        ("col1 intro-survey (Dy30, excluded)", len(idor["intro_survey_organoids"])),
    ]
    return pd.DataFrame(rows, columns=["metric", "count"])


def main():
    votes = _vote_split_table(ALL_DATA_PATH)
    cascade = _cohort_cascade(ALL_DATA_PATH)

    print("=== Table 1: Dy30 vote-split (IDOR classified at Dy30, N=248) ===\n")
    print(votes.to_string(index=False))

    print("\n=== Table 2: IDOR sample (BA1 + BA2) cohort cascade ===\n")
    print(cascade.to_string(index=False))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    votes_path = OUTPUT_DIR / "idor_eda_vote_splits.csv"
    counts_path = OUTPUT_DIR / "idor_eda_counts.csv"
    votes.to_csv(votes_path, index=False)
    cascade.to_csv(counts_path, index=False)
    print(f"\nSaved to {votes_path}\n        {counts_path}")


if __name__ == "__main__":
    main()
