#!/usr/bin/env python3
"""Cohort construction for the metabolite-prediction analysis on our sample.

Both cohorts are restricted to the IDOR col2 set (the 248 BA1+BA2 organoids
actually classified at Dy30). They differ only in how the Dy30 survey vote is
turned into a binary label:

- ``strong-consensus`` (198): supermajority labels — uses ``paper_label_fn``,
  which reads the merge-step consensus ``value`` (set only when >= MIN_VOTES of
  the 5 regular votes agree). The 50 no-consensus organoids (3-2 / 2-3 splits)
  have ``value = None`` and are dropped. 165 Acceptable / 33 Not Acceptable.
- ``full`` (248): every col2 organoid, with the ambiguous 3-2 / 2-3 splits
  resolved by simple majority of the 5 regular votes (``simple_majority_label_fn``).
  191 Acceptable / 57 Not Acceptable.

All label/vote logic reads through ``pipeline.data_loader`` accessors (no raw
``json.load`` of ``all_data.json``; AGENTS.md rule #3). ``_load_idor_organoid_ids``
reads the IDOR partner CSV, not ``all_data.json``, and is the same helper the
public ``idor_organoid_filter`` is built on.
"""

from collections import Counter
from collections.abc import Callable

from pipeline.data_loader import (
    HIGH_QUALITY_BATCHES,
    LABEL_DAY,
    OrganoidDataset,
    _load_idor_organoid_ids,
    get_survey_vote_counts,
    paper_label_fn,
    require_batches,
)

ALL_DATA_PATH = "data/all_data.json"

# Expected cohort sizes and label splits on the current all_data.json. These are
# asserted at build time so a silent upstream data drift fails loudly here.
COHORT_EXPECTATIONS = {
    "strong-consensus": {"n": 198, "Acceptable": 165, "Not Acceptable": 33},
    "full": {"n": 248, "Acceptable": 191, "Not Acceptable": 57},
}


def col2_membership_filter(csv_path: str | None = None) -> Callable:
    """Keep only organoids in the IDOR col2 set (248 Dy30-classified BA1+BA2).

    The set is loaded once at construction; downstream calls are O(1). Mirrors
    the convention of ``idor_organoid_filter`` (closure over the CSV-derived ids).
    """
    _col1, col2_pairs = _load_idor_organoid_ids(csv_path)
    col2_ids = {oid for oid, _ in col2_pairs}

    def f(org_id: str, records: dict) -> bool:
        return org_id in col2_ids

    f.__doc__ = f"col2_membership_filter({len(col2_ids)} organoids)"
    return f


def simple_majority_label_fn(
    org_id: str, records: dict, label_day: str = LABEL_DAY, **_
) -> str | None:
    """Label from a simple majority of the Dy30 *regular* survey votes.

    Resolves the 3-2 / 2-3 splits that ``paper_label_fn`` leaves unlabeled.
    Returns 'Acceptable' / 'Not Acceptable', or None if the organoid has no Dy30
    regular votes. Five regular votes is odd, so a tie is impossible; the
    ``acc == nacc`` guard only protects against partial-vote data drift.
    """
    rec = records.get(label_day)
    if rec is None:
        return None
    acc, total = get_survey_vote_counts(rec)  # regular bucket only, caps at 5
    if total == 0:
        return None
    nacc = total - acc
    if acc == nacc:
        return None
    return "Acceptable" if acc > nacc else "Not Acceptable"


# Cohort name → label function. Both share the col2 + BA1/BA2 filters.
COHORT_LABEL_FNS = {
    "strong-consensus": paper_label_fn,
    "full": simple_majority_label_fn,
}


def build_cohort(
    name: str, all_data_path: str = ALL_DATA_PATH, *, csv_path: str | None = None
) -> tuple[OrganoidDataset, dict]:
    """Build the OrganoidDataset for a cohort and assert its expected makeup.

    Returns (dataset, label_counts). Raises KeyError for an unknown name and
    AssertionError if the cohort size / label split drifts from expectations.
    """
    if name not in COHORT_LABEL_FNS:
        raise KeyError(f"unknown cohort {name!r}; choose from {list(COHORT_LABEL_FNS)}")

    ds = OrganoidDataset(
        all_data_path,
        filters=[require_batches(*HIGH_QUALITY_BATCHES), col2_membership_filter(csv_path)],
        label_fn=COHORT_LABEL_FNS[name],
    )
    counts = dict(Counter(ds.organoid_labels().values()))

    exp = COHORT_EXPECTATIONS[name]
    n = len(ds.organoid_ids)
    assert n == exp["n"], f"{name}: {n} organoids, expected {exp['n']}"
    for label in ("Acceptable", "Not Acceptable"):
        assert counts.get(label, 0) == exp[label], (
            f"{name}: {label}={counts.get(label, 0)}, expected {exp[label]}"
        )
    return ds, counts
