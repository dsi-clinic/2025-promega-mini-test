"""Vote-strength stratification invariants (AGENTS.md rules 11 & 17).

The three strata (5-0 / 4-1 / 3-2) must *partition* the full 248 cohort: every
organoid lands in exactly one stratum, the counts sum to 248, and no organoid
appears in two strata. The 5-0 + 4-1 strata must reconstruct the strong-consensus
cohort (>= 4/5 agreement) exactly. Observed on the current all_data.json:

    5-0 = 119   4-1 = 79   3-2 = 50   (sum = 248)
    5-0 + 4-1 = 198  == strong-consensus cohort size
    label split within strata (Acceptable / Not Acceptable):
        5-0: 107 / 12    4-1: 58 / 21    3-2: 26 / 24

The 5-0 + 4-1 Acceptable (107 + 58 = 165) and Not-Acceptable (12 + 21 = 33)
counts equal the strong-consensus label split exactly, confirming the strata are
just the vote-strength decomposition of that cohort.
"""

import pytest
from invariants import assert_count_conserved, assert_organoid_count

# Exact expectations on the committed all_data.json. Duplicated from the observed
# run (not imported from the module) so a change to the source can't self-certify.
EXPECTED_STRATA = {
    "5-0": {"Acceptable": 107, "Not Acceptable": 12, "total": 119},
    "4-1": {"Acceptable": 58, "Not Acceptable": 21, "total": 79},
    "3-2": {"Acceptable": 26, "Not Acceptable": 24, "total": 50},
}
FULL_COHORT_N = 248
STRONG_CONSENSUS_N = 198
STRONG_CONSENSUS_ACCEPTABLE = 165
STRONG_CONSENSUS_NOT_ACCEPTABLE = 33


@pytest.fixture(scope="session")
def strata(full_cohort):
    from vote_strength import stratum_by_organoid

    return stratum_by_organoid(full_cohort)


def test_strata_partition_the_cohort(full_cohort, strata):
    """Rule 11/17: the three strata partition the 248 cohort exactly."""
    ids = list(full_cohort.organoid_ids)
    assert_organoid_count(ids, FULL_COHORT_N, context="full cohort")

    # Every organoid is stratified, exactly once, into one of the three buckets.
    assert set(strata) == set(ids), "stratification did not cover the cohort 1:1"
    assert set(strata.values()) == {"5-0", "4-1", "3-2"}, "unexpected stratum label"

    buckets = {"5-0": [], "4-1": [], "3-2": []}
    for oid, s in strata.items():
        buckets[s].append(oid)

    # No organoid in two strata; counts sum to the cohort size.
    all_bucketed = buckets["5-0"] + buckets["4-1"] + buckets["3-2"]
    assert_organoid_count(all_bucketed, FULL_COHORT_N, context="partition (no overlap)")
    assert_count_conserved(ids, all_bucketed, context="cohort -> strata partition")
    assert sum(len(v) for v in buckets.values()) == FULL_COHORT_N

    for s, exp in EXPECTED_STRATA.items():
        assert len(buckets[s]) == exp["total"], f"stratum {s}: {len(buckets[s])}"


def test_distribution_label_split(full_cohort, strata):
    """The stratum x label distribution matches the observed cross-tab."""
    from vote_strength import build_distribution

    table = build_distribution(full_cohort, strata)
    for s, exp in EXPECTED_STRATA.items():
        assert table[s]["Acceptable"] == exp["Acceptable"], f"{s} Acceptable"
        assert table[s]["Not Acceptable"] == exp["Not Acceptable"], f"{s} N.A."
        assert sum(table[s].values()) == exp["total"], f"{s} total"

    grand = sum(table[s][lab] for s in table for lab in table[s])
    assert grand == FULL_COHORT_N, "distribution does not cover the cohort"


def test_strong_consensus_equals_5_0_plus_4_1(strata, full_cohort):
    """5-0 + 4-1 reconstructs the strong-consensus cohort (size and label split).

    Documents the relationship the task asks us to pin down: the strong-consensus
    (>= 4/5) cohort is exactly the union of the two strongest vote strata.
    """
    from vote_strength import build_distribution

    strong = sum(1 for s in strata.values() if s in ("5-0", "4-1"))
    assert strong == STRONG_CONSENSUS_N, f"5-0 + 4-1 = {strong}, expected 198"
    assert strong == EXPECTED_STRATA["5-0"]["total"] + EXPECTED_STRATA["4-1"]["total"]

    table = build_distribution(full_cohort, strata)
    acc = table["5-0"]["Acceptable"] + table["4-1"]["Acceptable"]
    nacc = table["5-0"]["Not Acceptable"] + table["4-1"]["Not Acceptable"]
    assert acc == STRONG_CONSENSUS_ACCEPTABLE, f"5-0+4-1 Acceptable = {acc}"
    assert nacc == STRONG_CONSENSUS_NOT_ACCEPTABLE, f"5-0+4-1 N.A. = {nacc}"


def test_stratum_rejects_non_five_vote_record():
    """vote_strength_stratum fails loud (rule 15) on a record without 5 votes."""
    from vote_strength import vote_strength_stratum

    bad = {"label": {"regular_votes": {"Acceptable": 2, "Not Acceptable": 1}}}
    with pytest.raises(ValueError):
        vote_strength_stratum(bad)
