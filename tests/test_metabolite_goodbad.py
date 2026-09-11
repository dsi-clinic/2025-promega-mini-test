"""Good/bad split integrity for the Figure-2 through-time summary (AGENTS.md
rules 11 & 17). ``split_ids_by_label`` partitions the cohort by Dy30 label; these
tests assert the split conserves the organoid count, neither group is empty, and
the partition is stable across calls.
"""

from invariants import assert_count_conserved, assert_organoid_count

# Canonical full-cohort split on the current all_data.json (mirrors
# cohorts.COHORT_EXPECTATIONS["full"]); duplicated so a change to the source
# constant can't silently pass its own test.
EXPECTED = {"n": 248, "Acceptable": 191, "Not Acceptable": 57}


def test_goodbad_split_counts_sum_to_cohort(full_cohort):
    from metabolite_summary_goodbad import GROUPS, split_ids_by_label

    ds = full_cohort
    groups = split_ids_by_label(ds)

    # Neither group is empty and each matches the known label split.
    for g in GROUPS:
        assert groups[g], f"group {g!r} is empty"
    assert len(groups["Acceptable"]) == EXPECTED["Acceptable"], "Acceptable count"
    assert len(groups["Not Acceptable"]) == EXPECTED["Not Acceptable"], "N.A. count"

    # good + bad == cohort total (191 + 57 == 248) — count conserved (rule 11).
    combined = groups["Acceptable"] + groups["Not Acceptable"]
    assert (
        len(groups["Acceptable"]) + len(groups["Not Acceptable"]) == EXPECTED["n"]
    ), "good + bad must sum to the cohort total"
    assert_organoid_count(combined, EXPECTED["n"], context="goodbad-split")
    # The split neither adds nor drops an organoid vs the source cohort.
    assert_count_conserved(combined, ds.organoid_ids, context="goodbad-split")


def test_goodbad_split_is_disjoint_and_stable(full_cohort):
    from metabolite_summary_goodbad import split_ids_by_label

    ds = full_cohort
    groups = split_ids_by_label(ds)

    # Groups are disjoint (no organoid labeled both good and bad).
    good, bad = set(groups["Acceptable"]), set(groups["Not Acceptable"])
    assert good.isdisjoint(bad), "an organoid appears in both groups"

    # Ids are sorted (deterministic ordering, rule 17).
    assert groups["Acceptable"] == sorted(groups["Acceptable"]), "good ids not sorted"
    assert groups["Not Acceptable"] == sorted(groups["Not Acceptable"]), "bad ids not sorted"

    # The split is stable: a second call reproduces it byte-for-byte.
    groups2 = split_ids_by_label(ds)
    assert groups == groups2, "split is not stable across calls"
