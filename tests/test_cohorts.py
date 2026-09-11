"""Cohort construction is the primary guard for organoid-count conservation
(AGENTS.md rule 11): ``build_cohort`` asserts exact sizes + label splits, so
these tests fail loudly if ``data/all_data.json`` drifts.
"""

import pytest
from invariants import assert_organoid_count

# The canonical expectations on the current all_data.json. Mirrors
# cohorts.COHORT_EXPECTATIONS; duplicated here so a change to the source
# constant can't silently pass its own test.
EXPECTED = {
    "strong-consensus": {"n": 198, "Acceptable": 165, "Not Acceptable": 33},
    "full": {"n": 248, "Acceptable": 191, "Not Acceptable": 57},
}


@pytest.mark.parametrize("name", ["strong-consensus", "full"])
def test_cohort_counts(all_data_path, name):
    from cohorts import build_cohort

    ds, counts = build_cohort(name, all_data_path)
    exp = EXPECTED[name]
    assert_organoid_count(ds.organoid_ids, exp["n"], context=name)
    assert counts.get("Acceptable", 0) == exp["Acceptable"], f"{name}: Acceptable count"
    assert counts.get("Not Acceptable", 0) == exp["Not Acceptable"], f"{name}: N.A. count"
    # Label counts must sum to the sample size (no unlabeled organoid slips in).
    assert sum(counts.values()) == exp["n"], f"{name}: labels don't cover the cohort"


def test_unknown_cohort_raises(all_data_path):
    from cohorts import build_cohort

    with pytest.raises(KeyError):
        build_cohort("does-not-exist", all_data_path)
