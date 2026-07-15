"""Tables 1-2 must report the known cohort makeup exactly (AGENTS.md rule 11).

These guard the descriptive-stats module against silent drift in
``data/all_data.json``: the reported counts have to match the frozen cohort
totals (198 / 248) and label splits (165/33, 191/57), and the per-batch
subtotals must partition each cohort with no organoid double-counted.
"""

import math

import pytest
from invariants import assert_organoid_count

# Frozen expectations, duplicated from cohorts.COHORT_EXPECTATIONS so a change
# to that constant can't silently pass its own test.
EXPECTED = {
    "strong-consensus": {"n": 198, "Acceptable": 165, "Not Acceptable": 33},
    "full": {"n": 248, "Acceptable": 191, "Not Acceptable": 57},
}


def _t1_value(table1, metric: str, column: str) -> int:
    row = table1[table1["metric"] == metric]
    assert len(row) == 1, f"metric {metric!r} not unique in Table 1"
    return int(row.iloc[0][column])


def test_table1_cohort_sizes_and_labels():
    from basic_stats_tables import build_table1

    t1 = build_table1()
    col = {"strong-consensus": "strong_consensus", "full": "full"}
    for name, exp in EXPECTED.items():
        c = col[name]
        assert _t1_value(t1, "cohort_size", c) == exp["n"], f"{name}: cohort_size"
        assert _t1_value(t1, "label_Acceptable", c) == exp["Acceptable"], f"{name}: Acceptable"
        assert _t1_value(t1, "label_Not_Acceptable", c) == exp["Not Acceptable"], (
            f"{name}: Not Acceptable"
        )
        # Labels partition the cohort.
        assert (
            _t1_value(t1, "label_Acceptable", c) + _t1_value(t1, "label_Not_Acceptable", c)
            == exp["n"]
        ), f"{name}: labels don't cover the cohort"


def test_table1_batch_subtotals_partition_cohort():
    from basic_stats_tables import build_table1

    t1 = build_table1()
    col = {"strong-consensus": "strong_consensus", "full": "full"}
    for name, exp in EXPECTED.items():
        c = col[name]
        batch_total = _t1_value(t1, "batch_BA1", c) + _t1_value(t1, "batch_BA2", c)
        assert batch_total == exp["n"], f"{name}: batch subtotals sum to {batch_total}, expected {exp['n']}"
        # batch x label cells must also partition the cohort exactly.
        cells = ["BA1_Acceptable", "BA1_Not_Acceptable", "BA2_Acceptable", "BA2_Not_Acceptable"]
        cell_total = sum(_t1_value(t1, m, c) for m in cells)
        assert cell_total == exp["n"], f"{name}: batch x label cells sum to {cell_total}, expected {exp['n']}"


@pytest.mark.parametrize("name", ["strong-consensus", "full"])
def test_no_organoid_double_counted(all_data_path, name):
    from basic_stats_tables import cohort_batch_label_counts
    from cohorts import build_cohort

    ds, _ = build_cohort(name, all_data_path)
    counts, id_cell = cohort_batch_label_counts(ds)
    exp = EXPECTED[name]
    # Every organoid mapped to exactly one (batch, label) cell, no dupes.
    assert_organoid_count(id_cell.keys(), exp["n"], context=f"{name} batch-label map")
    assert sum(counts.values()) == exp["n"], f"{name}: cells don't cover the cohort"


def test_funnel_is_monotone_and_lands_on_cohorts():
    from basic_stats_tables import count_funnel

    f = count_funnel()
    assert f["total_all_batches"] >= f["BA1_BA2"] >= f["IDOR_col2"] >= f["strong_consensus"]
    assert f["IDOR_col2"] == EXPECTED["full"]["n"]
    assert f["strong_consensus"] == EXPECTED["strong-consensus"]["n"]


def test_table2_growth_and_voting():
    from basic_stats_tables import build_table2

    t2 = build_table2()
    # Growth: one row per day, n never exceeds the full cohort.
    growth = t2[t2["section"] == "growth_by_day"]
    assert len(growth) == 11, "expected 11 day rows"
    assert (growth["n"] + growth["n_missing"] == EXPECTED["full"]["n"]).all(), (
        "growth n + n_missing must equal the full cohort on every day"
    )

    # Voting: the three margin buckets partition the full cohort, and
    # consensus + no_consensus also sum to it.
    splits = t2[t2["section"] == "vote_split"]
    assert int(splits["n"].sum()) == EXPECTED["full"]["n"], "vote splits don't cover the cohort"
    consensus = t2[t2["section"] == "consensus"]
    assert int(consensus["n"].sum()) == EXPECTED["full"]["n"], "consensus rows don't cover the cohort"
    # Consensus fraction is a valid probability.
    cons_frac = float(consensus[consensus["group"].str.startswith("consensus")].iloc[0]["fraction"])
    assert 0.0 <= cons_frac <= 1.0 and not math.isnan(cons_frac)
