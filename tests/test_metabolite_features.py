"""Feature/label join integrity for the metabolite matrices (AGENTS.md rules
11 & 14). The feature builder joins per-day metabolite records to labels; these
tests assert the join neither invents nor mismatches rows.
"""

import numpy as np
from invariants import assert_subset

DAY = "Dy30"


def test_feature_matrix_join_integrity(full_cohort):
    ds = full_cohort
    X, y, names, ids = ds.get_metabolite_features(
        "all", DAY, include_growth=False, include_initial=True,
        normalize_by_size=False, winsorize=False,
    )
    # Rule 14: the X / y / ids join must be internally consistent.
    assert X.shape[0] == len(y) == len(ids), "X/y/ids row-count mismatch"
    assert X.shape[1] == len(names), "feature-name count != X columns"
    assert len(set(ids)) == len(ids), "duplicate organoid ids in feature matrix"
    # A given day can miss some organoids, but the join must never *add* one.
    assert_subset(ids, ds.organoid_ids, context=f"features@{DAY}")
    assert set(np.unique(y)).issubset({0, 1}), "labels must be binary 0/1"
    assert not np.isnan(X).any(), "feature matrix has NaNs"


def test_growth_features_only_reduce(full_cohort):
    ds = full_cohort
    _, _, _, ids_nogrow = ds.get_metabolite_features("all", DAY, include_growth=False)
    _, _, _, ids_grow = ds.get_metabolite_features("all", DAY, include_growth=True)
    # Growth needs a previous timepoint, so it can only drop organoids, never
    # add them (rule 11).
    assert_subset(ids_grow, ids_nogrow, context="growth vs no-growth")


def test_winsorized_features_preserve_organoids(full_cohort):
    ds = full_cohort
    _, _, _, ids_raw = ds.get_metabolite_features("all", DAY, winsorize=False)
    _, _, _, ids_win = ds.get_metabolite_features("all", DAY, winsorize=True)
    # Winsorizing clips values; it must not change *which* organoids are present.
    assert set(ids_win) == set(ids_raw), "winsorize changed the organoid set"
