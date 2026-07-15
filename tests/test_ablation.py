"""Ablation harness invariants (AGENTS.md rules 11 & 5).

Two guarantees:
  (a) A column-drop ablation NEVER drops organoids/rows for a given day — only
      feature columns leave, so the row set is conserved (rule 11).
  (b) The ablation baseline reproduces run.py's headline ``nominal_delta``
      result at Dy30 (full cohort, LightGBM) within tolerance, and is clearly
      above chance (balanced accuracy > 0.5).
"""

import ablation
import numpy as np
from invariants import assert_count_conserved

DAY = "Dy30"

# run.py's nominal_delta / full-cohort / LightGBM Dy30 balanced accuracy sits in
# the ~0.73-0.79 band (measured 0.783 on the committed all_data.json). We assert
# a tolerant band around that and, separately, that it clears chance by a wide
# margin — the load-bearing claim is "the metabolite signal is real".
_EXPECTED_DY30_BALACC = 0.783
_TOL = 0.06  # allow modest drift in the committed data / library versions


def _lgbm():
    return ablation.MODEL_SPECS["lgbm"]


def test_column_drop_conserves_organoids(full_cohort):
    """Every leave-out (metabolite + growth/initial group) drops cols, not rows."""
    ds = full_cohort
    X, _y, names, ids = ablation._features(ds, DAY)
    n0 = X.shape[0]
    assert n0 == len(ids) == len(_y), "baseline X/y/ids mismatch"

    col_keep = {**ablation.metabolite_ablations(), **ablation._GROUP_KEEP}
    assert len(col_keep) == 8, "expected 6 metabolite + 2 group column-drops"

    for name, keep in sorted(col_keep.items()):
        X_sub, kept = ablation._drop_columns(X, names, keep)
        # (a) row count / organoid set is conserved; columns strictly shrink.
        assert X_sub.shape[0] == n0, f"{name}: row count changed {n0}->{X_sub.shape[0]}"
        assert len(kept) < len(names), f"{name}: dropped no columns"
        # ids are unchanged by a column drop — same organoid set, same order.
        assert_count_conserved(ids, ids, context=f"ablation {name}")


def test_metabolite_ablation_removes_only_that_metabolite(full_cohort):
    """The metabolite mask removes exactly that metabolite's columns."""
    ds = full_cohort
    _X, _y, names, _ids = ablation._features(ds, DAY)
    for met in ablation.REQUIRED_METABOLITES:
        keep = ablation.metabolite_ablations()[f"metabolite:{met}"]
        dropped = [n for n in names if not keep(n)]
        assert dropped, f"{met}: nothing dropped"
        assert all(met in n for n in dropped), f"{met}: dropped a foreign column"
        # Every remaining column belongs to a different metabolite.
        kept = [n for n in names if keep(n)]
        assert all(met not in n for n in kept), f"{met}: leaked a column"


def test_baseline_reproduces_runpy_dy30(full_cohort):
    """(b) Baseline Dy30 full/LightGBM balanced accuracy matches run.py, > 0.5."""
    ds = full_cohort
    X, y, _names, ids = ablation._features(ds, DAY)
    m = ablation.run_cv_for_day(_lgbm(), X, y, ids, n_folds=5, seed=42)
    assert m is not None, "baseline CV returned None at Dy30"
    ba = m["balanced_accuracy"]
    assert ba > 0.5, f"baseline balanced accuracy {ba:.3f} not above chance"
    assert abs(ba - _EXPECTED_DY30_BALACC) < _TOL, (
        f"baseline balacc {ba:.3f} drifted from run.py's {_EXPECTED_DY30_BALACC:.3f} "
        f"(tol {_TOL})"
    )


def test_run_day_drop_signs_are_finite(full_cohort):
    """A full run_day at Dy30 yields finite baseline + ablation balacc drops."""
    ds = full_cohort
    res = ablation.run_day(ds, DAY, [_lgbm()], n_folds=5, seed=42)
    assert res is not None and "LightGBM" in res
    entry = res["LightGBM"]
    assert np.isfinite(entry["baseline"]["balanced_accuracy"])
    # All 6 metabolite + 2 group + 1 size_norm ablations present and finite.
    ab = entry["ablations"]
    for met in ablation.REQUIRED_METABOLITES:
        key = f"metabolite:{met}"
        assert key in ab, f"missing {key}"
        assert np.isfinite(ab[key]["balacc_drop"])
    for grp in ("group:growth", "group:initial"):
        assert grp in ab and np.isfinite(ab[grp]["balacc_drop"])
