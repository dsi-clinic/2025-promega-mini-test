"""Proof that the winsorized metabolite data we use is regenerable from source
(AGENTS.md rules 3 & 16). For each well-behaved metabolite we recompute
``win`` as ``k * per-day-winsorized(concentration_uM)`` straight from
``all_data.json`` and require a high per-record match rate within a small
relative tolerance; MalateGlo is asserted to be the documented exception (a
separately-cleaned noise-floor signal that no winsorization reproduces).
"""

import numpy as np

from pipeline.metabolites.winsorize import (
    MALATE,
    MALATE_MATCH_CEIL,
    MATCH_FLOOR,
    MATCH_TOL,
    WELL_BEHAVED,
    _collect_raw_and_win,
    verify_winsorization,
    winsorize_per_day,
)

# The three metabolites whose win regenerates almost exactly — held to a much
# tighter bound as an extra guardrail (measured ~0.96-0.98 within 1%).
TIGHT_TRIO = ("GlucoseGlo", "LactateGlo", "BCAAGlo")


def test_win_regenerable_from_source(all_data):
    """verify_winsorization proves regeneration (raises if any metabolite fails)."""
    report = verify_winsorization(all_data)  # asserts internally

    for m in WELL_BEHAVED:
        assert m in report, f"{m} missing from regeneration report"
        # per-record match rate clears the metabolite's floor within MATCH_TOL
        assert report[m]["match_rate"] >= MATCH_FLOOR[m], (
            f"{m}: match {report[m]['match_rate']:.3f} < floor {MATCH_FLOOR[m]}"
        )
        # and the bulk lines up (median relative residual is small)
        assert report[m]["median_rel_resid"] < 0.03, f"{m}: median resid too high"

    # MalateGlo must FAIL both checks — it is not a winsorization of the raw.
    assert report[MALATE]["match_rate"] < MALATE_MATCH_CEIL, "Malate unexpectedly regenerates"
    assert report[MALATE]["median_rel_resid"] > 0.03, "Malate median resid unexpectedly small"


def test_clean_trio_regenerates_within_one_percent(all_data):
    """Glucose / Lactate / BCAA reproduce >= 95% of records within 1% relative."""
    for m in TIGHT_TRIO:
        raw, win, days = _collect_raw_and_win(all_data, m)
        our = winsorize_per_day(raw, days)
        nz = np.abs(our) > 1e-9
        k = float(np.median(win[nz] / our[nz]))
        rel = np.abs(win[nz] - k * our[nz]) / np.abs(k * our[nz])
        rate = float(np.mean(rel < 0.01))
        assert rate >= 0.95, f"{m}: only {rate:.1%} regenerate within 1% (expected >= 95%)"


def test_match_floors_leave_margin_over_malate(all_data):
    """Every well-behaved floor sits well above what a non-winsorized signal scores."""
    report = verify_winsorization(all_data)
    malate_rate = report[MALATE]["match_rate"]
    for m in WELL_BEHAVED:
        # the floor (and the actual rate) must clear Malate's rate by a wide gap,
        # so the test can actually distinguish "regenerable" from "not".
        assert MATCH_FLOOR[m] - malate_rate > 0.3, f"{m}: floor too close to Malate"
        assert report[m]["match_rate"] - malate_rate > 0.4, f"{m}: rate too close to Malate"
    assert MATCH_TOL == 0.05  # guards the documented tolerance this suite assumes
