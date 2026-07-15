"""Stability of the winsorization-scope determination (beads qp7).

We were told MalateGLO's stored ``win`` was winsorized over the *whole dataset*
(all days pooled) while the other five metabolites were winsorized *per-day*.
``verify_winsorize_scope`` tests that empirically. These tests pin the resulting
determination so it fails loudly if the data or the metric drifts:

- the five "well-behaved" metabolites fit **per-day** (per-day match rate clears
  the fit bar AND strictly beats whole-dataset), and
- MalateGlo fits **neither** scope -- its ``win`` is a separately-cleaned,
  noise-floor signal, non-monotonic in ``concentration_uM``, so it is not any
  winsorization of the raw concentration. This **refutes** the whole-dataset
  hypothesis for Malate (whole-dataset does not beat per-day).

Values come from the committed ``data/all_data.json`` (AGENTS.md rule 3).
"""

import json

import pytest
from verify_winsorize_scope import (
    ALL_METABOLITES,
    DEFAULT_TOL,
    FIT_THRESHOLD,
    MALATE,
    WELL_BEHAVED,
    evaluate_all,
    scope_match_rates,
)

# A per-day match rate this high counts as "clearly fits"; a rate this low counts
# as "clearly does not fit". The gap between them is what makes the call stable.
FITS_BAR = 0.60
NOFIT_BAR = 0.15
# Per-day must beat whole-dataset by at least this much for a well-behaved
# metabolite (a genuine, non-numerical margin).
MARGIN = 0.005


@pytest.fixture(scope="module")
def scope_rows(all_data_path):
    with open(all_data_path) as f:
        all_data = json.load(f)
    rows = evaluate_all(all_data)
    return {r["metabolite"]: r for r in rows}


def test_all_metabolites_reported(scope_rows):
    assert set(scope_rows) == set(ALL_METABOLITES)
    assert FIT_THRESHOLD <= FITS_BAR, "FITS_BAR must clear the script's fit threshold"


@pytest.mark.parametrize("met", sorted(WELL_BEHAVED))
def test_well_behaved_fit_per_day(scope_rows, met):
    r = scope_rows[met]
    # Per-day clearly reproduces win...
    assert r["verdict"] == "per-day", f"{met}: verdict {r['verdict']!r}, expected per-day"
    assert r["per_day_rate"] >= FITS_BAR, (
        f"{met}: per-day match rate {r['per_day_rate']:.3f} < {FITS_BAR}"
    )
    # ...and does so better than the whole-dataset scope (per-day is the scope
    # the lab used for these five).
    assert r["per_day_rate"] - r["whole_rate"] >= MARGIN, (
        f"{met}: per-day ({r['per_day_rate']:.3f}) does not beat "
        f"whole-dataset ({r['whole_rate']:.3f}) by >= {MARGIN}"
    )


def test_malate_fits_neither_scope(scope_rows):
    r = scope_rows[MALATE]
    assert r["verdict"] == "neither", f"MalateGlo verdict {r['verdict']!r}, expected neither"
    # Neither scope reproduces Malate's win (both far below the fit bar).
    assert r["per_day_rate"] < NOFIT_BAR, f"Malate per-day {r['per_day_rate']:.3f} unexpectedly high"
    assert r["whole_rate"] < NOFIT_BAR, f"Malate whole {r['whole_rate']:.3f} unexpectedly high"


def test_malate_whole_dataset_hypothesis_refuted(scope_rows):
    # The hypothesis was that Malate is winsorized whole-dataset. Empirically the
    # whole-dataset scope does NOT beat per-day (it is essentially tied and, if
    # anything, marginally worse), so the hypothesis is refuted.
    r = scope_rows[MALATE]
    assert r["whole_rate"] <= r["per_day_rate"] + MARGIN, (
        f"whole-dataset ({r['whole_rate']:.3f}) beats per-day ({r['per_day_rate']:.3f}) "
        "for Malate by a clear margin -- the whole-dataset hypothesis would be supported, "
        "not refuted; revisit the finding."
    )


def test_record_count_conserved(scope_rows, all_data_path):
    # Rule 11: winsorization must not add/drop records. scope_match_rates already
    # asserts per-scope size == n0 internally; here we re-derive n and confirm the
    # reported count is the same non-trivial per-metabolite total on a re-run
    # (deterministic, rule 17).
    with open(all_data_path) as f:
        all_data = json.load(f)
    for met in ALL_METABOLITES:
        r = scope_rows[met]
        again = scope_match_rates(all_data, met, tol=DEFAULT_TOL)
        assert again["n"] == r["n"], f"{met}: record count not deterministic"
        assert r["n"] > 2000, f"{met}: only {r['n']} assay records -- unexpected drop"
