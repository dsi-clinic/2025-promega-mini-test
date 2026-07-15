"""Shared pytest fixtures for the test suite.

Tests run against the committed ``data/all_data.json`` — the single source of
truth (AGENTS.md rule 3) — so they exercise the real schema rather than a mock.
The repo root and the digit-prefixed ``analysis/2026_06_metabolite_pred``
package (which is imported by path, not as a module) are put on ``sys.path``.
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
_METAB_PKG = REPO_ROOT / "analysis" / "2026_06_metabolite_pred"
for _p in (REPO_ROOT, _METAB_PKG):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

ALL_DATA = REPO_ROOT / "data" / "all_data.json"


@pytest.fixture(scope="session")
def all_data_path() -> str:
    assert ALL_DATA.exists(), f"missing {ALL_DATA}; run `make pipeline-merge` first"
    return str(ALL_DATA)


@pytest.fixture(scope="session")
def all_data(all_data_path) -> dict:
    """The raw ``all_data.json`` dict (record-id -> record). Loaded once."""
    import json

    with open(all_data_path) as f:
        return json.load(f)


@pytest.fixture(scope="session")
def full_cohort(all_data_path):
    """The ``full`` cohort (248) with a single 'all' split, as run.py builds it.

    Session-scoped so the ~22MB all_data.json is loaded once for the suite.
    """
    from cohorts import build_cohort

    from pipeline.splits import Splits

    ds, _counts = build_cohort("full", all_data_path)
    ds.apply_splits(
        Splits.from_dict(
            {oid: "all" for oid in ds.organoid_ids},
            name="cv_all_full",
            provenance="test fixture (conftest.full_cohort)",
        ),
        strict=True,
    )
    return ds
