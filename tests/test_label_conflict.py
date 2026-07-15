"""Regression tests for the merge-step organoid-label conflict logic
(``pipeline/merge/normalized_records.py``), guarding beads zsr.

The conflict logic must fire ONLY on same-day split variants that disagree — it
must NOT clear an organoid because two *different days* disagree (the historical
bug that wiped ``BA1 96_1 E10``, which is Not Acceptable at Dy28 but Acceptable
at the canonical Dy30). Silent label loss is this repo's highest-impact failure
mode (AGENTS.md rule 11), so both the logic and the shipped data are pinned here.
"""

from collections import defaultdict

from pipeline.merge.normalized_records import (
    SURVEY_LABEL_DAY,
    OrganoidRecordBuilder,
    RecordMetrics,
)


def _builder() -> OrganoidRecordBuilder:
    return OrganoidRecordBuilder(record_metrics=RecordMetrics())


# ---- logic-level: exercise _get_organoid_labels directly (no data needed) ----

def test_cross_day_disagreement_does_not_clear():
    """Dy28 'Not Acceptable' then Dy30 'Acceptable' must keep the Dy30 label."""
    b = _builder()
    # Non-canonical day is ignored for the organoid-wide label (returned as-is).
    r_dy28 = b._get_organoid_labels("BA1_96_1_E10", "src_dy28",
                                    {"value": "Not Acceptable"}, "Dy28")
    assert r_dy28 == {"value": "Not Acceptable"}
    # Canonical-day label must survive the earlier-day disagreement.
    r_dy30 = b._get_organoid_labels("BA1_96_1_E10", "src_dy30",
                                    {"value": "Acceptable"}, SURVEY_LABEL_DAY)
    assert r_dy30 == {"value": "Acceptable"}, "cross-day disagreement wrongly cleared Dy30"
    assert "BA1_96_1_E10" not in b.conflicted_organoids


def test_same_day_split_conflict_still_clears():
    """Two SURVEY_LABEL_DAY variants of one organoid that disagree ARE cleared."""
    b = _builder()
    r1 = b._get_organoid_labels("BA2_96_2_B3", "src_split1",
                                {"value": "Acceptable"}, SURVEY_LABEL_DAY)
    assert r1 == {"value": "Acceptable"}
    r2 = b._get_organoid_labels("BA2_96_2_B3", "src_split2",
                                {"value": "Not Acceptable"}, SURVEY_LABEL_DAY)
    assert r2 == {}, "conflicting same-day split labels should clear"
    assert "BA2_96_2_B3" in b.conflicted_organoids


def test_none_then_definitive_upgrades():
    """A no-majority label is upgraded by a later definitive same-day label."""
    b = _builder()
    b._get_organoid_labels("ORG_X", "s1", {"value": None}, SURVEY_LABEL_DAY)
    r = b._get_organoid_labels("ORG_X", "s2", {"value": "Acceptable"}, SURVEY_LABEL_DAY)
    assert r == {"value": "Acceptable"}
    assert "ORG_X" not in b.conflicted_organoids
    assert b.organoid_dict["ORG_X"]["label"]["value"] == "Acceptable"


def test_agreeing_same_day_labels_no_conflict():
    b = _builder()
    b._get_organoid_labels("ORG_Y", "s1", {"value": "Acceptable"}, SURVEY_LABEL_DAY)
    r = b._get_organoid_labels("ORG_Y", "s2", {"value": "Acceptable"}, SURVEY_LABEL_DAY)
    assert r == {"value": "Acceptable"}
    assert "ORG_Y" not in b.conflicted_organoids


# ---- data-level: pin the shipped all_data.json (guards a bad regeneration) ----

def test_e10_dy30_label_is_acceptable(all_data):
    """The historical victim reads its canonical Dy30 verdict, not a cleared label."""
    recs = [r for k, r in all_data.items()
            if isinstance(r, dict) and r.get("organoid_id") == "BA1_96_1_E10"
            and r["day"]["id"] == "Dy30"]
    assert recs, "BA1_96_1_E10 Dy30 record missing"
    assert (recs[0].get("label") or {}).get("value") == "Acceptable"


def test_no_surveyed_organoid_wrongly_cleared(all_data):
    """No surveyed, non-split BA1/BA2 organoid may have its Dy30 label cleared to {}.

    (Organoids with no survey at all legitimately have an empty label; split
    variants that disagree are cleared on purpose — neither is a bug.)
    """
    org = defaultdict(lambda: {"survey": False, "split": False, "dy30_label": None})
    for k, r in all_data.items():
        if not isinstance(r, dict):
            continue
        batch = (r.get("plate") or {}).get("batch") or ""
        if not (batch.startswith("BA1") or batch.startswith("BA2")):
            continue
        oid = r["organoid_id"]
        if "split" in k:
            org[oid]["split"] = True
        if r.get("survey"):
            org[oid]["survey"] = True
        if r["day"]["id"] == "Dy30":
            org[oid]["dy30_label"] = r.get("label")

    offenders = sorted(
        oid for oid, v in org.items()
        if v["survey"] and not v["split"] and v["dy30_label"] == {}
    )
    assert offenders == [], f"surveyed non-split organoids wrongly cleared: {offenders}"
