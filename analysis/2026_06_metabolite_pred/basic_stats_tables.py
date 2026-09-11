#!/usr/bin/env python3
"""Descriptive Tables 1-2 for the paper: organoid / growth / voting summaries.

Everything is derived at runtime from ``data/all_data.json`` through the
``pipeline.data_loader`` accessors and the ``cohorts`` helpers (AGENTS.md
rules 3/16 — no raw ``json.load`` of feature values, no live recomputation of
persisted quantities). ``build_cohort`` is the source of truth for cohort
membership and asserts the exact 198 / 248 sizes and label splits.

Table 1 (organoids & cohorts):
  * the count funnel: all batches -> BA1+BA2 -> IDOR col2 -> strong consensus
  * per-cohort counts by batch, by label, by batch x label, and by cell line

Table 2 (growth & voting):
  * growth: ``mask_area_um2`` mean / median / sd and n per day (full cohort)
  * voting: Dy30 regular-vote-split distribution (5-0 / 4-1 / 3-2) and the
    consensus rate

Writes ``table1_organoids.csv`` and ``table2_growth_voting.csv`` to
``$ANALYSIS_OUTPUT_DIR`` (gitignored) and prints both to stdout so they can go
straight into a PR body.

Run by path (package name starts with a digit):
    make run ARGS="analysis/2026_06_metabolite_pred/basic_stats_tables.py"
    PYTHONPATH=. python analysis/2026_06_metabolite_pred/basic_stats_tables.py
"""

import json
import logging
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd

from pipeline.data_loader import (
    ANALYSIS_OUTPUT_DIR,
    DAY_ORDER,
    HIGH_QUALITY_BATCHES,
    OrganoidDataset,
    _group_records_by_organoid,
    get_batch,
    get_mask_area_um2,
    get_survey_vote_counts,
)

# Sibling modules are imported top-level (package name starts with a digit).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cohorts import ALL_DATA_PATH, COHORT_EXPECTATIONS, build_cohort, col2_membership_filter

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

COHORTS = ("strong-consensus", "full")

# Vote-split margin buckets. Regular votes are 0..5 (odd, so no 2.5-2.5 tie);
# the key is the absolute margin between Acceptable and Not-Acceptable votes.
VOTE_BUCKETS = {5: "5-0", 3: "4-1", 1: "3-2"}
# 4-1 and 5-0 margins are the supermajority (consensus) buckets; 3-2 is not.
CONSENSUS_MARGINS = {3, 5}


# ---------------------------------------------------------------------------
# Table 1 building blocks
# ---------------------------------------------------------------------------

def count_funnel(all_data_path: str = ALL_DATA_PATH, *, csv_path: str | None = None) -> dict[str, int]:
    """Cohort-independent count funnel: all -> BA1+BA2 -> IDOR col2 -> consensus.

    Reads the grouped, unfiltered organoid set and applies the same batch /
    col2 predicates the cohorts use. The final ``strong_consensus`` figure is
    read from the asserted cohort expectations (source of truth).
    """
    with open(all_data_path) as f:
        all_data: dict = json.load(f)
    grouped = _group_records_by_organoid(all_data)

    by_batch: Counter[str] = Counter(info["batch"] for info in grouped.values())
    total_all = len(grouped)
    ba_ids = {oid for oid, info in grouped.items() if info["batch"] in HIGH_QUALITY_BATCHES}

    col2f = col2_membership_filter(csv_path)
    col2_ids = {oid for oid in ba_ids if col2f(oid, grouped[oid]["records_by_day"])}

    # Rule 11 / 14: the col2 restriction must land on exactly the full cohort.
    assert len(col2_ids) == COHORT_EXPECTATIONS["full"]["n"], (
        f"IDOR col2 = {len(col2_ids)}, expected {COHORT_EXPECTATIONS['full']['n']}"
    )

    funnel = {
        "total_all_batches": total_all,
        "BA1_BA2": len(ba_ids),
        "IDOR_col2": len(col2_ids),
        "strong_consensus": COHORT_EXPECTATIONS["strong-consensus"]["n"],
    }
    # Record the excluded batches so the "others excluded" note is data-driven.
    for batch in sorted(by_batch):
        if batch not in HIGH_QUALITY_BATCHES:
            funnel[f"excluded_batch_{batch}"] = by_batch[batch]
    return funnel


def cohort_batch_label_counts(ds: OrganoidDataset) -> tuple[dict[tuple[str, str], int], dict[str, tuple[str, str]]]:
    """(counts_by_(batch,label), org_id -> (batch,label)) for one cohort.

    Each organoid is assigned to exactly one (batch, label) cell. The per-id map
    is returned so callers/tests can prove no organoid is double-counted.
    """
    counts: Counter[tuple[str, str]] = Counter()
    id_cell: dict[str, tuple[str, str]] = {}
    for oid, info in sorted(ds.iter_organoids()):  # rule 17: deterministic order
        any_rec = next(iter(info["records"].values()))
        batch = get_batch(any_rec)
        label = info["label"]
        assert batch is not None, f"{oid}: missing batch"
        cell = (batch, label)
        counts[cell] += 1
        id_cell[oid] = cell
    return dict(counts), id_cell


def build_table1(all_data_path: str = ALL_DATA_PATH) -> pd.DataFrame:
    """Assemble Table 1 (long format: metric x cohort) as a DataFrame."""
    funnel = count_funnel(all_data_path)

    # Per-cohort building blocks.
    per_cohort: dict[str, dict] = {}
    for name in COHORTS:
        ds, label_counts = build_cohort(name, all_data_path)
        bl_counts, id_cell = cohort_batch_label_counts(ds)
        n = len(ds.organoid_ids)

        # Rule 14: batch x label cells must partition the cohort exactly, with
        # every organoid counted once (no double-counting, no silent drop).
        assert sum(bl_counts.values()) == n, (
            f"{name}: batch x label cells sum to {sum(bl_counts.values())}, expected {n}"
        )
        assert len(id_cell) == n, f"{name}: {len(id_cell)} ids mapped, expected {n}"
        batch_totals = Counter()
        for (batch, _label), c in bl_counts.items():
            batch_totals[batch] += c
        assert sum(batch_totals.values()) == n, f"{name}: batch subtotals != {n}"
        assert sum(label_counts.values()) == n, f"{name}: label subtotals != {n}"

        cell_lines = Counter()
        for _oid, info in ds.iter_organoids():
            any_rec = next(iter(info["records"].values()))
            cell_lines[any_rec.get("cell_line")] += 1

        per_cohort[name] = {
            "n": n,
            "label": label_counts,
            "batch": dict(batch_totals),
            "batch_label": bl_counts,
            "cell_line": dict(cell_lines),
        }

    batches = sorted(HIGH_QUALITY_BATCHES)
    labels = ["Acceptable", "Not Acceptable"]
    cell_lines = sorted({cl for c in per_cohort.values() for cl in c["cell_line"]})

    rows: list[dict] = []

    def _row(section: str, metric: str, sc: int, full: int) -> None:
        rows.append(
            {"section": section, "metric": metric, "strong_consensus": sc, "full": full}
        )

    # Funnel is cohort-independent (same value in both columns) except the last
    # row, which is by construction the strong-consensus cohort size.
    _row("funnel", "total_all_batches", funnel["total_all_batches"], funnel["total_all_batches"])
    for key in sorted(k for k in funnel if k.startswith("excluded_batch_")):
        batch = key.removeprefix("excluded_batch_")
        _row("funnel", f"excluded_{batch}", funnel[key], funnel[key])
    _row("funnel", "BA1_BA2", funnel["BA1_BA2"], funnel["BA1_BA2"])
    _row("funnel", "IDOR_col2", funnel["IDOR_col2"], funnel["IDOR_col2"])

    _row("cohort", "cohort_size", per_cohort["strong-consensus"]["n"], per_cohort["full"]["n"])

    for batch in batches:
        _row(
            "by_batch",
            f"batch_{batch}",
            per_cohort["strong-consensus"]["batch"].get(batch, 0),
            per_cohort["full"]["batch"].get(batch, 0),
        )
    for label in labels:
        _row(
            "by_label",
            f"label_{label.replace(' ', '_')}",
            per_cohort["strong-consensus"]["label"].get(label, 0),
            per_cohort["full"]["label"].get(label, 0),
        )
    for batch in batches:
        for label in labels:
            _row(
                "by_batch_label",
                f"{batch}_{label.replace(' ', '_')}",
                per_cohort["strong-consensus"]["batch_label"].get((batch, label), 0),
                per_cohort["full"]["batch_label"].get((batch, label), 0),
            )
    for cl in cell_lines:
        _row(
            "by_cell_line",
            f"cell_line_{cl}",
            per_cohort["strong-consensus"]["cell_line"].get(cl, 0),
            per_cohort["full"]["cell_line"].get(cl, 0),
        )

    return pd.DataFrame(rows, columns=["section", "metric", "strong_consensus", "full"])


# ---------------------------------------------------------------------------
# Table 2 building blocks
# ---------------------------------------------------------------------------

def growth_by_day(ds: OrganoidDataset) -> pd.DataFrame:
    """Per-day organoid-size (``mask_area_um2``) summary for one cohort.

    Reads the persisted ``mask_area_um2`` (rule 16 — never recompute). Organoids
    that lack a value on a given day are dropped from that day's stats and the
    dropped count is logged at WARNING (rule 15). Days are visited in
    ``DAY_ORDER`` (rule 17).
    """
    n0 = len(ds.organoid_ids)
    rows: list[dict] = []
    for day in DAY_ORDER:
        vals: list[float] = []
        for _oid, info in ds.iter_organoids():
            rec = info["records"].get(day)
            if rec is None:
                continue
            area = get_mask_area_um2(rec)
            if area is not None:
                vals.append(float(area))
        arr = np.array(vals, dtype=float)
        n_missing = n0 - arr.size
        if n_missing:
            logger.warning(
                "growth: %s dropped from mask_area_um2 stats on %s (no persisted area)",
                n_missing,
                day,
            )
        rows.append(
            {
                "section": "growth_by_day",
                "group": day,
                "n": int(arr.size),
                "n_missing": int(n_missing),
                "mean_um2": float(arr.mean()) if arr.size else float("nan"),
                "median_um2": float(np.median(arr)) if arr.size else float("nan"),
                "sd_um2": float(arr.std(ddof=1)) if arr.size > 1 else float("nan"),
                "fraction": float("nan"),
            }
        )
    return pd.DataFrame(rows)


def voting_summary(ds: OrganoidDataset) -> pd.DataFrame:
    """Dy30 regular-vote-split distribution and consensus rate for one cohort.

    Buckets each organoid by its vote margin (5-0 / 4-1 / 3-2) and reports the
    consensus rate = share of organoids whose margin is a supermajority.
    """
    n0 = len(ds.organoid_ids)
    buckets: Counter[str] = Counter()
    n_consensus = 0
    n_no_vote = 0
    for _oid, info in sorted(ds.iter_organoids()):  # rule 17
        rec = info["records"].get("Dy30")
        if rec is None:
            n_no_vote += 1
            continue
        acc, total = get_survey_vote_counts(rec)
        if total == 0:
            n_no_vote += 1
            continue
        margin = abs(acc - (total - acc))
        label = VOTE_BUCKETS.get(margin, f"margin_{margin}")
        buckets[label] += 1
        if margin in CONSENSUS_MARGINS:
            n_consensus += 1
    if n_no_vote:
        logger.warning("voting: %s organoids had no Dy30 regular votes", n_no_vote)

    counted = sum(buckets.values())
    assert counted + n_no_vote == n0, (
        f"voting: {counted}+{n_no_vote} != cohort {n0}"
    )

    rows: list[dict] = []
    for margin in sorted(VOTE_BUCKETS, reverse=True):  # 5-0, 4-1, 3-2
        label = VOTE_BUCKETS[margin]
        c = buckets.get(label, 0)
        rows.append(
            {
                "section": "vote_split",
                "group": label,
                "n": c,
                "n_missing": 0,
                "mean_um2": float("nan"),
                "median_um2": float("nan"),
                "sd_um2": float("nan"),
                "fraction": c / n0 if n0 else float("nan"),
            }
        )
    rows.append(
        {
            "section": "consensus",
            "group": "consensus (>=4-1)",
            "n": n_consensus,
            "n_missing": 0,
            "mean_um2": float("nan"),
            "median_um2": float("nan"),
            "sd_um2": float("nan"),
            "fraction": n_consensus / n0 if n0 else float("nan"),
        }
    )
    rows.append(
        {
            "section": "consensus",
            "group": "no_consensus (3-2)",
            "n": n0 - n_consensus,
            "n_missing": 0,
            "mean_um2": float("nan"),
            "median_um2": float("nan"),
            "sd_um2": float("nan"),
            "fraction": (n0 - n_consensus) / n0 if n0 else float("nan"),
        }
    )
    return pd.DataFrame(rows)


def build_table2(all_data_path: str = ALL_DATA_PATH) -> pd.DataFrame:
    """Assemble Table 2 (growth-by-day + voting) on the full 248 cohort."""
    ds, _counts = build_cohort("full", all_data_path)
    growth = growth_by_day(ds)
    voting = voting_summary(ds)
    return pd.concat([growth, voting], ignore_index=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(all_data_path: str = ALL_DATA_PATH) -> None:
    ANALYSIS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    table1 = build_table1(all_data_path)
    table2 = build_table2(all_data_path)

    t1_path = ANALYSIS_OUTPUT_DIR / "table1_organoids.csv"
    t2_path = ANALYSIS_OUTPUT_DIR / "table2_growth_voting.csv"
    table1.to_csv(t1_path, index=False)
    table2.to_csv(t2_path, index=False)

    with pd.option_context("display.max_rows", None, "display.width", 120):
        print("\n=== Table 1: organoids & cohorts ===")
        print(table1.to_string(index=False))
        print("\n=== Table 2: growth (full cohort, mask_area_um2) & voting ===")
        print(table2.to_string(index=False))

    logger.info("\nSaved %s", t1_path)
    logger.info("Saved %s", t2_path)


if __name__ == "__main__":
    main()
