"""Lookup builder for Promega-normalized metabolite CSV.

Reads ``data/normalized/CONC_*.csv`` and returns a mapping from canonical
record_id (e.g. ``"BA1 96_1 Dy03 A1"``) to per-assay normalized values
(``win``, ``win_vol_norm``). Used by ``metabolite_mapper`` to fold the
normalized fields into the metabolite intermediate alongside the raw
xlsx-derived fields.

The record_id format mirrors ``metabolite_mapper.get_organoid_id`` exactly so
the merge join is exact.
"""

import logging
import math
import pathlib

import pandas as pd

METABOLITES = ("GlucoseGlo", "GlutamateGlo", "LactateGlo", "PyruvateGlo", "MalateGlo", "BCAAGlo")
NORMALIZED_FIELDS = ("win", "win_vol_norm")


def _null(val) -> float | None:
    if val is None:
        return None
    try:
        if math.isnan(val):
            return None
    except TypeError:
        pass
    return val


def _record_id(organoid: str, day_int: int) -> str | None:
    """Build the canonical metabolite-mapper record_id from CSV columns.

    Mirrors ``pipeline.metabolites.metabolite_mapper.get_organoid_id``:
    - Organoid ``"BA1_96_1_A1"`` → batch/plate ``"BA1 96_1"`` and well ``"A1"``.
    - Day int → ``"Dy{:02d}"``, with ``20``/``21`` aliased to ``"Dy20.5"``.
    """
    parts = organoid.split("_")
    if len(parts) != 4:
        return None
    ba_plate = f"{parts[0]} {parts[1]}_{parts[2]}"
    well = parts[3]
    if day_int in (20, 21):
        day = "Dy20.5"
    else:
        day = f"Dy{day_int:02d}"
    return f"{ba_plate} {day} {well}"


def build_normalized_lookup(csv_path: pathlib.Path) -> dict:
    """Build ``{record_id: {assay: {win, win_vol_norm}}}`` from the CONC CSV.

    Rows with malformed Organoid or Day are skipped with a debug log.
    """
    df = pd.read_csv(csv_path)
    lookup: dict = {}
    skipped = 0

    for _, row in df.iterrows():
        organoid = row.get("Organoid")
        day_raw = row.get("Day")
        if not isinstance(organoid, str) or pd.isna(day_raw):
            skipped += 1
            continue
        try:
            day_int = int(day_raw)
        except (TypeError, ValueError):
            skipped += 1
            continue

        record_id = _record_id(organoid, day_int)
        if record_id is None:
            skipped += 1
            continue

        per_assay: dict = {}
        for met in METABOLITES:
            per_assay[met] = {
                "win": _null(row.get(f"{met}_win")),
                "win_vol_norm": _null(row.get(f"{met}_win_vol_norm")),
            }
        lookup[record_id] = per_assay

    logging.info(
        "Normalized lookup: %d records (%d CSV rows skipped due to malformed Organoid/Day)",
        len(lookup),
        skipped,
    )
    return lookup
