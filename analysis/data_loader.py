"""
Core runtime data loader for paper reproducibility.

Loads all_data.json + organoid_splits.csv and provides composable filters
and label derivation. Everything except split assignment is derived at runtime
from all_data.json, making it trivial to swap filters for new experiments.

Usage:
    from analysis.data_loader import OrganoidDataset, default_filters, paper_label_fn

    ds = OrganoidDataset("data/all_data.json", "data/organoid_splits.csv")
    ds.summary()

    train = ds.get_split("train", day="Dy13")
    X, y, ids = ds.get_metabolite_features("train", day="Dy13")
"""

import csv
import json
import re
import warnings
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Constants matching the paper
# ---------------------------------------------------------------------------

REQUIRED_METABOLITES = ["GlucoseGlo", "GlutamateGlo", "LactateGlo", "PyruvateGlo"]
CONDITIONAL_METABOLITES = {"MalateGlo": lambda day_num: day_num > 10}
EXCLUDED_METABOLITES = {"BCAAGlo"}

LABEL_DAY = "Dy30"
HIGH_QUALITY_BATCHES = ("BA1", "BA2")
MIN_VOTES = 4

# Day ordering used throughout analysis
DAY_ORDER = [
    "Dy03", "Dy06", "Dy08", "Dy10", "Dy13",
    "Dy15", "Dy17", "Dy20_5", "Dy24", "Dy28", "Dy30",
]

# Map raw day IDs to canonical day IDs (Dy20/Dy21 → Dy20_5)
DAY_ALIAS = {"Dy20": "Dy20_5", "Dy21": "Dy20_5"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_organoid_id(record_key: str) -> str:
    """Strip the day component to get the organoid identity.

    'BA1 96_1 Dy30 A1' → 'BA1 96_1 A1'
    """
    m = re.match(r"^(.*)\s+Dy\d+\s+(.*)$", record_key)
    return f"{m.group(1)} {m.group(2)}" if m else record_key


def _get_batch(record: dict) -> Optional[str]:
    """Extract top-level batch prefix (e.g. 'BA1') from a record."""
    ba = record.get("BA", "")
    return ba.split()[0] if ba else None


def _get_day_number(day_id: str) -> Optional[int]:
    """'Dy13' → 13, 'Dy20_5' → 20."""
    m = re.match(r"Dy(\d+)", day_id or "")
    return int(m.group(1)) if m else None


def _canonicalize_day(day_id: str) -> str:
    return DAY_ALIAS.get(day_id, day_id)


def _compute_majority_label(
    evaluations: list, min_votes: int = MIN_VOTES
) -> Optional[str]:
    """Return consensus label if ≥ min_votes agree, else None."""
    if not evaluations or len(evaluations) != 5:
        return None
    votes: Dict[str, int] = {}
    for e in evaluations:
        v = e.get("evaluation", "")
        if v:
            votes[v] = votes.get(v, 0) + 1
    for label in ("Acceptable", "Not Acceptable"):
        if votes.get(label, 0) >= min_votes:
            return label
    return None


# ---------------------------------------------------------------------------
# Composable filter functions
# ---------------------------------------------------------------------------
# Each filter is  (organoid_id, records_by_day: dict) → bool (True = keep)

def require_batches(*batches: str) -> Callable:
    """Keep organoids belonging to the specified batches."""
    batch_set = set(batches)

    def f(org_id: str, records: dict) -> bool:
        # Use the first available record to get batch
        for rec in records.values():
            return _get_batch(rec) in batch_set
        return False

    f.__doc__ = f"require_batches({', '.join(batches)})"
    return f


def require_complete_metabolites(
    required: Sequence[str] = REQUIRED_METABOLITES,
) -> Callable:
    """Keep organoids that have all required metabolites on days where metabolite
    data is expected.

    Days with no metabolite data at all (e.g. Dy20, a pure imaging timepoint)
    are skipped — they don't disqualify the organoid.  The check is: for every
    day that has *any* metabolite dict, all ``required`` metabolites must be
    present with non-null concentration_uM.
    """

    def f(org_id: str, records: dict) -> bool:
        has_any_met_day = False
        for day_id, rec in records.items():
            mets = rec.get("metabolites")
            if not mets:
                continue  # day has no metabolite data at all — skip
            has_any_met_day = True
            for m in required:
                if m not in mets:
                    return False
                conc = mets[m].get("concentration_uM")
                if conc is None:
                    return False
        return has_any_met_day  # must have at least one day with metabolites

    f.__doc__ = "require_complete_metabolites"
    return f


def require_valid_images() -> Callable:
    """Keep organoids where every day has processed img_path and mask_path."""

    def f(org_id: str, records: dict) -> bool:
        for rec in records.values():
            proc = rec.get("processed")
            if not proc or "img_path" not in proc or "mask_path" not in proc:
                return False
        return True

    f.__doc__ = "require_valid_images"
    return f


def exclude_classification(*types: str) -> Callable:
    """Drop organoids if ANY day has a classification_verification in *types*.

    Common types: 'Stitched', 'Split', 'SplitStitched', 'PreSplit'.
    The verification field uses combined tokens like 'NoSplitNoStitched',
    'SplitNoStitched', etc.  We check the main_id for split/stitch tokens.
    """
    # We check via the main_id which encodes split/stitch info
    exclude_stitched = any("stitch" in t.lower() for t in types)
    exclude_split = any("split" in t.lower() for t in types)

    def f(org_id: str, records: dict) -> bool:
        for rec in records.values():
            mid = (rec.get("main_id") or "").lower()
            img_path = (rec.get("processed", {}).get("img_path") or "").lower()
            combined = mid + " " + img_path
            if exclude_stitched and "stitched" in combined and "nostitch" not in combined:
                return False
            if exclude_split and "split" in combined and "nosplit" not in combined:
                return False
        return True

    f.__doc__ = f"exclude_classification({', '.join(types)})"
    return f


# ---------------------------------------------------------------------------
# Label derivation
# ---------------------------------------------------------------------------

def paper_label_fn(
    org_id: str,
    records: dict,
    min_votes: int = MIN_VOTES,
    label_day: str = LABEL_DAY,
) -> Optional[str]:
    """Derive label from survey consensus at label_day.

    Returns 'Acceptable', 'Not Acceptable', or None (excluded).
    """
    rec = records.get(label_day)
    if rec is None:
        return None
    survey = rec.get("survey")
    if not survey:
        return None
    evaluations = survey.get("evaluations", [])
    return _compute_majority_label(evaluations, min_votes=min_votes)


# ---------------------------------------------------------------------------
# Default configuration matching paper
# ---------------------------------------------------------------------------

def default_filters() -> List[Callable]:
    """Filters used in the paper: BA1+BA2, complete metabolites, valid images."""
    return [
        require_batches(*HIGH_QUALITY_BATCHES),
        require_complete_metabolites(),
        require_valid_images(),
    ]


# ---------------------------------------------------------------------------
# OrganoidDataset
# ---------------------------------------------------------------------------

class OrganoidDataset:
    """Runtime dataset built from all_data.json + split CSV.

    Groups records by organoid, applies filters, derives labels, and provides
    accessors for metabolite features and image paths by split and day.
    """

    def __init__(
        self,
        all_data_path: str,
        splits_csv_path: str,
        filters: Optional[List[Callable]] = None,
        label_fn: Optional[Callable] = None,
    ):
        self.all_data_path = Path(all_data_path)
        self.splits_csv_path = Path(splits_csv_path)
        self.filters = filters if filters is not None else default_filters()
        self.label_fn = label_fn or paper_label_fn

        # Load sources
        with open(self.all_data_path) as f:
            self.all_data: dict = json.load(f)
        self._csv_splits = self._load_splits_csv()

        # Build dataset
        self._organoids: Dict[str, dict] = {}  # org_id → {label, split, records}
        self._build()

    # -- construction --------------------------------------------------------

    def _load_splits_csv(self) -> Dict[str, str]:
        """Load CSV → {organoid_id: split}."""
        splits = {}
        with open(self.splits_csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                splits[row["organoid_id"]] = row["split"]
        return splits

    def _build(self):
        """Group all_data records by organoid, apply filters, derive labels."""
        # Step 1: group records by organoid_id
        grouped: Dict[str, Dict[str, dict]] = {}  # org_id → {canonical_day: record}
        for key, rec in self.all_data.items():
            org_id = _extract_organoid_id(key)
            day_raw = rec.get("dayID", "")
            day = _canonicalize_day(day_raw)
            grouped.setdefault(org_id, {})[day] = rec

        # Step 2: keep only organoids present in the CSV
        csv_ids = set(self._csv_splits.keys())
        dropped_not_in_csv = 0

        for org_id, records in grouped.items():
            if org_id not in csv_ids:
                continue

            # Step 3: apply filters
            keep = True
            for filt in self.filters:
                if not filt(org_id, records):
                    keep = False
                    break
            if not keep:
                dropped_not_in_csv += 1
                continue

            # Step 4: derive label
            label = self.label_fn(org_id, records)
            if label is None:
                dropped_not_in_csv += 1
                continue

            self._organoids[org_id] = {
                "label": label,
                "split": self._csv_splits[org_id],
                "records": records,
            }

        # Warn about CSV organoids that got dropped by filters
        csv_only = csv_ids - set(self._organoids.keys())
        if csv_only:
            warnings.warn(
                f"{len(csv_only)} organoids in CSV were dropped by filters/label derivation"
            )

    # -- accessors -----------------------------------------------------------

    @property
    def organoid_ids(self) -> List[str]:
        return list(self._organoids.keys())

    @property
    def splits(self) -> List[str]:
        return sorted(set(o["split"] for o in self._organoids.values()))

    @property
    def days(self) -> List[str]:
        """All canonical days present across all organoids, sorted."""
        ds = set()
        for o in self._organoids.values():
            ds.update(o["records"].keys())
        return [d for d in DAY_ORDER if d in ds]

    def get_split(
        self, split: str, day: Optional[str] = None
    ) -> Dict[str, dict]:
        """Get organoids for a split, optionally filtered to those having a specific day.

        Returns: {org_id: {label, records: {day: record, ...}}}
        """
        result = {}
        for org_id, info in self._organoids.items():
            if info["split"] != split:
                continue
            if day is not None and day not in info["records"]:
                continue
            result[org_id] = info
        return result

    def get_metabolite_features(
        self,
        split: str,
        day: str,
        include_growth: bool = False,
        include_initial: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
        """Extract metabolite feature matrix for split + day.

        Returns: (X, y, feature_names, organoid_ids)
            X: (n_samples, n_features) float array
            y: (n_samples,) int array  (1=Not Acceptable, 0=Acceptable)
            feature_names: list of feature column names
            organoid_ids: list of organoid IDs corresponding to rows
        """
        day_num = _get_day_number(day)
        subset = self.get_split(split, day=day)

        # Determine which metabolites to include for this day
        active_mets = list(REQUIRED_METABOLITES)
        for met, cond_fn in CONDITIONAL_METABOLITES.items():
            if day_num is not None and cond_fn(day_num):
                active_mets.append(met)

        # Build feature names
        feat_names = []
        for met in active_mets:
            feat_names.append(f"{met}_concentration_uM")
            if include_initial:
                feat_names.append(f"{met}_initial_concentration")

        rows = []
        labels = []
        ids = []

        for org_id, info in subset.items():
            rec = info["records"].get(day)
            if rec is None:
                continue
            mets = rec.get("metabolites", {})

            row = []
            skip = False
            for met in active_mets:
                met_data = mets.get(met, {})
                conc = met_data.get("concentration_uM")
                if conc is None:
                    skip = True
                    break
                row.append(conc)
                if include_initial:
                    row.append(met_data.get("initial_concentration", np.nan))
            if skip:
                continue

            rows.append(row)
            labels.append(1 if info["label"] == "Not Acceptable" else 0)
            ids.append(org_id)

        if not rows:
            return (
                np.empty((0, len(feat_names))),
                np.empty(0, dtype=int),
                feat_names,
                [],
            )

        X = np.array(rows, dtype=float)
        y = np.array(labels, dtype=int)

        # Optionally add growth features (difference from previous day)
        if include_growth and day_num is not None:
            X, feat_names, ids_out = self._add_growth_features(
                X, feat_names, ids, split, day, active_mets, include_initial
            )
            y_out = []
            id_set = set(ids_out)
            for org_id in ids_out:
                y_out.append(
                    1 if self._organoids[org_id]["label"] == "Not Acceptable" else 0
                )
            y = np.array(y_out, dtype=int)
            ids = ids_out

        return X, y, feat_names, ids

    def _add_growth_features(
        self,
        X: np.ndarray,
        feat_names: List[str],
        org_ids: List[str],
        split: str,
        day: str,
        active_mets: List[str],
        include_initial: bool,
    ) -> Tuple[np.ndarray, List[str], List[str]]:
        """Add growth (delta) features from the previous available day."""
        day_idx = DAY_ORDER.index(day) if day in DAY_ORDER else -1
        if day_idx <= 0:
            # No previous day available
            return X, feat_names, org_ids

        prev_day = DAY_ORDER[day_idx - 1]
        prev_day_num = _get_day_number(prev_day)

        # Determine previous-day active metabolites
        prev_mets = list(REQUIRED_METABOLITES)
        for met, cond_fn in CONDITIONAL_METABOLITES.items():
            if prev_day_num is not None and cond_fn(prev_day_num):
                prev_mets.append(met)

        # Only compute growth for metabolites available in both days
        growth_mets = [m for m in active_mets if m in prev_mets]

        growth_names = [f"{m}_growth" for m in growth_mets]
        new_rows = []
        new_ids = []
        keep_indices = []

        for i, org_id in enumerate(org_ids):
            info = self._organoids[org_id]
            prev_rec = info["records"].get(prev_day)
            if prev_rec is None:
                continue

            prev_mets_data = prev_rec.get("metabolites", {})
            growth_row = []
            skip = False
            for m in growth_mets:
                curr_data = info["records"][day].get("metabolites", {}).get(m, {})
                prev_data = prev_mets_data.get(m, {})
                curr_c = curr_data.get("concentration_uM")
                prev_c = prev_data.get("concentration_uM")
                if curr_c is None or prev_c is None:
                    skip = True
                    break
                growth_row.append(curr_c - prev_c)
            if skip:
                continue

            new_rows.append(growth_row)
            new_ids.append(org_id)
            keep_indices.append(i)

        if not new_rows:
            return X, feat_names, org_ids

        X_kept = X[keep_indices]
        growth_arr = np.array(new_rows, dtype=float)
        X_combined = np.hstack([X_kept, growth_arr])
        return X_combined, feat_names + growth_names, new_ids

    def get_image_paths(
        self, split: str, day: str, mode: str = "overlay"
    ) -> List[Tuple[str, str, str]]:
        """Get image paths for split+day.

        Args:
            mode: 'img' | 'mask' | 'overlay'

        Returns: list of (organoid_id, label, path)
        """
        key_map = {"img": "img_path", "mask": "mask_path", "overlay": "overlay_path"}
        path_key = key_map.get(mode, mode)

        subset = self.get_split(split, day=day)
        result = []
        for org_id, info in subset.items():
            rec = info["records"].get(day)
            if rec is None:
                continue
            proc = rec.get("processed", {})
            path = proc.get(path_key)
            if path:
                result.append((org_id, info["label"], path))
        return result

    def get_record(self, org_id: str, day: str) -> Optional[dict]:
        """Get the raw record for an organoid+day."""
        info = self._organoids.get(org_id)
        if info is None:
            return None
        return info["records"].get(day)

    # -- summary -------------------------------------------------------------

    def summary(self) -> str:
        """Return a human-readable dataset summary."""
        lines = []
        lines.append(f"OrganoidDataset: {len(self._organoids)} organoids")
        lines.append(f"  Source: {self.all_data_path}")
        lines.append(f"  Splits CSV: {self.splits_csv_path}")
        lines.append(f"  Days: {', '.join(self.days)}")
        lines.append("")

        for split in ["train", "val", "test"]:
            subset = self.get_split(split)
            if not subset:
                continue
            labels = Counter(v["label"] for v in subset.values())
            total = sum(labels.values())
            lines.append(
                f"  {split:5s}: {total:3d} organoids  "
                f"(Acceptable={labels.get('Acceptable', 0)}, "
                f"Not Acceptable={labels.get('Not Acceptable', 0)})"
            )

            # Day coverage
            day_counts = Counter()
            for v in subset.values():
                for d in v["records"]:
                    day_counts[d] += 1
            day_str = "  ".join(
                f"{d}={day_counts[d]}" for d in DAY_ORDER if d in day_counts
            )
            lines.append(f"         {day_str}")
            lines.append("")

        return "\n".join(lines)

    def __repr__(self):
        return (
            f"OrganoidDataset({len(self._organoids)} organoids, "
            f"splits={self.splits}, days={len(self.days)})"
        )
