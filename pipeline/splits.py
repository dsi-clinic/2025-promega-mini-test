"""First-class train/val/test split assignments for organoids.

A ``Splits`` carries the organoid→partition mapping plus a name and provenance
string, so callers can identify which split they're using and compare two
splits for agreement. Constructed via factory classmethods, never via
``__init__`` directly.

Typical use::

    from pipeline.splits import Splits

    splits = Splits.canonical()                          # repo default
    splits = Splits.from_csv("data/splits/other.csv")    # alternate file
    splits = Splits.stratified_random(                    # quick ad-hoc split
        ds.organoid_labels(),
        ratios={"train": 0.8, "val": 0.1, "test": 0.1},
        seed=42,
        name="rand_80_10_10",
    )

    canon = Splits.canonical()
    harriet = Splits.from_csv("data/splits/harriet_2026_05.csv")
    print(canon.agreement_with(harriet))

The mapping is organoid-level (one assignment per ``organoid_id``, applied to
every day's record for that organoid) per AGENTS.md rule #2.
"""

from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Set

from sklearn.model_selection import train_test_split

CANONICAL_PATH = Path("data/splits/canonical_2026_winter.csv")


@dataclass(frozen=True)
class Splits:
    mapping: Mapping[str, str]
    name: str
    provenance: str

    # ---- factories ----

    @classmethod
    def from_csv(
        cls,
        path,
        name: Optional[str] = None,
        provenance: Optional[str] = None,
    ) -> "Splits":
        """Load from a CSV with at minimum columns ``organoid_id`` and ``split``.
        Extra columns (e.g. a redundant ``label``) are ignored.
        """
        path = Path(path)
        mapping: Dict[str, str] = {}
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None or "organoid_id" not in reader.fieldnames or "split" not in reader.fieldnames:
                raise ValueError(
                    f"CSV at {path} must have columns 'organoid_id' and 'split'; got {reader.fieldnames}"
                )
            for row in reader:
                mapping[row["organoid_id"]] = row["split"]
        return cls(
            mapping=mapping,
            name=name or path.stem,
            provenance=provenance or str(path),
        )

    @classmethod
    def from_dict(
        cls,
        mapping: Mapping[str, str],
        name: str,
        provenance: str = "in-memory dict",
    ) -> "Splits":
        return cls(mapping=dict(mapping), name=name, provenance=provenance)

    @classmethod
    def from_partition(
        cls,
        *,
        name: str,
        provenance: str = "from partition lists",
        **partitions: Iterable[str],
    ) -> "Splits":
        """Build from disjoint ID lists, e.g.::

            Splits.from_partition(train=train_ids, val=val_ids, test=test_ids, name="...")
        """
        mapping: Dict[str, str] = {}
        for split_name, ids in partitions.items():
            for oid in ids:
                if oid in mapping:
                    raise ValueError(
                        f"organoid_id {oid!r} appears in both '{mapping[oid]}' and '{split_name}'"
                    )
                mapping[oid] = split_name
        return cls(mapping=mapping, name=name, provenance=provenance)

    @classmethod
    def canonical(cls) -> "Splits":
        """The repo's canonical 2026-winter student splits."""
        return cls.from_csv(
            CANONICAL_PATH,
            name="canonical_2026_winter",
            provenance=str(CANONICAL_PATH),
        )

    @classmethod
    def stratified_random(
        cls,
        organoid_labels: Mapping[str, str],
        *,
        ratios: Mapping[str, float],
        seed: int = 42,
        name: str,
    ) -> "Splits":
        """Build a stratified random split from ``{organoid_id: label}``.

        ``ratios`` is split_name → fraction. 2-way (e.g. ``{"train": 0.8,
        "test": 0.2}``) and 3-way (with ``"val"``) are supported. Fractions
        must sum to ~1.0. Strata are the label values. Provenance records
        the ratios + seed for reproducibility.
        """
        if not organoid_labels:
            raise ValueError("organoid_labels is empty")
        total = sum(ratios.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"ratios must sum to 1.0, got {total}")
        if len(ratios) not in (2, 3):
            raise ValueError(f"ratios must have 2 or 3 splits, got {len(ratios)}")

        ids = sorted(organoid_labels.keys())
        labels = [organoid_labels[i] for i in ids]

        if len(ratios) == 2:
            (first_name, first_frac), (second_name, _) = list(ratios.items())
            ids_first, ids_second, _, _ = train_test_split(
                ids, labels, train_size=first_frac, random_state=seed, stratify=labels,
            )
            mapping = {oid: first_name for oid in ids_first}
            mapping.update({oid: second_name for oid in ids_second})
        else:
            # 3-way: split off test first (by ratio), then val from the rest
            split_names = list(ratios.keys())
            if "test" not in ratios:
                raise ValueError("3-way ratios must include 'test'")
            train_name = next(n for n in split_names if n not in ("val", "test"))
            test_frac = ratios["test"]
            val_frac = ratios["val"]

            ids_tv, ids_test, lbl_tv, _ = train_test_split(
                ids, labels, test_size=test_frac, random_state=seed, stratify=labels,
            )
            val_size_relative = val_frac / (1.0 - test_frac)
            ids_train, ids_val, _, _ = train_test_split(
                ids_tv, lbl_tv, test_size=val_size_relative, random_state=seed, stratify=lbl_tv,
            )
            mapping = {oid: train_name for oid in ids_train}
            mapping.update({oid: "val" for oid in ids_val})
            mapping.update({oid: "test" for oid in ids_test})

        ratio_str = ", ".join(f"{k}={v:.3f}" for k, v in ratios.items())
        return cls(
            mapping=mapping,
            name=name,
            provenance=f"stratified_random(ratios={{{ratio_str}}}, seed={seed})",
        )

    # ---- accessors ----

    def head_counts(self) -> Dict[str, int]:
        return dict(Counter(self.mapping.values()))

    def split_names(self) -> Set[str]:
        return set(self.mapping.values())

    def organoid_ids(self) -> Set[str]:
        return set(self.mapping.keys())

    def __getitem__(self, organoid_id: str) -> str:
        return self.mapping[organoid_id]

    def __contains__(self, organoid_id: object) -> bool:
        return organoid_id in self.mapping

    def __len__(self) -> int:
        return len(self.mapping)

    def __repr__(self) -> str:
        counts = " ".join(f"{k}={v}" for k, v in sorted(self.head_counts().items()))
        return f"Splits(name={self.name!r}, {len(self)} orgs, {counts})"

    # ---- comparison ----

    def agreement_with(self, other: "Splits") -> Dict[str, Any]:
        """Compare two Splits at the organoid level.

        Returns a dict with shared/only-in-self/only-in-other counts, the
        number of partition agreements vs disagreements on the shared set,
        and a confusion matrix keyed by ``(self_split, other_split)``.
        """
        shared = self.organoid_ids() & other.organoid_ids()
        only_self = self.organoid_ids() - other.organoid_ids()
        only_other = other.organoid_ids() - self.organoid_ids()

        confusion: Dict[Any, int] = Counter()
        identical = 0
        for oid in shared:
            pair = (self.mapping[oid], other.mapping[oid])
            confusion[pair] += 1
            if pair[0] == pair[1]:
                identical += 1

        return {
            "shared_ids": len(shared),
            "only_in_self": len(only_self),
            "only_in_other": len(only_other),
            "identical": identical,
            "disagreements": len(shared) - identical,
            "confusion": dict(confusion),
        }

    # ---- I/O ----

    def to_csv(self, path) -> None:
        """Write a 2-column ``organoid_id,split`` CSV, sorted by organoid_id."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["organoid_id", "split"])
            for oid in sorted(self.mapping):
                w.writerow([oid, self.mapping[oid]])
