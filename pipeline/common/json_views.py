"""
Common utilities for JSON views.

This module contains common utilities for JSON views.

It defines the `ViewEmitter` protocol and the `BaseViewEmitter` class, which are used to create concrete view emitters.

The `ViewEmitter` protocol defines the interface for view emitters, and the `BaseViewEmitter` class provides a base implementation of the protocol.

The `ViewEmitter` protocol is used to create concrete view emitters, and the `BaseViewEmitter` class is used to provide a base implementation of the protocol.

"""

from typing import Any, Dict, Protocol

from pipeline.merge.normalized_records import OrganoidRecord

SchemaDict = Dict[str, Any]

class ViewEmitter(Protocol):
    """Protocol for view emitters."""

    name: str

    def process(self, record: OrganoidRecord) -> None:
        raise NotImplementedError

    def finalize(self) -> SchemaDict:
        raise NotImplementedError


class BaseViewEmitter:
    """Shared scaffolding for concrete emitters."""

    name: str
    label_list = [0,1]

    def process(self, record: OrganoidRecord) -> None:
        raise NotImplementedError

    def finalize(self) -> SchemaDict:
        fields = ("id", "img_path", "label", "mask_path", "overlay_path")

        records = {"records": {}, "metadata": {}}
        for day, rows in self._records_by_day.items():
            day_data = {name: [row.get(name) for row in rows] for name in fields}

            # remove fields where every value is None
            for name, values in list(day_data.items()):
                if all(value is None for value in values):
                    day_data.pop(name)

            records["records"][day] = day_data

        for day, skipped in self._skipped_records_by_day.items():
            records["records"].setdefault(day, {})["skipped"] = skipped

        records["metadata"]["total_skipped"] = sum(
            len(skipped) for skipped in self._skipped_records_by_day.values()
        )

        return records
