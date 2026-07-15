"""Reusable invariant assertions shared across the test suite.

These encode AGENTS.md rules 11 (organoid count is conserved) and 14 (assert
every join). Import them in tests and in analysis code that wants the same
guard at runtime.
"""

from collections.abc import Iterable


def assert_organoid_count(ids: Iterable[str], expected: int, context: str = "") -> None:
    """Rule 11: assert an organoid-id collection has exactly ``expected`` members.

    Also asserts uniqueness — a silent duplicate is as much a bug as a silent
    drop. Pass ``context`` to identify the call site in the failure message.
    """
    ids = list(ids)
    dupes = len(ids) - len(set(ids))
    assert dupes == 0, f"{context}: {dupes} duplicate organoid id(s)"
    assert len(ids) == expected, f"{context}: {len(ids)} organoids, expected {expected}"


def assert_count_conserved(before: Iterable[str], after: Iterable[str], context: str = "") -> None:
    """Rule 11: assert a transform neither added nor dropped organoids.

    ``after`` must be exactly the same set as ``before``. For steps that are
    *meant* to reduce the sample, use ``assert_subset`` + an explicit expected
    count instead — silence about an intended drop is still a bug.
    """
    before, after = set(before), set(after)
    added = after - before
    dropped = before - after
    assert not added, f"{context}: added {len(added)} organoids: {sorted(added)[:5]}"
    assert not dropped, f"{context}: dropped {len(dropped)} organoids: {sorted(dropped)[:5]}"


def assert_subset(after: Iterable[str], before: Iterable[str], context: str = "") -> None:
    """Rule 11: assert a transform only ever *reduces* the organoid set.

    Useful for steps that legitimately drop rows (e.g. growth features need a
    prior day) but must never introduce a new organoid.
    """
    added = set(after) - set(before)
    assert not added, f"{context}: added {len(added)} organoids not in the source: {sorted(added)[:5]}"
