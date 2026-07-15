#!/usr/bin/env python3
"""
Centralized regex patterns and helpers for parsing organoid identifiers.

This module exposes only patterns and helpers that have external callers in the
current codebase. Older variants (full key parsers, validators, batch/well
extractors) were removed in 2026-04 — recover them from git history if
re-needed.
"""

import re


class OrganoidPatterns:
    """Regex patterns for organoid identifier parsing.

    Used externally:
        DUPLICATE_IMAGE, PARTIAL_IMAGE, PLATE_PATTERN, PLATE_REMOVE,
        REMOVE_PARENS, STITCHED, WELL_STRICT
    Used internally by OrganoidNormalizer:
        DAY_EXTRACT, Z_LEVEL, REMOVE_BRACKETS, KEEP_ALPHANUMERIC,
        NORMALIZE_SPACES, SPLIT_TOKEN, SPLIT_PAREN, SPLIT_HYPHEN
    """

    # Day patterns
    DAY_EXTRACT = re.compile(r"[Dd][Yy]?(\d+)", re.IGNORECASE)

    # Plate patterns
    PLATE_PATTERN = re.compile(r"\b(96_[12]|PT1)\b", re.IGNORECASE)
    PLATE_REMOVE = re.compile(r"96_[12]", re.IGNORECASE)

    # Well ID
    WELL_STRICT = re.compile(r"(?<!BA)\b([A-H])(\d{1,2})\b", re.IGNORECASE)

    # Cleaning
    REMOVE_BRACKETS = re.compile(r"\[.*?\]")
    REMOVE_PARENS = re.compile(r"\(.*?\)")
    KEEP_ALPHANUMERIC = re.compile(r"[^A-Za-z0-9\s_]")
    NORMALIZE_SPACES = re.compile(r"\s+")

    # File markers
    Z_LEVEL = re.compile(r" Z(\d+)", re.IGNORECASE)
    PARTIAL_IMAGE = re.compile(r"\(\d+\s+of\s+\d+\)")
    DUPLICATE_IMAGE = re.compile(r"\((\d+)\)")
    STITCHED = re.compile(r"\(stitched\)", re.IGNORECASE)

    # Split markers (after the split day)
    SPLIT_PAREN = re.compile(r"\(\s*(\d{1,2})\s*\)\s*%?", re.IGNORECASE)  # "(2)%", "(12)"
    SPLIT_HYPHEN = re.compile(r"[-_ ](\d{1,2})\s*[-_ ]?%(?=\b|[^0-9])", re.IGNORECASE)  # "-2-%", "_12_%"
    SPLIT_TOKEN = re.compile(r"\bsplit[_-]?(\d{1,2})\b", re.IGNORECASE)


class OrganoidNormalizer:
    """Stateless helpers used across the pipeline mappers.

    Kept as a class for namespacing only — all methods are pure functions over
    their inputs.
    """

    @staticmethod
    def clean_string(text: str) -> str:
        """Strip brackets/parens, drop non-alphanumeric/space/underscore, collapse whitespace."""
        text = OrganoidPatterns.REMOVE_BRACKETS.sub("", text)
        text = OrganoidPatterns.REMOVE_PARENS.sub("", text)
        text = OrganoidPatterns.KEEP_ALPHANUMERIC.sub(" ", text)
        return OrganoidPatterns.NORMALIZE_SPACES.sub(" ", text).strip()

    @staticmethod
    def extract_day_number(text: str) -> int | None:
        """Extract day number as integer ('Dy03' → 3, 'Day 12' → 12). Lossy for half-days."""
        match = OrganoidPatterns.DAY_EXTRACT.search(text)
        return int(match.group(1)) if match else None

    @staticmethod
    def extract_z_level(text: str) -> int:
        """Extract Z level from filename (' Z2.tif' → 2). Returns -1 if absent."""
        match = OrganoidPatterns.Z_LEVEL.search(text)
        return int(match.group(1)) if match else -1

    @staticmethod
    def extract_split_info(raw_name: str) -> dict:
        """Parse split / stitched / partial markers from a raw filename or key.

        Returns a dict with keys: is_split, split_index, stitched, partial.
        """
        s = raw_name.lower()
        info = {"is_split": False, "split_index": None, "stitched": False, "partial": False}

        for pat in (OrganoidPatterns.SPLIT_TOKEN, OrganoidPatterns.SPLIT_PAREN, OrganoidPatterns.SPLIT_HYPHEN):
            m = pat.search(s)
            if m:
                info["is_split"] = True
                info["split_index"] = int(m.group(1))
                break

        if OrganoidPatterns.STITCHED.search(s):
            info["stitched"] = True
        if OrganoidPatterns.PARTIAL_IMAGE.search(s):
            info["partial"] = True
        return info


def clean_id_for_json(text: str) -> str:
    """Module-level alias for OrganoidNormalizer.clean_string (used by mappers)."""
    return OrganoidNormalizer.clean_string(text)
