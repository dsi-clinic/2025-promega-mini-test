#!/usr/bin/env python3
import hashlib
import os
import shutil
import sys
from typing import Dict, List, Tuple


def compute_file_sha256(file_path: str, chunk_size: int = 1024 * 1024) -> str:
    """Compute SHA256 hash of a file in a memory-efficient way."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            sha256.update(chunk)
    return sha256.hexdigest()


def build_index(base_dir: str) -> Tuple[Dict[str, List[str]], Dict[str, List[str]], Dict[str, List[str]]]:
    """Build indices:
    - name_to_paths: filename -> list of full paths under base_dir
    - size_to_paths: filesize (as str) -> list of full paths
    - hash_to_paths: sha256 -> list of full paths (computed lazily as needed)

    Note: For performance, we only pre-build name and size indices. Hashes are computed on demand.
    """
    name_to_paths: Dict[str, List[str]] = {}
    size_to_paths: Dict[str, List[str]] = {}
    # hash_to_paths will be filled lazily when needed
    hash_to_paths: Dict[str, List[str]] = {}

    for root, _dirs, files in os.walk(base_dir):
        for fname in files:
            full_path = os.path.join(root, fname)
            try:
                stat = os.stat(full_path)
            except FileNotFoundError:
                continue
            name_to_paths.setdefault(fname, []).append(full_path)
            size_to_paths.setdefault(str(stat.st_size), []).append(full_path)

    return name_to_paths, size_to_paths, hash_to_paths


def find_target_paths(
    src_path: str,
    name_to_paths: Dict[str, List[str]],
    size_to_paths: Dict[str, List[str]],
    hash_to_paths: Dict[str, List[str]],
) -> List[str]:
    """Find candidate original paths in base_dir that match src_path.

    Strategy:
    1) Filter by filename.
    2) Narrow by file size.
    3) Disambiguate by SHA256 if needed.
    """
    fname = os.path.basename(src_path)
    try:
        src_stat = os.stat(src_path)
    except FileNotFoundError:
        return []

    name_candidates = name_to_paths.get(fname, [])
    if not name_candidates:
        # Fallback: try by size first, then disambiguate by hash
        size_key = str(src_stat.st_size)
        size_only_candidates = [p for p in size_to_paths.get(size_key, []) if os.path.exists(p)]
        if not size_only_candidates:
            return []
        src_hash = compute_file_sha256(src_path)
        matched_by_hash = []
        for cand in size_only_candidates:
            try:
                if compute_file_sha256(cand) == src_hash:
                    matched_by_hash.append(cand)
            except Exception:
                continue
        return matched_by_hash

    size_candidates = [p for p in name_candidates if os.path.exists(p) and os.path.getsize(p) == src_stat.st_size]
    if not size_candidates:
        return []

    if len(size_candidates) == 1:
        return size_candidates

    # Compute hash for src and candidates if ambiguous
    src_hash = compute_file_sha256(src_path)

    # Populate hash_to_paths lazily
    matched_by_hash: List[str] = []
    for cand in size_candidates:
        # Compute cand hash
        cand_hash = compute_file_sha256(cand)
        if cand_hash == src_hash:
            matched_by_hash.append(cand)
            hash_to_paths.setdefault(cand_hash, []).append(cand)

    return matched_by_hash if matched_by_hash else size_candidates


def restore_images(
    misclassified_dir: str,
    base_dir: str,
    dry_run: bool = False,
    log_path: str = "restore_images.log",
    prefer_folder_mapping: bool = True,
) -> None:
    abs_mis = os.path.abspath(misclassified_dir)
    abs_base = os.path.abspath(base_dir)
    abs_log = os.path.abspath(log_path)

    if not os.path.isdir(abs_mis):
        print(f"ERROR: misclassified directory not found: {abs_mis}", file=sys.stderr)
        sys.exit(1)

    if not os.path.isdir(abs_base):
        print(f"ERROR: base directory not found: {abs_base}", file=sys.stderr)
        sys.exit(1)

    name_to_paths, size_to_paths, hash_to_paths = build_index(abs_base)

    moved_count = 0
    skipped_count = 0
    ambiguous_count = 0
    missing_count = 0

    with open(abs_log, "w", encoding="utf-8") as logf:
        logf.write(f"Restoring images from: {abs_mis}\n")
        logf.write(f"Back to base tree:    {abs_base}\n")
        logf.write("\n")

        for root, _dirs, files in os.walk(abs_mis):
            for fname in files:
                src = os.path.join(root, fname)
                if not os.path.isfile(src):
                    continue

                # Strategy 0: Direct folder mapping if possible
                target_file = None
                if prefer_folder_mapping:
                    rel_path = os.path.relpath(src, abs_mis)
                    # Use the first path component under misclassified_dir as the destination folder name
                    parts = rel_path.split(os.sep)
                    if len(parts) >= 2:
                        dest_folder_name = parts[0]
                        candidate_dir = os.path.join(abs_base, dest_folder_name)
                        if os.path.isdir(candidate_dir):
                            target_file = os.path.join(candidate_dir, os.path.basename(src))

                # Strategy 1: Name/size/hash search if folder mapping failed
                if target_file is None:
                    candidates = find_target_paths(src, name_to_paths, size_to_paths, hash_to_paths)
                    if not candidates:
                        logf.write(f"MISSING: {src} -> no match in base by folder/name/size/hash\n")
                        missing_count += 1
                        continue

                    if len(candidates) > 1:
                        logf.write(
                            "AMBIGUOUS: {} -> multiple matches ({}), using first.\n".format(
                                src, len(candidates)
                            )
                        )
                        for c in candidates[:5]:
                            logf.write(f"  candidate: {c}\n")
                        ambiguous_count += 1

                    target_file = candidates[0]
                target_dir = os.path.dirname(target_file)

                # If the exact file already exists at target and is identical, just remove src
                if os.path.exists(target_file):
                    try:
                        if os.path.getsize(target_file) == os.path.getsize(src):
                            if compute_file_sha256(target_file) == compute_file_sha256(src):
                                if not dry_run:
                                    os.remove(src)
                                logf.write(f"SKIP (already in place): {src} == {target_file}\n")
                                skipped_count += 1
                                continue
                    except Exception as e:
                        logf.write(f"WARN: could not compare existing {target_file} and {src}: {e}\n")

                    # If not identical, write with suffix to avoid overwrite
                    base_name, ext = os.path.splitext(os.path.basename(target_file))
                    new_name = f"{base_name}__restored{ext}"
                    target_file = os.path.join(target_dir, new_name)

                os.makedirs(target_dir, exist_ok=True)
                action = "MOVE" if not dry_run else "DRY-RUN would move"
                logf.write(f"{action}: {src} -> {target_file}\n")
                if not dry_run:
                    try:
                        shutil.move(src, target_file)
                        moved_count += 1
                    except Exception as e:
                        logf.write(f"ERROR: failed to move {src} -> {target_file}: {e}\n")

        logf.write("\n")
        logf.write(f"Moved: {moved_count}\n")
        logf.write(f"Skipped (already there): {skipped_count}\n")
        logf.write(f"Ambiguous: {ambiguous_count}\n")
        logf.write(f"Missing: {missing_count}\n")

    print(
        f"Done. Moved={moved_count}, Skipped={skipped_count}, Ambiguous={ambiguous_count}, Missing={missing_count}.\nLog: {abs_log}"
    )


def main() -> None:
    # Defaults based on user's environment
    default_mis = os.path.join(os.getcwd(), "misclassified_images")
    default_base = "/net/projects2/promega/data-analysis/output/infer_resized_512x384/auto_processed/"

    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Restore images from a 'misclassified_images' folder back to their original "
            "locations under the specified base directory by matching name, size, and hash."
        )
    )
    parser.add_argument(
        "--mis-dir",
        default=default_mis,
        help="Path to the misclassified images directory (default: ./misclassified_images)",
    )
    parser.add_argument(
        "--base-dir",
        default=default_base,
        help="Base directory containing the original dataset tree",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not move files, only log intended actions",
    )
    parser.add_argument(
        "--log",
        default="restore_images.log",
        help="Path to the log file (default: restore_images.log)",
    )

    args = parser.parse_args()
    restore_images(args.mis_dir, args.base_dir, dry_run=args.dry_run, log_path=args.log)


if __name__ == "__main__":
    main()


