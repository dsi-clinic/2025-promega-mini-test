#!/usr/bin/env python3
# rename_to_common_key.py
import json, csv, hashlib, shutil, re, sys
from pathlib import Path
from typing import Optional, Tuple
import argparse
import cv2
from tqdm import tqdm
from config import ALL_DATA_JSON
all_data_path = Path(str(ALL_DATA_JSON))


# ---------- defaults ----------
DEFAULT_DEST_ROOT = Path("/net/projects2/promega/data-analysis/output/predictions_renamed")

# ---------- utils ----------
def sha256_of_file(p: Path, bufsize: int = 1<<20) -> Optional[str]:
    if not p.exists():
        return None
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(bufsize), b""):
            h.update(chunk)
    return h.hexdigest()

def mask_shape(p: Path) -> Optional[Tuple[int,int]]:
    try:
        m = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if m is None:
            return None
        h, w = m.shape[:2]
        return int(h), int(w)
    except Exception:
        return None

PLATE_RE = re.compile(r"^(96_[12]|PT1)$", re.IGNORECASE)
DAY_RE   = re.compile(r"^Dy\d{1,2}$", re.IGNORECASE)

def split_common_key(common_key: str) -> Tuple[str, Optional[str], Optional[str]]:
    """
    Returns (BA, plate_or_None, DyXX_or_None) from a common_key like:
      BA2_96_1_Dy03_A1_nosplit_nostitch
      BA1_Dy20_A7_presplit_nostitch
    """
    parts = common_key.split("_")
    if not parts:
        return "", None, None
    ba = parts[0]
    plate = None
    idx = 1
    if idx < len(parts) and PLATE_RE.match(parts[idx] or ""):
        plate = parts[idx]
        idx += 1
    day = None
    for p in parts[idx:]:
        if DAY_RE.match(p or ""):
            day = p
            break
    return ba, plate, day

def find_mask_path(proc: dict) -> Optional[str]:
    """
    Be liberal: try a few likely keys in processed dict.
    """
    if not isinstance(proc, dict):
        return None
    for k in ("mask_path", "pred_mask_path", "pred_mask", "mask"):
        v = proc.get(k)
        if isinstance(v, str) and v:
            return v
    # last resort: any string ending with common mask extensions
    for v in proc.values():
        if isinstance(v, str) and re.search(r"\.(png|tif|tiff)$", v, re.IGNORECASE):
            return v
    return None

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(
        description="Duplicate + rename predicted masks to common_key naming and emit a verification manifest."
    )
    ap.add_argument("--all-data", type=Path, default=None,
                    help="Path to all_data.json (if omitted, tries to import from config.ALL_DATA_JSON).")
    ap.add_argument("--dest-root", type=Path, default=DEFAULT_DEST_ROOT,
                    help=f"Destination root folder (default: {DEFAULT_DEST_ROOT})")
    ap.add_argument("--layout", choices=["nested", "flat"], default="nested",
                    help="nested: BA[_plate]/DyXX subfolders; flat: all files in root.")
    ap.add_argument("--overwrite", action="store_true",
                    help="Overwrite existing destination files.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print what would happen; do not copy.")
    args = ap.parse_args()

    # Resolve all_data path
    all_data_path = args.all_data
    if all_data_path is None:
        try:
            from config import ALL_DATA_JSON
            all_data_path = Path(str(ALL_DATA_JSON))
        except Exception:
            print("ERROR: --all-data not provided and config.ALL_DATA_JSON not available.", file=sys.stderr)
            sys.exit(2)

    if not all_data_path.exists():
        print(f"ERROR: all_data.json not found: {all_data_path}", file=sys.stderr)
        sys.exit(2)

    # Load
    with all_data_path.open() as f:
        data = json.load(f)

    dest_root: Path = args.dest_root
    if not args.dry_run:
        dest_root.mkdir(parents=True, exist_ok=True)

    rows = []
    total = 0
    copied = 0
    skipped = 0
    missing = 0
    mismatched = 0

    # Iterate
    for raw_k, entry in tqdm(list(data.items()), desc="Renaming masks"):
        if not isinstance(entry, dict):
            continue
        common_key = entry.get("common_key")
        proc = entry.get("processed", {})
        mpath = find_mask_path(proc)
        if not common_key or not mpath:
            missing += 1
            continue

        src_path = Path(mpath)
        if not src_path.exists():
            # Some processed JSONs store relative paths with a base folder in the wrapper.
            # Try to resolve if _base_folder exists at top-level (optional).
            base_folder = data.get("_base_folder")  # unlikely here; just in case
            if base_folder:
                alt = Path(base_folder) / src_path
                if alt.exists():
                    src_path = alt
            if not src_path.exists():
                missing += 1
                rows.append({
                    "common_key": common_key,
                    "old_path": str(mpath),
                    "new_path": None,
                    "old_size": None,
                    "new_size": None,
                    "old_sha256": None,
                    "new_sha256": None,
                    "old_shape": None,
                    "new_shape": None,
                    "verify": "MISSING_SOURCE",
                })
                continue

        # destination path
        ext = src_path.suffix.lower() or ".png"
        new_filename = f"{common_key}_predmask{ext}"

        if args.layout == "nested":
            ba, plate, day = split_common_key(common_key)
            ba_part = f"{ba}_{plate}" if plate else ba
            dest_dir = dest_root / ba_part / (day or "DyXX")
        else:
            dest_dir = dest_root

        dest_path = dest_dir / new_filename

        # ensure folder
        if not args.dry_run:
            dest_dir.mkdir(parents=True, exist_ok=True)

        # Copy or skip
        did_copy = False
        total += 1

        if dest_path.exists() and not args.overwrite:
            # verify equality; if same, mark skipped; if different, mark mismatched
            src_hash = sha256_of_file(src_path)
            dst_hash = sha256_of_file(dest_path)
            src_size = src_path.stat().st_size
            dst_size = dest_path.stat().st_size
            src_sh   = mask_shape(src_path)
            dst_sh   = mask_shape(dest_path)
            same = (src_hash == dst_hash) and (src_size == dst_size) and (src_sh == dst_sh)

            rows.append({
                "common_key": common_key,
                "old_path": str(src_path),
                "new_path": str(dest_path),
                "old_size": src_size,
                "new_size": dst_size,
                "old_sha256": src_hash[:16] if src_hash else None,
                "new_sha256": dst_hash[:16] if dst_hash else None,
                "old_shape": f"{src_sh}" if src_sh else None,
                "new_shape": f"{dst_sh}" if dst_sh else None,
                "verify": "OK(SKIP)" if same else "DIFF_EXISTS",
            })
            skipped += 1 if same else 0
            mismatched += 1 if not same else 0
        else:
            if not args.dry_run:
                shutil.copy2(src_path, dest_path)
                did_copy = True
            # verify
            src_hash = sha256_of_file(src_path)
            dst_hash = sha256_of_file(dest_path) if not args.dry_run else None
            src_size = src_path.stat().st_size
            dst_size = dest_path.stat().st_size if not args.dry_run and dest_path.exists() else None
            src_sh   = mask_shape(src_path)
            dst_sh   = mask_shape(dest_path) if not args.dry_run and dest_path.exists() else None

            verify_ok = (not args.dry_run) and (dst_hash is not None) and (
                (src_hash == dst_hash) and (src_size == dst_size) and (src_sh == dst_sh)
            )
            rows.append({
                "common_key": common_key,
                "old_path": str(src_path),
                "new_path": str(dest_path),
                "old_size": src_size,
                "new_size": dst_size,
                "old_sha256": src_hash[:16] if src_hash else None,
                "new_sha256": dst_hash[:16] if dst_hash else None,
                "old_shape": f"{src_sh}" if src_sh else None,
                "new_shape": f"{dst_sh}" if dst_sh else None,
                "verify": "OK" if verify_ok else ("DRYRUN" if args.dry_run else "FAIL"),
            })
            if did_copy:
                copied += 1
                if not verify_ok:
                    mismatched += 1

    # Write manifest
    if rows:
        manifest_path = args.dest_root / "mask_rename_manifest.csv"
        if not args.dry_run:
            with manifest_path.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
            print(f"\nWrote manifest: {manifest_path}")
        else:
            print("\n(DRY-RUN) Manifest not written.")

    # Summary
    ok_count = sum(1 for r in rows if r["verify"].startswith("OK"))
    print("\nSummary")
    print(f"  total considered : {total}")
    print(f"  copied           : {copied}")
    print(f"  skipped (same)   : {skipped}")
    print(f"  missing source   : {missing}")
    print(f"  verify OK        : {ok_count}")
    print(f"  mismatched/other : {mismatched}")

if __name__ == "__main__":
    main()
