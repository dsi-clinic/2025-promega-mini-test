# `scripts/splits/` — cohort splitter

Single entry point: `make_splits.py`. One config produces one cohort (both
`full/` and `series/` views), plus a `MANIFEST.json` recording full
provenance.

## Run it

```bash
conda activate /net/projects2/promega

# IDOR cohort (the curated 266-organoid list from data/idor_organoids.csv)
python scripts/splits/make_splits.py scripts/splits/configs/idor.json

# Expanded cohort (IDOR + BA4, with BA4 filtered the same way as the rest)
python scripts/splits/make_splits.py scripts/splits/configs/expanded.json
```

Outputs land in `data/cohorts/<name>/` per the config's `output_dir`.

## Adding a new cohort

Copy one of the existing configs and edit it. The config tells the splitter
three things:

1. **Which base wells are in the cohort.**
   `cohort.identifiers_csv` is the mandatory starting set. `cohort.add_batches`
   adds every base well whose batch prefix (e.g. `BA4`) matches — useful when
   you want to extend the curated list with a whole batch.

2. **Stage 1 quality filter (per-day).**
   `stage1.max_edge_fraction` and `stage1.exclude_classifications` define
   which records are dropped on a per-day basis. Applied uniformly to every
   record regardless of cohort membership.

3. **Split parameters.**
   `split.seed`, `split.test_size`, `split.val_size`. Splitting happens at
   the base-well level (same well in presplit → both daughters → same
   partition, preventing leakage).

`full/` and `series/` are always written together from one run — they share
the same base-well → partition assignment, so a well's partition never
depends on which view a student loads.

## Manifest contents

`<output_dir>/MANIFEST.json` records:

- `created_at_utc`, `git_sha`
- `config` (the full config as-run)
- `all_data_sha256`, `identifiers_csv_sha256`
- `seed`, `expected_days`, `label_day`
- `cohort_stats`, `skipped_counts`
- `full_summary`, `series_summary` (per-partition counts)

With these, any split can be reproduced or audited long after the fact.

## Superseded scripts

The following are kept in `scripts/` for reference but should not be run for
new work — their behavior is folded into `make_splits.py`:

- `split_series_reproducible.py` → now produces the `series/` view of the IDOR cohort
- `split_data_reproducible.py`   → superseded by the `full/` view
- `split_data_no_stitch.py`      → superseded; stitched records are not in the cohort after Stage 1 if needed (configurable via `exclude_classifications`)

Once the new `data/cohorts/*` outputs are validated against the old
`data_splits/*` files for any in-flight experiments, the legacy scripts can
move into an `_archive/` subfolder.
