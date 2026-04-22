# `data/` — start here

This folder holds all **data** for the project. Code lives elsewhere
(`file_utils/`, `scripts/`, `analysis/`). If you just want to train a model
or run an analysis, everything you need is below.

## The files

```
data/
├── all_data.json              # canonical merged dataset (all records, all days)
├── idor_organoids.csv         # curated IDOR cohort (266 organoids); copied from the cluster
├── Summary_Table.csv          # batch-level summary
├── cohorts/                   # TRAIN/VAL/TEST SPLITS FOR MODELS ← you are probably looking for this
│   ├── idor/
│   │   ├── full/              #   per-day records (no series-completeness requirement)
│   │   │   ├── train.json
│   │   │   ├── val.json
│   │   │   ├── test.json
│   │   │   └── summary.json
│   │   ├── series/            #   complete 11-timepoint series only (for LSTM / time-series)
│   │   │   ├── train.json
│   │   │   ├── val.json
│   │   │   ├── test.json
│   │   │   └── summary.json
│   │   └── MANIFEST.json      #   seed, git SHA, input hashes — full provenance
│   └── expanded/              # IDOR + BA4 (BA4 also filtered by edge_fraction and doublet-well rules)
│       ├── full/{train,val,test,summary}.json
│       ├── series/{train,val,test,summary}.json
│       └── MANIFEST.json
└── data_splits/               # LEGACY — do not use for new work; see "Legacy splits" below
```

## What each cohort means

- **`cohorts/idor/`** — the curated IDOR list of 266 organoids. This is what
  the initial paper / figure 1 uses. Excludes BA4, excludes split-organoid
  wells, and (after Stage 1) excludes any day where the mask touches the
  image edge (`edge_fraction > 0.05`).
- **`cohorts/expanded/`** — same as IDOR *plus* BA4 added back in. BA4 wasn't
  in the original IDOR curation, so we apply the same per-day quality filter
  to it ourselves. Use this when you want the largest training set.

## What `full` vs `series` means

Every cohort ships in two views, built from the same underlying split:

- **`full/`** — per-day records grouped by organoid. A well appears here as
  long as at least one day passed Stage 1 and Dy30 was labeled. Use this for
  single-timepoint image/metabolite models and per-day ablations.
- **`series/`** — only organoids with **all 11 expected timepoints** present
  (after Stage 1). Use this for LSTM / time-series models.

The base-well → partition assignment is identical across the two views, so
any well in `series/train` is also in `full/train`, etc. You can compare
models across views without worrying about leakage.

### Expected timepoints

`Dy03, Dy06, Dy08, Dy10, Dy13, Dy15, Dy17, Dy20/Dy21, Dy24, Dy28, Dy30`
(Dy20 and Dy21 map to the same `mdl_day = 20.5`.)

### Labels

Every organoid carries a label computed from the Dy30 survey
(`Acceptable` / `Not Acceptable`), pre-computed in `all_data.json` at
`value.label.value`.

## How to regenerate a cohort

Splits are reproducible — same seed + same `all_data.json` + same
`idor_organoids.csv` always give the same partitions. To regenerate:

```bash
conda activate /net/projects2/promega
python scripts/splits/make_splits.py scripts/splits/configs/idor.json
python scripts/splits/make_splits.py scripts/splits/configs/expanded.json
```

`MANIFEST.json` in each cohort folder records: git SHA, sha256 of inputs,
seed, config, and per-partition counts — enough to audit or reproduce any
split later.

## Legacy splits

`data/data_splits/` (nested) and `data_splits/` (at repo root) are leftover
from earlier pipelines. Don't build new work on them; regenerate with
`make_splits.py` instead. They'll be removed once the new cohorts are
confirmed to replace them.

## How the pipeline fits together

```
raw sources ──► mappers ──► merge_all_data.py ──► data/all_data.json
                                                       │
                   data/idor_organoids.csv ────────────┤
                                                       ▼
                       scripts/splits/make_splits.py  (seed-controlled)
                                                       │
                                 ┌─────────────────────┴─────────────────────┐
                                 ▼                                           ▼
                     data/cohorts/idor/                          data/cohorts/expanded/
                       full/  series/  MANIFEST.json               full/  series/  MANIFEST.json
```
