# `data/` — start here

This folder holds the **inputs** the analysis code reads. Per-day record dumps
are not stored here — those are built at runtime from `all_data.json` by
`pipeline.data_loader`. See [Building cohorts and splits](#building-cohorts-and-splits) below.

## Files

```
data/
├── all_data.json                       # canonical merged dataset (all records, all days)
├── idor_organoids.csv                  # IDOR curated list (266 organoids; col1=evaluated, col2=classified)
├── normalized/                         # Promega-residualized metabolite source CSVs (see normalized/README.md)
└── splits/                             # named train/val/test assignments (see splits/README.md)
    ├── canonical_2026_winter.csv       # repo default; loaded via Splits.canonical()
    └── harriet_2026_05.csv             # alternate variant from PR #110
```

## Building cohorts and splits

The canonical split is loaded as a first-class `Splits` object:

```python
from pipeline.data_loader import OrganoidDataset, filters_for_mode
from pipeline.splits import Splits

ds = OrganoidDataset(
    "data/all_data.json",
    splits=Splits.canonical(),
    filters=filters_for_mode("base"),
)
```

For ad-hoc / IDOR-series work where the canonical CSV doesn't apply:

```python
from pipeline.data_loader import OrganoidDataset, filters_for_mode, split_organoids
from pipeline.splits import Splits

# Filter+label first (no split assigned yet)
ds = OrganoidDataset("data/all_data.json", filters=filters_for_mode("series_idor"))

# Build a Splits via well-grouped stratified split, then apply
train_ids, val_ids, test_ids = split_organoids(ds, seed=42, test_size=0.2, val_size=0.1)
splits = Splits.from_partition(train=train_ids, val=val_ids, test=test_ids,
                                name="series_idor_seed42")
ds.apply_splits(splits)
```

For a quick ad-hoc stratified split from a labels dict:

```python
splits = Splits.stratified_random(
    ds.organoid_labels(),
    ratios={"train": 0.8, "test": 0.2},     # 2-way or 3-way
    seed=42,
    name="rand_80_20",
)
ds.apply_splits(splits)
```

### Available filter modes

- `"base"` — BA1+BA2, complete metabolites, valid images. Paper default.
- `"switch1"` / `"switch2"` / `"switch3"` — modality-asymmetric variants. See `filters_for_mode` docstring.
- `"series_idor"` — IDOR cohort (266 partner-curated wells) with complete 11-day
  series, per-day `edge_fraction <= 0.05`, no Split/SplitStitched/blank,
  clipped_meanfill image present. The CNN-LSTM trainers use this.

### Splits at the well level

`split_organoids` partitions at the **base_well** level (not organoid),
stratified by per-well majority Dy30 label, seed=42. Daughter organoids
from the same well always co-locate in the same partition.

## IDOR cohort verification

The 266-organoid IDOR list ships with a build-time contract verifier:

```bash
make analysis-verify-idor
```

This asserts the partner's stated semantics (col1 = 266 BA1+BA2 evaluated
organoids, no splits; col2 ⊆ col1 = Day-30-classified; col1\\col2 = 13
didn't-reach-Dy30 + 5 intro-survey).
