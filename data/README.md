# `data/` — start here

This folder holds the **inputs** the analysis code reads. Train/val/test
splits are not stored here — they're built at runtime from `all_data.json`
by `pipeline.data_loader`. See [Building cohorts and splits](#building-cohorts-and-splits) below.

## Files

```
data/
├── all_data.json                       # canonical merged dataset (all records, all days)
├── idor_organoids.csv                  # IDOR curated list (266 organoids; col1=evaluated, col2=classified)
└── 2026_winter_student_splits.csv      # frozen splits used by paper-replication scripts
```

## Building cohorts and splits

Don't write derivative split JSONs to disk. The loader builds them
deterministically from `all_data.json`:

```python
from pipeline.data_loader import OrganoidDataset, filters_for_mode, split_organoids

ds = OrganoidDataset(
    "data/all_data.json",
    split_ratios={"train": 1.0},      # placeholder — split_organoids overrides
    split_seed=42,
    filters=filters_for_mode("series_idor"),
)
train_ids, val_ids, test_ids = split_organoids(ds, seed=42, test_size=0.2, val_size=0.1)
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
