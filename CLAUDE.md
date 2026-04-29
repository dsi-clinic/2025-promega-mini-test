# Project Notes for Claude

This file captures system-specific context that's useful when assisting with code changes. For setup, layout, and how-to-run instructions see `README.md`. For project conventions / rules (env, splits, schema invariants) see `AGENTS.md`. For where to put new code see `notes/CODE_ORGANIZATION.md`.

## System Overview

Two-phase data flow:

```
Raw inputs ──pipeline/ steps 1-16──► data/all_data.json ──analysis/ steps 17+──► models, figures, paper outputs
```

- **`pipeline/`** is deterministic data prep. Reads `$RAW_DIR`, writes `$INTERMEDIATE_DIR` and `$MODELS_DIR/mmseg`. Outputs `data/all_data.json` as the single source of truth.
- **`analysis/`** consumes `all_data.json` plus the organoid splits CSV. All filtering happens at runtime via `pipeline.data_loader.OrganoidDataset`; no derived JSONs are materialized to disk (`AGENTS.md` rule #3).

## Key Files

| File | Role |
|---|---|
| `pipeline/merge/merge_all_data.py` | Step 16 — combines metabolite/survey/image maps into `all_data.json`. |
| `pipeline/merge/normalized_records.py` | Canonical schema (`OrganoidRecord`, `OrganoidRecordBuilder`, `RecordMetrics`). The on-disk JSON shape is defined here. |
| `pipeline/data_loader.py` | `OrganoidDataset` — the loader every analysis script should use. Includes `filters_for_mode(mode, modality)` presets for paper modes (base / switch1 / switch2 / switch3). |
| `pipeline/common/organoid_patterns.py` | Centralized regex patterns for organoid keys. |
| `analysis/paper_2026_04/*.py` | Self-contained scripts to reproduce paper tables/figures. Schema-stable against current `all_data.json`. |

## Schema Reminders

- The on-disk `all_data.json` uses the **normalized** schema (`plate.batch`, `day.id`, `images.*`, `metabolite` singular). The pre-normalization schema (`BA`, `dayID`, `processed`, `metabolites`) only exists in memory inside `merge_all_data.py`.
- Day canonicalization: `Dy3 → Dy03`, `Dy20 / Dy20.5 / Dy21 → Dy20_5`. Centralized in `pipeline.data_loader`.
- Organoid IDs strip the day component: `BA1 96_1 Dy03 A1` → `BA1 96_1 A1`.

## Common Pitfalls

- **Don't** import schema fields by name from old code without checking — many fields renamed in the 2026-04 normalized-records refactor.
- **Don't** materialize filtered subsets to disk; always use `OrganoidDataset(filters=...)`.
- **Don't** hard-code data paths; use `$DATA_ROOT` / Makefile variables.
- **Don't** `pip install` outside `core_env`; add to `core_env.yaml` and rebuild.

## Status as of 2026-04

- `all_data.json` regenerated 2026-04-17 (5,168 records, 22 MB).
- `pipeline.data_loader` rewritten to read the normalized schema (was reading the in-memory intermediate; this caused a "0 organoids" bug fixed early April).
- Paper-replication scripts moved into `analysis/paper_2026_04/` and patched for the new schema.
- LightGBM Dy30 result regressed from 0.9444 (paper) to 0.7849 on fresh data — under investigation, see `STATUS.md`.
