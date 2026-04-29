# Code Organization

One-page reference for where code and data live after the `pipeline/` vs `analysis/` reorg. Pair with `CLAUDE.md` and `AGENTS.md`.

## Top-level layout

- **`pipeline/`** — Everything that turns raw inputs into `data/all_data.json` (steps 1-16). Deterministic. No ML training. No paper-specific logic.
- **`analysis/`** — Everything that consumes `all_data.json`: model heads, exploratory notebooks, paper replication.
- **`data/`** — Checked-in outputs: `all_data.json`, splits CSV. Derived split JSONs are *not* committed (see rule #3 below).
- **`scripts/`** — Standalone utilities that aren't part of the pipeline.
- **`paper/`**, **`notes/`** — Writing, figures, feedback, working notes.

## Pipeline (`pipeline/`) — where to add things

- **`pipeline/identifiers/`** — step 1 (main record index)
- **`pipeline/metabolites/`** — step 2 (Excel → `metabolite_map.json`)
- **`pipeline/surveys/`** — step 3 (Excel → `survey_map.json`)
- **`pipeline/images/`** — steps 4-15
  - `image_mapper.py` — step 4
  - `segmentation_mmseg/` — steps 5, 7, 8, 9 (requires `mmcv_env`)
  - `resize/` — steps 6, 14
  - `quality/` — steps 10, 11
  - `series/` — steps 12, 13
  - `postprocess/` — step 15
- **`pipeline/merge/`** — step 16 (`all_data.json`)
- **`pipeline/common/`** — shared helpers (organoid_patterns, json_views, etc.)
- **`pipeline/data_loader.py`** — canonical `OrganoidDataset` for downstream consumers

Rule: if it reads from `$RAW_DIR` or writes to `$INTERMEDIATE_DIR` / `$MODELS_DIR/mmseg`, it belongs in `pipeline/`. Every module should be callable as `make stepN`.

## Analysis (`analysis/`) — where to add things

- **`analysis/imagequality_classification/`** — step 17 (PyTorch ViT/ResNet/CNN image quality classifier)
- **`analysis/image_survey_classification/`** — step 18 (TensorFlow ResNet50V2 survey classifier)
- **`analysis/multimodal/`** — combined image + metabolite models
- **`analysis/paper_2026_04/`** — exact scripts needed to reproduce the 2026-04 paper (Tables 1-3, Figures 5-11). Built against `pipeline.data_loader` (public API only).
- **`analysis/outputs/`** — generated figures and CSVs (gitignored)
- **`analysis/cnn_lstm_legacy/`**, **`analysis/metabolites_legacy/`**, **`analysis/legacy_paper_2026_04/`** — frozen reference implementations from before the 2026-04 data-loader rewrite. Not Makefile-wired.

Rule: if the input is `data/all_data.json` or something derived at runtime from it, it belongs in `analysis/`. Paper-specific scripts go under `analysis/<paper-tag>/` so they stay reproducible even as the core analysis evolves.

## Data — where to put files

Everything lives under `$DATA_ROOT` (default `/net/projects2/promega/2026_04_15_data/`). The Makefile exposes `RAW_DIR`, `INTERMEDIATE_DIR`, `MODELS_DIR`, `ANALYSIS_OUTPUT_DIR` derived from `DATA_ROOT`.

- **`raw/`** — inputs that can't be regenerated: raw images, hand-labeled masks, metabolite.xlsx, surveys, Sample-Tracing.xlsx. **Pipeline never writes here.**
- **`intermediate/`** — everything `make pipeline-all` produces. Safe to wipe: `make clean` does. Contains `indexes/`, `resized_*/`, `overlays/`, `masks_processed/`, `lstm_ready/`, etc.
- **`models/`** — trained checkpoints: `mmseg/`, `imagequality_classification/`, `image_survey_classification/`.
- **`analysis_output/`** — manual figures, reports, ad-hoc outputs.

In-repo data: only `data/all_data.json` (the merged source of truth) and `data/2026_winter_student_splits.csv` (split assignments, organoid-keyed) are committed. Do **not** materialize filtered views into separate JSON files — consumers go through `pipeline.data_loader.OrganoidDataset`, which reads `all_data.json` + the splits CSV at runtime (AGENTS.md rule #3).

## Environments

- **`core_env`** (from `core_env.yaml`) — everything except mmseg.
- **`mmcv_env`** — only for segmentation steps 8-9. Activated automatically by the Makefile.
- Never `pip install` into system Python or outside these envs.

## Running

- `make pipeline-all` — steps 1-16 end-to-end
- `make train-all` — steps 17-18
- `make stepN` — single step
- `OVERWRITE=1` (default) regenerates outputs. Pass `OVERWRITE=` to skip outputs that already exist.
- `make run ARGS="-m analysis.paper_2026_04.metabolite_boxplot"` — one-off scripts with correct env + `PYTHONPATH`.
