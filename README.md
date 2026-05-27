# Promega Organoid Analysis System

Pipeline + analysis code for predicting organoid quality from multimodal data: microscopy images, metabolite assays, and human survey ratings across an 11-day developmental time series.

```
Raw inputs ──► pipeline/ (steps 1-16) ──► data/all_data.json ──► analysis/ (steps 17+, paper repro)
```

The `pipeline/` package turns raw images, Excel sheets, and segmentation masks into a single canonical `all_data.json`. The `analysis/` package consumes that JSON for model training, paper replication, and exploratory work. **One source of truth** — no derived JSONs are checked in (see `AGENTS.md` rule #3).

## Contributors
- Amanda Johnson
- Harriet Zu
- Liya Ding
- Lucy Li
- Nick Ross
- Nikki Tebaldi
- Sophie Feng
- Summer Han
- Yichen Ding


## Quick Start

```bash
# 1. Create env (once)
conda env create -f core_env.yaml

# 2. End-to-end
make pipeline-all             # steps 1-16: raw → all_data.json
make train-all                # steps 17-18: train classifiers

# 3. Single step or paper script
make step9-late                                          # one pipeline step
make run ARGS="-m analysis.paper_2026_04.descriptive_stats"  # one-off analysis
make help                                                # list all targets
```

`make` handles conda env activation and `PYTHONPATH` automatically. See `AGENTS.md` for project conventions.

## Repo Layout

```
pipeline/                           # Steps 1-16: data prep, deterministic
  identifiers/  metabolites/  surveys/
  images/{image_mapper, segmentation_mmseg, resize, quality, series, postprocess}
  merge/        common/             # all_data.json builder + shared helpers
  data_loader.py                    # OrganoidDataset (canonical analysis loader)

analysis/                           # Steps 17+: ML heads, exploration, paper repro
  imagequality_classification/      # Step 17 (PyTorch ViT/ResNet/CNN/DINOv2)
  image_survey_classification/      # Step 18 (TensorFlow ResNet50V2)
  images/cnn_lstm/                  # CNN-LSTM temporal ablation (EfficientNet + attention)
  images/classifier/                # Per-day image classifiers, Fig 7, Table 2 scripts
  metabolites/  multimodal/         # Misc model heads
  paper_2026_04/                    # Self-contained paper-replication scripts
  legacy_paper_2026_04/             # SLURM submission scripts for paper-replication runs

data/                               # Committed: all_data.json + organoid splits CSV only
notes/                              # Working docs (see CODE_ORGANIZATION.md)
paper/                              # Paper drafts, figures, feedback
scripts/                            # Standalone SLURM scripts (Fig 7, Table 2, organoid strips)
core_env.yaml                       # Canonical conda env spec
Makefile                            # Single source of truth for all step invocations
```

See `notes/CODE_ORGANIZATION.md` for the one-page rule on where to put new code.

## Data

All bulk data lives at `$DATA_ROOT` (default `/net/projects2/promega/2026_04_15_data/`):

| Dir | Contents | Regenerable? |
|---|---|---|
| `raw/` | Raw images, manual masks, metabolite/survey/Sample-Tracing XLSX | **Never written by pipeline** |
| `intermediate/` | All step 1-16 outputs (indexes, resized images, overlays, etc.) | `make clean` wipes |
| `models/` | Step 17/18 checkpoints + per-model metrics JSONs | `make train-all` regenerates |
| `analysis_output/` | Manual figures, paper-repro outputs | `make run` writes here |

Override per command: `make step1 DATA_ROOT=/path/to/your/data`.

In-repo data:
- `data/all_data.json` — merged source of truth (5,168 records, ~22MB)
- `data/splits/canonical_2026_winter.csv` — organoid-level train/val/test assignments (canonical; loaded via `Splits.canonical()`). Alternate named splits sit alongside under `data/splits/`.

`pipeline.data_loader.OrganoidDataset` reads `all_data.json` and applies a `pipeline.splits.Splits` at runtime; downstream code should never materialize filtered subsets to disk.

### `all_data.json` Schema (sketch)

The full schema lives in `pipeline/merge/normalized_records.py`. High-level structure:

```
{
  "schema_version": 1,
  "generated_at": "...",
  "stats": { ... },
  "records": {
    "BA1 96_1 Dy03 A1": {
      "id": "...",  "organoid_id": "BA1_96_1_A1",
      "day": {"id": "Dy3", "number": 3.0, "original": 3},
      "plate": {"batch": "BA1 96_1", "well": "A1"},
      "cell_line": "GM23279A",
      "images": {
        "main_id", "img_path", "mask_path", "overlay_path",
        "manual_mask_path",
        "aspect_ratio": {...},        # 575×575 geometry-preserving variant
        "clipped_meanfill": {...}     # mean-filled background variants
      },
      "metabolite": {
        "GlucoseGlo": {
          "concentration_uM": 9.83,           # raw (assay)
          "initial_concentration": 19654.49,  # raw (assay)
          "is_outlier": false, "well_384": "B2",
          "win": 18.09,                       # Promega-normalized (winsorized + scaled)
          "win_vol_norm": 1.50e-06             # Promega-normalized + volume-normalized
        },
        ...
      },
      "survey":     {"evaluations": [...], "quality_scores": [...]},
      "label":      {"value": "Acceptable", "votes": {...}, "source": "..."}
    },
    ...
  }
}
```

## Pipeline Steps

Each step is a Make target. Inputs/outputs and CLI flags are documented in the corresponding module's docstring.

| Step | Module | What |
|---|---|---|
| 1 | `pipeline.identifiers.retrieve_main_identifiers` | Build canonical record index from `image_verification.csv` |
| 2 | `pipeline.metabolites.metabolite_mapper` | Excel → `metabolite_map.json` |
| 3 | `pipeline.surveys.surveys_mapper` | Excel → `survey_map.json` |
| 4 | `pipeline.images.image_mapper` | Resolve raw images, Z-stacks, splits → `image_map.json` |
| 5 | `pipeline.images.segmentation_mmseg.preprocessing.manual_masks_mapping` | Pair manual masks with image entries |
| 6 | `pipeline.images.resize.resize_remap_images` | Resize images + masks to 512×384 |
| 7 | `pipeline.images.segmentation_mmseg.preprocessing.test_split` | Train/val/test JSONs for mmseg |
| 8 | `pipeline.images.segmentation_mmseg.train` | Train early + late segmentation models (`mmcv_env`) |
| 9 | `pipeline.images.segmentation_mmseg.predict_masks` | Run inference per phase (`mmcv_env`) |
| 10 | `pipeline.images.quality.image_mask_overlay` | RGB×mask overlays |
| 11 | `pipeline.images.quality.mask_edge_fraction` | Compute per-mask edge-touching fraction |
| 12 | `pipeline.images.series.filter_complete_series` | Filter to organoids with complete 11-day series |
| 13 | `pipeline.images.series.preprocess_for_lstm` | Uniform-physical-scale 768×768 for LSTM |
| 14 | `pipeline.images.resize.resize_aspect_ratio` | Aspect-ratio-preserving 575×575 |
| 15 | `pipeline.images.postprocess.meanfill_clip` | Background mean-fill (uses masks) |
| 16 | `pipeline.merge.merge_all_data` | Merge → `all_data.json` + `summary.json` |
| 17 | `analysis.imagequality_classification.train_model_accuracy` | Per-day image classifier (ViT/ResNet/CNN, two-phase) |
| 18 | `analysis.image_survey_classification.simple_classifier` | Survey classifier (TF ResNet50V2 + mask CNN) |

`OVERWRITE=1` (default) regenerates outputs each time. Pass `OVERWRITE=` to skip outputs that already exist. Steps 8-9 require `mmcv_env`; everything else uses `core_env`.

## Environments

Two conda envs:

- **`core_env`** — defined by `core_env.yaml`. Used for everything except segmentation training/inference.
- **`mmcv_env`** — separate, version-pinned for `mmseg` (steps 8-9). The Makefile activates it automatically for those targets.

If you need a new package, add it to `core_env.yaml` and rebuild — never `pip install` ad-hoc (`AGENTS.md` rule #1).

## Resource Requirements

| Workload | GPU | Memory | Time |
|---|---|---|---|
| Pipeline steps 1-7, 10-16 | None (CPU) | 8-16 GB | ~30 min total |
| Step 8 (segmentation training) | A100 ×1 | 32 GB | ~6 hr per phase |
| Step 9 (segmentation inference) | A100 ×1 | 32 GB | ~30 min per phase |
| Step 17 (image classifier) | A100 ×1 | 32 GB | ~2-3 hr |
| Step 18 (survey classifier) | A100 ×1 | 32 GB | ~1-2 hr |

Cluster jobs go through SLURM (see `analysis/imagequality_classification/run_*.sh` for sweep templates that aren't covered by a single `make` target).

## Documentation Map

| File | What |
|---|---|
| `AGENTS.md` | Project conventions / rules (env, splits, schema invariants); the canonical rule reference |
| `CLAUDE.md` | One-line stub pointing at the docs that matter — kept so Claude Code finds it |
| `STATUS.md` | Current analysis state, paper feedback, open decisions |
| `notes/CODE_ORGANIZATION.md` | One-page rule for where to put new code |
| `notes/table_replication.md` | Master Tables 1/2/3 reproduction summary |
| `notes/table2_reproducibility.md`, `notes/table3_reproducibility.md` | Deep findings + variance analyses for the per-table reproduction |
| `data/splits/README.md`, `data/normalized/README.md` | Co-located docs for the splits and Promega-normalized metabolite directories |
| `REPLICATION_STATUS.md` | Table 1/2/3 and Fig 7 replication status and balanced-accuracy results |
| `VERIFICATION_REPORT.md` | End-to-end verification report: data coverage, model results, coding-standards audit |

---

**Document Version**: 3.2
**Last Updated**: 2026-05-27
