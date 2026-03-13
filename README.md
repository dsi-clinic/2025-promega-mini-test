# Promega Organoid Classification – Final Submission

Organoid quality classification using image and metabolite data. This repository contains three classifier pipelines:

- **Image classifier** – CNN, EfficientNet, ViT, CNN-LSTM temporal models
- **Metabolite classifier** – LightGBM on metabolite features
- **Combined classifier** – Joint image + metabolite LightGBM model

## Directory Structure

```
├── README.md              (this file)
├── all_data.json          (merged organoid data; required)
├── config.py              (paths, env vars)
├── split_data.py          (train/val/test split generation)
├── split_data_reproducible.py  (reproducible splits by seed)
├── data_splits/           (JSON splits: both_train_base.json, etc.)
├── file_utils/            (data mapping: images, metabolites, surveys)
├── image_classifier/       (see image_classifier/README.md)
├── metabolite_classifier/ (see metabolite_classifier/README.md)
└── combined_classifier/   (see combined_classifier/README.md)
```

## Path configuration

Scripts and SLURM jobs use **`YOUR_GITHUB_USERNAME`** as a placeholder for your local path. Configure in one of two ways:

1. **Replace the placeholder** – In shell/SLURM scripts, change `/home/YOUR_GITHUB_USERNAME/promega-classifier` to your actual repo path (e.g. `/home/jsmith/promega-classifier`).

2. **Export `PROJ_ROOT`** (recommended) – Set before running:
   ```bash
   export PROJ_ROOT=/home/yourname/promega-classifier
   ```
   For SLURM:
   ```bash
   sbatch --export=PROJ_ROOT=/path/to/repo image_classifier/regeneration/submit_regeneration_run.slurm
   ```
   Or add `PROJ_ROOT` to your `.env` so it’s picked up automatically.

Scripts that respect `PROJ_ROOT` include: regeneration scripts, comparison runs, training, metabolite/combined classifiers, and all `submit_*.slurm` jobs.

## Quick Start

### Prerequisites

- Python 3.9+
- CUDA (for image models)
- See `pyproject.toml` for dependencies

### Run via Docker

```bash
docker build . -t promega-classifier
docker run -p 8888:8888 -v ${PWD}:/workspace promega-classifier
```

### Data Pipeline

1. Ensure `all_data.json` exists (merge of images, metabolites, surveys).
2. Generate splits: `python split_data.py` or `python split_data_reproducible.py --seed 42`.
3. Train image models: `image_classifier/training/train_model_accuracy.py`.
4. Train metabolite models: `metabolite_classifier/train_metabolites_cpu.py`.
5. Train combined model: `combined_classifier/train_combined_lgbm.py`.

### Environment

Set env vars in `.env` or export: `BASE_PATH`, `OUTPUT_FOLDER`, `RAW_IMAGE_DATA`, etc. See `config.py` for full list.

## Links

- [Image classifier](image_classifier/README.md)
- [Metabolite classifier](metabolite_classifier/README.md)
- [Combined classifier](combined_classifier/README.md)
