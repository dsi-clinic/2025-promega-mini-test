# Combined Classifier

Joint image + metabolite model using LightGBM on concatenated image (CNN) features and metabolite features.

## Path configuration

Shell scripts (`run_multimodal.s`) and SLURM (`submit_combined.slurm`) use `PROJ_ROOT` or `/home/YOUR_GITHUB_USERNAME/promega-classifier`. See the [root README](../README.md#path-configuration) for setup. Example:

```bash
export PROJ_ROOT=/home/yourname/promega-classifier
sbatch --export=PROJ_ROOT combined_classifier/submit_combined.slurm
```

## Training

Run from repo root:

```bash
python combined_classifier/train_combined_lgbm.py
```

## Scripts

- `train_combined_lgbm.py` – Combined image+metabolite LightGBM
- `train_multimodal.py` – Multimodal fusion (CNN + metabolite)
- `train_metabolites_cpu.py` – Metabolite-only training (shared with metabolite_classifier)
- `stitched_preprocessing.py` – Moved to `image_classifier/preprocessing/`

## Data

Requires `all_data.json` and `data_splits/`. Image features come from a pretrained CNN; metabolite features from the metabolite pipeline.
