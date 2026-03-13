# Metabolite Classifier

LightGBM-based organoid quality classification using metabolite features (GlucoseGlo, GlutamateGlo, LactateGlo, PyruvateGlo, etc.).

## Path configuration

Shell scripts (`run_metabolites_gpu.s`) use `PROJ_ROOT` or `/home/YOUR_GITHUB_USERNAME/promega-classifier`. See the [root README](../README.md#path-configuration) for setup. Example:

```bash
export PROJ_ROOT=/home/yourname/promega-classifier
# or edit run_metabolites_gpu.s to set PROJ_ROOT
```

## Training

Run from repo root:

```bash
python metabolite_classifier/train_metabolites_cpu.py
python metabolite_classifier/train_metabolites_gpu.py   # if GPU available
```

## Scripts

- `train_metabolites_cpu.py` – CPU training
- `train_metabolites_gpu.py` – GPU training
- `train_metabolites_logreg.py` – Logistic regression variant
- `train_metabolites_trajectory.py` – Trajectory-based features
- `extract_feature_importance.py` – Feature importance export
- `analyze_results.py` – Results analysis

## Data

Uses `all_data.json` and `data_splits/both_{train,val,test}_base.json`. Metabolite features must be present in the merged records.
