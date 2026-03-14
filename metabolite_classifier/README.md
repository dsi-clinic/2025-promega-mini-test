# Metabolite Classifier

LightGBM-based organoid quality classification using metabolite features (GlucoseGlo, GlutamateGlo, LactateGlo, PyruvateGlo).

## Files

| File | Description |
|------|-------------|
| `train_metabolites_cpu.py` | Main CPU training script |
| `train_metabolites_gpu.py` | GPU training script |
| `train_metabolites_logreg.py` | Logistic regression variant |
| `train_metabolites_logreg_nogrowth.py` | Logistic regression without growth-rate features |
| `train_metabolites_trajectory.py` | Trajectory-based temporal features |
| `train_metabolites_trajectory_legacy.py` | Legacy trajectory training |
| `train_metabolites_allpreds.py` | All-organoid prediction export |
| `train_metabolites_legacy.py` | Legacy training script (reference only) |
| `extract_feature_importance.py` | Feature importance analysis and export |
| `analyze_results.py` | Results analysis and summary |
| `run_metabolites_gpu.s` | SLURM job script for GPU training |
| `submit_gpu_experiments.sh` | Batch submission helper |

## How to Run

Set `PROJ_ROOT` and run from the repo root. See the [root README](../README.md#how-to-run) for full environment setup.

```bash
export PROJ_ROOT=/home/yourname/promega-classifier

python metabolite_classifier/train_metabolites_cpu.py
python metabolite_classifier/train_metabolites_gpu.py   # if GPU available
```

## Data

**Features**: metabolite assay concentrations (uM) for GlucoseGlo, GlutamateGlo, LactateGlo, and PyruvateGlo, sourced from `metabolite_data_07_23_25.xlsx` and mapped via `metabolite_map.json` into `all_data.json`. Optional trajectory features use values across multiple timepoints (Dy03-Dy30). **Labels**: Acceptable / Not Acceptable from survey consensus (Dy30, 4/5 vote). Splits: `data_splits/both_{train,val,test}_base.json`. See [root README](../README.md#data) for full data description.
