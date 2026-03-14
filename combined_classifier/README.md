# Combined Classifier

Joint image + metabolite model using LightGBM on concatenated image (CNN) features and metabolite features.

## Files

| File | Description |
|------|-------------|
| `train_combined_lgbm.py` | Main combined image+metabolite LightGBM training |
| `train_multimodal.py` | Multimodal fusion (CNN + metabolite) training |
| `train_metabolites_cpu.py` | Metabolite-only training (shared logic with metabolite_classifier) |
| `run_multimodal.s` | SLURM job script for multimodal training |
| `submit_combined.slurm` | SLURM submission script for combined model |

## How to Run

Set `PROJ_ROOT` and run from the repo root. See the [root README](../README.md#how-to-run) for full environment setup.

```bash
export PROJ_ROOT=/home/yourname/promega-classifier

python combined_classifier/train_combined_lgbm.py
```

SLURM submission:

```bash
sbatch --export=PROJ_ROOT=$PROJ_ROOT combined_classifier/submit_combined.slurm
```

## Data

Requires `all_data.json` and `data_splits/`. **Image features**: extracted from a pretrained CNN (e.g. EfficientNet) on the processed organoid images; one feature vector per sample. **Metabolite features**: GlucoseGlo, GlutamateGlo, LactateGlo, PyruvateGlo concentrations from the metabolite mapping pipeline. The combined model concatenates both feature sets and trains a LightGBM classifier. Labels from survey consensus (Dy30). See [root README](../README.md#data) for full data description.
