# Image Classifier

Image-based organoid quality classification using CNN, EfficientNet, ViT, and temporal (CNN-LSTM, effnet_ts) models.

## Structure

- **training/** -- Per-day models (EfficientNet, ViT, ResNet), soft labels, ensembles
- **cnn_lstm/** -- Temporal models (CNN-LSTM, effnet_ts, temporal change)
- **segmentation/** -- MMSegmentation mask generation
- **preprocessing/** -- Quality, resize, series prep, stitched preprocessing
- **comparison_runs/** -- Per-day study, threshold tuning, metrics export
- **day13_15_challenge/** -- Day 13/15 robustness experiments (grayscale norm, filled mask)
- **baseline/** -- EfficientNet single-timepoint baseline results
- **regeneration/** -- Seed-rotation reproducibility scripts
- **surveys/** -- Survey agreement aggregation, label preprocessing

## How to Run

Set `PROJ_ROOT` and run from the repo root. See the [root README](../README.md#how-to-run) for full environment setup.

```bash
export PROJ_ROOT=/home/yourname/promega-classifier

python image_classifier/training/train_model_accuracy.py
python image_classifier/training/train_efficientnet_improved_tnr.py
```

For comparison runs (per-day study, threshold study):

```bash
python image_classifier/comparison_runs/run_per_day_study.py --model_type effnet_ts --day 30
python image_classifier/comparison_runs/run_threshold_study.py
```

SLURM submission:

```bash
sbatch --export=PROJ_ROOT=$PROJ_ROOT image_classifier/comparison_runs/submit_per_day_study_effnet_ts.slurm
sbatch --export=PROJ_ROOT=$PROJ_ROOT image_classifier/regeneration/submit_regeneration_run.slurm
```

## Data

Inputs: **processed organoid images** (`img_path` from `all_data.json`), resized to the target dimensions and paired with MMSegmentation masks. **Labels** come from the survey consensus at Dy30 (Acceptable / Not Acceptable, 4/5 vote). Splits are read from `data_splits/both_{train,val,test}_*.json`; the image pipeline supports base, base_no_stitch, and style variants. See [root README](../README.md#data) for full data sources and merge pipeline.

## Dependencies

- PyTorch, torchvision, timm
- mmsegmentation (for segmentation)
- scikit-image, scipy (for preprocessing)
