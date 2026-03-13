# Promega Organoid Analysis System

This repository contains a comprehensive system for analyzing organoid quality using multimodal data including images, metabolites, and survey assessments for time series prediction.

## Team
- Ethan Waggoner
- Tony Luo
- Darin Keng
- Raabiyaal Ishaq

## Data Description

This project analyzes organoid development across **11 timepoints (Day 1–30)** using microscopy images, metabolite signals, and survey evaluations.

### Dataset Composition
- Four batches (Batch 3 & 4 excluded due to quality issues)
- 96 organoids per batch
- Some organoids removed due to incomplete data

### 1. Image Data
- Microscopy images collected at each timepoint
- Standardized resizing for model input
- Includes both split and stitched images

### 2. Metabolite Data
- GlucoseGlo, GlutamateGlo, LactateGlo, PyruvateGlo, MalateGlo, BCAAGlo

### 3. Survey Label Data
- Evaluated by 5 experts
- Only keep the Majority vote
- Used as ground truth labels for training



## Quick Start

### 1. Environment Setup
```bash
# The conda environment is located at:
/net/projects2/promega

# You don't need to activate it manually - the SLURM scripts will use it
```

### 2. Generate Master Data File
```bash
# From the project root directory, run:
cd /home/YOUR_GITHUB_USERNAME/MINITEST_DIRECTORY  # Replace YOUR_GITHUB_USERNAME with your actual GitHub username
/net/projects2/promega/bin/python file_utils/merge/merge_all_data.py

# This generates all_data.json with 5,168+ merged records
# Output: /net/projects2/promega/data-analysis/output/all_data.json
```

### IMPORTANT: Update Paths Before Running Analysis

**Before submitting any jobs**, you must update the hardcoded paths in the SLURM scripts to match your setup:

Update the `PROJ_ROOT` variable in SLURM scripts to match your setup:

**`analysis/images/image_classifier/run_training.s`**
    - Line 22: `PROJ_ROOT=${PROJ_ROOT:-/home/YOUR_GITHUB_USERNAME/MINITEST_DIRECTORY}`

Example:
```bash
# If your GitHub username is jsmith and you cloned to /home/jsmith/promega-analysis
# Change: PROJ_ROOT=${PROJ_ROOT:-/home/YOUR_GITHUB_USERNAME/MINITEST_DIRECTORY}
# To:     PROJ_ROOT=${PROJ_ROOT:-/home/jsmith/promega-analysis}
```

**Note:** The `${PROJ_ROOT:-default}` syntax allows overriding via environment variable: `export PROJ_ROOT=/custom/path` before running `sbatch`.

### 3. Run Analysis on GPU Computation Nodes

**⚠️ Important Note**: This section applies **only to the Image Classifier**. The metabolite classifier is lightweight and should be run locally (see `analysis/metabolites/classifier/README.md` for local execution instructions).

#### Image Classifier (GPU Required - Cluster Only)

The image classifier requires GPU computation due to the complexity of deep learning models (EfficientNet, ResNet, Vision Transformers) and large image datasets. **This assumes you have access to a cluster with GPU nodes** (e.g., SLURM-managed cluster).

**If you do not have cluster/GPU access**, you will need to:
- Use a cloud GPU service (Google Colab, AWS, etc.)
- Adapt the training scripts for local GPU execution
- Or contact your institution about cluster access

**For users with cluster access:**

```bash
# Navigate to classifier directory
cd /home/YOUR_GITHUB_USERNAME/MINITEST_DIRECTORY/analysis/images/image_classifier

# Submit the training job to SLURM
sbatch run_training.s --input-path-key img_path

# Monitor job status
squeue -u $USER

# Check logs
tail -f logs/train-img_<JOBID>.out
```

The image classifier will train models for each day (Dy3, Dy6, Dy8, etc.) sequentially.
Results are saved in the output directory specified (or defaults based on configuration).

**Why GPU is required for image classifier:**
- Deep learning models (EfficientNet, ResNet, Vision Transformers) require significant computational resources
- Large image datasets (512x384 resolution, multiple timepoints per organoid)
- Training multiple models per day with extensive hyperparameter tuning
- Typical training time: several hours per day on GPU vs. days/weeks on CPU

**Metabolite classifier (run locally):**
The metabolite classifier uses LightGBM (gradient boosting) which is computationally lightweight and runs efficiently on CPU. See `analysis/metabolites/classifier/README.md` for local execution instructions.

## Configuration System

The system uses a centralized `config.py` file that loads configuration from environment variables. Key variables:

- `BASE_PATH` - Root directory for raw data
- `OUTPUT_FOLDER` - Location for processed outputs
- `SURVEY_RESULTS` - Directory containing Excel survey files
- `METABOLITE_DATA_DIR` - Directory for metabolite Excel files
- `TARGET_WIDTH` / `TARGET_HEIGHT` - Image processing dimensions

Create a `.env` file in the project root with these variables set to your local paths.

## Data Processing Pipeline

1. **Individual Mappers**: Process raw data sources
   - `file_utils/images/image_mapper_main.py` - Maps image files to metadata
   - `file_utils/metabolites/metabolite_mapper.py` - Processes metabolite Excel data
   - `file_utils/surveys/surveys_mapper.py` - Processes survey Excel data

2. **Master Merger**: Combines all data sources
   - `file_utils/merge/merge_all_data.py` - Creates unified `all_data.json`

3. **Reproducible Data Splits**: Creates train/validation splits for ML models
   - `split_data.py` - Splits data by organoid (prevents data leakage) and extracts metabolite features
   - Extracts both `concentration_uM` and `initial_concentration` for each metabolite (e.g., `GlucoseGlo_concentration_uM`, `GlucoseGlo_initial_concentration`)
   - Outputs saved to `data_splits/` directory

4. **Analysis**: Uses `all_data.json` as single source of truth
   - All analysis code in `analysis/` directory
   - No direct access to raw data files
   - Standardized organoid key format: `"BA1 96_1 Dy30 A1"`

## Data Structure

The `all_data.json` file contains unified organoid data with structure:
```json
{
  "BA1 96_1 Dy03 A1": {
    "dayID": "Dy03",
    "BA": "BA1 96_1",
    "wellID": "A1",
    "day_num": 3,
    "mdl_day": 3.0,
    "Best Z Filename": "/path/to/image.tif",
    "256x192": { "img_path": "...", "mask_path": "..." },
    "512x384": { "img_path": "...", "mask_path": "..." },
    "metabolites": { "GlucoseGlo": {...}, "ATP": {...} },
    "survey": { "evaluations": [...], "quality_scores": [...] }
  }
}
```
A file **`AllData_Summary.xlsx`** is included for organoid-wise and general summary of the all_data.json.


## Data Split Script

The `split_data.py` script creates reproducible train/validation/test splits for image and metabolite models. It splits by **organoid** (not individual samples) to prevent data leakage when using early days to predict Dy30 outcomes.

**Key features:**
- Uses Dy30 labels with 4/5 expert consensus
- Only BA1 and BA2 batches (high quality)
- Requires complete metabolite data and valid processed images
- Final ratios: 72% Train / 8% Val / 20% Test
- Fixed random seed (42) for reproducibility

**Usage:**
```bash
# Default (include all images)
python split_data.py

# Exclude stitched images only
python split_data.py --switch exclude_stitched_only

# Exclude split/presplit images only
python split_data.py --switch exclude_split_only

# Exclude both stitched and split images
python split_data.py --switch exclude_both

# Generate all four modes
python split_data.py --all
```

Outputs are saved to `data_splits/` directory. See `split_data.py` for detailed documentation.

## Classifiers

- **Image Classifier**: ViT/ResNet/EfficientNet backbones with focal loss. See `analysis/images/image_classifier/README.md` for details.
- **Metabolite Classifier**: Per-day LightGBM classifiers with class weighting and threshold tuning. See `analysis/metabolites/classifier/README.md` for details.

## Development Guidelines

- **Environment**: Always activate conda environment first: `conda activate /net/projects2/promega`
- **Configuration**: Use `config.py` for all path and setting management
- **Data Access**: Use `all_data.json` as single source of truth
- **Analysis Location**: Place all analysis code in `analysis/` directory
- **Execution**: Run everything from project root directory
