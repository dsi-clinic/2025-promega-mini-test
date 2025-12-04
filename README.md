# Promega Organoid Analysis System

This repository contains a comprehensive system for analyzing organoid quality using multimodal data including images, metabolites, and survey assessments for time series prediction.

## Recent Changes (November 2025)

### Data Reorganization
The data pipeline has been reorganized to use a normalized records structure:

- **Normalized Records System**: Introduced `file_utils/merge/normalized_records.py` with `OrganoidRecord`, `OrganoidRecordBuilder`, and `RecordMetrics` to canonicalize organoid data
- **Unified Data Structure**: `all_data.json` now uses a `records` map with standardized organoid IDs plus a `stats` block containing metadata (totals, vote counts, metabolite outliers, skipped items)
- **View-Specific Outputs**: The merge process generates specialized views:
  - `all_data.json` - Complete unified dataset with all organoid records
  - `image_classifier.json` - Day-indexed records optimized for image classifier training
  - `survey_classifier.json` - Day-indexed records optimized for survey classifier training
- **Enhanced Logging**: Rich-based structured logging throughout the merge process
- **Metadata Tracking**: Each view includes metadata about skipped records, vote counts, and data quality metrics

### Classifier Updates
- **Image Classifier** and **Survey Classifier**: Updated to read from new normalized JSON structure (`records` map instead of flat files)
- **Deterministic Training**: Added support for reproducible training runs with deterministic operations and seed control

## Project Structure

```mermaid
flowchart TD
    %% ========= INPUT STAGE ========= %%
    A1([Raw Images])
    A2([Metabolite Excels])
    A3([Survey Excels])
    A4([Config & Env Vars<br/>config.py / core_env.yaml])

    %% ========= FILE_UTILS PROCESSING ========= %%
    subgraph B[file_utils - Data Mapping & Integration]
        B1[file_utils/images/scripts<br/>image_mapper_main.py<br/>Image metadata → JSON]
        B1b[file_utils/common/organoid_patterns.py<br/>Pattern normalization helpers]
        B2[file_utils/metabolites/metabolite_mapper.py<br/>Metabolite Excel → JSON]
        B3[file_utils/surveys/surveys_mapper.py<br/>Survey Excel → JSON]
        B4[file_utils/merge/merge_all_data.py<br/>Merge image + metabolite + survey JSON → all_data.json]
    end

    %% ========= ANALYSIS PIPELINE ========= %%
    subgraph C[analysis - Downstream Analysis & ML]
        subgraph C1[analysis/images]
            C11[resize<br/>Standardize image size + pixel scale]
            C12[metrics/shape_metrics<br/>Organoid shape features]
            C13[segmentation_mmseg<br/>MMSeg training & inference]
            C14[classifier<br/>Image classifiers – ViT / CNN]
            C15[series/preprocess<br/>Filter complete time series + normalize masks]
        end

        subgraph C2[analysis/metabolites]
            C21[classifier<br/>Metabolite-based classifiers]
        end

        subgraph C3[analysis/surveys]
            C31[agreement_aggregations<br/>Survey agreement analysis]
            C32[classifier<br/>Survey-based classifiers]
            C33[simulations<br/>Reliability simulations]
        end

        C4[multimodal<br/>CNN fusion of image + metabolite + survey features]
    end

    %% ========= DATA FLOW ========= %%
    A1 --> B1
    B1b --> B1
    A2 --> B2
    A3 --> B3
    B1 --> B4
    B2 --> B4
    B3 --> B4

    %% From merged data to analyses
    B4 --> C11
    C11 --> C13
    C13 --> C14
    C14 --> C15
    C15 --> C4

    B4 --> C21
    B4 --> C31
    B4 --> C4
```

## Setup and Installation

### Development Setup (Full Installation)

For local development with full code access:

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd 2025-promega-mini-test
   ```

2. **Set up Python environment**:
   ```bash
   # On cluster, use the shared conda environment:
   # /net/projects2/promega

   # For local development, create a conda environment:
   conda create -n promega python=3.10
   conda activate promega
   pip install -r requirements.txt  # If available, or install manually:
   # torch, torchvision, timm, tensorflow, keras, numpy, scikit-learn,
   # pillow, matplotlib, rich, python-dotenv
   ```

3. **Configure environment variables** (optional):
   Create a `.env` file in the project root:
   ```bash
   BASE_PATH=/path/to/raw/data
   OUTPUT_FOLDER=/path/to/output
   SURVEY_RESULTS=/path/to/survey/excel/files
   METABOLITE_DATA_DIR=/path/to/metabolite/excel/files
   TARGET_WIDTH=512
   TARGET_HEIGHT=384
   ```

### Runtime Setup (Cluster Only)

For running existing code on the cluster without development:

1. **Access the cluster** and navigate to your project directory
2. **Use the shared conda environment**: `/net/projects2/promega`
3. **Update SLURM script paths** (see Quick Start section)
4. **No additional installation needed** - all dependencies are in the shared environment

## Quick Start

### 1. Environment Setup

**On Cluster**:
```bash
# The conda environment is located at:
/net/projects2/promega

# You don't need to activate it manually - the SLURM scripts will use it
```

**Local Development**:
```bash
# Activate your local conda environment
conda activate promega  # or your environment name
```

### 2. Generate Master Data File

**On Cluster**:
```bash
# From the project root directory, run:
cd /home/YOUR_GITHUB_NAME/MINITEST_DIRECTORY  # Replace with your actual path
/net/projects2/promega/bin/python file_utils/merge/merge_all_data.py \
    --in-dir /net/projects2/promega/data-analysis \
    --out-dir /net/projects2/promega/data-analysis/output

# This generates:
# - all_data.json (5,168+ merged records)
# - image_classifier.json (day-indexed view for image training)
# - survey_classifier.json (day-indexed view for survey training)
# Output location: /net/projects2/promega/data-analysis/output/json/
```

**Local Development**:
```bash
# From project root:
python file_utils/merge/merge_all_data.py \
    --in-dir /path/to/input/data \
    --out-dir ./data/output
```

### ⚠️ IMPORTANT: Update Paths Before Running Analysis

**Before submitting any jobs**, you must update the hardcoded paths in the SLURM scripts to match your setup:

Replace `/home/tonyluo/minitest` with `/home/YOUR_GITHUB_NAME/YOUR_MINITEST_DIRECTORY` in:

1. **`analysis/images/classifier/run_accuracy.s`**
   - Line 13: `PROJ_ROOT=/home/YOUR_GITHUB_NAME/YOUR_MINITEST_DIRECTORY`

2. **`analysis/surveys/classifier/run_survey_classifier.s`**
   - Line 12: `PROJ_ROOT=/home/YOUR_GITHUB_NAME/YOUR_MINITEST_DIRECTORY`

Example:
```bash
# If your username is jsmith and you cloned to /home/jsmith/promega-analysis
# Change: PROJ_ROOT=/home/tonyluo/minitest
# To:     PROJ_ROOT=/home/jsmith/promega-analysis
```

### 3. Run Analysis

#### Cluster (GPU Required)

**Important**: Analysis must be run on computation nodes (not login nodes) using SLURM job submission.

##### 3a. Image Classifier
```bash
# Navigate to classifier directory
cd /home/YOUR_GITHUB_NAME/MINITEST_DIRECTORY/analysis/images/classifier

# Submit the training job to SLURM
sbatch run_accuracy.s

# Monitor job status
squeue -u $USER

# Check logs
tail -f logs/soft-label_<JOBID>.out
```

The image classifier will train models for each day (Dy3, Dy6, Dy8, etc.) sequentially.
Results are saved in `outputs_512x384_Regular_image_with_train_augment_with_auroc/vit/DyXX/`

##### 3b. Survey Classifier
```bash
# Navigate to survey classifier directory
cd /home/YOUR_GITHUB_NAME/MINITEST_DIRECTORY/analysis/surveys/classifier

# Submit the survey classifier job
sbatch run_survey_classifier.s

# Check completion
squeue -u $USER
cat logs/survey_<JOBID>.out
```

The survey classifier trains a ResNet50V2+CNN dual-input model on Day 30 organoids using survey evaluation labels.
Results include trained model (`.h5`), training curves, and confusion matrix.

#### Local Development

For local testing and development:

```bash
# Image Classifier
cd analysis/images/classifier
python train_model_accuracy.py \
    --out-dir ./outputs \
    --batch-size 8 \
    --epoch1 10 \
    --epoch2 20 \
    --test-frac 0.1 \
    --val-frac 0.1 \
    --target-width 512 \
    --target-height 384 \
    --seed 1 \
    --deterministic

# Survey Classifier
cd analysis/surveys/classifier
python simple_classifier.py \
    --out-dir ./outputs \
    --batch-size 8 \
    --epoch1 10 \
    --epoch2 20 \
    --target-day Dy30 \
    --target-width 224 \
    --target-height 224 \
    --seed 1 \
    --deterministic
```

**Note**: Local development requires GPU access for training. For CPU-only testing, use very small batch sizes and epochs, or test with a subset of data.

**Note**: Before submitting jobs, update the SLURM scripts:
- `analysis/images/classifier/run_accuracy.s` - Update `PROJ_ROOT` variable (line 13)
- `analysis/surveys/classifier/run_survey_classifier.s` - Update `PROJ_ROOT` variable (line 13)

## Configuration System

The system uses a centralized `config.py` file that loads configuration from environment variables. Key variables:

- `BASE_PATH` - Root directory for raw data
- `OUTPUT_FOLDER` - Location for processed outputs
- `SURVEY_RESULTS` - Directory containing Excel survey files
- `METABOLITE_DATA_DIR` - Directory for metabolite Excel files
- `TARGET_WIDTH` / `TARGET_HEIGHT` - Image processing dimensions

Create a `.env` file in the project root with these variables set to your local paths.

## Command Line Arguments

### Data Merge (`file_utils/merge/merge_all_data.py`)

**Entry Point**: `python file_utils/merge/merge_all_data.py`

**Required Arguments**:
- `--in-dir`: Path to input directory containing organoid data
- `--out-dir`: Path to output directory where JSON files will be saved

**Optional Arguments**:
- `--min-survey-votes`: Minimum votes for survey label (default: 4)
- `--survey-day`: Day that survey was conducted (default: 30)
- `--target-width`: Target image width in pixels (default: 512)
- `--target-height`: Target image height in pixels (default: 384)

**Example**:
```bash
python file_utils/merge/merge_all_data.py \
    --in-dir /path/to/input \
    --out-dir /path/to/output \
    --min-survey-votes 4 \
    --target-width 512 \
    --target-height 384
```

### Image Classifier (`analysis/images/classifier/train_model_accuracy.py`)

**Entry Point**: `python train_model_accuracy.py`

**Required Arguments**:
- `--out-dir`: Path to output directory where results will be saved

**Optional Arguments**:
- `--epoch1`: Number of training epochs for phase 1 (frozen backbone) (default: 100)
- `--epoch2`: Number of training epochs for phase 2 (unfrozen backbone) (default: 300)
- `--batch-size`: Training batch size (default: 16)
- `--val-batch-size`: Validation/Test batch size (defaults to batch-size)
- `--test-frac`: Fraction of data used for testing (default: 0.1)
- `--val-frac`: Fraction of data used for validation (default: 0.1)
- `--use-mask`: Include mask tensors and a mask branch in the classifier (default: False)
- `--input-path-key`: JSON field to use as image input ('img_path' or 'overlay_path') (default: 'img_path')
- `--target-width`: Target input image width in pixels (default: 512)
- `--target-height`: Target input image height in pixels (default: 384)
- `--num-workers`: Number of subprocesses for data loading (default: 0)
- `--seed`: Random seed for reproducibility (default: 1)
- `--deterministic`: Use deterministic operations for reproducibility (default: False)

**Example**:
```bash
python train_model_accuracy.py \
    --out-dir ./outputs \
    --batch-size 16 \
    --epoch1 50 \
    --epoch2 150 \
    --test-frac 0.1 \
    --val-frac 0.1 \
    --target-width 512 \
    --target-height 384 \
    --seed 1 \
    --deterministic
```

### Survey Classifier (`analysis/surveys/classifier/simple_classifier.py`)

**Entry Point**: `python simple_classifier.py`

**Required Arguments**:
- `--out-dir`: Path to output directory where results will be saved

**Optional Arguments**:
- `--batch-size`: Training batch size (default: 8)
- `--epoch1`: Number of training epochs for phase 1 (frozen backbone) (default: 50)
- `--epoch2`: Number of training epochs for phase 2 (unfrozen backbone) (default: 150)
- `--target-day`: Target day for training (default: "Dy30")
- `--target-width`: Target input image width in pixels (default: 224)
- `--target-height`: Target input image height in pixels (default: 224)
- `--deterministic`: Use deterministic operations (default: False)
- `--seed`: Random seed for reproducibility (default: 1)

**Example**:
```bash
python simple_classifier.py \
    --out-dir ./outputs \
    --batch-size 8 \
    --epoch1 50 \
    --epoch2 150 \
    --target-day Dy30 \
    --target-width 224 \
    --target-height 224 \
    --seed 1 \
    --deterministic
```

## Input Data Types

The system processes three main types of input data:

### 1. Image Data
- **Raw Images**: Multi-Z-stack TIFF files (`.tif`) from microscopy
- **Processed Images**: Resized PNG files at multiple resolutions:
  - `256x192`: Lower resolution for quick processing
  - `512x384`: Standard resolution for training (default)
- **Masks**: Segmentation masks (predicted or manual) as PNG files
- **Overlays**: Image-mask overlay visualizations

**Location on Cluster**: `/net/projects2/promega/data-analysis/output/infer_resized_512x384/`

**Structure**:
```
images/
├── raw_images/          # Original TIFF files
├── infer_resized_512x384/  # Processed PNG files
└── ...
masks/
├── predicted/           # ML-generated masks
├── manual/              # Human-annotated masks
└── image_overlays/      # Visualization overlays
```

### 2. Metabolite Data
- **Format**: Excel spreadsheets (`.xlsx`)
- **Content**: Chemical assay measurements for 6 metabolites:
  - BCAAGlo, GlucoseGlo, GlutamateGlo, LactateGlo, MalateGlo, PyruvateGlo
- **Fields**: Concentration values, initial concentrations, outlier flags, well mappings

**Location on Cluster**: `/net/projects2/promega/data-analysis/metabolite_data/`

**Example File**: `metabolite_data_07_23_25.xlsx`

### 3. Survey Data
- **Format**: Excel spreadsheets (`.xlsx`)
- **Content**: Quality assessment evaluations from human raters
- **Structure**: Individual votes (5 per organoid) with "Acceptable" or "Not Acceptable" labels
- **Processing**: Majority voting (4+ votes) determines final label

**Location on Cluster**: `/net/projects2/promega/data-analysis/results_surveys/`

**Example File**: `organoid_surveys_aggregated.json` (generated from Excel)

### 4. Generated JSON Files

After running the merge process, the following JSON files are generated:

- **`all_data.json`**: Complete unified dataset with all organoid records
  - Structure: `{"schema_version": 1, "generated_at": "...", "stats": {...}, "records": {...}}`
  - Records keyed by organoid ID: `"BA1_96_1_Dy03_A1"`

- **`image_classifier.json`**: Day-indexed view for image classifier training
  - Structure: `{"metadata": {...}, "records": {"Dy3": {...}, "Dy6": {...}, ...}}`
  - Each day contains arrays: `img_path`, `mask_path`, `label`

- **`survey_classifier.json`**: Day-indexed view for survey classifier training
  - Structure: `{"metadata": {...}, "records": {"Dy30": {...}}}`
  - Contains arrays: `img_path`, `mask_path`, `label` (computed from survey evaluations)

## Resource Requirements

### Cluster Resources (SLURM)

**Image Classifier**:
- **GPU**: 1x A100 (required)
- **Memory**: 32GB RAM
- **Time**: ~2 hours per job
- **Storage**: ~10GB for model checkpoints and outputs per training run

**Survey Classifier**:
- **GPU**: 1x A100 (required)
- **Memory**: 32GB RAM
- **Time**: ~2 hours per job
- **Storage**: ~5GB for model checkpoints and outputs

**Data Merge**:
- **CPU**: Standard compute node (no GPU needed)
- **Memory**: 8GB RAM (sufficient for 5,168 records)
- **Time**: ~5-10 minutes
- **Storage**:
  - Input: ~50GB (raw images, processed images, masks)
  - Output: ~500MB (JSON files)

### Local Development

**Minimum Requirements**:
- **GPU**: NVIDIA GPU with CUDA support (recommended) or CPU-only for small-scale testing
- **Memory**: 16GB RAM minimum, 32GB recommended
- **Storage**:
  - Code: ~500MB
  - Data: Depends on subset size (see Test Data section)
  - Models: ~2-5GB per training run

**Recommended for Full Training**:
- **GPU**: NVIDIA GPU with 8GB+ VRAM (RTX 3070/3080, A100, etc.)
- **Memory**: 32GB+ RAM
- **Storage**: 100GB+ free space

## Test Data and Quick Development

### Test Data Availability

Currently, there is **no dedicated test dataset** for quick local development. However, you can:

1. **Use a subset of the full dataset**:
   ```python
   # In your training script, filter to a single day with fewer samples
   # Example: Use only Dy3 which typically has fewer organoids
   python train_model_accuracy.py --out-dir ./test_outputs --epoch1 5 --epoch2 10
   ```

2. **Reduce data size for testing**:
   - Train on a single day instead of all days
   - Use smaller batch sizes (4-8 instead of 16)
   - Reduce number of epochs (5-10 instead of 100-300)

3. **Create a minimal test set** (manual):
   - Copy 10-20 images and corresponding masks to a test directory
   - Create a minimal JSON file with just those records
   - Point your training script to this test data

### Agile Development Workflow

For iterative development:

1. **Start with minimal configuration**:
   ```bash
   # Quick test run with minimal epochs
   python train_model_accuracy.py \
       --out-dir ./test_outputs \
       --epoch1 2 \
       --epoch2 5 \
       --batch-size 4 \
       --test-frac 0.2 \
       --val-frac 0.2
   ```

2. **Use deterministic mode** for reproducible debugging:
   ```bash
   --deterministic --seed 1
   ```

3. **Monitor with smaller validation sets** to speed up iteration

4. **Test code changes** before running full training on cluster

## Data Sharing on Cluster

### Data Locations

**Shared Data Directory**: `/net/projects2/promega/data-analysis/`

**Structure**:
```
/net/projects2/promega/data-analysis/
├── output/
│   ├── json/
│   │   ├── all_data.json
│   │   ├── image_classifier.json
│   │   └── survey_classifier.json
│   ├── infer_resized_512x384/  # Processed images
│   └── ...
├── metabolite_data/
│   └── metabolite_data_07_23_25.xlsx
├── results_surveys/
│   └── organoid_surveys_aggregated.json
└── ...
```

### Sharing Your Results

1. **Model outputs**: Save to your home directory or a shared results directory
2. **Generated JSON files**: Can be shared via the shared data directory
3. **Logs**: Keep in your project's `logs/` directory

### Accessing Shared Data

All cluster users have read access to `/net/projects2/promega/data-analysis/`

## Data Processing Pipeline

1. **Individual Mappers**: Process raw data sources
   - `file_utils/images/image_mapper_main.py` - Maps image files to metadata
   - `file_utils/metabolites/metabolite_mapper.py` - Processes metabolite Excel data
   - `file_utils/surveys/surveys_mapper.py` - Processes survey Excel data

2. **Master Merger**: Combines all data sources
   - `file_utils/merge/merge_all_data.py` - Creates unified `all_data.json` and view-specific JSON files

3. **Analysis**: Uses normalized JSON files as single source of truth
   - All analysis code in `analysis/` directory
   - No direct access to raw data files
   - Standardized organoid key format: `"BA1_96_1_Dy03_A1"`

## Data Structure

The `all_data.json` file contains unified organoid data with structure:
```json
{
  "schema_version": 1,
  "generated_at": "2025-11-24T16:34:36.725704+00:00",
  "stats": {
    "total_records": 5168,
    "survey_matches": 393,
    "num_acceptable_votes": 1356,
    "num_not_acceptable_votes": 749,
    ...
  },
  "records": {
    "BA1_96_1_Dy03_A1": {
      "id": "BA1 96_1 Dy03 A1",
      "day": {
        "id": "Dy3",
        "number": 3.0,
        "original": "Dy03"
      },
      "cell_line": "GM23279A",
      "images": {
        "processed": {
          "img_path": "/path/to/image.png",
          "mask_path": "/path/to/mask.png",
          "overlay_path": "/path/to/overlay.png"
        }
      },
      "metabolites": {
        "GlucoseGlo": {
          "concentration_uM": 9.827,
          "is_outlier": false
        },
        ...
      },
      "survey": {
        "evaluations": [...],
        "label": {
          "value": "Acceptable",
          "acceptance_flag": 1
        }
      }
    }
  }
}
```

The view-specific files (`image_classifier.json`, `survey_classifier.json`) use a day-indexed structure:
```json
{
  "metadata": {
    "total_skipped": 2041,
    ...
  },
  "records": {
    "Dy3": {
      "img_path": ["/path/to/img1.png", ...],
      "mask_path": ["/path/to/mask1.png", ...],
      "label": [1, 0, 1, ...]
    },
    "Dy6": {...},
    ...
  }
}
```

## Key Features

- **Multimodal Data Integration**: Images, metabolites, and surveys in one structure
- **Time Series Analysis**: Organoid quality tracking across days (Dy3, Dy6, Dy8, etc.)
- **Standardized Processing**: Consistent image resolutions and metadata
- **Normalized Records**: Canonical organoid representation with metadata tracking
- **View-Specific Outputs**: Optimized JSON views for different analysis tasks
- **Environment-Based Configuration**: No hardcoded paths
- **Comprehensive Analysis Tools**: Classification, segmentation, and statistical analysis
- **Reproducible Training**: Deterministic mode and seed control for consistent results

## Development Guidelines

- **Environment**: Always activate conda environment first: `conda activate /net/projects2/promega` (cluster) or your local environment
- **Configuration**: Use `config.py` for all path and setting management
- **Data Access**: Use normalized JSON files (`all_data.json`, `image_classifier.json`, `survey_classifier.json`) as single source of truth
- **Analysis Location**: Place all analysis code in `analysis/` directory
- **Execution**: Run everything from project root directory
- **Reproducibility**: Use `--deterministic` and `--seed` flags for reproducible experiments

## Current Status

✅ **Fully Functional System** (Updated November 2025)
- Data reorganization completed with normalized records structure
- All immediate code quality fixes completed
- Working data generation pipeline producing complete 5,168-record dataset
- Multimodal data integration (images, metabolites, surveys) operational
- Centralized configuration and pattern management implemented
- Comprehensive error handling and validation added
- View-specific JSON outputs for optimized classifier training
- Deterministic training support for reproducible experiments

## Known Issues & Future Improvements

See `CLAUDE.md` for detailed code analysis and recommended architectural enhancements.


