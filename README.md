# Promega Organoid Analysis System

This repository contains a comprehensive system for analyzing organoid quality using multimodal data including images, metabolites, and survey assessments for time series prediction.

Table of Contents
=================

* [Promega Organoid Analysis System](#promega-organoid-analysis-system)
   * [Recent Changes (January 2026)](#recent-changes-january-2026)
      * [Data Reorganization](#data-reorganization)
      * [Classifier Updates](#classifier-updates)
      * [Label Propagation System](#label-propagation-system)
   * [Project Structure](#project-structure)
   * [Setup Promega Code](#setup-promega-code)
   * [Run Promega Code](#run-promega-code)
      * [1. Activate Environment](#1-activate-environment)
      * [2. Run Data Processing Pipeline Analysis](#2-run-data-processing-pipeline-analysis)
         * [2a. Prepare to run on Cluster](#2a-prepare-to-run-on-cluster)
         * [2b. Image Classifier](#2b-image-classifier)
         * [2c. Survey Classifier](#2c-survey-classifier)
   * [Data Processing Pipeline](#data-processing-pipeline)
      * [Overview](#overview)
      * [Prerequisites](#prerequisites)
      * [Raw Input Data Structure](#raw-input-data-structure)
      * [STEP 1: Retrieve Main Identifiers](#step-1-retrieve-main-identifiers)
      * [STEP 2: Map Metabolite Data](#step-2-map-metabolite-data)
      * [STEP 3: Map Survey Data](#step-3-map-survey-data)
      * [STEP 4: Map Image Data](#step-4-map-image-data)
      * [STEP 5: Map Manual Masks](#step-5-map-manual-masks)
      * [STEP 6: Resize and Remap Images](#step-6-resize-and-remap-images)
      * [STEP 7: Test Splits JSON](#step-7-test-splits-json)
      * [STEP 8: Train Segmentation Masks](#step-8-train-segmentation-masks)
      * [STEP 9: Predict Segmentation Masks](#step-9-predict-segmentation-masks)
      * [STEP 10: Image Mask Overlay](#step-10-image-mask-overlay)
      * [STEP 11: Mask Edge Fraction](#step-11-mask-edge-fraction)
      * [STEP 12: Filter Complete Series](#step-12-filter-complete-series)
      * [STEP 13: Preprocess for LSTM](#step-13-preprocess-for-lstm)
      * [STEP 14: Resize Aspect Ratio](#step-14-resize-aspect-ratio)
      * [STEP 15: Mean Fill Clip](#step-15-mean-fill-clip)
      * [STEP 16: Generate All Data JSON File](#step-16-generate-all-data-json-file)
      * [STEP 17: Image Classifier Training](#step-17-image-classifier-training)
      * [STEP 18: Survey Classifier Training](#step-18-survey-classifier-training)
   * [Data](#data)
      * [Main data file structure](#main-data-file-structure)
      * [Input Data Types](#input-data-types)
   * [Resource Requirements](#resource-requirements)
      * [Cluster Resources (SLURM)](#cluster-resources-slurm)
      * [Local Development](#local-development)
   * [Local development and testing](#local-development-and-testing)
   * [Current Status](#current-status)
   * [Known Issues & Future Improvements](#known-issues--future-improvements)

---
DATA ORGANIZATION

Use images.img_path for baseline models.
Use images.aspect_ratio.* if geometry matters.
Use images.clipped_meanfill.std or images.clipped_meanfill.ar if background artifacts matter.
Use images.edge_fraction to filter bad segmentations.

all_data.json
├── <record_id>
│   ├── images
│   │   ├── main_id
│   │   ├── img_path                      → 512×384 PNG (default)
│   │   ├── mask_path                     → predicted mask (512×384)
│   │   ├── overlay_path                  → QC only
│   │   ├── manual_mask_path              → manual mask PNG (if present)
│   │   ├── manual_mask_path_orginal      → original manual mask TIFF (if present)
│   │   ├── edge_fraction                 → mask touches border fraction
│   │   ├── dimensions_px
│   │   ├── um_per_px
│   │   ├── raw_images                    → original .tif Z-stack filenames
│   │   ├── best_z
│   │   ├── pre_split_days
│   │   ├── aspect_ratio                  → geometry-preserving view (575×575)
│   │   │   ├── ar_image
│   │   │   ├── ar_mask
│   │   │   └── ar_* metadata (scale, μm/px, padding)
│   │   └── clipped_meanfill              → segmentation-aware mean-filled views
│   │       ├── std
│   │       │   ├── cm_image_abs
│   │       │   ├── cm_source_image_abs
│   │       │   ├── cm_source_mask_abs
│   │       │   ├── cm_source_image_field
│   │       │   └── cm_source_mask_field
│   │       └── ar
│   │           ├── cm_image_abs
│   │           ├── cm_source_image_abs
│   │           ├── cm_source_mask_abs
│   │           ├── cm_source_image_field
│   │           └── cm_source_mask_field
│   │
│   ├── metabolite
│   ├── survey
│   ├── label
│   └── metadata / plate / day / cell_line


## Recent Changes (January 2026)

### Data Reorganization
The data pipeline has been reorganized to use a normalized records structure:

- **Normalized Records System**: Introduced `file_utils/merge/normalized_records.py` with `OrganoidRecord`, `OrganoidRecordBuilder`, and `RecordMetrics` to canonicalize organoid data
- **Unified Data Structure**: `all_data.json` now uses a `records` map with standardized organoid IDs plus a `summary.json` file containing metadata (totals, vote counts, metabolite outliers, skipped items)
- **View-Specific Outputs**: The merge process generates a main data file from which specialized views can be created by the image and survey classifiers:
  - `all_data.json` - Complete unified dataset with all organoid records
  - `image_classifier.json` - Day-indexed records optimized for image classifier training
  - `survey_classifier.json` - Day-indexed records optimized for survey classifier training
- **Metadata Tracking**: Each view includes metadata about skipped records, vote counts, and data quality metrics

### Classifier Updates
- **Image Classifier** and **Survey Classifier**: Updated to read from new normalized JSON structure (`records` map instead of flat files)
- **Deterministic Training**: Added support for reproducible training runs with deterministic operations and seed control

### Label Propagation System
- **Label Propagation**: Survey labels from Day 28/30 are now propagated to all earlier time points (Day 3-24) for each organoid
- **Consensus Threshold**: Requires 4+ votes (80% agreement) for label determination
- **Conflict Resolution**: Organoids with inconsistent labels across days or splits are filtered out
- **Results**: 220 unique organoids with propagated labels, 100% label consistency across all time points

---

## Project Structure

```mermaid
flowchart TD
    %% ========= INPUT STAGE ========= %%
    A1([Raw Images])
    A2([Metabolite Excels])
    A3([Survey Excels])

    %% ========= FILE_UTILS PROCESSING ========= %%
    subgraph B[file_utils - Data Mapping &<br/>Integration]
        B1[image_mapper_main.py<br/>Image metadata → JSON]
        B1b[organoid_patterns.py<br/>Pattern normalization helpers]
        B2[metabolite_mapper.py<br/>Metabolite Excel → JSON]
        B3[surveys_mapper.py<br/>Survey Excel → JSON]
        B4[merge_all_data.py<br/>image + metabolite + survey →<br/>all_data.json]
    end

    %% ========= ANALYSIS PIPELINE ========= %%
    subgraph C[analysis - Downstream Analysis & ML]
        subgraph C1[analysis/images]
            C10[classifier<br/>Image classifiers – ViT / CNN]
        end

        subgraph C2[analysis/metabolites]
            C20[classifier<br/>Metabolite-based classifiers]
        end

        subgraph C3[analysis/surveys]
            C30[classifier<br/>Survey-based classifiers]
        end
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
    B4 --> C10
    B4 --> C20
    B4 --> C30
```

---

## Setup Promega Code

For local development with full code access:

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd 2025-promega-mini-test
   ```

2. **Set up Python environment**:
   ```bash
   # Create the conda environment from the repo's spec:
   conda env create -f core_env.yaml
   # Or with micromamba:
   micromamba create -n core_env -f core_env.yaml
   ```

## Run Promega Code

### 1. Activate Environment

```bash
conda activate core_env
```

For segmentation steps (8-9), use the separate `mmcv_env` environment.

### Data Directory

Pipeline data lives at `/net/projects2/promega/2026_04_data/`. Override with:
```bash
make step1 DATA_DIR=/path/to/your/data
```

### 2. Run Data Processing Pipeline *Analysis*

The data processing pipeline consists of several sequential steps to generate the master data files needed for analysis. See the [**Data Processing Pipeline**](#data-processing-pipeline) section below for details of each step.

**Quick Overview**:
1. Retrieve main identifiers from verification CSV
2. Map metabolite data from Excel files
3. Map survey data from Excel files
4. Map image files and metadata
5. Map manual masks
6. Resize and remap images
7. Generate test splits JSON
8. Train segmentation masks
9. Predict segmentation masks
10. Generate image mask overlays
11. Calculate mask edge fractions
12. Filter complete series
13. Preprocess for LSTM
14. Resize with aspect ratio
15. Mean fill clip processing
16. Generate unified `all_data.json` file
17. Run image classifier training
18. Run survey classifier training

This section will cover the analysis which includes the image and survey classifiers.

#### 2a. Prepare to run on Cluster

**Important**: Analysis must be run on computation nodes (not login nodes) using SLURM job submission. A GPU is required when running on the cluster.

**⚠️ IMPORTANT: Update Paths Before Running Analysis**

**Before submitting any jobs**, you must update the hardcoded paths in the SLURM scripts to match your setup:

Replace:
- `/home/YOUR_GITHUB_NAME/YOUR_MINITEST_DIRECTORY` with the path to the GitHub repo directory on your machine
- `/path/to/data`  with the path to the pre-processed images (Image classifier) or survey directory (Survey classifier)
- `/path/to/all_data.json` with the path to the main data JSON file (Image classifier only)

Locations:
1. **`analysis/images/classifier/run_accuracy.s`**
   - Line 13: `PROJ_ROOT=/home/YOUR_GITHUB_NAME/YOUR_MINITEST_DIRECTORY`
   - Line 15: `DATA_DIR=/path/to/data/images`
   - Line 16: `ALL_DATA_JSON=/path/to/all_data.json`

2. **`analysis/surveys/classifier/run_survey_classifier.s`**
   - Line 13: `PROJ_ROOT=/home/YOUR_GITHUB_NAME/YOUR_MINITEST_DIRECTORY`
   - Line 15: `DATA_DIR=/path/to/data/surveys`

Example:
```bash
# If your username is jsmith and you cloned to /home/jsmith/promega-analysis
# Change: PROJ_ROOT=/home/YOUR_GITHUB_NAME/YOUR_MINITEST_DIRECTORY
# To:     PROJ_ROOT=/home/jsmith/promega-analysis
```

#### 2b. Image Classifier
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
Results are saved in `DATA_DIR` which is defined in `run_accuracy.s`

#### 2c. Survey Classifier
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
Results include trained model (`.h5`), training curves, and confusion matrix. Results are saved in `DATA_DIR` which is defined in `run_survey_classifier.s`

---

## Data Processing Pipeline

This section provides detailed step-by-step instructions for processing raw data into the unified dataset used for analysis.

### Overview

The complete pipeline flow processes raw microscopy images, metabolite measurements, and survey evaluations through multiple stages to create a unified dataset for machine learning analysis.

**Pipeline Stages**:
1. **Identifier Mapping** - Extract and normalize organoid identifiers
2. **Data Source Mapping** - Map metabolite and survey data to identifiers
3. **Image Processing** - Resize, segment, and prepare images
4. **Series Filtering** - Filter complete time series data
5. **Data Merging** - Combine all sources into unified JSON
6. **Model Training** - Train image and survey classifiers

### Prerequisites

Before starting, ensure you have:
- Python environment set up (see [Setup Promega Code](#setup-promega-code))
- Required input data files:
  - Image verification CSV file
  - Metabolite Excel files
  - Survey Excel files
  - Raw image files (TIFF format)
  - Sample tracing Excel file (metadata)
  - Manual segmentation masks

### Raw Input Data Structure

Starting data directory structure:

```
.
├── images
│   ├── image_verification.csv
│   ├── raw_images
│   │   ├── Ba1 96_1 Dy03 A10 Z0.tif
│   │   ├── Ba1 96_1 Dy03 A10 Z1.tif
│   │   ├── Ba1 96_1 Dy03 A10 Z2.tif
│   │   └── ... (30,000+ TIFF files)
│   └── Sample-Tracing.xlsx
├── masks
│   └── manual
│       ├── Manuais
│       │   ├── Mask_M Ba1 96_1 Dy03 A10 Z2.tif
│       │   └── ...
│       ├── masks-batch-1
│       │   ├── manual
│       │   │   ├── Mask_M Ba1 96_1 Dy03 A10 Z2.tif
│       │   │   └── ...
│       │   └── threshold
│       │       ├── Mask_T Ba1 96_1 Dy08 A10 Z2.tif
│       │       └── ...
│       ├── masks-batch-2_1
│       │   └── manual
│       │       ├── Mask_M Ba2 96_1 Dy03 A10 Z2.tif
│       │       └── ...
│       ├── masks-batch-2_2
│       │   └── manual
│       │       ├── Mask_M Ba2 96_2 Dy03 A10 Z2.tif
│       │       └── ...
│       └── Treshold
│           ├── Mask_T Ba1 96_1 Dy08 A10 Z2.tif
│           └── ...
├── metabolite
│   └── metabolite_data_07_23_25.xlsx
└── survey
    ├── Image Classification Form - Excel Report(2025-06-13).xlsx
    ├── Organoid Classification (Form A) - Part 1 of 3 - Excel Report(2025-06-13).xlsx
    ├── Organoid Classification (Form A) - Part 2 of 3 - Excel Report(2025-06-13).xlsx
    ├── Organoid Classification (Form A) - Part 3 of 3 - Excel Report(2025-06-13).xlsx
    ├── Organoid Classification (Form B) - Part 1 of 3 - Excel Report(2025-06-13).xlsx
    ├── Organoid Classification (Form B) - Part 2 of 3 - Excel Report(2025-06-13).xlsx
    ├── Organoid Classification (Form B) - Part 3 of 3 - Excel Report(2025-06-13).xlsx
    ├── Organoid Classification (Form C) - Part 1 of 3 - Excel Report(2025-06-13).xlsx
    ├── Organoid Classification (Form C) - Part 2 of 3 - Excel Report(2025-06-13).xlsx
    └── Organoid Classification (Form C) - Part 3 of 3 - Excel Report(2025-06-13).xlsx

15 directories, 30027 files
```

### STEP 1: Retrieve Main Identifiers

Extract and normalize main identifiers from the image verification CSV file.

**Inputs/Outputs**:
- **In**: `image_verification.csv`
- **Out**: `record_identifiers.json` (5,168 identifiers)

**Command**:
```bash
python3 -m file_utils.identifiers.retrieve_main_identifiers \
    --csv-file <path/to/image_verification.csv> \
    --out-file <path/to/record_identifiers.json>
```

**What it does**:
- Extracts and normalizes main identifiers from image filename bases in a CSV file
- Normalizes split markers: `(1)%` → `split_1`, `(2)%` → `split_2`
- Removes stitched markers: `(stitched)` → removed
- Normalizes case: `Ba` → `BA`

**Required Arguments**:
- `--csv-file`: Path to CSV file containing a `filename base` column
- `--out-file`: Path to output JSON file where normalized identifiers will be saved

**Output**: `main_identifiers.json` - Normalized identifier list

**Important Assumptions**:
- Batch 1 Day 20 and Day 21 from other batches are normalized to Day 20.5

### STEP 2: Map Metabolite Data

Process metabolite Excel files and map them to main identifiers.

**Inputs/Outputs**:
- **In**: `record_identifiers.json`, metabolite spreadsheets (`.xlsx`)
- **Out**: `metabolite_map.json` (4,154 identifiers matched)

**Command**:
```bash
python -m file_utils.metabolites.metabolite_mapper \
    --in-file <path/to/metabolite_data.xlsx> \
    --identifiers <path/to/record_identifiers.json> \
    --out-file <path/to/metabolite_map.json>
```

**What it does**:
- Reads metabolite concentration data from Excel file
- Maps metabolite data to normalized identifiers
- Handles day normalization (Day 20/21 → Day 20.5)
- Duplicates metabolite data across splits with the same main identifier

**Required Arguments**:
- `--in-file`: Path to Excel file containing metabolite data
- `--identifiers`: Path to main identifiers JSON file
- `--out-file`: Path to output JSON file where metabolite map will be saved

**Output**: `metabolite_map.json` - Metabolite data mapped to identifiers

**Important Assumptions**:
- Batch 1 Day 20 and Day 21 from other batches are treated as Day 20.5
- Metabolite data can be duplicated across splits

### STEP 3: Map Survey Data

Process survey Excel files and map evaluations to identifiers.

**Inputs/Outputs**:
- **In**: `record_identifiers.json`, survey spreadsheets (`.xlsx`)
- **Out**: `survey_map.json` (393 records with 2,105 total votes)

**Command**:
```bash
python -m file_utils.surveys.surveys_mapper \
    --in-dir <path/to/survey/excel/files> \
    --out-file <path/to/survey_map.json> \
    --identifiers <path/to/record_identifiers.json>
```

**What it does**:
- Processes survey Excel files from input directory
- Extracts evaluation data (employee, evaluation, quality scores)
- Maps survey data to normalized identifiers
- Computes labels using majority voting (4+ votes required for consensus)

**Required Arguments**:
- `--in-dir`: Path to directory containing survey Excel files
- `--identifiers`: Path to main identifiers JSON file
- `--out-file`: Path to output JSON file where survey map will be saved

**Optional Arguments**:
- `--min-survey-votes`: Minimum number of votes required to determine a majority (default: 4)

**Output**: `survey_map.json` - Survey evaluation data mapped to identifiers

**Note**: The survey map includes an `organoid_id` key layer to handle cases with multiple organoids per identifier.

### STEP 4: Map Image Data

Map raw image files to metadata and create image mapping JSON.

**Inputs/Outputs**:
- **In**: `record_identifiers.json`, `Sample-Tracing.xlsx`, `image_verification.csv`, raw images (`.tif`)
- **Out**: `image_map.json` (5,168 records)

**Command**:
```bash
python3 -m file_utils.images.image_mapper \
    --base-dir <path/to/raw_images> \
    --verify-csv <path/to/image_verification.csv> \
    --meta-xlsx <path/to/Sample-Tracing.xlsx> \
    --identifiers <path/to/record_identifiers.json> \
    --out-file <path/to/image_map.json>
```

**What it does**:
1. Loads metadata from Sample-Tracing Excel, adds pixel scale (`um_per_px`)
2. Groups metadata by day, batch plate, and well
3. Computes pre-split wells to track split organoids
4. Resolves images:
   - Finds candidate raw image files matching identifier
   - Sorts by Z-level
   - Handles split indexes and stitched images
   - Selects best focus image
5. Creates entries with classification (SplitStitched, Split, Regular, etc.)
6. Adds verification metadata (split, stitched, blank flags)

**Required Arguments**:
- `--base-dir`: Path to base directory containing raw images
- `--meta-xlsx`: Path to Sample-Tracing Excel file containing metadata
- `--out-file`: Path to output JSON file where image mapping will be saved

**Optional Arguments**:
- `--verify-csv`: Path to CSV file containing image verification data
- `--identifiers`: Path to main identifiers JSON file

**Output**: `image_map.json` - Complete image file mapping with metadata (5,168 entries)

### STEP 5: Map Manual Masks

Map manual segmentation masks to image entries.

**Inputs/Outputs**:
- **In**: `image_map.json`, manual masks (`.tif`)
- **Out**: `image_mapping_thresholded_and_manual.json` (2,091 entries with masks from 2,148 total masks)

**Command**:
```bash
python3 -m analysis.images.segmentation_mmseg.preprocessing.manual_masks_mapping \
    --image-json <path/to/image_map.json> \
    --masks-dir <path/to/masks/manual> \
    --output-file <path/to/image_mapping_thresholded_and_manual.json>
```

**What it does**:
- Discovers manual mask files in batch subdirectories
- Matches mask files to image entries
- Creates mapping JSON with manual mask paths
- Excludes split entries due to naming inconsistencies

**Required Arguments**:
- `--image-json`: Path to image mapping JSON
- `--masks-dir`: Path to directory containing manual masks
- `--output-file`: Path to output JSON file

**Output**: `image_mapping_thresholded_and_manual.json` - Mapping with manual mask paths (2,091 entries with masks)

**Statistics**:
- Total masks: 2,148
- Mapped entries: 2,091
- Skipped (no matching masks): 2,987
- Excluded split entries: 90

### STEP 6: Resize and Remap Images

Resize images and masks to target resolution and remap paths.

**Inputs/Outputs**:
- **In**: `image_map.json`, target width (512), target height (384)
- **Out**: `image_map_resized_512x384.json` (5,168 entries), resized images (`.png`)

**Command**:
```bash
python3 -m analysis.images.resize.resize_remap_images \
    --image-mapping-json <path/to/image_map.json> \
    --mask-mapping-json <path/to/image_mapping_thresholded_and_manual.json> \
    --out-dir <path/to/output/resized_512x384> \
    --out-mapping-json <path/to/image_map_resized_512x384.json>
```

**What it does**:
- Resizes raw images to target dimensions (default: 512x384)
- Calculates `um_per_px` values for resized images
- Adds manual mask paths to support mask training
- Creates updated mapping JSON with new image paths

**Required Arguments**:
- `--image-mapping-json`: Path to image mapping JSON
- `--mask-mapping-json`: Path to mask mapping JSON
- `--out-dir`: Output directory for resized images
- `--out-mapping-json`: Output JSON file path

**Optional Arguments**:
- `--target-width`: Target width in pixels (default: 512)
- `--target-height`: Target height in pixels (default: 384)
- `--overwrite`: Overwrite existing files (default: False)

**Output**: `image_map_resized_512x384.json` - Mapping with resized image paths (5,168 entries)

**Note**: The `um_per_px` calculations differ from previous versions due to possible Sample-Tracing data changes.

### STEP 7: Test Splits JSON

Create train/validation/test splits for model training.

**Inputs/Outputs**:
- **In**: `image_map_resized_512x384.json`
- **Out**: Train/val/test split JSON files (removes 36 bad entries, 5,132 remain)

**Command**:
```bash
python3 -m analysis.images.segmentation_mmseg.preprocessing.test_split \
    --resized-json <path/to/image_map_resized_512x384.json> \
    --splits-dir <path/to/output/splits> \
    --split-days
```

**What it does**:
- Removes 36 bad entries including duplicates for split organoids
- Creates train/val/test splits (80%/10%/10%)
- Generates day-specific splits:
  - Full mapping splits
  - Early days (Dy3-10) splits
  - Late days (Dy13-30) splits

**Required Arguments**:
- `--resized-json`: Path to resized image mapping JSON
- `--splits-dir`: Output directory for split JSON files

**Optional Arguments**:
- `--train-frac`: Training set fraction (default: 0.8)
- `--val-frac`: Validation set fraction (default: 0.1)
- `--split-days`: Split by day groups (default: False)

**Output**: Multiple split JSON files
- `image_map_resized_512x384_train.json` (4,105 entries)
- `image_map_resized_512x384_val.json` (513 entries)
- `image_map_resized_512x384_test.json` (514 entries)
- Day-specific splits for early (Dy3-10) and late (Dy13-30) days

**Note**: Script now evaluates 5,168 records (all resized images) compared to previous 2,153.

### STEP 8: Train Segmentation Masks

**Early vs Late models (current behavior)**  
Segmentation is trained as two separate models to better capture morphology differences across development:

- **early**: Dy03–Dy10  
- **late**: Dy13–Dy30  

Each model trains from the corresponding split JSONs and writes its own checkpoint/config under its own work directory.

**Inputs/Outputs**:
- **In**: Splits directory with split JSON files
- **Out**: Trained model checkpoints (`.pth`) and training logs

**Command**:
```bash
python -m analysis.images.segmentation_mmseg.train \
    --splits-dir <path/to/splits> \
    --work-dir <path/to/trained_models>
```

**What it does**:
- Loads train/val/test splits
- Trains segmentation model on organoid masks
- Skips records without manual masks
- Saves model checkpoints and training logs

**Required Arguments**:
- `--splits-dir`: Directory containing split JSON files
- `--work-dir`: Directory for model checkpoints and outputs

**Output**: Trained segmentation model checkpoints

**Statistics**:
- Training set: 2,592 entries → 1,094 with masks (42.2%)
- Validation set: 324 entries → 140 with masks (43.2%)
- Test set: 324 entries → 136 with masks (42.0%)

**Note**: Modified `day_datasets.py` to reference correct keys in resized JSON format.

### STEP 9: Predict Segmentation Masks

**Early vs Late inference (current behavior)**  
Mask prediction is run twice (once per model) and outputs are written to `masks/predicted/` (or the configured output dir). The image mapping JSON is updated to point at the predicted mask paths.

**Inputs/Outputs**:
- **In**: `image_map_resized_512x384.json`, trained models (config.py and checkpoint.pth)
- **Out**: Predicted masks (`.png`), updated `image_map_resized_512x384.json` with `predicted_mask_path` keys

**Command**:
```bash
# Early model
python -m analysis.images.segmentation_mmseg.predict_masks \
  --image-mapping-json <...> \
  --model-type early \
  --config <early_config.py> \
  --checkpoint <early_checkpoint.pth> \
  --out-dir <predicted_masks_dir>

# Late model
python -m analysis.images.segmentation_mmseg.predict_masks \
  --image-mapping-json <...> \
  --model-type late \
  --config <late_config.py> \
  --checkpoint <late_checkpoint.pth> \
  --out-dir <predicted_masks_dir>

**What it does**:
- Loads trained segmentation model
- Generates predicted masks for all 5,168 images
- Updates mapping JSON with predicted mask paths
- Creates visualization collage

**Required Arguments**:
- `--image-mapping-json`: Path to resized image mapping JSON
- `--out-dir`: Output directory for predicted masks
- `--config`: Path to model config file
- `--model-type (early or late)
- `--checkpoint`: Path to model checkpoint

**Optional Arguments**:
- `--model-type`: Model type (early/late) (default: late)
- `--write-collage`: Generate visualization collage (default: False)
- `--overwrite`: Overwrite existing predictions (default: False)

**Output**:
- Predicted mask PNG files (5,168 masks)
- Updated mapping JSON with mask paths
- Visualization collage

### STEP 10: Image Mask Overlay

Create overlay visualizations of images with masks.

**Inputs/Outputs**:
- **In**: `image_map_resized_512x384.json` (`processed_image` and `predicted_mask_path`)
- **Out**: Overlay images (`.png`), `summary.json`, updated JSON with `overlay_path` keys

**Command**:
```bash
python3 -m analysis.images.quality.image_mask_overlay \
    --image-mapping-json <path/to/image_map_resized_512x384.json> \
    --overlay-dir <path/to/overlays> \
    --overwrite
```

**What it does**:
- Creates overlay images showing organoid segmentation
- Combines original image with predicted mask
- Saves overlays as PNG files
- Updates mapping JSON with overlay paths

**Required Arguments**:
- `--image-mapping-json`: Path to image mapping JSON with mask paths
- `--overlay-dir`: Output directory for overlay images

**Optional Arguments**:
- `--overwrite`: Overwrite existing overlays (default: False)

**Output**:
- Overlay PNG files (5,168 overlays)
- Updated mapping JSON with overlay paths
- Summary JSON with processing statistics

### STEP 11: Mask Edge Fraction

Calculate fraction of mask touching image edges.

**Inputs/Outputs**:
- **In**: `image_map_resized_512x384.json` (`predicted_mask_path`)
- **Out**: Updated JSON with `edge_fraction` field (5,168 processed)

**Command**:
```bash
python3 -m analysis.images.quality.mask_edge_fraction \
    --image-mapping-json <path/to/image_map_resized_512x384.json>
```

**What it does**:
- Analyzes predicted masks to detect edge touching
- Calculates percentage of mask at image borders
- Updates mapping JSON with `edge_fraction` field

**Required Arguments**:
- `--image-mapping-json`: Path to image mapping JSON with mask paths

**Output**: Updated mapping JSON with `edge_fraction` values

**Important Note**:
- Edge fraction represents the percent of the cell that is off camera
- Values over 5% (0.05) are considered organoids that should be ignored for some analyses
- Currently produces 0.0 for most masks - **requires review**

**Open Questions**:
- What was `split_children` doing in original implementation?
- Which masks should this run on (predicted vs manual)?

### STEP 12: Filter Complete Series

Filter organoids with complete time series data across all 11 days.

**Inputs/Outputs**:
- **In**: `image_map_resized_512x384.json`
- **Out**: `complete_series_data_no_blanks.json` (4,993 entries), `complete_series_metadata_no_blanks.json`, `series_completeness_summary.json`

**Command**:
```bash
python3 -m analysis.images.series.filter_complete_series \
    --image-mapping-json <path/to/image_map_resized_512x384.json> \
    --out-dir <path/to/filter_complete_series> \
    --show-examples
```

**What it does**:
- Organizes data by genealogy (tracking splits and pre-split days)
- Identifies complete series (all 11 days present)
- Filters out incomplete series and blanks
- Validates split genealogy correctness

**Required Arguments**:
- `--image-mapping-json`: Path to image mapping JSON
- `--out-dir`: Output directory for filtered data

**Optional Arguments**:
- `--show-examples`: Show example organoids (default: False)

**Output**:
- `complete_series_data_no_blanks.json` - Filtered data (4,993 entries from 5,168)
- `complete_series_metadata_no_blanks.json` - Series metadata by organoid
- `series_completeness_summary.json` - Analysis summary

**Statistics**:
- Total unique wells: 475
- Complete series (no blanks): 461 (96.6% retention)
  - nosplit: 437
  - presplit+split1: 11
  - presplit+split2: 13
- Complete series with blanks: 3
- Incomplete series: 25

### STEP 13: Preprocess for LSTM

Prepare images for LSTM time series analysis with consistent physical scale.

**Inputs/Outputs**:
- **In**: Raw image data, `complete_series_data_no_blanks.json`
- **Out**: LSTM-ready images and masks (768×768 px), updated `complete_series_data_no_blanks.json` with `lstm_processed` field

**Command**:
```bash
python3 -m analysis.images.series.preprocess_for_lstm \
    --complete-series <path/to/complete_series_data_no_blanks.json> \
    --raw-image-dir <path/to/raw_images> \
    --out-dir <path/to/lstm>
```

**What it does**:
- Rescales images to uniform physical scale (6.0 um/px)
- Resizes to uniform dimensions (768×768 px)
- Applies white padding for images, black padding for masks
- Analyzes target dimensions and validates size sufficiency

**Required Arguments**:
- `--complete-series`: Path to complete series JSON
- `--raw-image-dir`: Path to raw images directory
- `--out-dir`: Output directory for LSTM-ready data

**Optional Arguments**:
- `--target-um-per-px`: Target physical scale (default: 6.0)
- `--target-size`: Target dimensions in pixels (default: 768)
- `--skip-analysis`: Skip dimension analysis (default: False)

**Output**:
- LSTM-ready images in `lstm_ready/images/` (4,993 images)
- LSTM-ready masks in `lstm_ready/masks/` (4,993 masks)
- Updated complete_series JSON with `lstm_processed` field

**Statistics**:
- Original scales: 1.69-2.24 um/px
- Target scale: 6.0 um/px
- Target dimensions: 768×768 px
- 95th percentile: width 604px, height 453px
- Images exceeding 512×512: 1,090 (21.8%)

**Note**: Modified to include `um_per_px` data values from `resize_remap_images.py`.

### STEP 14: Resize Aspect Ratio

Resize images maintaining aspect ratio for additional analysis.

**Inputs/Outputs**:
- **In**: `image_map_resized_512x384.json`, raw images
- **Out**: Resized square images and masks (575×575 px), updated JSON with `_ar` suffix

**Command**:
```bash
python3 -m analysis.images.resize.resize_aspect_ratio \
    --image-mapping-json <path/to/image_map_resized_512x384.json> \
    --raw-images-dir <path/to/raw_images> \
    --out-images-dir <path/to/resized_575_square> \
    --out-masks-dir <path/to/resized_575_square/masks> \
    --require-mask
```

**What it does**:
- Resizes images to specified physical scale and dimensions
- Maintains aspect ratio with appropriate padding
- Processes both images and masks

**Required Arguments**:
- `--image-mapping-json`: Path to image mapping JSON
- `--raw-images-dir`: Path to raw images directory
- `--out-images-dir`: Output directory for resized images
- `--out-masks-dir`: Output directory for resized masks

**Optional Arguments**:
- `--target-um-per-px`: Target physical scale (default: 9.0)
- `--target-size`: Target dimensions in pixels (default: 575)
- `--require-mask`: Only process entries with masks (default: False)
- `--overwrite`: Overwrite existing files (default: False)

**Output**:
- Resized images and masks
- Updated mapping JSON with `_ar` suffix

### STEP 15: Mean Fill Clip

Apply mean-fill background replacement to standardize image backgrounds.

**Inputs/Outputs**:
- **In**: `image_map_..._ar.json`, aspect ratio-sized images and masks
- **Out**: Mean-filled clipped images, `global_mean.npy`, processing stats JSON, updated JSON with `_meanfill` suffix

**Command**:
```bash
python3 -m analysis.images.postprocess.meanfill_clip \
    --image-mapping-json <path/to/image_map_..._ar.json> \
    --compute-mean \
    --save-computed-mean \
    --out-images-dir <path/to/mean_fill_clip> \
    --images-base <path/to/resized_575_square> \
    --masks-base <path/to/resized_575_square/masks> \
    --require-mask
```

**What it does**:
- Computes global mean RGB background value
- Replaces image background outside mask with mean value
- Creates standardized background across all images

**Required Arguments**:
- `--image-mapping-json`: Path to image mapping JSON
- `--out-images-dir`: Output directory for processed images

**Optional Arguments**:
- `--compute-mean`: Compute global mean from images (default: False)
- `--save-computed-mean`: Save computed mean to file (default: False)
- `--images-base`: Base directory for input images
- `--masks-base`: Base directory for input masks
- `--require-mask`: Only process entries with masks (default: False)
- `--mean-region`: Region for mean calculation (background/foreground) (default: background)

**Output**:
- Mean-filled images (5,168 processed)
- `global_mean.npy` - Computed global mean RGB values
- Updated mapping JSON with `_meanfill` suffix

**Statistics**:
- Computed global mean RGB: [0.695, 0.695, 0.695] (in 0-1 range)

**Note**: Prioritizes image and mask paths from aspect ratio processing step.

### STEP 16: Generate All Data JSON File

Merge all mapped data sources into unified `all_data.json` file.

**Inputs/Outputs**:
- **In**: `record_identifiers.json`, `metabolite_map.json`, `survey_map.json`, final `image_map_*.json` (typically `image_map_resized_512x384.json`)
- **Out**:
  - `all_data.json` (5,168 total records, 220 unique organoids with labels after filtering)
  - `image_classifier.json` (2,931 labeled records across 11 days)
  - `survey_classifier.json` (269 Day 30 records with survey labels)
  - `summary.json` (statistics and metadata)

**Command**:
```bash
python3 -m file_utils.merge.merge_all_data \
    --data-dir <path/to/data/directory> \
    --image-mapping-json <path/to/final/image_mapping.json>
```

**What it does**:
1. Builds survey map and normalizes keys
2. Loads metabolite map
3. Loads final processed image mapping
4. Merges all data sources for each identifier:
   - Combines image info, metabolites, survey data, manual masks
   - Propagates labels from Day 28/30 to all earlier days
   - Filters out organoids with conflicting labels
   - Extracts numerical day values
5. Builds normalized records with standardized structure
6. Validates schema (if enabled)
7. Generates view-specific outputs:
   - `all_data.json` - Complete unified dataset
   - `image_classifier.json` - Day-indexed view for image training
   - `survey_classifier.json` - Day-indexed view for survey training

**Required Arguments**:
- `--data-dir`: Path to data directory (must contain: `identifiers/`, `images/`, `metabolite/`, `survey/`)

**Optional Arguments**:
- `--image-mapping-json`: Path to final image mapping JSON (overrides auto-discovery)
- `--min-survey-votes`: Minimum votes for survey label (default: 4)
- `--survey-day`: Day that survey was conducted (default: 30)
- `--target-width`: Target image width in pixels (default: 512)
- `--target-height`: Target image height in pixels (default: 384)
- `--no-validate`: Skip schema validation (default: False, validation runs by default)

**Output Files**:
- `all_data.json` - Complete dataset with all organoid records (5,168 records)
- `summary.json` - Statistics and metadata
- `image_classifier.json` - 2,931 labeled records across 11 days
- `survey_classifier.json` - 269 Day 30 records with survey labels

**Label Statistics**:
- 220 unique organoids with propagated labels (BA1+BA2, ≥4/5 vote consensus at Day 30)
- 2,931 total labeled records (11 days × ~220 organoids)
- 100% label consistency across time points
- Distribution: 72.5% Acceptable, 27.5% Not Acceptable
- 13 organoids filtered due to conflicting labels

**Important Assumptions**:
- Survey label takes priority when populating the `label` field
- Day 20 and Day 21 are normalized to Day 20.5
- Organoids with vote conflicts between splits/days are excluded

### STEP 17: Image Classifier Training

Train image classification models for each day.

**Inputs/Outputs**:
- **In**: `image_classifier.json` (2,931 labeled records)
- **Out**: Trained models (`.h5`), training curves, confusion matrices, metrics

**On Cluster (SLURM)**:
```bash
cd analysis/images/classifier
sbatch run_accuracy.s
```

**Local Development**:
```bash
python3 -m analysis.images.classifier.train_model_accuracy \
    --epoch1 <num_epochs_phase1> \
    --epoch2 <num_epochs_phase2> \
    --val-frac <validation_fraction> \
    --test-frac <test_fraction> \
    --deterministic \
    --data-dir <path/to/data>
```

**What it does**:
1. Loads `image_classifier.json` (2,931 labeled records)
2. Splits data into train/validation/test sets
3. Trains three model architectures (ViT, ResNet, CNN) for each day:
   - **Phase 1**: Frozen backbone, trains classifier head (default: 100 epochs)
   - **Phase 2**: Unfrozen backbone, fine-tunes entire model (default: 300 epochs)
   - Uses early stopping and class weights for imbalanced data
4. Evaluates on validation and test sets
5. Saves model checkpoints, training curves, metrics

**Required Arguments**:
- `--data-dir`: Path to data directory containing organoid data

**Optional Arguments**:
- `--all-data-json`: Path to `all_data.json` file
- `--image-classifier-json`: Path to `image_classifier.json` file
- `--epoch1`: Number of epochs for phase 1 (default: 100)
- `--epoch2`: Number of epochs for phase 2 (default: 300)
- `--batch-size`: Training batch size (default: 16)
- `--test-frac`: Test set fraction (default: 0.1)
- `--val-frac`: Validation set fraction (default: 0.1)
- `--use-mask`: Include mask branch in classifier (default: False)
- `--deterministic`: Use deterministic operations (default: False)
- `--seed`: Random seed (default: 1)

**Output**: Trained models, metrics, plots, confusion matrices

### STEP 18: Survey Classifier Training

Train survey-based classification model on Day 30 organoids.

**Inputs/Outputs**:
- **In**: `survey_classifier.json` (269 Day 30 records with survey labels)
- **Out**: Trained model (`.h5`), training curves, confusion matrix, metrics

**On Cluster (SLURM)**:
```bash
cd analysis/surveys/classifier
sbatch run_survey_classifier.s
```

**Local Development**:
```bash
python3 -m analysis.surveys.classifier.simple_classifier \
    --epoch1 <num_epochs_phase1> \
    --epoch2 <num_epochs_phase2> \
    --deterministic \
    --data-dir <path/to/data>
```

**What it does**:
1. Loads `survey_classifier.json` (269 Day 30 records)
2. Preprocesses: extracts image paths, mask paths, and labels
3. Creates TensorFlow datasets with augmentation
4. Builds dual-input ResNet50V2 + CNN model:
   - Image input: Pretrained ResNet50V2 backbone
   - Mask input: Custom CNN for mask features
   - Combined classification head
5. Trains model:
   - **Phase 1**: Frozen ResNet50V2 (default: 50 epochs)
   - **Phase 2**: Unfreezes last 10 layers (default: 150 epochs)
6. Evaluates: accuracy, confusion matrix, training curves
7. Saves model weights, history, visualizations

**Required Arguments**:
- `--data-dir`: Path to data directory

**Optional Arguments**:
- `--all-data-json`: Path to `all_data.json` file
- `--survey-classifier-json`: Path to `survey_classifier.json` file
- `--batch-size`: Training batch size (default: 8)
- `--epoch1`: Epochs for phase 1 (default: 50)
- `--epoch2`: Epochs for phase 2 (default: 150)
- `--target-day`: Target day for training (default: "Dy30")
- `--deterministic`: Use deterministic operations (default: False)
- `--seed`: Random seed (default: 1)

**Output**: Trained model (`.h5`), training curves, confusion matrix, metrics

---

## Data

### Main data file structure

The `all_data.json` file contains unified organoid data with structure:
```json
{
  "schema_version": 1,
  "generated_at": "2026-01-21T16:34:36.725704+00:00",
  "stats": {
    "total_records": 5168,
    "num_organoids": 475,
    "num_labels": 2931,
    "num_acceptable_votes": 1356,
    "num_not_acceptable_votes": 749,
    ...
  },
  "records": {
    "BA1 96_1 Dy03 A1": {
      "id": "BA1 96_1 Dy03 A1",
      "organoid_id": "BA1_96_1_A1",
      "day": {
        "id": "Dy3",
        "number": 3.0,
        "original": 3
      },
      "cell_line": "GM23279A",
      "plate": {
        "batch": "BA1 96_1",
        "well": "A1"
      },
      "images": {
        "main_id": "BA1_96_1_Dy03_A1_nosplit_nostitch",
        "img_path": "/path/to/image.png",
        "mask_path": "/path/to/mask.png",
        "overlay_path": "/path/to/overlay.png",
        "manual_mask_path": "/path/to/manual_mask.tif"
      },
      "metabolite": {
        "GlucoseGlo": {
          "concentration_uM": 9.827,
          "is_outlier": false
        },
        ...
      },
      "survey": {
        "evaluations": [...],
        "quality_scores": [...]
      },
      "label": {
        "value": "Acceptable",
        "acceptance_flag": 1,
        "votes": {"Acceptable": 4, "Not Acceptable": 1},
        "total_evaluations": 5,
        "source": "survey.evaluations"
      }
    }
  }
}
```

The view-specific files (`image_classifier.json`, `survey_classifier.json`) use a day-indexed structure:
```json
{
  "metadata": {
    "total_records": 2931,
    "total_skipped": 2237,
    ...
  },
  "records": {
    "Dy3": {
      "id": ["BA1 96_1 Dy03 A1", ...],
      "img_path": ["/path/to/img1.png", ...],
      "mask_path": ["/path/to/mask1.png", ...],
      "label": [1, 0, 1, ...]
    },
    "Dy6": {...},
    ...
  }
}
```

### Input Data Types

The system processes three main types of input data:

**1. Image Data**
- **Raw Images**: Multi-Z-stack TIFF files (`.tif`) from microscopy
- **Processed Images**: Resized PNG files at multiple resolutions (512×384, 768×768)
- **Masks**: Segmentation masks (predicted or manual) as PNG/TIFF files
- **Overlays**: Image-mask overlay visualizations

**Location**: Raw images in `images/raw_images/`, processed in various output directories

**2. Metabolite Data**
- **Format**: Excel spreadsheets (`.xlsx`)
- **Content**: Chemical assay measurements for 6 metabolites:
  - BCAAGlo, GlucoseGlo, GlutamateGlo, LactateGlo, MalateGlo, PyruvateGlo
- **Fields**: Concentration values, initial concentrations, outlier flags

**Location**: `metabolite/metabolite_data_07_23_25.xlsx`

**3. Survey Data**
- **Format**: Excel spreadsheets (`.xlsx`)
- **Content**: Quality assessment evaluations from human raters
- **Structure**: 5 evaluations per organoid with "Acceptable" or "Not Acceptable" labels
- **Processing**: Requires 4+ votes (80% consensus) for label determination

**Location**: `survey/` directory with multiple Excel files (Forms A, B, C)

---

## Resource Requirements

### Cluster Resources (SLURM)

**Image Classifier**:
- **GPU**: 1x A100 (required)
- **Memory**: 32GB RAM
- **Time**: ~2 hours per job
- **Storage**: ~10GB for model checkpoints per run

**Survey Classifier**:
- **GPU**: 1x A100 (required)
- **Memory**: 32GB RAM
- **Time**: ~2 hours per job
- **Storage**: ~5GB for model checkpoints

**Data Pipeline**:
- **CPU**: Standard compute node (no GPU needed for most steps)
- **GPU**: Required for mask prediction (Step 9)
- **Memory**: 8-16GB RAM
- **Time**: ~2-4 hours for complete pipeline
- **Storage**:
  - Input: ~50GB (raw images, masks)
  - Output: ~30GB (processed images, JSON files)

### Local Development

**Minimum Requirements**:
- **GPU**: NVIDIA GPU with CUDA support (recommended) or CPU for testing
- **Memory**: 16GB RAM minimum, 32GB recommended
- **Storage**: 100GB+ free space

**Recommended for Full Training**:
- **GPU**: NVIDIA GPU with 8GB+ VRAM (RTX 3070/3080, A100)
- **Memory**: 32GB+ RAM
- **Storage**: 150GB+ free space

---

## Local development and testing

### Test dataset

Currently, there is **no dedicated test dataset** for quick local development. For testing:

1. **Use a subset of full dataset**: Filter to single day with fewer samples
2. **Reduce data size**: Smaller batch sizes (4-8), fewer epochs (5-10)
3. **Create minimal test set**: Copy 10-20 images/masks to test directory

### Example execution

```bash
# Image Classifier (reduced for testing)
cd analysis/images/classifier
python train_model_accuracy.py \
    --out-dir ./outputs \
    --batch-size 8 \
    --epoch1 10 \
    --epoch2 20 \
    --deterministic \
    --seed 1

# Survey Classifier (reduced for testing)
cd analysis/surveys/classifier
python simple_classifier.py \
    --out-dir ./outputs \
    --batch-size 8 \
    --epoch1 10 \
    --epoch2 20 \
    --deterministic \
    --seed 1
```

### Development Guidelines

- **Environment**: Always activate conda environment first
- **Data Access**: Use normalized JSON files as single source of truth
- **Execution**: Run from project root directory
- **Reproducibility**: Use `--deterministic` and `--seed` flags

---

## Current Status

✅ **Fully Functional System** (Updated January 2026)
- Complete preprocessing pipeline with 16 documented steps
- Data reorganization with normalized records structure
- Label propagation system from Day 28/30 to all earlier days
- Working data generation producing 5,168 total records
  - 2,931 labeled records (220 unique organoids across 11 days)
  - 269 Day 30 records for survey classifier
- Multimodal data integration (images, metabolites, surveys) operational
- View-specific JSON outputs for optimized classifier training
- Deterministic training support for reproducible experiments
- Comprehensive error handling and validation

## Known Issues & Future Improvements

**Known Issues**:
1. **Mask Edge Fraction**: Currently produces 0.0 for all masks - requires investigation
2. **Label Threshold**: Strict 4+ vote requirement reduces dataset size by 26.6%
3. **um_per_px values**: Differ from previous version, possibly due to Sample-Tracing data changes

**Future Improvements**:
1. **Label threshold adjustment**: Consider relaxing to 3-vote majority (60%) to recover ~80 samples
2. **Class balancing**: Implement weighted sampling for minority class
3. **Pipeline optimization**: Add caching and parallel processing where possible
4. **Test dataset**: Create minimal test dataset for rapid local development
5. **Documentation**: Add troubleshooting guide and FAQ section

See `CLAUDE.md` and analysis documents in `data/results analysis/` for detailed code analysis and recommendations.

---

**Document Version**: 2.0
**Last Updated**: 2026-01-21
**Contributors**: See git history for full list of contributors
