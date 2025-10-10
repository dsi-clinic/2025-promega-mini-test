# Promega Organoid Analysis System

This repository contains a comprehensive system for analyzing organoid quality using multimodal data including images, metabolites, and survey assessments for time series prediction.

## Project Structure

```
analysis/                               # All analysis code and experiments
├── images/
│   ├── resize/                        # Image resizing + updated pixel-scale metadata 
│   ├── classifier/                    # Image classification models (e.g., ViT, CNNs)
│   ├── metrics/                       # Image analysis tools
│   │   ├── shape_metrics/             # Organoid shape features
│   │   └── stitching/                 # Image stitching scripts
│   └── segmentation_mmseg/            # MMSegmentation training and inference
│       ├── datasets/                  # Dataset definitions for mmseg
│       ├── preprocessing/             # Mask/image preprocessing tools
│       └── utils/                     # Custom transforms and helpers
├── metabolites/
│   └── classifier/                    # Classifier using metabolite data
├── multimodal/                        # CNN classifier using merged modalities
└── surveys/
    ├── agreement_aggregations/        # Processed survey agreement data
    ├── classifier/                    # Survey-based classifiers
    ├── notebooks/                     # Statistical exploration
    └── simulations/                   # Survey reliability simulations

file_utils/                            # Data processing and mapping utilities
├── images/                            # Image-metadata mapping tools
│   ├── scripts/                       # Core image mapping logic
│   └── image_mapper_main.py           # Entry point for image mapping
├── merge/                             # Merges all data sources
│   └── merge_all_data.py              # Main merger script
├── metabolites/                       # Metabolite-metadata mapping
│   └── metabolite_mapper.py           # Processes Excel metabolite data
└── surveys/                           # Survey-metadata mapping
    └── surveys_mapper.py              # Processes Excel survey data

config.py                              # Centralized configuration (environment variables)
all_data.json                          # Master merged data file (generated, not in repo)
core_env.yaml                          # Conda environment specification
CLAUDE.md                              # Code analysis and documentation
```

## Quick Start

### 1. Environment Setup
```bash
# Activate the required conda environment
conda activate /net/projects2/promega

# Ensure your .env file is configured with required paths
```

### 2. Generate Master Data File
```bash
# From the root directory, generate all_data.json
python file_utils/merge/merge_all_data.py

# Add edge_fraction field (required for complete dataset)
python analysis/images/quality/mask_edge_fraction.py
```

### 3. Run Analysis
```bash
# All analysis runs from root directory
python analysis/images/classifier/train_model_accuracy.py
python analysis/surveys/classifier/simple_classifier.py
```

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
   - `analysis/images/quality/mask_edge_fraction.py` - Adds edge_fraction field to `all_data.json`

3. **Analysis**: Uses `all_data.json` as single source of truth
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
    "Best Z": 0,
    "Best Z Filename": "Ba1/Dy03/Ba1 96_1 Dy03 A1 Z0.tif",
    "Actual Z Value": 0,
    "Classification": "Regular",
    "um_per_px": 1.687,
    "all_files": ["Ba1/Dy03/Ba1 96_1 Dy03 A1 Z0.tif", "..."],
    "cellLine": "GM23279A",
    "treatment": "...",
    "main_id": "BA1_96_1_Dy03_A1_nosplit_nostitch",
    "processed": {
      "img_path": "/net/projects2/promega/data-analysis/output/...",
      "mask_path": "/net/projects2/promega/data-analysis/predictions/...",
      "overlay_path": "...",
      "orig_width_px": 1128,
      "orig_height_px": 832
    },
    "verification": {
      "classification_verification": "NoSplitNoStitched",
      "blank_verified": "NO",
      "blank": false
    },
    "metabolites": {
      "GlucoseGlo": {"concentration_uM": 12.138, "is_outlier": false},
      "GlutamateGlo": {"concentration_uM": 0.206, "is_outlier": false}
    },
    "survey": {
      "evaluations": [{"evaluation": "Acceptable", "employee": "..."}],
      "quality_scores": [{"quality": "Good"}]
    },
    "edge_fraction": 0.181
  }
}
```

**Note:** The `edge_fraction` field is added by running `mask_edge_fraction.py` after the initial merge.

### Field Descriptions

#### Top-Level Key
- **Key format** (e.g., `"BA1 96_1 Dy03 A1"`): Unique identifier combining batch, day, and well position

#### Core Metadata Fields
- **`dayID`**: Day identifier in format "DyXX" (e.g., "Dy03", "Dy30")
- **`BA`**: Batch identifier (e.g., "BA1 96_1", "BA2 96_2")
- **`wellID`**: Well position in the plate (e.g., "A1", "H11")
- **`main_id`**: Standardized identifier with processing flags (e.g., "BA1_96_1_Dy03_A1_nosplit_nostitch")
- **`cellLine`**: Cell line identifier (e.g., "GM23279A")
- **`treatment`**: Treatment condition applied to the organoid (may be "nan" if none)

#### Image Metadata Fields
- **`Best Z`**: Integer index of the best focus Z-plane selected for analysis
- **`Actual Z Value`**: The actual Z-plane value used
- **`Best Z Filename`**: Relative path to the best Z-plane image file
- **`all_files`**: Array of all available Z-stack image file paths
- **`Classification`**: Image classification category (e.g., "Regular", "Split", "Stitched")
- **`um_per_px`**: Micrometers per pixel - spatial resolution of the original image

#### Processed Image Data (`processed` object)
- **`img_path`**: Path to processed/resized image file
- **`mask_path`**: Path to predicted segmentation mask
- **`overlay_path`**: Path to image with mask overlay visualization
- **`main_id`**: Processing identifier
- **`orig_width_px`** / **`orig_height_px`**: Original image dimensions in pixels
- **`orig_um_per_px_x`** / **`orig_um_per_px_y`**: Original spatial resolution
- **`final_um_per_px_x`** / **`final_um_per_px_y`**: Spatial resolution after processing

#### Verification Data (`verification` object)
- **`classification_verification`**: Verified classification (e.g., "NoSplitNoStitched")
- **`blank_verified`**: Whether organoid was verified as blank ("YES"/"NO")
- **`blank`**: Boolean indicating if well is blank (no organoid present)
- **`gen_main_id`**: Generated main identifier for verification

#### Metabolite Data (`metabolites` object)
Each metabolite (e.g., "GlucoseGlo", "GlutamateGlo", "ATP", "LactateGlo", etc.) contains:
- **`concentration_uM`**: Measured concentration in micromolar (μM)
- **`initial_concentration`**: Initial concentration value
- **`is_outlier`**: Boolean indicating if measurement is a statistical outlier
- **`well_384`**: Well position in the 384-well metabolite assay plate

#### Survey Data (`survey` object)
- **`evaluations`**: Array of expert evaluations
  - **`image_id`**: Image identifier
  - **`evaluation`**: Classification ("Acceptable" or "Not Acceptable")
  - **`employee`**: Name of evaluator
  - **`source_file`**: Excel file containing the evaluation
  - **`BA`**, **`dayID`**, **`wellID`**: Identifiers linking to image
- **`quality_scores`**: Array of quality assessments
  - **`quality`**: Quality rating (e.g., "Good", "Reasonable", "Poor")
  - Other fields similar to evaluations

#### Quality Metrics
- **`edge_fraction`**: Fraction of mask pixels touching image edges (0.0-1.0). Added by `mask_edge_fraction.py`. Lower values indicate better-centered organoids. This field is crucial for quality filtering in downstream analysis.

## Key Features

- **Multimodal Data Integration**: Images, metabolites, and surveys in one structure
- **Time Series Analysis**: Organoid quality tracking across days (Dy3, Dy6, Dy8, etc.)
- **Standardized Processing**: Consistent image resolutions and metadata
- **Local Output Generation**: Outputs written to mini-test directory while reading from shared cluster data
- **Comprehensive Analysis Tools**: Classification, segmentation, and statistical analysis

## Development Guidelines

- **Environment**: Always activate conda environment first: `conda activate /net/projects2/promega`
- **Configuration**: Use `config.py` for all path and setting management
- **Data Access**: Use `all_data.json` as single source of truth
- **Analysis Location**: Place all analysis code in `analysis/` directory
- **Execution**: Run everything from project root directory

## Current Status

✅ **Fully Functional System** (Updated October 2025, read CHANGES.md for more info)
- All immediate code quality fixes completed
- Working data generation pipeline producing complete 5,168-record dataset (13MB)
- Multimodal data integration (images, metabolites, surveys) operational
- Centralized configuration and pattern management implemented
- Comprehensive error handling and validation added
- Complete 16-field dataset including edge_fraction quality metric

## Known Issues & Future Improvements

See `CLAUDE.md` for detailed code analysis and recommended architectural enhancements.


