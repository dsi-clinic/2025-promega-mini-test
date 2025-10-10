# Promega Organoid Analysis System

This repository contains a comprehensive system for analyzing organoid quality using multimodal data including images, metabolites, and survey assessments for time series prediction.

## Project Structure

```mermaid
%%{init: {"themeVariables": { "fontSize": "16px" }, "flowchart": { "rankSpacing": 50, "nodeSpacing": 45 }}}%%
flowchart TD
    %% ========= INPUT STAGE ========= %%
    A1([Raw Images])
    A2([Metabolite Excels])
    A3([Survey Excels])
    A4([Config and Env Vars<br/>config.py / core_env.yaml])
    
    %% ========= FILE_UTILS PROCESSING ========= %%
    subgraph B[file_utils - Data Mapping and Integration]
        B1[file_utils/images/scripts<br/>image_mapper_main.py<br/>Image metadata → JSON]
        B1b[file_utils/common/organoid_patterns.py<br/>Pattern normalization helpers]
        B2[file_utils/metabolites/metabolite_mapper.py<br/>Metabolite Excel → JSON]
        B3[file_utils/surveys/surveys_mapper.py<br/>Survey Excel → JSON]
        B4[file_utils/merge/merge_all_data.py<br/>Merge image, metabolite, and survey JSON<br/>→ all_data.json]
    end
    
    %% ========= ANALYSIS PIPELINE ========= %%
    subgraph C[analysis - Downstream Analysis]
        subgraph C1[analysis/images]
            C13[segmentation_mmseg<br/>MMSeg training and predictions]
            C16[edge_fraction<br/>Post-prediction edge analysis<br/>Adds edge metrics to all_data.json]
            C14[classifier<br/>ViT or CNN classifier<br/>Uses MMSeg-processed images]
            C11[resize<br/>Scale to physical size and aspect ratio]
            C12[metrics/shape_metrics<br/>Morphological feature analysis]
            
            subgraph C15[series - Time Series Analysis]
                C15a[series/filter_complete_series.py<br/>Filter to complete series<br/>no blanks, no edge issues<br/>→ complete_series_data_no_blanks.json]
                C15b[series/preprocess_for_lstm.py<br/>Resize to uniform scale 6.0 um/px<br/>Pad to 768×768, preserve aspect ratio<br/>Adds lstm_processed field to JSON<br/>→ lstm_ready/ images]
                C15c[series/cnn_lstm/train_organoid_lstm.py<br/>CNN-LSTM model training<br/>Predict organoid quality from time series]
            end
        end
        
        subgraph C2[analysis/metabolites]
            C21[classifier<br/>Metabolite-based models<br/>Combined with image-derived data]
        end
        
        subgraph C3[analysis/surveys]
            C32[classifier<br/>Survey-based models<br/>Combined with image-derived data]
            C33[agreement_aggregations<br/>Statistical analysis only<br/>Derived from all_data.json]
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
    
    B4 --> C13
    C13 --> C16
    C16 --> B4
    C16 --> C14
    C16 --> C11
    
    %% LSTM Pipeline Flow
    B4 --> C15a
    C15a --> C15b
    C15b --> C15c
    
    C11 --> C12
    B4 --> C14
    B4 --> C12
    B4 --> C21
    B4 --> C32
    B4 --> C33
    
    C14 --> C21
    C14 --> C32
    C12 --> C21
    C12 --> C32
    C15c --> C21
    C15c --> C32
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


