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

## Key Features

- **Multimodal Data Integration**: Images, metabolites, and surveys in one structure
- **Time Series Analysis**: Organoid quality tracking across days (Dy3, Dy6, Dy8, etc.)
- **Standardized Processing**: Consistent image resolutions and metadata
- **Environment-Based Configuration**: No hardcoded paths
- **Comprehensive Analysis Tools**: Classification, segmentation, and statistical analysis

## Development Guidelines

- **Environment**: Always activate conda environment first: `conda activate /net/projects2/promega`
- **Configuration**: Use `config.py` for all path and setting management
- **Data Access**: Use `all_data.json` as single source of truth
- **Analysis Location**: Place all analysis code in `analysis/` directory
- **Execution**: Run everything from project root directory

## Current Status

✅ **Fully Functional System** (Updated August 2025)
- All immediate code quality fixes completed
- Working data generation pipeline producing complete 4,276-record dataset (9.5MB)
- Multimodal data integration (images, metabolites, surveys) operational
- Centralized configuration and pattern management implemented
- Comprehensive error handling and validation added

## Known Issues & Future Improvements

See `CLAUDE.md` for detailed code analysis and recommended architectural enhancements.


