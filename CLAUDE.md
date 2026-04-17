# Code Analysis: all_data.json Generation System

## System Overview

The `all_data.json` generation system is designed to merge data from multiple sources:
- **Images**: Raw image metadata and processed image data at different resolutions
- **Metabolites**: Chemical assay data from Excel spreadsheets
- **Surveys**: Quality assessment data from Excel forms

The system generates a unified JSON structure for time series prediction analysis of organoid quality across multiple days (Dy1, Dy3, etc.).

## Data Flow

```
Raw Sources → Individual Mappers → merge_all_data.py → all_data.json
     ↓              ↓                     ↓              ↓
   Images      image_mapper.py      Normalization    Unified
Metabolites   metabolite_mapper.py     & Merge      Structure
  Surveys      surveys_mapper.py
```

## Key Files Analysis

### 1. `merge_all_data.py` (`pipeline/merge/`)

**Purpose**: Central merger that combines all data sources into `all_data.json`

**Key Components**:
- Normalizes keys using `norm_key()` function
- Merges data from 4 sources: base images, processed images, metabolites, surveys
- Adds day processing logic (`day_from_key`, `to_mdl_day`)

### 2. Individual Mappers (in `pipeline/`)

- **`pipeline/images/image_mapper.py`**: Entry point for image mapping
- **`pipeline/metabolites/metabolite_mapper.py`**: Processes Excel metabolite data
- **`pipeline/surveys/surveys_mapper.py`**: Processes Excel survey data


### 🟢 **CURRENT SYSTEM STATUS (UPDATED)**

**Data Generation**: ✅ **WORKING** - Successfully generates complete all_data.json (4,276 records, 9.5MB)
- Complete base image mapping data (raw images, Z-stacks, metadata)
- Multi-resolution processed images (256x192, 512x384)
- Metabolite assay data with concentration values
- Survey evaluation and quality score data
- Proper NaN → null conversion for valid JSON
- Centralized regex patterns eliminate code duplication

**Environment**: ✅ **DOCUMENTED** - Conda environment setup properly documented
- `conda activate core_env` required before running (defined by `core_env.yaml` in repo)
- `mmcv_env` separate environment for segmentation steps 8-9
- Data lives at `/net/projects2/promega/2026_04_data/`
- Legacy/archive data at `/net/projects2/promega/2026_04_non_env/`
- Proper PYTHONPATH configuration for imports

### Logic Issues (Lower Priority)

1. **Base path environment dependency**: The code relies on `BASE_PATH` environment variable but may fail silently if not set properly in merge context.

2. ~~**Error handling gaps**: Several `ValueError` exceptions in key normalization could crash the entire merge process.~~ ✅ **IMPROVED** - Added proper error handling with metadata key filtering

3. **Memory inefficiency**: Loads entire datasets into memory before merging.

## Areas for Improvement

### 1. DRY Principle Violations

- **Repeated JSON loading**: `load_json()` function is simple but could be enhanced with error handling

### 2. KISS Principle Violations

- **Nested dictionary structure**: Deep nesting in final JSON makes access complex
- **Mixed responsibilities**: `merge_all_data.py` handles both file I/O and business logic

### 3. Complexity Management Issues

- **Monolithic merge function**: Lines 118-128 could be broken into smaller functions
- **Hard-coded mappings**: BA_FOLDER_MAP and other mappings scattered across files
- **Missing abstraction**: No clear data model classes for organoid data

### 4. Data Structure Inefficiencies

- **Redundant key storage**: Keys are stored both as dictionary keys and within entries
- **Mixed data types**: Inconsistent use of strings vs numbers for day values
- **Resolution-specific nesting**: The `256x192`/`512x384` structure could be more generic
- **Survey data structure**: Complex nested structure that's hard to query

## Recommended Improvements

### Architectural Improvements (Medium Priority)

1. **Create data model classes**:
   ```python
   @dataclass
   class OrganoidData:
       key: str
       day_num: int
       ba: str
       well_id: str
       images: dict
       metabolites: dict
       survey: dict
   ```

2. **Implement proper error handling**:
   ```python
   def safe_load_json(path: Path) -> dict:
       try:
           with open(path) as f:
               return json.load(f)
       except (FileNotFoundError, JSONDecodeError) as e:
           logging.error(f"Failed to load {path}: {e}")
           return {}
   ```

3. **Add data validation**:
   ```python
   def validate_organoid_key(key: str) -> bool:
       pattern = r'^BA\d+(\s+96_[12])?\s+Dy\d+\s+[A-H]\d{1,2}$'
       return bool(re.match(pattern, key))
   ```

4. **Implement streaming/chunked processing** for large datasets

### Long-term Enhancements (Low Priority)

1. **Database integration**: Consider using SQLite or similar for structured queries
2. **Schema versioning**: Add version field to support data evolution
3. **Caching layer**: Implement caching for expensive operations
4. **Testing framework**: Add comprehensive unit tests

## Data Structure Analysis

### Current Structure Strengths
- **Unified access**: Single file contains all related data
- **Hierarchical organization**: Logical grouping by organoid
- **Extensible**: Easy to add new data types

### Current Structure Weaknesses
- **Large file size**: Single JSON file can become unwieldy
- **Query complexity**: No indexing for efficient data access  
- **Memory usage**: Must load entire structure for any query
- **Inconsistent nesting**: Some data is flat, some deeply nested

### Suggested Improvements
1. **Flatten resolution data**: Use generic `resolutions` array instead of fixed keys
2. **Normalize survey structure**: Standardize evaluation vs quality_scores
3. **Add metadata header**: Include generation timestamp, version, statistics
4. **Consider partitioning**: Split by batch or day for large datasets

## Environment Setup Notes

### Directory Layout

```
Code:    ~/2025-promega-mini-test/          (this repo)
  pipeline/                                 # Data prep steps 1-16
    ├── identifiers/   images/   metabolites/   surveys/
    ├── merge/         common/
    └── images/{resize, quality, series, postprocess, segmentation_mmseg}
  analysis/                                 # Model heads steps 17+ & exploratory
    ├── imagequality_classification/        # Step 17: PyTorch image quality classifier
    ├── image_survey_classification/        # Step 18: TensorFlow survey classifier
    ├── cnn_lstm/   metabolites/   multimodal/   surveys/
    └── data_loader.py   outputs/

Data:    /net/projects2/promega/2026_04_15_data/
  raw/                                        # NEVER WRITTEN by pipeline
    ├── images/raw_images/   masks/manual/
    ├── metabolite.xlsx   Sample-Tracing.xlsx
    └── surveys/   image_verification.csv
  intermediate/                               # Regenerated by make pipeline-all
    ├── indexes/   resized_512x384/   overlays/
    ├── resized_575_square/   mean_fill_clip/
    └── lstm_ready/   masks_processed/
  models/                                     # Regenerated by make train-all
    ├── mmseg/   imagequality_classification/
    └── image_survey_classification/
  analysis_output/                            # Manual figures & reports
Archive: /net/projects2/promega/2026_04_non_env/  (legacy student outputs)
```

### Conda Environment
The environment is defined by `core_env.yaml` in this repo:
```bash
conda activate core_env
```

For segmentation (steps 8-9), use the separate `mmcv_env`.

### Running the Pipeline
```bash
conda activate core_env
make pipeline-all                    # runs steps 1-16 (pipeline/)
make train-all                       # steps 17-18 (analysis/)
make imagequality-classification     # step 17 only
make image-survey-classification     # step 18 only
```

All paths are configured via Makefile variables (`DATA_ROOT`, `RAW_DIR`, `INTERMEDIATE_DIR`, `MODELS_DIR`).
Override with: `make step1 DATA_ROOT=/path/to/data`

## Testing Recommendations

1. **Unit tests** for key normalization functions
2. **Integration tests** for end-to-end data flow  
3. **Data validation tests** to catch malformed inputs
4. **Performance tests** for large datasets
5. **Schema validation** for JSON output structure

## Code Organization

The codebase is split into two top-level packages:

- **`pipeline/`** — Data preparation (steps 1-16). Transforms raw data into `all_data.json`. Contains mappers, image processing, segmentation, and merge logic. Formerly `file_utils/` + pipeline parts of `analysis/`.

- **`analysis/`** — Model training and exploratory analysis (steps 17+). Consumes `all_data.json` and referenced image/mask files. Each model head is self-contained with its own train script. Imports shared utilities from `pipeline.common` and `pipeline.merge`.