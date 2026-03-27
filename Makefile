# Promega Organoid Analysis Pipeline Makefile
# ===========================================
# This Makefile automates the 18-step preprocessing and classification pipeline
# documented in README.md
#
# Quick Start:
#   make help              - Show all available targets
#   make pipeline-all      - Run complete preprocessing pipeline (Steps 1-16)
#   make train-all         - Train both classifiers (Steps 17-18)
#
# Environment Variables:
#   DATA_DIR              - Base data directory (default: /net/projects2/promega/data_reorg/data)
#   PYTHON                - Python executable (default: conda run -p /net/projects2/promega python3)
#   PYTHON_MMCV           - Python with mmcv env (default: conda run -n mmcv_env python)

# -------- Configuration --------
DATA_DIR           ?= /net/projects2/promega/project_data/amanda_test
PYTHON             ?= conda run --no-capture-output -p /net/projects2/promega python3
PYTHON_MMCV        ?= conda run --no-capture-output -p /net/scratch2/ntebaldi/conda_envs/mmcv_env python
PYTHONPATH         := $(shell pwd)

# Input directories
IMAGE_VERIFICATION_CSV := $(DATA_DIR)/images/image_verification.csv
SAMPLE_TRACING_XLSX    := $(DATA_DIR)/images/Sample-Tracing.xlsx
RAW_IMAGES_DIR         := $(DATA_DIR)/images/raw_images
MANUAL_MASKS_DIR       := $(DATA_DIR)/masks/manual
METABOLITE_XLSX        := $(DATA_DIR)/metabolite/metabolite_data_07_23_25.xlsx
SURVEY_DIR             := $(DATA_DIR)/survey

# Output directories
IDENTIFIERS_DIR    := $(DATA_DIR)/identifiers
IMAGES_DIR         := $(DATA_DIR)/images
MASKS_DIR          := $(DATA_DIR)/masks
METABOLITE_DIR     := $(DATA_DIR)/metabolite
SURVEY_OUT_DIR     := $(DATA_DIR)/survey
SPLITS_DIR         := $(DATA_DIR)/images/resized_512x384_splits
LSTM_DIR           := $(DATA_DIR)/lstm

# Output files
RECORD_IDENTIFIERS     := $(IDENTIFIERS_DIR)/record_identifiers.json
METABOLITE_MAP         := $(METABOLITE_DIR)/metabolite_map.json
SURVEY_MAP             := $(SURVEY_OUT_DIR)/survey_map.json
IMAGE_MAP              := $(IMAGES_DIR)/image_map.json
IMAGE_MAP_MANUAL       := $(MASKS_DIR)/image_mapping_thresholded_and_manual.json
IMAGE_MAP_RESIZED      := $(IMAGES_DIR)/image_map_resized_512x384.json
IMAGE_MAP_PREDICTED    := $(IMAGES_DIR)/image_map_resized_512x384_predicted.json
IMAGE_MAP_OVERLAY      := $(IMAGES_DIR)/image_map_resized_512x384_predicted_overlay.json
# Overlay + edge_fraction + aspect-ratio (for step15 mean fill clip)
# Produced by step14 when given IMAGE_MAP_OVERLAY (stem + "_ar.json")
IMAGE_MAP_MEANFILL     := $(IMAGES_DIR)/image_map_resized_512x384_predicted_overlay_ar.json
# + meanfill (for step16 merge_all_data; has predicted_mask_path and full pipeline fields)
# Produced by step15 from IMAGE_MAP_MEANFILL (stem + "_meanfill.json")
IMAGE_MAP_MERGE        := $(IMAGES_DIR)/image_map_resized_512x384_predicted_overlay_ar_meanfill.json
ALL_DATA_JSON          := $(IDENTIFIERS_DIR)/all_data.json
IMAGE_CLASSIFIER_JSON  := $(IDENTIFIERS_DIR)/image_classifier.json
SURVEY_CLASSIFIER_JSON := $(IDENTIFIERS_DIR)/survey_classifier.json

# Options
OVERWRITE          ?=
TARGET_WIDTH       ?= 512
TARGET_HEIGHT      ?= 384
MIN_SURVEY_VOTES   ?= 4
TRAIN_FRAC         ?= 0.8
VAL_FRAC           ?= 0.1

# Segmentation training (early vs late datasets)
SEG_TRAIN_SCRIPT ?= analysis/images/segmentation_mmseg/train.py
SEG_CONFIG       ?= analysis/images/segmentation_mmseg/segformer_mitb0.py
SEG_WORK_ROOT    ?= $(DATA_DIR)/masks/mmseg_models
GPU              ?=
RESUME           ?=

# Segmentation models (outputs from step8)
# Config: latest timestamped run dir (e.g. .../late/late/20260209_103224/vis_data/config.py)
# Checkpoint: in phase dir, not run dir (mmseg saves .../late/late/iter_1000.pth)
EARLY_RUN_DIR    = $(shell ls -t $(SEG_WORK_ROOT)/early/early/ 2>/dev/null | grep -E '^[0-9]{8}_[0-9]{6}$$' | head -1)
LATE_RUN_DIR     = $(shell ls -t $(SEG_WORK_ROOT)/late/late/ 2>/dev/null | grep -E '^[0-9]{8}_[0-9]{6}$$' | head -1)
EARLY_CONFIG     ?= $(SEG_WORK_ROOT)/early/early/$(EARLY_RUN_DIR)/vis_data/config.py
EARLY_CHECKPOINT ?= $(SEG_WORK_ROOT)/early/early/iter_1000.pth
LATE_CONFIG      ?= $(SEG_WORK_ROOT)/late/late/$(LATE_RUN_DIR)/vis_data/config.py
LATE_CHECKPOINT  ?= $(SEG_WORK_ROOT)/late/late/iter_1000.pth


# Classifier training options
EPOCH1             ?= 100
EPOCH2             ?= 300
BATCH_SIZE         ?= 16
DETERMINISTIC      ?= 1
SEED               ?= 1

# -------- Phony Targets --------
.PHONY: help clean \
        step1 step2 step3 step4 step5 step6 step7 step8 step9 step10 \
        step11 step12 step13 step14 step15 step16 step17 step18 \
        pipeline-identifiers pipeline-mappers pipeline-preprocessing \
        pipeline-segmentation pipeline-quality pipeline-series \
        pipeline-merge pipeline-all train-all \
        validate-inputs

.PHONY: seg-train-early seg-train-late

seg-train-early:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON_MMCV) $(SEG_TRAIN_SCRIPT) \
		--config $(SEG_CONFIG) \
		--splits-dir $(SPLITS_DIR) \
		--split early \
		--work-dir $(SEG_WORK_ROOT)/early \
		$(if $(RESUME),--resume) \
		$(if $(GPU),--cfg-options launcher=pytorch gpu_ids=$(GPU))

seg-train-late:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON_MMCV) $(SEG_TRAIN_SCRIPT) \
		--config $(SEG_CONFIG) \
		--splits-dir $(SPLITS_DIR) \
		--split late \
		--work-dir $(SEG_WORK_ROOT)/late \
		$(if $(RESUME),--resume) \
		$(if $(GPU),--cfg-options launcher=pytorch gpu_ids=$(GPU))


# -------- Help --------
help:
	@echo "=================================="
	@echo "Promega Pipeline Makefile"
	@echo "=================================="
	@echo ""
	@echo "QUICK START:"
	@echo "  make pipeline-all          - Run complete preprocessing pipeline (Steps 1-16)"
	@echo "  make train-all             - Train both classifiers (Steps 17-18)"
	@echo ""
	@echo "PIPELINE SECTIONS:"
	@echo "  make pipeline-identifiers  - Step 1: Retrieve identifiers"
	@echo "  make pipeline-mappers      - Steps 2-4: Map metabolite, survey, image data"
	@echo "  make pipeline-preprocessing - Steps 5-7: Manual masks, resize, test splits"
	@echo "  make pipeline-segmentation - Steps 8-9: Train and predict segmentation"
	@echo "  make pipeline-quality      - Steps 10-11: Overlays and edge fraction"
	@echo "  make pipeline-series       - Steps 12-15: Filter series, LSTM prep, resize"
	@echo "  make pipeline-merge        - Step 16: Generate all_data.json"
	@echo ""
	@echo "INDIVIDUAL STEPS:"
	@echo "  make step1                 - Retrieve main identifiers"
	@echo "  make step2                 - Map metabolite data"
	@echo "  make step3                 - Map survey data"
	@echo "  make step4                 - Map image data"
	@echo "  make step5                 - Map manual masks"
	@echo "  make step6                 - Resize and remap images"
	@echo "  make step7                 - Create test splits"
	@echo "  make step8                 - Train segmentation model"
	@echo "  make step9                 - Predict segmentation masks"
	@echo "  make step10                - Create image-mask overlays"
	@echo "  make step11                - Calculate mask edge fraction"
	@echo "  make step12                - Filter complete series"
	@echo "  make step13                - Preprocess for LSTM"
	@echo "  make step14                - Resize with aspect ratio"
	@echo "  make step15                - Mean fill clip"
	@echo "  make step16                - Generate all_data.json"
	@echo "  make step17                - Train image classifier"
	@echo "  make step18                - Train survey classifier"
	@echo ""
	@echo "CONFIGURATION:"
	@echo "  DATA_DIR=$(DATA_DIR)"
	@echo "  OVERWRITE=$(OVERWRITE)"
	@echo "  TARGET_WIDTH=$(TARGET_WIDTH), TARGET_HEIGHT=$(TARGET_HEIGHT)"
	@echo ""
	@echo "EXAMPLES:"
	@echo "  make step1 DATA_DIR=/path/to/data"
	@echo "  make step6 OVERWRITE=1"
	@echo "  make pipeline-all OVERWRITE=1"
	@echo "  make step17 EPOCH1=50 EPOCH2=150 DETERMINISTIC=1"
	@echo ""

# ====================================
# STEP 1: Retrieve Main Identifiers
# ====================================
step1: validate-inputs
	@echo "===> STEP 1: Retrieving main identifiers"
	@mkdir -p $(IDENTIFIERS_DIR)
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m file_utils.identifiers.retrieve_main_identifiers \
		--csv-file $(IMAGE_VERIFICATION_CSV) \
		--out-file $(RECORD_IDENTIFIERS)
	@echo "===> Output: $(RECORD_IDENTIFIERS)"

# ====================================
# STEP 2: Map Metabolite Data
# ====================================
step2: $(RECORD_IDENTIFIERS)
	@echo "===> STEP 2: Mapping metabolite data"
	@mkdir -p $(METABOLITE_DIR)
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m file_utils.metabolites.metabolite_mapper \
		--in-file $(METABOLITE_XLSX) \
		--identifiers $(RECORD_IDENTIFIERS) \
		--out-file $(METABOLITE_MAP)
	@echo "===> Output: $(METABOLITE_MAP)"

# ====================================
# STEP 3: Map Survey Data
# ====================================
step3: $(RECORD_IDENTIFIERS)
	@echo "===> STEP 3: Mapping survey data"
	@mkdir -p $(SURVEY_OUT_DIR)
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m file_utils.surveys.surveys_mapper \
		--in-dir $(SURVEY_DIR) \
		--out-file $(SURVEY_MAP) \
		--identifiers $(RECORD_IDENTIFIERS) \
		--min-survey-votes $(MIN_SURVEY_VOTES)
	@echo "===> Output: $(SURVEY_MAP)"

# ====================================
# STEP 4: Map Image Data
# ====================================
step4: $(RECORD_IDENTIFIERS)
	@echo "===> STEP 4: Mapping image data"
	@mkdir -p $(IMAGES_DIR)
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m file_utils.images.image_mapper \
		--base-dir $(RAW_IMAGES_DIR) \
		--verify-csv $(IMAGE_VERIFICATION_CSV) \
		--meta-xlsx $(SAMPLE_TRACING_XLSX) \
		--identifiers $(RECORD_IDENTIFIERS) \
		--out-file $(IMAGE_MAP)
	@echo "===> Output: $(IMAGE_MAP)"

# ====================================
# STEP 5: Map Manual Masks & Resize
# ====================================
step5: $(IMAGE_MAP)
	@echo "===> STEP 5: Mapping + preprocessing manual masks"
	@mkdir -p $(MASKS_DIR)
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m analysis.images.segmentation_mmseg.preprocessing.manual_masks_mapping \
		--image-json $(IMAGE_MAP) \
		--masks-dir $(MANUAL_MASKS_DIR) \
		--output-file $(IMAGE_MAP_MANUAL) \
		--target-width $(TARGET_WIDTH) \
		--target-height $(TARGET_HEIGHT) \
		--processed-masks-dir $(MASKS_DIR)/manual_processed_$(TARGET_WIDTH)x$(TARGET_HEIGHT) \
		$(if $(OVERWRITE),--overwrite)


# ====================================
# STEP 6: Resize and Remap Images
# ====================================
step6: $(IMAGE_MAP) $(IMAGE_MAP_MANUAL)
	@echo "===> STEP 6: Resizing and remapping images"
	@mkdir -p $(IMAGES_DIR)/resized_$(TARGET_WIDTH)x$(TARGET_HEIGHT)
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m analysis.images.resize.resize_remap_images \
		--image-mapping-json $(IMAGE_MAP) \
		--mask-mapping-json $(IMAGE_MAP_MANUAL) \
		--out-dir $(IMAGES_DIR)/resized_$(TARGET_WIDTH)x$(TARGET_HEIGHT) \
		--out-mapping-json $(IMAGE_MAP_RESIZED) \
		--target-width $(TARGET_WIDTH) \
		--target-height $(TARGET_HEIGHT) \
		$(if $(OVERWRITE),--overwrite)
	@echo "===> Output: $(IMAGE_MAP_RESIZED)"

# ====================================
# STEP 7: Create Test Splits
# ====================================
step7: $(IMAGE_MAP_RESIZED)
	@echo "===> STEP 7: Creating test splits"
	@mkdir -p $(SPLITS_DIR)
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m analysis.images.segmentation_mmseg.preprocessing.test_split \
		--resized-json $(IMAGE_MAP_RESIZED) \
		--splits-dir $(SPLITS_DIR) \
		--train-frac $(TRAIN_FRAC) \
		--val-frac $(VAL_FRAC) \
		--split-days
	@echo "===> Output: $(SPLITS_DIR)/"

# ====================================
# STEP 8: Train Segmentation Model
# ====================================
# Default phase for segmentation steps (early or late)
PHASE ?= late

step8: step7
	@echo "===> STEP 8: Training segmentation model (PHASE=$(PHASE))"
	$(MAKE) seg-train-$(PHASE) DATA_DIR=$(DATA_DIR) GPU=$(GPU) RESUME=$(RESUME)



# ====================================
# STEP 9: Predict Segmentation Masks
# ====================================
# Note: Requires checkpoint from step8
# Choose correct model files based on PHASE
ifeq ($(PHASE),early)
  CONFIG_FILE = $(EARLY_CONFIG)
  CHECKPOINT  = $(EARLY_CHECKPOINT)
else ifeq ($(PHASE),late)
  CONFIG_FILE = $(LATE_CONFIG)
  CHECKPOINT  = $(LATE_CHECKPOINT)
else
  $(error PHASE must be early or late)
endif

# Which mapping JSON to run inference on (defaults to resized)
PRED_INPUT_JSON ?= $(IMAGE_MAP_RESIZED)

step9:
	@echo "===> STEP 9: Predicting segmentation masks (PHASE=$(PHASE))"
	@mkdir -p $(MASKS_DIR)/predicted
	@if [ ! -f "$(CHECKPOINT)" ]; then \
		echo "ERROR: Checkpoint not found: $(CHECKPOINT)"; \
		exit 1; \
	fi
	PYTHONPATH=$(PYTHONPATH) $(PYTHON_MMCV) -m analysis.images.segmentation_mmseg.predict_masks \
		--image-mapping-json $(PRED_INPUT_JSON) \
		--out-dir $(MASKS_DIR)/predicted \
		--model-type $(PHASE) \
		--config $(CONFIG_FILE) \
		--checkpoint $(CHECKPOINT) \
		$(if $(DAYS),--days $(DAYS)) \
		--write-collage \
		$(if $(OVERWRITE),--overwrite)
	@echo "===> Output masks: $(MASKS_DIR)/predicted/"

step9-early:
	$(MAKE) step9 PHASE=early OVERWRITE=$(OVERWRITE) DAYS=Dy03,Dy06,Dy08,Dy10 PRED_INPUT_JSON=$(IMAGE_MAP_RESIZED)

step9-late:
	$(MAKE) step9 PHASE=late OVERWRITE=$(OVERWRITE) DAYS=Dy13,Dy15,Dy17,Dy20,Dy21,Dy24,Dy28,Dy30 \
	  PRED_INPUT_JSON=$(IMAGE_MAP_RESIZED:.json=_predicted.json)


# ====================================
# STEP 10: Create Image-Mask Overlays
# ====================================
# Uses predicted mapping (step9 output): overlays need both processed_image and predicted_mask_path
step10: $(IMAGE_MAP_PREDICTED)
	@echo "===> STEP 10: Creating image-mask overlays"
	@mkdir -p $(IMAGES_DIR)/overlays
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m analysis.images.quality.image_mask_overlay \
		--image-mapping-json $(IMAGE_MAP_PREDICTED) \
		--overlay-dir $(IMAGES_DIR)/overlays \
		$(if $(OVERWRITE),--overwrite)
	@echo "===> Output: $(IMAGES_DIR)/overlays/"

# ====================================
# STEP 11: Calculate Mask Edge Fraction
# ====================================
# Uses overlay mapping (step10 output): has predicted_mask_path and overlay_path
step11: step10
	@echo "===> STEP 11: Calculating mask edge fraction"
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m analysis.images.quality.mask_edge_fraction \
		--image-mapping-json $(IMAGE_MAP_OVERLAY)
	@echo "===> Updated: $(IMAGE_MAP_OVERLAY) (with edge_fraction field)"

# ====================================
# STEP 12: Filter Complete Series
# ====================================
step12: $(IMAGE_MAP_RESIZED)
	@echo "===> STEP 12: Filtering complete series"
	@mkdir -p $(IMAGES_DIR)/filter_complete_series
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m analysis.images.series.filter_complete_series \
		--image-mapping-json $(IMAGE_MAP_RESIZED) \
		--out-dir $(IMAGES_DIR)/filter_complete_series \
		--show-examples
	@echo "===> Output: $(IMAGES_DIR)/filter_complete_series/"

# ====================================
# STEP 13: Preprocess for LSTM
# ====================================
step13: step12
	@echo "===> STEP 13: Preprocessing for LSTM"
	@mkdir -p $(LSTM_DIR)
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m analysis.images.series.preprocess_for_lstm \
		--complete-series $(IMAGES_DIR)/filter_complete_series/complete_series_data_no_blanks.json \
		--raw-image-dir $(RAW_IMAGES_DIR) \
		--out-dir $(LSTM_DIR)
	@echo "===> Output: $(LSTM_DIR)/lstm_ready/"

# ====================================
# STEP 14: Resize with Aspect Ratio
# ====================================
# Uses overlay mapping (step10+11) so output has overlay+edge+ar; step15 consumes this
step14: step11
	@echo "===> STEP 14: Resizing with aspect ratio"
	@mkdir -p $(IMAGES_DIR)/resized_575_square
	@mkdir -p $(MASKS_DIR)/resized_575_square
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m analysis.images.resize.resize_aspect_ratio \
		--image-mapping-json $(IMAGE_MAP_OVERLAY) \
		--raw-images-dir $(RAW_IMAGES_DIR) \
		--out-images-dir $(IMAGES_DIR)/resized_575_square \
		--out-masks-dir $(MASKS_DIR)/resized_575_square \
		--require-mask \
		$(if $(OVERWRITE),--overwrite)
	@echo "===> Output: $(IMAGES_DIR)/resized_575_square/"
	@echo "===> Output: $(IMAGE_MAP_MEANFILL)"

# ====================================
# STEP 15: Mean Fill Clip
# ====================================
# Uses IMAGE_MAP_MEANFILL produced by step14 (overlay_ar.json)
step15: step14
	@echo "===> STEP 15: Applying mean fill clip"
	@mkdir -p $(IMAGES_DIR)/mean_fill_clip
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m analysis.images.postprocess.meanfill_clip \
		--image-mapping-json $(IMAGE_MAP_MEANFILL) \
		--compute-mean \
		--save-computed-mean \
		--out-images-dir $(IMAGES_DIR)/mean_fill_clip \
		--images-base $(IMAGES_DIR)/resized_575_square \
		--masks-base $(MASKS_DIR)/resized_575_square \
		--require-mask \
		$(if $(OVERWRITE),--overwrite)
	@echo "===> Output: $(IMAGES_DIR)/mean_fill_clip/"

# ====================================
# STEP 16: Generate All Data JSON
# ====================================
step16: $(METABOLITE_MAP) $(SURVEY_MAP) $(IMAGE_MAP_MERGE)
	@echo "===> STEP 16: Generating all_data.json"
	@mkdir -p $(IDENTIFIERS_DIR)
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m file_utils.merge.merge_all_data \
		--data-dir $(DATA_DIR) \
		--image-mapping-json $(IMAGE_MAP_MERGE) \
		--min-survey-votes $(MIN_SURVEY_VOTES) \
		--target-width $(TARGET_WIDTH) \
		--target-height $(TARGET_HEIGHT)
	@echo "===> Output: $(ALL_DATA_JSON)"

# ====================================
# STEP 17: Train Image Classifier
# ====================================
step17: step16
	@echo "===> STEP 17: Training image classifier"
	@mkdir -p $(IMAGES_DIR)/image_classifier
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m analysis.images.classifier.train_model_accuracy \
		--data-dir $(DATA_DIR) \
		--image-classifier-json $(IMAGE_CLASSIFIER_JSON) \
		--all-data-json $(ALL_DATA_JSON) \
		--epoch1 $(EPOCH1) \
		--epoch2 $(EPOCH2) \
		--batch-size $(BATCH_SIZE) \
		--test-frac 0.1 \
		--val-frac 0.1 \
		$(if $(DETERMINISTIC),--deterministic) \
		--seed $(SEED)
	@echo "===> Output: $(IMAGES_DIR)/image_classifier/"

# ====================================
# STEP 18: Train Survey Classifier
# ====================================
step18: step16
	@echo "===> STEP 18: Training survey classifier"
	@mkdir -p $(SURVEY_OUT_DIR)/survey_classifier
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m analysis.surveys.classifier.simple_classifier \
		--data-dir $(DATA_DIR) \
		--survey-classifier-json $(SURVEY_CLASSIFIER_JSON) \
		--all-data-json $(ALL_DATA_JSON) \
		--batch-size 8 \
		--epoch1 50 \
		--epoch2 150 \
		$(if $(DETERMINISTIC),--deterministic) \
		--seed $(SEED)
	@echo "===> Output: $(SURVEY_OUT_DIR)/survey_classifier/"

# ====================================
# Pipeline Convenience Targets
# ====================================

pipeline-identifiers: step1

pipeline-mappers: step2 step3 step4

pipeline-preprocessing: step5 step6 step7

pipeline-segmentation: step8 step9

pipeline-quality: step10 step11

pipeline-series: step12 step13 step14 step15

pipeline-merge: step16

pipeline-all: pipeline-identifiers pipeline-mappers pipeline-preprocessing pipeline-segmentation pipeline-quality pipeline-series pipeline-merge
	@echo ""
	@echo "========================================="
	@echo "PREPROCESSING PIPELINE COMPLETE!"
	@echo "========================================="
	@echo "Generated files:"
	@echo "  - $(ALL_DATA_JSON)"
	@echo ""
	@echo "Next steps:"
	@echo "  make train-all              - Train both classifiers"
	@echo ""

train-all: step17 step18
	@echo ""
	@echo "========================================="
	@echo "CLASSIFIER TRAINING COMPLETE!"
	@echo "========================================="
	@echo "Trained models:"
	@echo "  - Image classifier: $(IMAGES_DIR)/image_classifier/"
	@echo "  - Survey classifier: $(SURVEY_OUT_DIR)/survey_classifier/"
	@echo ""

# ====================================
# Validation and Utilities
# ====================================

validate-inputs:
	@echo "===> Validating input files..."
	@if [ ! -f "$(IMAGE_VERIFICATION_CSV)" ]; then \
		echo "ERROR: Image verification CSV not found: $(IMAGE_VERIFICATION_CSV)"; \
		exit 1; \
	fi
	@if [ ! -f "$(SAMPLE_TRACING_XLSX)" ]; then \
		echo "ERROR: Sample tracing Excel not found: $(SAMPLE_TRACING_XLSX)"; \
		exit 1; \
	fi
	@if [ ! -d "$(RAW_IMAGES_DIR)" ]; then \
		echo "ERROR: Raw images directory not found: $(RAW_IMAGES_DIR)"; \
		exit 1; \
	fi
	@if [ ! -d "$(MANUAL_MASKS_DIR)" ]; then \
		echo "ERROR: Manual masks directory not found: $(MANUAL_MASKS_DIR)"; \
		exit 1; \
	fi
	@if [ ! -f "$(METABOLITE_XLSX)" ]; then \
		echo "ERROR: Metabolite Excel not found: $(METABOLITE_XLSX)"; \
		exit 1; \
	fi
	@if [ ! -d "$(SURVEY_DIR)" ]; then \
		echo "ERROR: Survey directory not found: $(SURVEY_DIR)"; \
		exit 1; \
	fi
	@echo "===> All input files validated successfully!"

clean:
	@echo "Cleaning generated files..."
	@echo "This will remove:"
	@echo "  - $(IDENTIFIERS_DIR)/*.json"
	@echo "  - $(METABOLITE_DIR)/*.json"
	@echo "  - $(SURVEY_OUT_DIR)/*.json"
	@echo "  - $(IMAGES_DIR)/*.json"
	@echo "  - $(MASKS_DIR)/*.json"
	@echo ""
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		rm -f $(RECORD_IDENTIFIERS) $(METABOLITE_MAP) $(SURVEY_MAP); \
		rm -f $(IMAGE_MAP) $(IMAGE_MAP_MANUAL) $(IMAGE_MAP_RESIZED); \
		rm -f $(ALL_DATA_JSON) $(IMAGE_CLASSIFIER_JSON) $(SURVEY_CLASSIFIER_JSON); \
		rm -rf $(IMAGES_DIR)/image_classifier $(SURVEY_OUT_DIR)/survey_classifier; \
		echo "Cleaned!"; \
	else \
		echo "Cancelled."; \
	fi

# Show current configuration
config:
	@echo "========================================="
	@echo "Current Configuration"
	@echo "========================================="
	@echo "DATA_DIR           = $(DATA_DIR)"
	@echo "PYTHON             = $(PYTHON)"
	@echo "PYTHON_MMCV        = $(PYTHON_MMCV)"
	@echo ""
	@echo "Input Files:"
	@echo "  IMAGE_VERIFICATION_CSV = $(IMAGE_VERIFICATION_CSV)"
	@echo "  SAMPLE_TRACING_XLSX    = $(SAMPLE_TRACING_XLSX)"
	@echo "  RAW_IMAGES_DIR         = $(RAW_IMAGES_DIR)"
	@echo "  MANUAL_MASKS_DIR       = $(MANUAL_MASKS_DIR)"
	@echo "  METABOLITE_XLSX        = $(METABOLITE_XLSX)"
	@echo "  SURVEY_DIR             = $(SURVEY_DIR)"
	@echo ""
	@echo "Output Files:"
	@echo "  RECORD_IDENTIFIERS     = $(RECORD_IDENTIFIERS)"
	@echo "  METABOLITE_MAP         = $(METABOLITE_MAP)"
	@echo "  SURVEY_MAP             = $(SURVEY_MAP)"
	@echo "  IMAGE_MAP              = $(IMAGE_MAP)"
	@echo "  IMAGE_MAP_RESIZED      = $(IMAGE_MAP_RESIZED)"
	@echo "  ALL_DATA_JSON          = $(ALL_DATA_JSON)"
	@echo ""
	@echo "Options:"
	@echo "  OVERWRITE          = $(OVERWRITE)"
	@echo "  TARGET_WIDTH       = $(TARGET_WIDTH)"
	@echo "  TARGET_HEIGHT      = $(TARGET_HEIGHT)"
	@echo "  MIN_SURVEY_VOTES   = $(MIN_SURVEY_VOTES)"
	@echo "  DETERMINISTIC      = $(DETERMINISTIC)"
	@echo "  SEED               = $(SEED)"
	@echo ""
