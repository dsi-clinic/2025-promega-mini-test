
# -------- Defaults (override at call time) --------
BATCHES     ?= 1
DAYS        ?= 03
OVERWRITE   ?=
DRY_RUN     ?=
SMOKE       ?=
MODEL_TYPE  ?= early     # early|late

PHASE        ?= late
DATE         ?= $(shell date +%Y%m%d)
MODEL_CONFIG ?= segformer_mitb0.py
WORK_ROOT    ?= /net/projects2/promega/data-analysis/plots/segformer_masks
WORK_DIR     ?= $(WORK_ROOT)/$(DATE)/$(PHASE)
GPU          ?=
RESUME_FROM  ?=
EXTRA        ?=

.PHONY: help clean \
        data data-image-mapper data-metabolite-mapper data-surveys-mapper data-merge \
        analysis-images-resize \
        analysis-images-predict-mapping \
        analysis-images-overlays \
        train-images-segmentation \
        train-classifier train-survey-classifier

# -------- Help --------
help:
	@echo "Targets:"
	@echo "  data                              - run all mappers then merge"
	@echo "  analysis-images-resize            - resize + remap raw images"
	@echo "  analysis-images-predict-mapping   - run mmseg inference (mmcv_env)"
	@echo "  analysis-images-overlays          - build overlays from predictions"
	@echo "  train-images-segmentation         - train mmseg model (mmcv_env)"
	@echo "  train-classifier                  - train image classifier (core_env)"
	@echo "  train-survey-classifier           - train survey classifier (core_env)"
	@echo
	@echo "Examples:"
	@echo "  make analysis-images-resize BATCHES=1,2 DAYS=03 OVERWRITE=1"
	@echo "  make analysis-images-predict-mapping MODEL_TYPE=late BATCHES=1 DAYS=28 OVERWRITE=1"
	@echo "  make analysis-images-overlays OVERWRITE=1 STRICT=1"
	@echo "  make train-images-segmentation PHASE=late MODEL_CONFIG=segformer_mitb0.py DATE=20250815"
	@echo "  make train-classifier"
	@echo "  make train-survey-classifier"

# =========================
# 1) Mappers (core_env)
# =========================
data-image-mapper:
	PYTHONPATH=. conda run -p /net/projects2/promega \
	    python file_utils/images/image_mapper_main.py

data-metabolite-mapper:
	PYTHONPATH=. conda run -p /net/projects2/promega \
	    python file_utils/metabolites/metabolite_mapper.py

data-surveys-mapper:
	PYTHONPATH=. conda run -p /net/projects2/promega \
	    python file_utils/surveys/surveys_mapper.py

data-merge:
	PYTHONPATH=. conda run -p /net/projects2/promega \
	    python file_utils/merge/merge_all_data.py

data: data-image-mapper data-metabolite-mapper data-surveys-mapper data-merge

# =========================
# 2) Image prep (core_env)
# =========================
analysis-images-resize:
	PYTHONPATH=. conda run -p /net/projects2/promega \
	    python analysis/images/resize/resize_remap_images.py \
	    --batches $(BATCHES) --days $(DAYS) $(if $(OVERWRITE),--overwrite)

# =========================
# 3) Inference (mmcv_env)
# =========================
analysis-images-predict-mapping:
	PYTHONPATH=. conda run -n mmcv_env \
	  python analysis/images/segmentation_mmseg/predict_masks.py \
	    --model_type $(MODEL_TYPE) \
	    --batches $(BATCHES) \
	    --days $(DAYS) \
	    $(if $(OVERWRITE),--overwrite) \
	    $(if $(DRY_RUN),--dry-run) \
	    $(if $(SMOKE),--smoke $(SMOKE))

# =========================
# 4) Overlays (core_env)
# =========================
analysis-images-overlays:
	PYTHONPATH=. conda run -p /net/projects2/promega \
	  python analysis/images/quality/image_mask_overlay.py \
	    $(if $(OVERWRITE),--overwrite) \
	    $(if $(STRICT),--strict)

# =========================
# 5) Training (mmcv_env)
# =========================
train-images-segmentation:
	@echo ">>> Training cfg: $(MODEL_CONFIG)"
	@echo ">>> Work dir    : $(WORK_DIR)"
	@mkdir -p "$(WORK_DIR)"
	PYTHONPATH=. conda run -n mmcv_env \
	  python analysis/images/segmentation_mmseg/train.py \
	    $(MODEL_CONFIG) \
	    --work-dir "$(WORK_DIR)" \
	    $(if $(RESUME_FROM),--resume-from "$(RESUME_FROM)") \
	    $(if $(GPU),--gpu-ids $(GPU)) \
	    $(EXTRA)

# =========================
# 6) Classification (core_env)
# =========================
train-classifier:
	PYTHONPATH=. conda run -p /net/projects2/promega \
	  python analysis/images/classifier/train_model_accuracy.py

train-survey-classifier:
	PYTHONPATH=. conda run -p /net/projects2/promega \
	  python analysis/surveys/classifier/simple_classifier.py

# -------- Clean (stub) --------
clean:
	@echo "Add rm -rf commands here if you want a cleaning step."
