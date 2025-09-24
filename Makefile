.PHONY: data clean help

# image mapper
data-image-mapper:
	PYTHONPATH=. conda run -p /net/projects2/promega python file_utils/images/image_mapper_main.py

data-metabolite-mapper:
	PYTHONPATH=. conda run -p /net/projects2/promega python file_utils/metabolites/metabolite_mapper.py

data-surveys-mapper:
	PYTHONPATH=. conda run -p /net/projects2/promega python file_utils/surveys/surveys_mapper.py

analysis-images-resize:
	PYTHONPATH=. conda run -p /net/projects2/promega \
	    python analysis/images/resize/resize_remap_images.py \
	    --batches $(BATCHES) --days $(DAYS)

train-images-segmentation:
	????
	PYTHONPATH=. conda run -p /net/projects2/promega python analysis/images/segmentation_mmseg/train.py ADD ARGS HERE

.PHONY: analysis-images-predict-mapping
analysis-images-predict-mapping:
	PYTHONPATH=. conda run -n mmcv_env \
	  python analysis/images/segmentation_mmseg/predict_masks.py \
	    --model_type $(MODEL_TYPE) \
	    --batches $(BATCHES) \
	    --days $(DAYS) \
	    $(if $(OVERWRITE),--overwrite,) \
	    $(if $(DRY_RUN),--dry-run,) \
	    $(if $(SMOKE),--smoke $(SMOKE),)


data-merge:
	PYTHONPATH=. conda run -p /net/projects2/promega python file_utils/merge/merge_all_data.py

# Generate the master data file with plate identifiers preserved
data: data-image-mapper data-metabolite-mapper data-surveys-mapper data-merge


# Below is related to the classification problem
train-classifier:
	PYTHONPATH=. conda run -p /net/projects2/promega python analysis/images/classifier/train_model_accuracy.py