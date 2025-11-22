Notes on draft branch:

in_dir  → contains data to read (images, masks)
out_dir → contains metadata to interpret those images

CLI:
--in-dir /path/to/shared/project/data
--out-dir /home/<user>/my_run/


1. in_dir = all orginal and processed data

Contains:
- the images
- images/raw_images/
- images/infer_resized_512x384/
- masks
- masks/predicted/
- masks/manual/
- masks/image_overlays/
- the mapping JSONs
- image_mapping.json
- metabolite_map.json
- organoid_surveys_aggregated.json
- preprocessed/*.json


2. out_dir = user-specific output folder

This contains only the derived JSONs:
- all_data.json
- image_classifier.json
- survey_classifier.json
- anything else generated downstream


# Dataset Structure Overview

## `in_dir` = Canonical Dataset (Shared, Read-Only)

This directory contains **all raw and derived input artifacts** needed to build the unified metadata table.  
It lives in shared PI storage (e.g., `/net/projects2/...`).

Contents include:

### Raw Data
- `images/`
- `masks/`

### Processed / Preprocessed JSONs
Generated during inference and preprocessing:
- `images/infer_resized_*/.../image_mapping_*_processed.json`
- `json/preprocessed/*`

### Mapping JSONs (Global Metadata)
Stored in `in_dir/json/`:
- `image_mapping.json`
- `image_mapping_thresholded_and_manual.json`
- `metabolite_map.json`
- `organoid_surveys_aggregated.json`

### Access Control
- **Read-only** for most collaborators  
- Editable by Amanda, Nikki, Nick, Liya
- Represents the authoritative “source of truth” for the dataset

---

## `out_dir` = Per-User Derived Metadata (Local, Writable)

Each user has their own output directory (e.g., `/home/amandabrooke/...`).  
Here the merge script writes all derived, merged metadata tables.

### Key Output Files
- `all_data.json`  
  - Master “table of contents” for the dataset
  - Defines:
    - Which organoids exist
    - How each maps to images and masks
    - Survey labels
    - Metabolite information
    - Manual mask associations
  - Contains clean schema for model training

- `image_classifier.json`
- `survey_classifier.json`

###################################################################
# Prop org

1. Ingest mappers
2025-promega-mini-test/
├── ingest_data/
│   ├── build_image_json.py          # raw image data + metadata cleaning
│   ├── build_survey_json.py         # from raw survey exports → aggregated json
│   ├── build_metabolite_json.py     # from Excel → metabolite_map.json
│
├── preprocess/
│   ├── make_canonical_layout.py     # raw → canonical `input/` tree on cluster
│   ├── resize_for_canonical.py      # raw → infer_resized_512x384 PNGs
│   ├── sync_manual_masks.py         # copy/import manual masks into canonical layout
│   └── README.md
│
├── segmentation/
│   ├── train.py                     # train seg model(s)
│   ├── infer_to_canonical.py        # run seg model → write masks/overlays + processed JSONs INTO canonical input
│   ├── datasets.py                  # seg-specific dataloaders (runtime resizing, aug)
│   └── README.md
│
├── integration/
│   ├── merge_all_data.py            # builds all_data.json, image/survey views
│   ├── normalized_records.py        # OrganoidRecordBuilder, emit_views, metrics
│   ├── classifiers/
│   │   ├── train_image_classifier.py
│   │   ├── train_survey_classifier.py
│   │   ├── datasets.py              # uses all_data.json, does classifier-only resize
│   │   └── eval.py
│   └── README.md
│
├── file_utils/
│   ├── common/
│   │   ├── organoid_patterns.py     # OrganoidNormalizer etc.
│   │   └── io.py                    # path helpers, loading helpers
│   └── __init__.py
│
├── configs/
│   ├── core_env.yaml
│   ├── merge_all_data.yaml          # optional default args
│   └── model_configs/...
│
├── scripts/
│   ├── run_merge_all.sh
│   ├── run_segmentation.sh
│   └── run_classifiers.sh
└── README.md
