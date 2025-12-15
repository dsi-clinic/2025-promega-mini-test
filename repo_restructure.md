# Organoid Data Pipeline – Preprocessing, mmseg, and Merging

This document summarizes how image, mask, metabolite, and survey data move through the pipeline, and where the final `merge_all_data` step should sit.

---

organoid-pipeline/
├─ config/
│  ├─ paths.yaml              # root paths, mmseg config refs, env vars
│  ├─ ids.yaml                # which ID is canonical, pointers to identifiers.json
│  └─ mmseg_train_config.py   # model + loss configuration (imported by train scripts)
│
├─ ids/
│  ├─ retrieve_main_identifiers.py   # builds identifiers.json from verification CSV
│  ├─ image_mapper.py                # maps raw images → IDs + QC
│  ├─ metabolite_mapper.py           # Excel → ID-keyed metabolites
│  ├─ survey_mapper.py               # Excel → ID-keyed surveys
│  └─ utils_id_normalization.py      # clean_id_for_json, main_id helpers, etc.
│
├─ mmseg_pipeline/
│  ├─ prep_train/
│  │  ├─ make_train_subset.py        # select manual-mask subset
│  │  └─ resize_train_data.py        # uniform H×W for training (images + masks)
│  ├─ train/
│  │  └─ train_mmseg.py              # main training entrypoint
│  ├─ infer_all/
│  │  ├─ resize_all_for_inference.py # resize ALL images to mmseg input size
│  │  └─ predict_all_masks.py        # run mmseg on all images
│  └─ postprocess/
│     ├─ build_complete_series.py    # filter by blanks, edge thresholds, etc.
│     └─ resize_aspect_ratio.py      # aspect-ratio–preserving resize + remap
│
├─ merge/
│  └─ merge_all_data.py              # final merge: images/masks + metabolites + surveys
│
├─ docs/
│  ├─ PIPELINE_OVERVIEW.md           # the high-level description you just wrote
│  └─ ID_CONTRACT.md                 # “record_id vs main_id” decision, canonical key
│
└─ notebooks/
   └─ exploratory/                   # ad hoc analysis, sanity checks

---

## 0. Modalities and IDs

- **Images**: raw microscopy images (per organoid, per timepoint).
- **Manual masks**: hand-drawn masks for a subset of images (for training).
- **Metabolites**: Excel → metabolite mapper → JSON keyed by ID.
- **Surveys**: Excel → survey mapper → JSON keyed by ID.
- **Image mapping**: image mapper → JSON mapping IDs → image metadata.

All image-related processing assumes a **canonical organoid ID** (e.g. `main_id` or `record_id`), used consistently across mappers.

---

## 1. Raw Images → Image Mapper

Purpose: connect raw images to canonical IDs and verification/QC.

- Input: raw image folders + verification CSV.
- Script: `image_mapper`.
- Output: `image_mapping.json` (or similar) containing, for each organoid ID:
  - locations of raw images
  - verification info (blank, blank_verified, etc.)
  - ID fields (record_id / main_id variants).

This is the first step that turns “files on disk” into structured metadata.

---

## 2. Preprocessing for mmseg Training (Subset with Manual Masks)

Purpose: create a small, uniform training set for mmseg.

- Take a subset of organoids with **manual masks**.
- Preprocess these images:
  - resize/crop to a **uniform size** (fixed H×W) for mmseg
  - apply the same transform to the corresponding manual masks.
- Map resized images and masks back to the canonical IDs.
- Result: a training dataset `{ID → (resized_raw_image, resized_manual_mask)}`.

This dataset is used only for training, not for full-dataset inference.

---

## 3. Train mmseg

Purpose: learn a segmentation model from the preprocessed subset.

- Use the preprocessed subset (uniform images + manual masks).
- Configure mmseg (model, loss functions, etc.).
- Train until satisfactory performance.

Output: a trained mmseg model capable of predicting masks on any resized image.

---

## 4. Preprocess All Images for mmseg Inference (Uniform Size)

Purpose: get **every** image into a form mmseg can predict on.

- Take **all** organoid images from `image_mapping.json`.
- Preprocess them to the same **uniform size** used in training.
- Result: `{ID, timepoint → resized_image_for_inference}` for all data.

This creates a complete inference-ready image set.

---

## 5. mmseg Prediction on All Images

Purpose: generate masks for the full dataset.

- Run the trained mmseg model on all resized images.
- Save predicted masks alongside the resized images.
- Result: `{ID, timepoint → (resized_image, predicted_resized_mask)}` for all images.

At this point, every timepoint has a uniform-size image and mask.

---

## 6. Time-Series Filtering (Complete Series JSON)

Purpose: select organoids with usable, complete time series.

- Input: `{ID, timepoint → resized_image + predicted_mask}` plus verification/QC info.
- Script: “complete series” script:
  - drops organoids that are:
    - blank
    - below an edge threshold
    - missing too many timepoints
  - keeps organoids with complete/usable series.
- Output: `complete_series.json`:
  - a filtered mapping of IDs → time series that pass QC.

This is a **subset** of the full dataset.

---

## 7. Aspect-Ratio–Conserving Resize and Remap

Purpose: get images and masks into a final, analysis-friendly size while preserving aspect ratio.

- Script: resize+remap function.
- Current behavior:
  - uses `complete_series.json` as input.
  - resizes images and masks with **aspect ratio preserved**.
  - generates new paths and mappings for these resized assets.
- Planned change (recommended):
  - run this on **all data**, not just `complete_series.json`, so the final mapping covers:
    - all valid organoids
    - all timepoints
    - with both:
      - uniform mmseg size (for model)
      - aspect-ratio-preserved size (for analysis/visualization).

Result: a final image/mask mapping JSON where each ID has:
- uniform-size image/mask (mmseg scale)
- aspect-ratio–conserved image/mask (analysis scale)

---

## 8. Metabolite and Survey Pipelines (Independent, Non-Image)

These run in parallel and do **not** touch image pixels.

- **Metabolite mapper**:
  - Excel → metabolite JSON keyed by organoid ID.
- **Survey mapper**:
  - Excel → survey JSON keyed by organoid ID.
- Both rely on the same ID convention used in the image mapping.

They remain independent of image resizing and mmseg, but must share the same IDs.

---

## 9. Where `merge_all_data` Should Sit

Current idea: `merge_all_data` should happen **after all image processing** is stable.

Recommended placement:

- Inputs to `merge_all_data`:
  - final image/mask mapping JSON:
    - includes mmseg-uniform and aspect-ratio–preserved paths
  - metabolite JSON
  - survey JSON
  - any other modality-specific mappings
- `merge_all_data` joins everything on the **canonical organoid ID**.

That way:
- the merged structure always points to the **final** image/mask locations.
- you don’t have to re-merge every time the image preprocessing changes.
- downstream analysis always sees consistent paths and resolutions.

---
