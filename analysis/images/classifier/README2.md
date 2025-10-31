# 🧬 Image Classifier — Code & Files (Deep Technical README)

# 0) Quick Locator: Where things live

```
analysis/
└─ images/
   ├─ classifier/                    # (this README focuses here)
   ├─ quality/                       # overlay generation + QC helpers
   ├─ resize/                        # deterministic resize/remap of images
   ├─ segmentation_mmseg/            # MMSeg configs, inference, training
   ├─ series/                        # time-series filtering/normalization
   ├─ metrics/                       # plotting utilities
   ├─ shape_metrics/                 # shape stats (not core to classifier)
   └─ stitching/                     # utilities to stitch chosen images to panels
```

---

# 1) `analysis/images/classifier/` — Module Overview

The classifier trains **per-day** 2-class models (Accepted vs Not) using different **input modes** (RGB, overlay, mask combinations) and **label modes** (hard vs soft). It saves **checkpoints**, **metrics JSON/CSVs**, and optional **misclassification panels**.

### 1.1 Directory Tree (what you will actually see)

```
analysis/images/classifier/
├─ data/                                   # All inputs the trainers read (already split by day/label mode)
│  └─ preprocessed/512x384/                # Canonical image size used by classifiers
│     ├─ Dy03/ Dy06/ Dy08/ ... Dy30/       # One folder per imaging day (each trains a separate model)
│     │  ├─ majority.json                  # Hard labels via majority rule (≥4/5 “Accepted” votes)
│     │  ├─ complete.json                  # Hard labels with unanimous agreement (5/5)
│     │  └─ raw_votes.json                 # Soft labels: fraction of “Accepted” votes in [0,1]
│     └─ (same structure if 256x192 runs are used)  # Alternative lower-res variant for quick sweeps
│
├─ logs/                                   # SLURM job outputs (stdout/err) from run_*.s; useful for debugging
│
├─ misclassified_images/                   # PNG contact sheets of errors (visual QC for wrong predictions)
├─ misclassifiedimages/                    # CSV tables listing each misclassified sample + metadata
│
├─ outputs_*                               # All experiment outputs (weights, metrics, plots) grouped by variant
│  ├─ outputs_nomask_image*/               # RGB-only hard-label runs (baseline morphology)
│  ├─ outputs_nomask_overlay*/             # Overlay-only hard-label runs (fluorescence-derived grids)
│  ├─ outputs_mask_image*/                 # RGB + mask hard-label runs (geometry prior added)
│  ├─ outputs_mask_overlay*/               # Overlay + mask hard-label runs (bio signal + geometry)
│  ├─ outputs_512x384_softlabels*/         # Soft-label runs (probabilistic targets + calibration metrics)
│  ├─ outputs_512x384_*train_augment*      # Runs with explicit data augmentation + AUROC tracking
│  └─ ...                                  # Other historical/alternate naming roots used in the repo
│
├─ stitched/                               # Contact sheets built from arbitrary CSV selections (not only errors)
│
├─ HOW_TO_RUN.MD                           # Operational quick-start: commands to generate data/train/evaluate
├─ README.md                               # Code-centric README (this one): what each file does and why
│
├─ aggegate_mis.py                         # Merge per-run misclassification CSVs into project-level summaries
├─ aggregated_misclassified_by_all_models.csv   # Rows misclassified by every backbone (hard cases)
├─ aggregated_misclassified_details.csv          # Long-form table of misclassifications with run/day/backbone
│
├─ generate_training_metadata.py           # Builds per-day JSONs (majority/complete/raw_votes) + train/val/test splits
├─ find_misclassified_images.py            # Extracts misclassified rows and (optionally) triggers panel stitching
├─ per_model_summary.py                    # Collates per-day metrics into summary JSON/CSVs + model comparison plots
├─ plot_auc_curve.py                       # Recomputes and plots ROC curves from saved predictions per day/backbone
├─ plot_model_metrics.py                   # Draws accuracy/F1/ROC-AUC by day across backbones for a run root
│
├─ train_model_accuracy.py                 # Main hard-label trainer (ViT/ResNet/EfficientNet; optional masks; AUROC)
├─ train_model_enhanced.py                 # Hard-label trainer with two-tier aug + deterministic flips/normalization
├─ train_model_soft_labels.py              # Soft-label trainer (BCE on fractions, Brier/RMSE, PR-AUC, thresh sweep)
├─ train_model_deep_ensemble.py            # Per-day backbone sweep (no/low aug), pick best by validation metric
│
├─ run_accuracy.s                          # SLURM wrapper for a standard hard-label sweep (sets env/paths/args)
├─ run_nomask_image.s                      # SLURM: RGB-only hard-label job launcher
├─ run_nomask_overlay.s                    # SLURM: overlay-only hard-label job launcher
├─ run_mask_image.s                        # SLURM: RGB + mask hard-label job launcher
├─ run_mask_overlay.s                      # SLURM: overlay + mask hard-label job launcher
├─ run_soft_label.s                        # SLURM: soft-label (raw_votes) job launcher
└─ run_all.s                               # SLURM meta-runner that submits all four hard-label variants
```

---

# 2) Core Data Objects

### 2.1 Preprocessed JSON (by day)

Each file in `data/preprocessed/512x384/DyXX/*.json` is a list of records:

```json
{
  "organoid_key": "BA1 96_1 Dy10 A1",
  "img_path": "/abs/path/to/brightfield_or_overlay.tif",
  "mask_path": "/abs/path/to/mask.tif",                 // optional
  "label_hard": 0,                                      // majority/complete
  "label_soft": 0.8,                                    // raw_votes (0..1)
  "channels": "rgb|overlay|rgb+mask|overlay+mask",
  "split": "train|val|test"                             // 0.8/0.1/0.1 split
}
```

Generated by `generate_training_metadata.py` with:

* **vote strategy**: `majority` (≥4/5), `complete` (5/5), `raw_votes` (fraction).
* **day normalization**: merges Dy20/Dy21 → `Dy20_5` for analysis parity.

---

# 3) Script-by-Script (Classifier)

Below are **the files you’ll read and modify most** and **how they work**.

## 3.1 `generate_training_metadata.py`

**Purpose:** Build per-day JSON datasets with label modes and splits.

**Inputs:**

* `/net/projects2/promega/data-analysis/output/all_data.json`
* Env/config from `config.py` (paths), expects image/mask paths to exist.

**Key CLI (typical):**

* `--all` (builds all days, all label modes)
* `--classification` (restricts to classifier-relevant fields)
* `--emit-raw-votes` (add soft fractional targets)
* `--raw-votes-min-n` (minimum votes to accept a soft label)

**Core logic:**

* Parses merged organoid records.
* Infers per-day **hard labels** (majority / complete) and **soft** (fraction).
* Normalizes day names (e.g., Dy20, Dy21 → `Dy20_5`).
* Splits into train/val/test with a fixed seed for reproducibility.
* Writes `DyXX/{majority|complete|raw_votes}.json`.

**Outputs:**

* `analysis/images/classifier/data/preprocessed/512x384/DyXX/*.json`

---

## 3.2 `train_model_accuracy.py`  *(hard labels; main baseline)*

**Purpose:** Train per-day **binary** classifiers (hard labels) with AUROC tracking and optional mask branch.

**Backbones:** ViT-B/16 (`timm`), ResNet-50 (`torchvision`), EfficientNet-B0 (`timm`), and a light custom CNN (if enabled).

**Inputs:**

* Per-day JSON from `data/preprocessed/512x384/DyXX/majority.json` or `complete.json`.
* Image tensors built as:

  * **RGB**: 3 channels
  * **Overlay**: 1–3 channels depending on preprocessed overlay encoding
  * **Mask**: appended as an extra channel if `--use-mask` is on

**Key args you’ll see in runners:**

* `--backbones vit resnet efficientnet`
* `--use-mask` (turn on 4th channel or parallel mask branch)
* `--augment` (random flips/jitter; some runs disable it)
* `--batch-size`, `--epochs`, `--lr`, `--patience`
* `--track-auroc` (log AUROC each epoch)

**Training loop:**

* BCEWithLogitsLoss on {0,1} labels.
* Optimizer: Adam or SGD (project defaults to Adam).
* LR schedule: step or plateau (check code block in file).
* Early stopping on **validation AUROC** or loss.
* Saves **best checkpoint** by validation metric.

**Outputs (typical root name depends on runner):**

* `outputs_nomask_image*/` or `outputs_mask_overlay*/` etc.

  * `vit/|resnet/|efficientnet/`

    * `DyXX/`

      * `model_best.pth` (weights)
      * `metrics_val.json`, `metrics_test.json`
      * `curves.png` (if enabled)

**Metrics (hard-label JSON):**

* `test_accuracy`, `test_f1`, `test_roc_auc`, `test_num`, `actual_good`, `predicted_good`
* `val_accuracy`, `val_roc_auc`

---

## 3.3 `train_model_soft_labels.py`  *(soft labels; uncertainty-aware)*

**Purpose:** Train per-day models with **fractional targets** (probabilities from votes).

**Loss/metrics:**

* BCE (with fractional target ∈ [0,1])
* `brier` (calibration error)
* `rmse`  (prediction vs target probability)
* AUROC, PR-AUC for ranking
* Threshold sweep to report:

  * `acc@0.5`, `f1@0.5`
  * `acc@t*`, `f1@t*` (best over validation)
  * `threshold_used` + `threshold_method` (e.g., `max_f1`)

**Inputs:**

* `data/preprocessed/512x384/DyXX/raw_votes.json` (must exist)

**Outputs:**

* `outputs_512x384_softlabels*/soft_{vit|resnet|efficientnet}/DyXX/metrics_*.json`, `model_best.pth`.

---

## 3.4 `train_model_deep_ensemble.py`  *(per-day backbone sweep)*

**Purpose:** For each day, train **multiple backbones** (usually **no augmentation**), early-stop, and **pick the best** by validation accuracy. Produces a per-day “winner” summary.

**What it does:**

* Loops over backbones: ViT, ResNet, EfficientNet.
* Trains each quickly with identical splits.
* Selects the best-performing backbone per day.
* Writes aggregate CSV/JSON of winners and test metrics.

**Outputs:**

* `outputs_512x384_Regular_image_without_train_augment/`

  * `vit/`, `resnet/`, `efficientnet/` trees
  * Per-day CSVs summarizing misclassified examples
  * Optional `*_misclassified_by_all_models.csv` to show overlap

---

## 3.5 `train_model_enhanced.py`  *(strong single-model baseline)*

**Purpose:** Hard-label training with stronger **two-tier augmentation**:

* Geometric (flips/rotations)
* Photometric (brightness/contrast/jitter)
  Plus deterministic flip duplication & configurable normalization. Useful for stress-testing augmentation hypotheses.

**Outputs:** typically `outputs_512x384_two_level_aug/` (if that root is used).

---

## 3.6 `find_misclassified_images.py`

**Purpose:** Generate CSVs of misclassified samples for a given trained model/day, and optionally **stitch** panels for review.

**Inputs:**

* A trained run folder (e.g., `outputs_nomask_image/vit/Dy20_5/model_best.pth` + `metrics_test.json` + internal cached predictions).
* Auto-discovers ground-truth JSONs under `data/preprocessed/...`.

**Outputs:**

* `misclassifiedimages/*.csv` rows with:

  * organoid_key, y_true, y_pred, p, day, backbone, run_root
* Optionally triggers stitching (see `stitching/` utilities) into `misclassified_images/*.png`.

---

## 3.7 `aggegate_mis.py`

**Purpose:** Merge per-run/per-day misclassification CSVs into project-level summaries:

* `aggregated_misclassified_by_all_models.csv` → samples **missed by every model** in a family
* `aggregated_misclassified_details.csv` → long-form audit table

Useful for **intersection/union** error analysis across experiments.

---

## 3.8 `per_model_summary.py`

**Purpose:** Summarize accuracy/F1/AUROC **by day** for each backbone in a run root into a single CSV/JSON and optionally plot per-day curves.

**Reads:**

* `*/<backbone>/DyXX/metrics_test.json`
* `*/<backbone>/DyXX/metrics_val.json`

**Writes/Displays:**

* `final_test_summary.json`
* `accuracy_by_model.png`, `f1_by_model.png`, `rocauc_by_model.png`
* `day_summary.csv` (counts per split per day)

---

## 3.9 Plotters

### `plot_model_metrics.py`

* Batch-loads all `metrics_*` JSONs in a run root and draws:

  * Per-day accuracy/F1 lines per backbone
  * Optional error bars (if repeated seeds exist)
* Saves `*_accuracy_by_day.png`, `*_metrics_by_day.png`

### `plot_auc_curve.py`

* Recomputes ROC from saved logits/probabilities
* Plots ROC per day, per backbone
* Saves `roc_auc_by_day.png` or per-day ROC PDFs

---

# 4) Runner Scripts (SLURM)

These are **thin wrappers** that:

1. activate the right environment (`/net/projects2/promega` or `mmcv_env`),
2. set `PROJ_ROOT`,
3. call the appropriate `train_model_*.py` with arguments.

* `run_nomask_image.s`      → RGB hard-label training
* `run_nomask_overlay.s`    → overlay-only hard-label training
* `run_mask_image.s`        → RGB + mask hard-label training
* `run_mask_overlay.s`      → overlay + mask hard-label training
* `run_soft_label.s`        → soft-label training on `raw_votes.json`
* `run_all.s`               → submits the four **hard-label** variants

> These write to different `outputs_*` roots so you can compare variants side by side (see next section).

---

# 5) Outputs — How to read them

### 5.1 Output roots (what each means)

| Output root                                       | Inputs         | Labels | Augment | Notes                        |
| ------------------------------------------------- | -------------- | ------ | ------- | ---------------------------- |
| `outputs_nomask_image*`                           | RGB            | Hard   | varies  | “baseline RGB” family        |
| `outputs_nomask_overlay*`                         | Overlay only   | Hard   | varies  | fluorescence channel(s) only |
| `outputs_mask_image*`                             | RGB + mask     | Hard   | varies  | mask as extra channel/branch |
| `outputs_mask_overlay*`                           | Overlay + mask | Hard   | varies  | joint biological + geometry  |
| `outputs_512x384_softlabels*`                     | RGB            | Soft   | varies  | brier/rmse/pr_auc/thresholds |
| `outputs_512x384_Regular_image_*`                 | RGB            | Hard   | varies  | alternate naming in old runs |
| `outputs_512x384_*with_train_augment*_with_auroc` | varies         | Hard   | **on**  | explicit AUROC logging       |
| `outputs_512x384_without_train_data_augmentation` | varies         | Hard   | **off** | disable augmentation         |

> Inside each root: `vit/`, `resnet/`, `efficientnet/` → each has `DyXX/` subfolders with `metrics_*.json` and `model_best.pth`.

### 5.2 Key files

* `final_test_summary.json` (top of each `outputs_*` root)
  Folded summary of every day & backbone. E.g.:

  ```json
  {
    "per_model": {
      "vit": { "per_day": { "Dy3": { "test_accuracy": ..., "test_roc_auc": ... }, ... } },
      "resnet": { "per_day": { ... } },
      "cnn": { "per_day": { ... } }
    },
    "batch_size_train": 16,
    "split_fractions": {"train": 0.8, "val": 0.1, "test": 0.1}
  }
  ```

* `DyXX/metrics_test.json` (per backbone)

  * **Hard-label runs:** `test_accuracy`, `test_f1`, `test_roc_auc`, `test_num`, `val_*`
  * **Soft-label runs:** `brier`, `rmse`, `roc_auc`, `pr_auc`, `acc@0.5`, `f1@0.5`, `acc@t*`, `f1@t*`, `threshold_used`, `threshold_method`, `val_brier_for_selection`

* `accuracy_by_model.png`, `f1_by_model.png`, `rocauc_by_model.png`
  Quick comparison by backbone across days.

---

# 6) Sibling Modules (High-Level)

These are **supporting** modules under `analysis/images/` you’ll reference but rarely edit when focused only on classification.

## 6.1 `quality/`

* **`image_mask_overlay.py`** — builds fluorescence *overlay images* from masks + intensities for visualization and the “overlay-only” classifier inputs.
* `manual_test.py`, `plate_test.py` — small harnesses to validate overlays in controlled subsets.
* `mask_edge_fraction.py` — checks how much mask area is near edges (sanity QC).
* `overlay_decode_failures.json` — records images where overlay decode failed.
* Notebooks `inspect_masks.ipynb`, `inspect_splits_stitched.ipynb` — EDA/QC.

## 6.2 `resize/`

* **`resize_remap_images.py`** — deterministic resize to **512×384** (or 256×192), updates JSON paths; ensures every model sees the same spatial size and pixel normalization.

## 6.3 `segmentation_mmseg/`

* **`predict_masks.py`** — MMSeg **inference** to generate/refresh `mask_path` used by classifier variants requiring masks.
* **`train.py`** + `segformer_mitb0.py` — MMSeg **training config** (SegFormer backbone).
* `mmcv_env.yaml` — the conda env name expected for MMSeg (separate from core).
* `mmseg_paths.py`, `env_config_handler.py` — binds repo paths to MMSeg.
* `datasets/`, `preprocessing/`, `utils/` — MMSeg data adapters and helpers.
* `perfs.md` — notes on mask model performance.

## 6.4 `series/`

* `filter_complete_series.py`, `preprocess_for_lstm.py` — build complete time series and normalization (used for sequence models; tangential to per-day classifiers).
* `check_sizes.py` — asserts consistent image sizes across days.
* Notebooks: `check_series.ipynb` — series sanity checks.

## 6.5 `metrics/` and `shape_metrics/`

* Plotting utilities (`metrics/plot_metrics.ipynb`) and basic **morphology features** (`shape_metrics/metrics.py`). Not required for running classifiers but useful for analysis.

## 6.6 `stitching/`

* **`stitch_images.py`** — takes a CSV of image paths/keys and creates contact-sheet PNGs. Used by `find_misclassified_images.py` to visualize errors.

---

Absolutely — here is a **cleaner**, **more intentional**, **scientifically motivated** rewrite of that section.
Instead of just listing terms, this follows the **real experimental flow of a single organoid** through the analysis pipeline — and **why** we built each branch and option.

You can drop this directly into your README:

---

# ✅ 7) Why We Have Multiple Model Variants (Biology → Imaging → ML Rationale)

Each organoid can be analyzed using **different signals** and **different assumptions**.
Every variant exists to answer a **distinct biological or modeling hypothesis**.

---

### 🧫 Step 1 — What visual information should we learn from?

| Input Type               | What it encodes                             | Scientific question it answers                              |
| ------------------------ | ------------------------------------------- | ----------------------------------------------------------- |
| **RGB (Brightfield)**    | Shape, texture, edges                       | *Does morphology alone predict viability?*                  |
| **Fluorescence Overlay** | Protein/gene expression via intensity grids | *Is biological function visible before structural changes?* |
| **Mask**                 | Boundary of the organoid                    | *If we remove background noise, does shape tell the story?* |

→ These form the **4 core input modes**:

| Variant            | What the model sees      | Why it exists                               |
| ------------------ | ------------------------ | ------------------------------------------- |
| **RGB only**       | Pure morphology          | Baseline; easiest real-world deployment     |
| **Overlay only**   | Pure biological signal   | Tests signal value independent of structure |
| **RGB + Mask**     | Morphology + shape prior | Focus on organoid; ignore plates, bubbles   |
| **Overlay + Mask** | Biology + geometry       | Best of both worlds; hypothesis-strong      |

✅ This is where you compare **biology vs physics** vs **both**.

---

### 🧪 Step 2 — How confident are we in the “ground truth”?

| Label Mode      | Values                                     | Why use it                                                  |
| --------------- | ------------------------------------------ | ----------------------------------------------------------- |
| **Hard Labels** | 0 or 1 only                                | For clear “Accepted/Rejected” downstream decisions          |
| **Soft Labels** | Real values in [0,1] = fraction good votes | Captures expert disagreement + removes artificial certainty |

Soft labeling allows the model to:

* learn confidence rather than force binary decisions
* improve **probability calibration**
* still support thresholding at deployment (e.g., 0.75 → accept)

✅ This is where we test **uncertainty vs black-and-white QC**.

---

### 🔁 Step 3 — Do we simulate biological variation?

| Factor                | Meaning              | Why use it                                      |
| --------------------- | -------------------- | ----------------------------------------------- |
| **Data Augmentation** | Rotation/flip/jitter | Generalization across microscopes & technicians |
| **No Augmentation**   | Pure data            | Evaluate absolute raw predictability            |

✅ This asks: *Is generalization required to avoid batch bias?*

---

### 🔍 Step 4 — Which backbone best extracts early-day signal?

| Model               | Strength                   | Difference in behavior                   |
| ------------------- | -------------------------- | ---------------------------------------- |
| **ResNet-50**       | Local textures             | Classic CNN inductive bias               |
| **EfficientNet-B0** | Efficiency                 | Good with small data                     |
| **ViT-B/16**        | Global relational features | Better if subtle spatial patterns matter |

✅ This tells us whether cues are **local or structural / distributed**.

---

### 📈 Step 5 — How do we measure success?

Each metric answers a different deployment-critical question:

| Metric                 | What it measures                     | Why it matters                           |
| ---------------------- | ------------------------------------ | ---------------------------------------- |
| **AUROC**              | Ranking signal — “who looks better?” | Best for early-day separation            |
| **F1**                 | Balance of false neg vs false pos    | Critical if “passing” is scarce          |
| **Accuracy**           | Raw proportion correct               | Simple interpretability                  |
| **PR-AUC** (soft only) | Performance under imbalance          | When “accepted” ratio varies across days |
| **Brier Score** (soft) | Probability calibration              | *Does 0.9 confidence mean 90% correct?*  |
| **RMSE** (soft)        | Deviation from true vote fraction    | Measures uncertainty modeling            |

✅ Together they show:

* how early we can predict…
* how reliably…
* and how **biologically meaningful** the confidence is.

---

### 🎯 The Scientific Story These Variants Answer

| Question                                        | Variant(s) that answer it  |
| ----------------------------------------------- | -------------------------- |
| *Are early structural cues enough?*             | RGB-only runs              |
| *Do early biological signals emerge first?*     | Overlay-only runs          |
| *Is geometry essential for classification?*     | Mask runs                  |
| *How do we handle uncertainty?*                 | Soft-label runs            |
| *When is morphology predictive?*                | Per-day AUROC curves       |
| *Which features generalize across experiments?* | Augment vs No-augment runs |
| *Which ML inductive biases align with biology?* | CNN vs ViT comparisons     |

✅ These aren’t duplicate models —
each is **testing a different scientific hypothesis**.

---

### 🧩 Compact Summary Flow

```
Image signal choice → (RGB / Overlay / Mask)
     ↓
Label confidence choice → (Hard / Soft)
     ↓
Generalization need → (Augment / No-augment)
     ↓
Inductive bias → (ViT / ResNet / EfficientNet)
     ↓
Per-day models → (Dy03 → Dy30)
     ↓
Metrics → Determine earliest reliable QC day
```

---

# 8) Typical Code Flow (Classifier)

1. `generate_training_metadata.py` → writes `data/preprocessed/512x384/DyXX/*.json` (majority, complete, raw_votes).
2. A **runner** (e.g., `run_mask_overlay.s`) calls:

   * `train_model_accuracy.py` with `--use-mask` and overlay input OR
   * `train_model_soft_labels.py` for `raw_votes`.
3. The trainer loads data, **augments** (if enabled), trains **per-day** models for each backbone, and writes:

   * `outputs_*/<backbone>/DyXX/model_best.pth`
   * `metrics_val.json`, `metrics_test.json`, plots.
4. `find_misclassified_images.py` → CSVs + optional stitched PNGs of errors.
5. `per_model_summary.py`, `plot_model_metrics.py`, `plot_auc_curve.py` → **final summaries and plots**.
