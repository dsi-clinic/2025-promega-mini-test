# 🧬 Image Classifier Module — Full Technical Guide
This guides new contributors through:

✅ End-to-end execution of image classifiers
✅ Understanding all model variants
✅ Interpreting outputs and performance
✅ Project-specific terminology

---

## 📌 Execution Overview (TL;DR)

From **repo root:**

```bash
cd ~/2025-promega-mini-test
```

---

### ✅ Step 1 — Build the unified dataset

Merges all image + survey metadata into JSON:

```bash
/net/projects2/promega/bin/python file_utils/merge/merge_all_data.py
```

📍 Output:

```
/net/projects2/promega/data-analysis/output/all_data.json
```

---

### ✅ Step 2 — Create classifier input metadata

Generates per-day classification JSON (all label modes):

```bash
/net/projects2/promega/bin/python analysis/images/classifier/generate_training_metadata.py \
  --all /net/projects2/promega/data-analysis/output/all_data.json \
  --emit-raw-votes --raw-votes-min-n 1
```

📍 Output root:

```
analysis/images/classifier/data/preprocessed/512x384/
```

Contains:

* **majority** (hard labels)
* **complete** (strict hard labels)
* **raw_votes** (soft labels)

---

### ✅ Step 3 — Submit SLURM training jobs

Move into classifier module:

```bash
cd analysis/images/classifier
```

#### 🔹 One-shot runs

```bash
# RGB only
sbatch run_nomask_image.s

# Fluorescent overlay only
sbatch run_nomask_overlay.s

# RGB + segmentation mask
sbatch run_mask_image.s

# Overlay + mask
sbatch run_mask_overlay.s

# Soft-label (probabilistic) training
sbatch run_soft_label.s
```

#### 🔹 All four hard-label variants at once

```bash
sbatch run_all.s
```

---

### ✅ Step 4 — Monitor progress

```bash
squeue -u $USER
tail -f logs/train-img-nomask_<JOBID>.out
tail -f logs/train-overlay-mask_<JOBID>.out
tail -f logs/soft-label_<JOBID>.out
```

---

## 🎯 What Models Are Being Trained?

Every script trains the same **backbones**:

| Backbone                          | Family         | Strength                   |
| --------------------------------- | -------------- | -------------------------- |
| **Vision Transformer (ViT-B/16)** | Self-attention | Global morphology patterns |
| **ResNet-50**                     | Convolutional  | Texture + spatial cues     |
| **EfficientNet-B0**               | Convolutional  | Light-weight, efficient    |

Training is:

* Per-day (**11 models per backbone**)
* 80/10/10 train/val/test
* With or without augmentation
* Binary Cross-Entropy (**hard**) or BCE on probabilities (**soft**)
* Early stopping enabled

---

## 🧩 Model Input Modes

| Variant        | Script                 | Inputs                       | Why                        |
| -------------- | ---------------------- | ---------------------------- | -------------------------- |
| RGB            | `run_nomask_image.s`   | Brightfield                  | Baseline morphology        |
| Overlay only   | `run_nomask_overlay.s` | Fluorescence                 | Biological signal emphasis |
| RGB + Mask     | `run_mask_image.s`     | 4-channel image              | Explicit organoid boundary |
| Overlay + Mask | `run_mask_overlay.s`   | Structural + bio signals     | Combined hypothesis        |
| Soft-label RGB | `run_soft_label.s`     | Probabilistic success target | Uncertainty-aware          |

---

## 🧠 Label Strategies

| Label Mode    | Source        | Range             | Purpose                            |
| ------------- | ------------- | ----------------- | ---------------------------------- |
| **majority**  | ≥4/5 raters   | {0,1}             | Standard QC training               |
| **complete**  | 5/5 unanimous | {0,1}             | High-confidence subset             |
| **raw_votes** | Fraction good | [0,1] real-valued | Probabilistic labels & calibration |

Soft-label outputs include:

* Brier score
* PR-AUC
* Test accuracy at 0.5 **and** optimal threshold (`acc@t*`)

Hard-label outputs include:

* Test AUROC
* Accuracy / F1
* Actual vs predicted good counts

---

## 📂 Output Organization

Everything lands under:

```
analysis/images/classifier/outputs_*/
```

Each family contains:

| File                      | Meaning                                   |
| ------------------------- | ----------------------------------------- |
| `accuracy_by_model.png`   | Per-day accuracy curves and model winners |
| `f1_by_model.png`         | Class balance performance                 |
| `rocauc_by_model.png`     | Best measure of ranking/separation        |
| `day_summary.csv`         | Counts & proportions per day split        |
| `final_test_summary.json` | ✅ Master results report                   |

Inside any backbone dir (e.g. `vit/`):

```
Dy03/
Dy06/
Dy08/
Dy10/
... Dy30/
```

Inside each day:

```
metrics_test.json
metrics_val.json
model.pth
training_curves.png
```

This gives:

* **Performance numbers**
* **Saved weights for inference**
* **Plots for QC**

---

## 🔍 Quick Result Inspection

Best single-file summaries:

```bash
cat outputs_nomask_image/final_test_summary.json
cat outputs_mask_overlay/final_test_summary.json
```

Search AUROC for a specific day across every experiment:

```bash
grep -R '"test_roc_auc"' outputs_*/**/Dy20_5/metrics_test.json
```

View soft-label threshold selection:

```bash
grep -R '"threshold_used"' outputs_512x384_softlabels*
```

---

## 📚 Project Term Glossary

| Term        | Definition                                        |
| ----------- | ------------------------------------------------- |
| `DyXX`      | Culture imaging day (distinct per-day classifier) |
| Mask        | Segmentation map isolating organoid pixels        |
| Overlay     | Fluorescent intensity mapped onto image grid      |
| Soft labels | Fractional probability that organoid passes QC    |
| Hard labels | Binary QC status inherited from Day-30            |
| AUROC       | True separation of pass/fail quality              |
| Brier score | Probability error (lower = better)                |
| model.pth   | Trained weights for new data inference            |

---

## ✅ What This Module Enables

* Predict organoid acceptance **10–15 days before** Day-30 evaluation
* Train uncertainty-aware models (soft label mode)
* Compare different biological inputs (RGB vs overlay vs segmentation)
* Extract interpretable misclassification panels for biologists
* Provide per-day embeddings for **multimodal fusion** with metabolomics and survey classifiers

---

### 👥 New-Contributor Checklist

✅ Confirm `all_data.json` exists
✅ Run `generate_training_metadata.py --all`
✅ Submit `run_all.s` + optionally `run_soft_label.s`
✅ Watch `outputs_*` metrics & AUROC to select best checkpoint
✅ Load saved `model.pth` for inference or multimodal modeling
✅ Use metrics plots + misclassified panels for reporting


