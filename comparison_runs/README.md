# Image classifier comparison and threshold study

**What:** Overlay setup — **per_day** (single-day EfficientNet) vs **effnet_ts** (EfficientNet time-series). Same split for all: `data_splits/both_*_base.json`.

**Result CSV (in this folder):** `overlay_threshold_study_results.csv` — primary metric `balanced_acc` = (sensitivity + specificity) / 2.

---

## How to run EfficientNet per_day and time-series (effnet_ts)

All commands below are run from **amanda_temporal** (project root). The directory contains `comparison_runs/`; data and `data_splits/` are in `2025-promega-mini-test/`.

**Prerequisites**

- Python env with PyTorch and sklearn.
- `.env` in the repo root with any paths your `config.py` needs (e.g. `BASE_PATH`, `OUTPUT_FOLDER`).
- Train/val/test splits in the repo: `data_splits/both_train_base.json`, `data_splits/both_val_base.json`, `data_splits/both_test_base.json`. Generate them with `split_data_reproducible.py --mode base` if missing.

**Models**

- **per_day (EfficientNet per day):** One EfficientNet model per day; each model sees only the **single image** at that day (e.g. day 15 only). Outputs go under `comparison_runs/per_day_study_overlay/per_day/day_<day>_0/`.
- **effnet_ts (EfficientNet time-series):** One model per “cutoff” day; each model sees **all frames from day 6 up to that day** (accumulated time series). Outputs go under `comparison_runs/per_day_study_overlay/effnet_ts/day_<day>_0/`.

**Days used:** 6, 8, 10, 13, 15, 17, 20.5, 24, 26, 28, 30 (11 timepoints).

### Option 1: Run one (model, day) at a time

From repo root:

```bash
# EfficientNet per_day — single image at that day (e.g. day 15)
python comparison_runs/run_per_day_study.py --model_type per_day --day 15 --input_mode overlay

# EfficientNet time-series — frames from day 6 up to that day (e.g. day 15)
python comparison_runs/run_per_day_study.py --model_type effnet_ts --day 15 --input_mode overlay
```

- **per_day** output: `comparison_runs/per_day_study_overlay/per_day/day_15_0/` (e.g. `results.json`, `model_day_15.0.pth`).
- **effnet_ts** output: `comparison_runs/per_day_study_overlay/effnet_ts/day_15_0/` (e.g. `results.json`, `best_model.pth`, `global_mean.npy`).

Use any of the 11 days: `--day 6`, `--day 8`, `--day 10`, `--day 13`, `--day 15`, `--day 17`, `--day 20.5`, `--day 24`, `--day 26`, `--day 28`, `--day 30`.

### Option 2: Run all 11 days for both models (full threshold study), then export CSV

This runs **per_day** and **effnet_ts** for all 11 days (22 runs total), then writes the summary CSV:

```bash
python comparison_runs/run_threshold_study.py
python comparison_runs/make_overlay_csv.py
```

- Training/eval outputs: `comparison_runs/per_day_study_overlay/per_day/day_*_0/` and `comparison_runs/per_day_study_overlay/effnet_ts/day_*_0/`.
- Summary CSV: `comparison_runs/overlay_threshold_study_results.csv`.

Add `--save_model` to `run_threshold_study.py` if you want to keep `.pth` checkpoints (otherwise they are not saved to save space).

### Option 3: SLURM (run on cluster)

From repo root, after fixing paths in the SLURM scripts (see “One-time setup” below):

```bash
# Single per_day run (one day)
sbatch comparison_runs/submit_per_day_study_overlay_per_day.slurm   # edit script to set day

# Single effnet_ts run (one day)
sbatch comparison_runs/submit_per_day_study_overlay_effnet_ts.slurm   # edit script to set day

# Full threshold study (all 11 days, both models) then CSV
sbatch comparison_runs/submit_threshold_study.slurm
sbatch comparison_runs/submit_make_overlay_csv.slurm
```

---

## 1. One-time setup: fix paths (optional)

If you use SLURM scripts, from the **repo root** (`2025-promega-mini-test`):

```bash
sed -i 's|/home/your_name/image_classifier_ts|/home/YOUR_USERNAME/YOUR_REPO_PATH|g' comparison_runs/submit_*.slurm comparison_runs/monitor_threshold_study.sh
```

Replace with your username and path to this repo. For running Python by hand, no path change is needed: scripts detect whether they run from inside the repo or from a parent workspace.

---

## 2. Reproduce the threshold study → CSV (short version)

From the **repo root**:

```bash
python comparison_runs/run_threshold_study.py
python comparison_runs/make_overlay_csv.py
```

→ Writes `comparison_runs/overlay_threshold_study_results.csv`.

---

## 3. Results (this folder)

| What | Where |
|------|--------|
| Overlay threshold study (per_day + effnet_ts, 11 days, thresholds 0.5–0.9 + optimal) | `overlay_threshold_study_results.csv` |
| Columns | setup, fold, model, day, threshold, accuracy, precision, recall, f1, TNR, TPR, sensitivity, specificity, balanced_acc, TN, FP, FN, TP, optimal_threshold |

**Not in repo:** `.pth`, `*.npy`, `per_day_study*/`, logs. Run the scripts above to regenerate.

---

## Proving overlay vs RGB (and rgb_mask)

The comparison_runs setup lets you **compare overlay vs RGB (and rgb_mask)** so you can show that overlay is better.

**Scripts (all in this folder)**

1. **Run the same models for each input type** (same data, same days):
   - **RGB:** `python comparison_runs/run_per_day_study.py --model_type per_day --day 15 --input_mode rgb` (and repeat for other days; or run the full study for rgb by using `run_threshold_study.py` logic with `input_mode=rgb` — for a full comparison you’d run per_day and effnet_ts for all 11 days for each of `rgb`, `overlay`, `rgb_mask`).
   - **Overlay:** `--input_mode overlay` (outputs in `per_day_study_overlay/`).
   - **RGB + mask:** `--input_mode rgb_mask` (outputs in `per_day_study_rgb_mask/`).
2. **Export all setups to one CSV** (so you can compare by `setup` and by `balanced_acc`):
   ```bash
   python comparison_runs/export_metrics_csv.py
   ```
   This writes **`comparison_runs/metrics.csv`** with a **`setup`** column (`rgb`, `overlay`, `rgb_mask`) and the same metric columns (e.g. `balanced_acc`, sensitivity, specificity). It reads from `per_day_study/`, `per_day_study_overlay/`, and `per_day_study_rgb_mask/` (only setups that exist on disk are included).

**Results**

- **In the repo:** Only the **overlay** summary is committed: `overlay_threshold_study_results.csv` (no `setup` column; it’s overlay-only). So the repo does **not** contain a committed CSV that side-by-side compares rgb vs overlay.
- **On your machine:** If you have run all three input modes, the **results** live in `per_day_study/` (rgb), `per_day_study_overlay/` (overlay), and `per_day_study_rgb_mask/` (rgb_mask). Running `export_metrics_csv.py` with no `--setup` produces **`metrics.csv`** in this folder with one row per (setup, model, day, threshold); you can sort or filter by `setup` and compare `balanced_acc` (and other metrics) to show overlay is better than rgb.
- **Summary:** The comparison_runs directory **contains the scripts** to run rgb, overlay, and rgb_mask and to export a combined comparison CSV. The **proof** that overlay is better is the combined `metrics.csv` you generate (and/or the underlying `results.json` files in each `per_day_study*` dir); that combined CSV is not in the repo, but the script to create it is.

---

## Other runs (optional)

- **Single (model, day):** See **How to run EfficientNet per_day and time-series** above; use `--model_type per_day` or `--model_type effnet_ts` with `--day` and `--input_mode overlay`. For CNN-LSTM use `--model_type cnn_lstm`.
- **K-fold:** Set `K` in `run_per_day_study_kfold.py`, then run by array index or SLURM.
- **Threshold tuning:** `python comparison_runs/tune_thresholds.py` → `comparison_runs/threshold_tuning_results.json`.

---

## Path reference (if not using the sed above)

| Where | What to change |
|-------|----------------|
| **This repo** | |
| `analysis/images/cnn_lstm/load_split_data.py` | `base_dir = Path(".../data_splits")` in `if __name__ == "__main__"`. |
| `submit_cnn_lstm_*.slurm` (repo root) | `cd /home/.../2025-promega-mini-test` → your repo path. |
| `analysis/images/cnn_lstm/comparisons/run_686612_vs_bundled/comparison_report.py` | `amanda_path`, `output_dir`. |
| `analysis/images/cnn_lstm/plot_metrics.py` | Docstring example `--outdir`. |
| `generate_all_summary_tables.py` | `RESULTS_BASE_DIR`. |
| **This folder** `comparison_runs/submit_*.slurm` | `cd /home/.../2025-promega-mini-test` → your repo path. |

---

## Scripts (in this folder)

| Script | Purpose |
|--------|--------|
| `run_threshold_study.py` | Overlay threshold study (per_day + effnet_ts). |
| `make_overlay_csv.py` | → `overlay_threshold_study_results.csv`. |
| `run_per_day_study.py` | Single-split per_day / effnet_ts / cnn_lstm. |
| `run_per_day_study_kfold.py` | K-fold; set `K` then run by index. |
| `tune_thresholds.py` | Optimal threshold (TNR/f1) on val. |
| `export_metrics_csv.py`, `add_threshold_results_to_existing.py` | Export / backfill. |

Training: `analysis/images/cnn_lstm/` (e.g. `train_base_model.py`, `train_temporal_ablation_attn.py`).
