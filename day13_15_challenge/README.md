# Day 13/15 Challenge

Scripts and outputs for the Day 13 / Day 15 effnet_ts collapse investigation. All runs target **days 13 and 15 only** (TS uses frames up to that day). Use GPU when available.

## Layout

| Item | Description |
|------|-------------|
| **README_REPORT.md** | Task 3: Reporting narrative (fixed t=0.5, balanced accuracy, document Day13/15 collapse). |
| **01_compute_grayscale_mean_std.py** | Task 1a: Compute mean/std from **grayscale** pixels (train, frames ≤ day). Saves `grayscale_mean_std_day13.npy`, `grayscale_mean_std_day15.npy`. |
| **02_audit_transforms.py** | Task 2: Audit post-transform tensors (Day 13 vs Day 28 overlay TS). Writes `audit_transforms_summary.json` and optional `audit_transforms_histogram.png`. |
| **run_1_grayscale_norm.py** | Task 1: Train **effnet_ts** and **per_day** for days 13 & 15 with **grayscale-derived normalization**. Outputs under `runs_grayscale_norm/`. |
| **run_4_filled_mask.py** | Task 4: Train **effnet_ts** and **per_day** for days 13 & 15 with input **(gray, gray, filled_mask)**. Outputs under `runs_filled_mask/`. |
| **dataset_grayscale_norm.py** | Dataset helpers: grayscale mean/std and TS/single-day datasets using that normalization. |
| **dataset_filled_mask.py** | Dataset variants: TS and single-day with (gray, gray, filled_mask) 3ch input. |
| **run_all_gpu.slurm** | Slurm job script: 1 GPU, 12h, runs `run_all_gpu.py`. Submit with `sbatch run_all_gpu.slurm`. |

## How to run

**Automated (all steps on GPU, one after another):**

**On a Slurm cluster (recommended):**
```bash
cd day13_15_challenge
# Edit run_all_gpu.slurm to set --partition and --account if needed, then:
sbatch run_all_gpu.slurm
# Logs: slurm_day1315_challenge_<jobid>.out, .err
```

**Interactive / login node with GPU:**
```bash
cd day13_15_challenge
python run_all_gpu.py
# or: CUDA_VISIBLE_DEVICES=0 python run_all_gpu.py
# or: chmod +x run_all_gpu.sh && ./run_all_gpu.sh
```

Steps run in order: (1) grayscale mean/std, (2) transform audit, (3) grayscale-norm training, (4) filled-mask training. If a step fails, the pipeline stops and the error is shown so you can fix and re-run.

**Manual (run scripts individually):**

```bash
cd day13_15_challenge
python 01_compute_grayscale_mean_std.py
python 02_audit_transforms.py
python run_1_grayscale_norm.py
python run_4_filled_mask.py
```

Data is read from `2025-promega-mini-test/data_splits/` (both_train_base.json, both_val_base.json, both_test_base.json).

## Outputs

- **runs_grayscale_norm/** — Task 1: `effnet_ts_grayscale_norm/day_13/`, `day_15/`, `per_day_grayscale_norm/day_13/`, `day_15/`, each with `results.json`, model checkpoints.
- **runs_filled_mask/** — Task 4: `effnet_ts_filled_mask/day_13/`, `day_15/`, `per_day_filled_mask/day_13/`, `day_15/`.
- **audit_transforms_summary.json** — Task 2: percentile (1/50/99) and min/max/mean/std for Day 13 vs Day 28 post-transform tensors.
- **grayscale_mean_std_day13.npy**, **grayscale_mean_std_day15.npy** — Task 1a: `{"mean": (3,), "std": (3,), "max_day": ...}`.
