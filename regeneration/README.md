# Regeneration: reproducible overlay runs

**Purpose:** Check that overlay per_day and effnet_ts results can be regenerated (reproducibility). No code logic was changed.

**Status**
- **per_day:** Proven reproducible. Baseline and run_1/run_2 match exactly (same optimal threshold and test metrics). Run_3 matched on most days with minor drift on a few (e.g. day 28), likely node or timing variation.
- **effnet_ts (time-series):** Was not fully reproducible because the attention trainer did not set `PYTHONHASHSEED`, `cudnn.deterministic`, or `cudnn.benchmark=False`. That is now fixed in `train_temporal_ablation_attn.py` (same seed lock as the base trainer). **New regeneration runs should make effnet_ts reproducible like per_day.**

**Layout:**
- `baseline/` — Copy of current results before any re-run: `per_day_study_overlay/` (all JSONs + .pth), `metrics.csv`, `overlay_threshold_study_results.csv`.
- `run_1/`, `run_2/`, `run_3/` — Each contains one full overlay threshold study re-run: `per_day_study_overlay/`, `overlay_threshold_study_results.csv`, `metrics.csv`. Output directories are separate so nothing overwrites.

**How to run the 3 regeneration runs:**
```bash
cd /home/tonyluo/amanda_temporal
bash regeneration/submit_all_regeneration_jobs.sh
```
Writes to `run_1/`, `run_2/`, `run_3/`. To re-run **without overwriting** (e.g. to prove reproducibility after the effnet_ts seed fix):
```bash
bash regeneration/submit_all_regeneration_jobs_seed_fix.sh
```
That submits 3 jobs writing to `run_1_seed_fix/`, `run_2_seed_fix/`, `run_3_seed_fix/`. Expect ~22 trainings per run; total runtime is long.

**Paths:**
- Repo: `2025-promega-mini-test` (threshold study writes to repo’s `comparison_runs/per_day_study_overlay`).
- Baseline and run_* live under: `amanda_temporal/regeneration/`.
