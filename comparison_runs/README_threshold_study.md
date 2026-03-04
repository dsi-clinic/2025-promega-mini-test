# Threshold Study: Final CSV Production

## Final Output
**`comparison_runs/overlay_threshold_study_results.csv`**

This CSV contains:
- **Setup:** overlay only (best for both models)
- **Models:** per_day, effnet_ts (2 models)
- **Days:** 6, 8, 10, 13, 15, 17, 20.5, 24, 26, 28, 30 (11 days)
- **Thresholds:** 0.5, 0.6, 0.7, 0.8, 0.9, and optimal (6 thresholds per run)
- **Total:** 132 rows (22 runs × 6 thresholds)
- **Primary metric:** balanced_acc = (Sensitivity + Specificity) / 2

## Columns
setup, fold, model, day, threshold, accuracy, precision, recall, f1, TNR, TPR, sensitivity, specificity, balanced_acc, TN, FP, FN, TP, optimal_threshold

## How to Produce the CSV

### Option 1: Quick Export (if results already exist)
```bash
cd /home/your_name/image_classifier_ts/comparison_runs
python3 make_overlay_csv.py --export_only
```

### Option 2: Full Pipeline (backfill + export)
```bash
cd /home/your_name/image_classifier_ts/comparison_runs
python3 make_overlay_csv.py
```

### Option 3: SLURM Job (runs on compute node)
```bash
cd /home/your_name/image_classifier_ts/comparison_runs
sbatch submit_make_overlay_csv.slurm
```

## Scripts

- **`make_overlay_csv.py`**: Main script that produces the CSV
  - `--export_only`: Skip backfill, just export existing results
  - Without flag: Runs backfill (adds threshold_results via inference) then exports

- **`submit_make_overlay_csv.slurm`**: SLURM script (2h, 4 CPUs, 16GB RAM, no GPU)
  - Runs on compute node (has guard to prevent login-node execution)
  - Produces final CSV

- **`run_threshold_study.py`**: Full study runner (backfill + train missing + export)
  - Skips runs that already have threshold_results
  - Only trains if results missing

- **`export_metrics_csv.py`**: CSV exporter (can filter by setup)
  - `--setup overlay`: Only export overlay results

## Status
✅ CSV is ready: `overlay_threshold_study_results.csv` (132 rows)
✅ Scripts verified and working
✅ SLURM scripts configured for compute nodes
