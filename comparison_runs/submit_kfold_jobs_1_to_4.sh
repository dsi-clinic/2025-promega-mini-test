#!/bin/bash
# Submit only k-fold jobs 1–4 (array 99–197, 198–296, 297–395, 396–494).
# Use this when job 0 (698390, array 0–98) is already submitted and the queue allows more.
# Usage: cd comparison_runs && bash submit_kfold_jobs_1_to_4.sh

cd "$(dirname "$0")"
echo "Submitting k-fold jobs 1–4 (arrays 99–197, 198–296, 297–395, 396–494)..."
for i in 1 2 3 4; do
  if sbatch submit_per_day_study_kfold_${i}.slurm; then
    echo "  Job $i submitted."
  else
    echo "  Job $i failed (limit?). Re-run this script later to retry."
  fi
done
echo "Done. Check: squeue -u \$USER"
