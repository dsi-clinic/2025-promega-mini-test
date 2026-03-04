#!/bin/bash
# Submit all 5 k-fold jobs at once; they will sit in queue and run as slots free.
# If you hit QOSMaxSubmitJobPerUserLimit, cancel other jobs or run this when the
# queue is light; you can re-run this script to submit any that failed.
# Usage: cd comparison_runs && bash submit_all_kfold.sh

cd "$(dirname "$0")"
echo "Submitting k-fold jobs 0-4 (arrays 0-98, 99-197, 198-296, 297-395, 396-494)..."
for i in 0 1 2 3 4; do
  if sbatch submit_per_day_study_kfold_${i}.slurm; then
    echo "  Job $i submitted."
  else
    echo "  Job $i failed (limit?). Re-run this script later to retry."
  fi
done
echo "Done. Check: squeue -u \$USER"
