#!/bin/bash
# Run overlay threshold study 3 times and save each run to regeneration/run_1, run_2, run_3.
# No code logic changes: uses existing run_threshold_study.py and export_metrics_csv.py.
# Output dirs are separate so nothing overwrites (run_1, run_2, run_3).

set -e
# CONFIGURE: Set PARENT to repo root, or export PROJ_ROOT before running.
PARENT="${PROJ_ROOT:-/home/YOUR_GITHUB_USERNAME/promega-classifier}"
REGEN="$PARENT/image_classifier/regeneration"

for i in 1 2 3; do
  echo "========== Regeneration run $i =========="
  cd "$PARENT"
  python image_classifier/comparison_runs/run_threshold_study.py --output_dir "$REGEN/run_$i"
  python image_classifier/comparison_runs/export_metrics_csv.py --setup overlay --overlay_dir "$REGEN/run_$i/per_day_study_overlay" -o "$REGEN/run_$i/metrics.csv"
  echo "Done run $i."
done
echo "All 3 regeneration runs complete. Results in $REGEN/run_1, run_2, run_3."
