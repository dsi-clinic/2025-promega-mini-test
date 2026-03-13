#!/bin/bash
# Submit 3 more regeneration runs (with seed-fixed effnet_ts); store in run_4, run_5, run_6.
#
# CONFIGURE: Set PROJ_ROOT or replace YOUR_GITHUB_USERNAME below.
# Run from: cd /home/YOUR_GITHUB_USERNAME/promega-classifier && bash image_classifier/regeneration/submit_regeneration_runs_456.sh

set -e
PROJ_ROOT="${PROJ_ROOT:-/home/YOUR_GITHUB_USERNAME/promega-classifier}"
REGEN="$PROJ_ROOT/image_classifier/regeneration"
mkdir -p "$REGEN/logs"
cd "$PROJ_ROOT"

J4=$(sbatch --export=RUN_NUMBER=4 "$REGEN/submit_regeneration_run.slurm" | awk '{print $4}')
J5=$(sbatch --export=RUN_NUMBER=5 "$REGEN/submit_regeneration_run.slurm" | awk '{print $4}')
J6=$(sbatch --export=RUN_NUMBER=6 "$REGEN/submit_regeneration_run.slurm" | awk '{print $4}')
echo "Submitted 3 runs (4, 5, 6) in parallel: $J4 (run_4), $J5 (run_5), $J6 (run_6)"
echo "Check: squeue -u \$USER"
