#!/bin/bash
# Submit all 3 regeneration runs as SLURM jobs in parallel (no dependency).
# Each job writes to its own image_classifier/regeneration/run_N/ so they don't overwrite.
#
# CONFIGURE: Set PROJ_ROOT or replace YOUR_GITHUB_USERNAME below.
# Run from: cd /home/YOUR_GITHUB_USERNAME/promega-classifier && bash image_classifier/regeneration/submit_all_regeneration_jobs.sh

set -e
PROJ_ROOT="${PROJ_ROOT:-/home/YOUR_GITHUB_USERNAME/promega-classifier}"
REGEN="$PROJ_ROOT/image_classifier/regeneration"
mkdir -p "$REGEN/logs"
cd "$PROJ_ROOT"

J1=$(sbatch --export=RUN_NUMBER=1 "$REGEN/submit_regeneration_run.slurm" | awk '{print $4}')
J2=$(sbatch --export=RUN_NUMBER=2 "$REGEN/submit_regeneration_run.slurm" | awk '{print $4}')
J3=$(sbatch --export=RUN_NUMBER=3 "$REGEN/submit_regeneration_run.slurm" | awk '{print $4}')
echo "Submitted all 3 runs in parallel: $J1 (run_1), $J2 (run_2), $J3 (run_3)"
echo "Check: squeue -u \$USER"
