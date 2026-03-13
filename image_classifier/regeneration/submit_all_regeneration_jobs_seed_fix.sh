#!/bin/bash
# Submit the same 3 regeneration runs again, with seed-fixed effnet_ts.
# Writes to run_1_seed_fix, run_2_seed_fix, run_3_seed_fix (does not overwrite run_1/2/3).
#
# CONFIGURE: Set PROJ_ROOT or replace YOUR_GITHUB_USERNAME below.
# Run from: cd /home/YOUR_GITHUB_USERNAME/promega-classifier && bash image_classifier/regeneration/submit_all_regeneration_jobs_seed_fix.sh

set -e
PROJ_ROOT="${PROJ_ROOT:-/home/YOUR_GITHUB_USERNAME/promega-classifier}"
REGEN="$PROJ_ROOT/image_classifier/regeneration"
mkdir -p "$REGEN/logs"
cd "$PROJ_ROOT"

J1=$(sbatch --export=RUN_NUMBER=1,RUN_SUFFIX=_seed_fix "$REGEN/submit_regeneration_run.slurm" | awk '{print $4}')
J2=$(sbatch --export=RUN_NUMBER=2,RUN_SUFFIX=_seed_fix "$REGEN/submit_regeneration_run.slurm" | awk '{print $4}')
J3=$(sbatch --export=RUN_NUMBER=3,RUN_SUFFIX=_seed_fix "$REGEN/submit_regeneration_run.slurm" | awk '{print $4}')
echo "Submitted 3 seed-fix runs in parallel: $J1 (run_1_seed_fix), $J2 (run_2_seed_fix), $J3 (run_3_seed_fix)"
echo "Check: squeue -u \$USER"
