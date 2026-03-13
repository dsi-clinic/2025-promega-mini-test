#!/bin/bash
# Submit regeneration runs 3–6 with effnet_ts only.
#
# CONFIGURE: Set PROJ_ROOT or replace YOUR_GITHUB_USERNAME below.
# Run from: cd /home/YOUR_GITHUB_USERNAME/promega-classifier && bash image_classifier/regeneration/submit_effnet_ts_regen.sh

set -e
PROJ_ROOT="${PROJ_ROOT:-/home/YOUR_GITHUB_USERNAME/promega-classifier}"
REGEN="$PROJ_ROOT/image_classifier/regeneration"
mkdir -p "$REGEN/logs"
cd "$PROJ_ROOT"

for n in 3 4 5 6; do
  J=$(sbatch --export=ALL,RUN_NUMBER=$n,MODELS=effnet_ts "$REGEN/submit_regeneration_run.slurm" | awk '{print $4}')
  echo "run_$n -> job $J"
done
echo "Done submitting. Check: squeue -u \$USER"
