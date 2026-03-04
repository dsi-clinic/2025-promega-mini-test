#!/bin/bash
# Submit all 3 regeneration runs as SLURM jobs in parallel (no dependency).
# Each job writes to its own regeneration/run_N/ so they don't overwrite.
# Run from: cd /home/tonyluo/amanda_temporal && bash regeneration/submit_all_regeneration_jobs.sh

set -e
REGEN="/home/tonyluo/amanda_temporal/regeneration"
mkdir -p "$REGEN/logs"
cd /home/tonyluo/amanda_temporal

J1=$(sbatch --export=RUN_NUMBER=1 "$REGEN/submit_regeneration_run.slurm" | awk '{print $4}')
J2=$(sbatch --export=RUN_NUMBER=2 "$REGEN/submit_regeneration_run.slurm" | awk '{print $4}')
J3=$(sbatch --export=RUN_NUMBER=3 "$REGEN/submit_regeneration_run.slurm" | awk '{print $4}')
echo "Submitted all 3 runs in parallel: $J1 (run_1), $J2 (run_2), $J3 (run_3)"
echo "Check: squeue -u \$USER"
