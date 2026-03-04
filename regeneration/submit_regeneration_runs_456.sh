#!/bin/bash
# Submit 3 more regeneration runs (with seed-fixed effnet_ts); store in run_4, run_5, run_6.
# Run from: cd /home/tonyluo/amanda_temporal && bash regeneration/submit_regeneration_runs_456.sh

set -e
REGEN="/home/tonyluo/amanda_temporal/regeneration"
mkdir -p "$REGEN/logs"
cd /home/tonyluo/amanda_temporal

J4=$(sbatch --export=RUN_NUMBER=4 "$REGEN/submit_regeneration_run.slurm" | awk '{print $4}')
J5=$(sbatch --export=RUN_NUMBER=5 "$REGEN/submit_regeneration_run.slurm" | awk '{print $4}')
J6=$(sbatch --export=RUN_NUMBER=6 "$REGEN/submit_regeneration_run.slurm" | awk '{print $4}')
echo "Submitted 3 runs (4, 5, 6) in parallel: $J4 (run_4), $J5 (run_5), $J6 (run_6)"
echo "Check: squeue -u \$USER"
