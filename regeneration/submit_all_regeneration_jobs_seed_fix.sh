#!/bin/bash
# Submit the same 3 regeneration runs again, with seed-fixed effnet_ts.
# Writes to run_1_seed_fix, run_2_seed_fix, run_3_seed_fix (does not overwrite run_1/2/3).
# Run from: cd /home/tonyluo/amanda_temporal && bash regeneration/submit_all_regeneration_jobs_seed_fix.sh

set -e
REGEN="/home/tonyluo/amanda_temporal/regeneration"
mkdir -p "$REGEN/logs"
cd /home/tonyluo/amanda_temporal

J1=$(sbatch --export=RUN_NUMBER=1,RUN_SUFFIX=_seed_fix "$REGEN/submit_regeneration_run.slurm" | awk '{print $4}')
J2=$(sbatch --export=RUN_NUMBER=2,RUN_SUFFIX=_seed_fix "$REGEN/submit_regeneration_run.slurm" | awk '{print $4}')
J3=$(sbatch --export=RUN_NUMBER=3,RUN_SUFFIX=_seed_fix "$REGEN/submit_regeneration_run.slurm" | awk '{print $4}')
echo "Submitted 3 seed-fix runs in parallel: $J1 (run_1_seed_fix), $J2 (run_2_seed_fix), $J3 (run_3_seed_fix)"
echo "Check: squeue -u \$USER"
