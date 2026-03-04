#!/bin/bash
set -e
REGEN="/home/tonyluo/amanda_temporal/regeneration"
mkdir -p "$REGEN/logs"
cd /home/tonyluo/amanda_temporal

for n in 3 4 5 6; do
  J=$(sbatch --export=ALL,RUN_NUMBER=$n,MODELS=effnet_ts "$REGEN/submit_regeneration_run.slurm" | awk '{print $4}')
  echo "run_$n -> job $J"
done
echo "Done submitting. Check: squeue -u tonyluo"
