#!/bin/bash
# Submit seed-rotation experiments: 3 models × 3 seeds = 9 runs across 2 GPU jobs.
#
# Seeds: 7, 13, 99  (all different from original seed 42)
# Models: per_day EfficientNet, effnet_ts (time-series), combined LGB
#
# Job A (~8h):  seed 7 full (per_day + effnet_ts + LGB) + seed 13 LGB + seed 99 LGB
# Job B (~10h): seed 13 (per_day + effnet_ts) + seed 99 (per_day + effnet_ts)
#
# Output structure:
#   regeneration/seed_rotation_s7/   -- per_day_study_overlay/ + combined_lgbm/
#   regeneration/seed_rotation_s13/  -- per_day_study_overlay/ + combined_lgbm/
#   regeneration/seed_rotation_s99/  -- per_day_study_overlay/ + combined_lgbm/
#   regeneration/seed_rotation_splits/s{7,13,99}/data_splits/  -- split JSONs
#
# Run from: cd /home/tonyluo/amanda_temporal && bash regeneration/submit_seed_rotation.sh

set -e
REGEN="/home/tonyluo/amanda_temporal/regeneration"
mkdir -p "$REGEN/logs"
cd /home/tonyluo/amanda_temporal

JA=$(sbatch "$REGEN/submit_seed_rotation_a.slurm" | awk '{print $4}')
JB=$(sbatch "$REGEN/submit_seed_rotation_b.slurm" | awk '{print $4}')

echo "Submitted seed-rotation jobs:"
echo "  Job A (seed 7 full + all LGBs):       $JA"
echo "  Job B (seed 13 + 99 threshold study):  $JB"
echo ""
echo "Monitor: squeue -u \$USER"
echo "Logs:    tail -f $REGEN/logs/seed_rotation_{a,b}_*.out"
