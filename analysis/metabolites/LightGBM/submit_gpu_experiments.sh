#!/bin/bash
# Submit 8 GPU metabolite classifier experiments to SLURM
#
# Experiments:
#   - Scoring: f1_notacceptable, recall_notacceptable (2)
#   - Imbalance: class_weight, scale_pos_weight, both, focal_loss (4)
#   Total: 2 x 4 = 8 jobs
#
# Usage: bash submit_gpu_experiments.sh
#
# Notes:
#   - Each job requests 1 A100 GPU
#   - Jobs run independently on the cluster
#   - Monitor with: squeue -u $USER

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLURM_SCRIPT="${SCRIPT_DIR}/run_metabolites_gpu.s"

echo "Submitting 8 GPU experiments..."
echo "SLURM script: ${SLURM_SCRIPT}"
echo ""

JOB_COUNT=0

for SCORING in f1_notacceptable recall_notacceptable; do
    for IMBALANCE in class_weight scale_pos_weight both focal_loss; do
        # Create short job name: e.g., "lgbm-f1na-cw"
        SCORE_SHORT=$(echo $SCORING | sed 's/f1_notacceptable/f1na/' | sed 's/recall_notacceptable/recna/')
        IMB_SHORT=$(echo $IMBALANCE | sed 's/class_weight/cw/' | sed 's/scale_pos_weight/spw/' | sed 's/focal_loss/fl/' | sed 's/both/both/')
        JOB_NAME="lgbm-gpu-${SCORE_SHORT}-${IMB_SHORT}"
        
        echo "Submitting: ${JOB_NAME}"
        echo "  --scoring ${SCORING} --imbalance ${IMBALANCE}"
        
        sbatch --job-name="${JOB_NAME}" "${SLURM_SCRIPT}" \
            --scoring "${SCORING}" \
            --imbalance "${IMBALANCE}"
        
        JOB_COUNT=$((JOB_COUNT + 1))
        echo ""
    done
done

echo "=============================================="
echo "Submitted ${JOB_COUNT} jobs"
echo "Monitor with: squeue -u $USER"
echo "Check logs in: logs/"
echo "=============================================="
