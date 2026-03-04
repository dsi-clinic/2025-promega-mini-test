#!/bin/bash
#SBATCH --job-name=combined-lgbm
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=6:00:00
#SBATCH --output=logs/combined_%j.out
#SBATCH --error=logs/combined_%j.err

# Combined Image + Metabolite LightGBM Training
# GPU needed only for EfficientNet feature extraction; LightGBM trains on CPU.
#
# Usage:
#   sbatch submit_combined.slurm                     # default run
#   sbatch submit_combined.slurm --day_filter Dy03   # smoke test one day

set -euo pipefail

PROJ_ROOT=/home/tonyluo/amanda_temporal/combined_model
DATA_ROOT=/home/tonyluo/amanda_temporal/2025-promega-mini-test
CONDA_PREFIX=/net/projects2/promega

EXTRA_ARGS="${@}"

mkdir -p "${PROJ_ROOT}/logs"

if command -v module >/dev/null 2>&1; then
    module purge 2>/dev/null || true
fi

echo "=============================================="
echo "Combined Image+Metabolite LightGBM"
echo "=============================================="
echo "Job ID:      ${SLURM_JOB_ID}"
echo "Node:        ${SLURM_NODELIST}"
echo "GPU:         ${CUDA_VISIBLE_DEVICES:-none}"
echo "Project:     ${PROJ_ROOT}"
echo "Data root:   ${DATA_ROOT}"
echo "Extra args:  ${EXTRA_ARGS}"
echo "Start:       $(date)"
echo ""

nvidia-smi || true
echo ""

cd "${PROJ_ROOT}"

${CONDA_PREFIX}/bin/python3 -u train_combined_lgbm.py \
    --data_dir "${DATA_ROOT}" \
    --scoring f1_notaccept \
    --imbalance class_weight \
    --n_folds 3 \
    --n_image_features 12 \
    ${EXTRA_ARGS}

EXIT_CODE=$?

echo ""
echo "=============================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "DONE - Results in ${PROJ_ROOT}/outputs/"
else
    echo "FAILED (exit $EXIT_CODE)"
fi
echo "End: $(date)"
echo "=============================================="

exit $EXIT_CODE
