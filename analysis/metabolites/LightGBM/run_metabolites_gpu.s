#!/bin/bash
#SBATCH --partition=general
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Metabolite LightGBM training script (GPU version)
# Usage: sbatch --job-name=lgbm-gpu run_metabolites_gpu.s [args]
#
# Examples:
#   sbatch --job-name=lgbm-gpu-f1na-cw run_metabolites_gpu.s --scoring f1_notacceptable --imbalance class_weight
#   sbatch --job-name=lgbm-gpu-recna-both run_metabolites_gpu.s --scoring recall_notacceptable --imbalance both

set -euo pipefail

# ====== adjust these paths ======
PROJ_ROOT=${PROJ_ROOT:-/home/waggonere/2025-promega-mini-test}
CONDA_PREFIX=/net/projects2/promega
# ================================

mkdir -p logs

if command -v module >/dev/null 2>&1; then
  module purge
fi

echo "Project root: ${PROJ_ROOT}"
echo "Conda prefix: ${CONDA_PREFIX}"
echo "Starting GPU LightGBM training at $(date)"
echo "=============================================="
nvidia-smi || true

cd "${PROJ_ROOT}"
export PYTHONPATH="${PROJ_ROOT}:${PYTHONPATH:-}"

SCRIPT="analysis/metabolites/LightGBM/train_metabolites_gpu.py"

echo "Running: ${SCRIPT} $@"
${CONDA_PREFIX}/bin/python3 -u $SCRIPT "$@"

EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo "ERROR: Training failed with exit code $EXIT_CODE"
    exit $EXIT_CODE
fi

echo ""
echo "=============================================="
echo "Training completed at $(date)"
echo "=============================================="
