#!/bin/bash
#SBATCH --partition=general
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Metabolite Classifier Training Script (GPU)
# Usage: sbatch --job-name=metab-train run_metabolite_training.s [OPTIONS]
#
# Examples:
#   # Default run (f1_notaccept, class_weight, 5-fold, 200 search configs)
#   sbatch --job-name=metab-default run_metabolite_training.s
#
#   # With SMOTE
#   sbatch --job-name=metab-smote run_metabolite_training.s --imbalance smote
#
#   # High recall run
#   sbatch --job-name=metab-recall run_metabolite_training.s --scoring recall_notaccept
#
#   # With acceleration features
#   sbatch --job-name=metab-accel run_metabolite_training.s --use_second_order_growth

set -euo pipefail

# ====== ADJUST THESE PATHS ======
# Replace YOUR_GITHUB_USERNAME with your actual GitHub username
# This should be the root directory of your cloned repository
PROJ_ROOT=${PROJ_ROOT:-/home/YOUR_GITHUB_USERNAME/MINITEST_DIRECTORY}
CONDA_PREFIX=/net/projects2/promega
# ================================

# Forward all arguments to Python script
EXTRA_ARGS="$@"

mkdir -p logs

if command -v module >/dev/null 2>&1; then
  module purge
fi

echo "=============================================="
echo "Metabolite Classifier Training (GPU)"
echo "=============================================="
echo "Project root: ${PROJ_ROOT}"
echo "Conda prefix: ${CONDA_PREFIX}"
echo "Extra arguments: ${EXTRA_ARGS}"
echo ""

nvidia-smi || true

cd "${PROJ_ROOT}"
export PYTHONPATH="${PROJ_ROOT}:${PYTHONPATH:-}"

PY_SCRIPT="${PROJ_ROOT}/analysis/metabolites/classifier/train_metabolites_gpu.py"

echo "Running: ${PY_SCRIPT} ${EXTRA_ARGS}"
echo ""

${CONDA_PREFIX}/bin/python3 -u "${PY_SCRIPT}" ${EXTRA_ARGS}

EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    echo "ERROR: Training failed with exit code $EXIT_CODE"
    exit $EXIT_CODE
fi

echo ""
echo "=============================================="
echo "Training Complete!"
echo "=============================================="
