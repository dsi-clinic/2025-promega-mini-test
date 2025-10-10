#!/bin/bash
#SBATCH --job-name=soft-label
#SBATCH --partition=general
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

# ====== adjust these paths ======
PROJ_ROOT=/net/scratch/jiaweizhang/2025-promega-mini-test
PY=${PROJ_ROOT}/analysis/images/classifier/train_model_soft_labels.py
DATA_DIR=${PROJ_ROOT}/analysis/images/classifier/data/preprocessed/512x384/majority
CONDA_PREFIX=/net/projects2/promega                           # conda env path (same you used before)
# ================================

mkdir -p logs


# Optional: purge modules if your cluster uses them
if command -v module >/dev/null 2>&1; then
  module purge
fi

echo "Conda prefix: ${CONDA_PREFIX}"
nvidia-smi || true
echo "Running find_misclassified_images.py"
echo "DATA_DIR=${DATA_DIR}"

# Run the script (uses simple defaults from the python file)
PYTHONPATH=. conda run -p "${CONDA_PREFIX}" python "${PY}" \
  --data_dir "${DATA_DIR}" \
  --batch-size 16 \
  --val-frac 0.10 \
  --test-frac 0.10

echo "Done."
