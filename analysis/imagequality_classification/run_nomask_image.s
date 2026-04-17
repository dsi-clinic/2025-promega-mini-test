#!/bin/bash
#SBATCH --job-name=train-img-nomask
#SBATCH --partition=general
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

# ====== adjust these paths ======
PROJ_ROOT=/net/scratch/jiaweizhang/2025-promega-mini-test
PY=${PROJ_ROOT}/analysis/images/classifier/train_model_accuracy.py
DATA_DIR=${PROJ_ROOT}/analysis/images/classifier/data/preprocessed/512x384/majority
OUT_DIR=${PROJ_ROOT}/analysis/images/classifier/outputs_nomask_image_noaugment
CONDA_PREFIX=/net/projects2/promega
# ================================

mkdir -p logs "${OUT_DIR}"

if command -v module >/dev/null 2>&1; then
  module purge
fi

INPUT_KEY=img_path
USE_MASK=false

echo "Conda prefix: ${CONDA_PREFIX}"
nvidia-smi || true
echo "Running ${PY}"
echo "DATA_DIR=${DATA_DIR}"
echo "Combination: input_key=${INPUT_KEY}, use_mask=${USE_MASK}"

ARGS=(
  --data_dir "${DATA_DIR}"
  --batch-size 16
  --val-frac 0.10
  --test-frac 0.10
  --input-path-key "${INPUT_KEY}"
  --outdir "${OUT_DIR}"
)

PYTHONPATH=. conda run -p "${CONDA_PREFIX}" python "${PY}" "${ARGS[@]}"

echo "Done."
