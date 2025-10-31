#!/bin/bash
#SBATCH --job-name=dino-img
#SBATCH --partition=general
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

# ====== adjust these paths ======
PROJ_ROOT=/home/tonyluo/minitest
PY=${PROJ_ROOT}/analysis/images/classifier/train_model_dinov2.py
DATA_DIR=${PROJ_ROOT}/analysis/images/classifier/data/preprocessed/512x384/majority
OUT_DIR=/net/projects2/promega/tony_results/outputs_nomask_image_dinov2
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

cd "${PROJ_ROOT}"
export PYTHONPATH="${PROJ_ROOT}:$PYTHONPATH"
${CONDA_PREFIX}/bin/python3 -u "${PY}" "${ARGS[@]}"

echo "Done."

