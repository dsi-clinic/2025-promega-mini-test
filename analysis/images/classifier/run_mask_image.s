#!/bin/bash
#SBATCH --job-name=train-img-mask
#SBATCH --partition=general
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

# ====== adjust these paths ======
PROJ_ROOT=${PROJ_ROOT:-/home/tonyluo/minitest}
PY=${PROJ_ROOT}/analysis/images/classifier/train_model_accuracy.py
TRAIN_SPLIT=${PROJ_ROOT}/data_splits/both_train_base.json
VAL_SPLIT=${PROJ_ROOT}/data_splits/both_val_base.json
TEST_SPLIT=${PROJ_ROOT}/data_splits/both_test_base.json
OUT_DIR=${OUT_DIR:-/net/projects2/promega/results/outputs_mask_image}
CONDA_PREFIX=/net/projects2/promega
# ================================

mkdir -p logs "${OUT_DIR}"

if command -v module >/dev/null 2>&1; then
  module purge
fi

INPUT_KEY=img_path
USE_MASK=true

echo "Conda prefix: ${CONDA_PREFIX}"
nvidia-smi || true
echo "Running ${PY} with fixed splits"
echo "Train split: ${TRAIN_SPLIT}"
echo "Val split: ${VAL_SPLIT}"
echo "Test split: ${TEST_SPLIT}"
echo "Combination: input_key=${INPUT_KEY}, use_mask=${USE_MASK}"

ARGS=(
  --train-split "${TRAIN_SPLIT}"
  --val-split "${VAL_SPLIT}"
  --test-split "${TEST_SPLIT}"
  --batch-size 16
  --val-batch-size 16
  --input-path-key "${INPUT_KEY}"
  --outdir "${OUT_DIR}"
)

if [[ "${USE_MASK}" == true ]]; then
  ARGS+=(--use-mask)
fi

cd "${PROJ_ROOT}"
export PYTHONPATH="${PROJ_ROOT}:$PYTHONPATH"
${CONDA_PREFIX}/bin/python3 -u "${PY}" "${ARGS[@]}"

echo "Done."

