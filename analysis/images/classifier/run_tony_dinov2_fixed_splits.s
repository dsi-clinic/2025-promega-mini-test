#!/bin/bash
#SBATCH --job-name=tony-dinov2-fixed-splits
#SBATCH --partition=general
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%A.out
#SBATCH --error=logs/%x_%A.err

set -euo pipefail

PROJ_ROOT=/home/tonyluo/minitest
PY=${PROJ_ROOT}/analysis/images/classifier/train_model_accuracy_tony_dinov2.py
TRAIN_SPLIT=${PROJ_ROOT}/data_splits/both_train_base.json
VAL_SPLIT=${PROJ_ROOT}/data_splits/both_val_base.json
OUT_DIR=/net/projects2/promega/tony_results/outputs_tony_dinov2_fixed_splits
CONDA_PREFIX=/net/projects2/promega

mkdir -p logs "${OUT_DIR}"

module purge 2>/dev/null || true

echo "Conda prefix: ${CONDA_PREFIX}"
which nvidia-smi >/dev/null 2>&1 && nvidia-smi || true

echo "Running ${PY} with fixed splits"
echo "Train split: ${TRAIN_SPLIT}"
echo "Val split: ${VAL_SPLIT}"
export PYTHONPATH="${PROJ_ROOT}:$PYTHONPATH"
"${CONDA_PREFIX}/bin/python3" -u "${PY}" \
  --train-split "${TRAIN_SPLIT}" \
  --val-split "${VAL_SPLIT}" \
  --batch-size 16 \
  --val-batch-size 16 \
  --test-frac 0.10 \
  --input-path-key img_path \
  --outdir "${OUT_DIR}"

echo "Done training with fixed splits."






