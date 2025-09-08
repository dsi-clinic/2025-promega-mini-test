#!/bin/bash
#SBATCH --job-name=promega-images-classifier
#SBATCH --partition=general
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

# --- Adjust these paths ---
PY=/net/scratch/jiaweizhang/2025-promega-mini-test/analysis/images/classifier/train_model_enhanced.py
DATA_DIR=/net/scratch/jiaweizhang/2025-promega-mini-test/analysis/images/classifier/data/preprocessed/512x384/majority
OUT_ROOT=/net/scratch/jiaweizhang/2025-promega-mini-test/analysis/images/classifier/organoid_cls
CONDA_PREFIX=/net/projects2/promega
# --------------------------

mkdir -p logs
mkdir -p "${OUT_ROOT}"

# Purge modules only if available
if command -v module >/dev/null 2>&1; then
  module purge
fi

echo "Conda prefix: ${CONDA_PREFIX}"
nvidia-smi || true

declare -a AUGS=("none" "geom" "geom_photo")
declare -a DUPS=("nodup" "dup")

for AUG in "${AUGS[@]}"; do
  for DUP in "${DUPS[@]}"; do
    EXP_NAME="v1_${AUG}_${DUP}"
    echo "=== Running ${EXP_NAME} ==="

    BASE_ARGS=(
      --data-dir "${DATA_DIR}"
      --out-root "${OUT_ROOT}"
      --exp-name "${EXP_NAME}"
      --augment-level "${AUG}"
      --batch-size 16
      --val-frac 0.10
      --test-frac 0.10
      --rot-deg 15
      --rrc-min-scale 0.9
      --rrc-max-scale 1.1
      --affine-deg 5
      --affine-trans 0.02
      --affine-min-scale 0.98
      --affine-max-scale 1.02
      --photo-noise-std 0.01
    )

    if [[ "${DUP}" == "dup" ]]; then
      BASE_ARGS+=( --duplicate-train-hflip )
    fi

    PYTHONPATH=. conda run -p "${CONDA_PREFIX}" python "${PY}" "${BASE_ARGS[@]}"
  done
done

echo "All runs completed."
