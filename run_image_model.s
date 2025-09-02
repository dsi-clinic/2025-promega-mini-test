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
PY=analysis/images/classifier/train_model_enhanced.py     # your Python entrypoint
DATA_DIR=/net/scratch/jiaweizhang/2025-promega-mini-test/analysis/images/classifier/data           # Dy*.json folder (input)
OUT_ROOT=/net/scratch/jiaweizhang/2025-promega-mini-test/analysis/images/classifier/organoid_cls   # outputs folder
CONDA_PREFIX=/net/projects2/promega                                                               # conda env prefix
# --------------------------

mkdir -p logs

module purge
# (Optional) load CUDA/toolchain modules if your cluster requires them, e.g.:
# module load cuda/12.1

echo "Conda prefix: ${CONDA_PREFIX}"
nvidia-smi || true

# Matrix of runs — now includes "none"
declare -a AUGS=("none" "geom" "geom_photo")
declare -a BACKBONES=("vit" "resnet" "efficientnet")

for AUG in "${AUGS[@]}"; do
  for BB in "${BACKBONES[@]}"; do
    EXP_NAME="v1_${AUG}_${BB}_dupflip"
    echo "=== Running ${EXP_NAME} ==="

    # Run via conda without activating the shell; set PYTHONPATH for the run.
    PYTHONPATH=. conda run -p "${CONDA_PREFIX}" python "${PY}" \
      --data-dir "${DATA_DIR}" \
      --out-root "${OUT_ROOT}" \
      --exp-name "${EXP_NAME}" \
      --augment-level "${AUG}" \
      --duplicate-train-hflip \
      --batch-size 16 \
      --val-frac 0.10 \
      --test-frac 0.10 \
      --rot-deg 15 \
      --rrc-min-scale 0.9 \
      --rrc-max-scale 1.1 \
      --affine-deg 5 \
      --affine-trans 0.02 \
      --affine-min-scale 0.98 \
      --affine-max-scale 1.02 \
      --photo-noise-std 0.01
  done
done

echo "All runs completed."
