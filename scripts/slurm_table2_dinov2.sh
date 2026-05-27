#!/bin/bash
#SBATCH --job-name=table2_dinov2
#SBATCH --output=logs/table2_dinov2_%j.out
#SBATCH --error=logs/table2_dinov2_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=general

set -euo pipefail

PROJ_ROOT="${HOME}/2025-promega-mini-test"
CONDA_PREFIX="${HOME}/.conda/envs/core_env"
PYTHON="${CONDA_PREFIX}/bin/python3"

mkdir -p "${PROJ_ROOT}/logs"

module purge 2>/dev/null || true

echo "Project root: ${PROJ_ROOT}"
nvidia-smi || true

cd "${PROJ_ROOT}"
export PYTHONPATH="${PROJ_ROOT}"
export DATA_DIR=/net/projects2/promega/2026_04_15_data

echo "=== Table 2: DINOv2 + ResNet + EfficientNet per-day classifiers ==="
echo "Image dir: ${DATA_DIR}/intermediate/resized_512x384"
echo "Split dir: ${DATA_DIR}/intermediate/data_splits"

"${PYTHON}" -u analysis/images/classifier/train_model_accuracy_tony_dinov2.py \
    --train-split data_splits/both_train_base.json \
    --val-split data_splits/both_val_base.json \
    --test-split data_splits/both_test_base.json

echo "Done at $(date)"
