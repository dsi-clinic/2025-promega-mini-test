#!/bin/bash
#SBATCH --job-name=table2_backbones
#SBATCH --output=logs/table2_backbones_%j.out
#SBATCH --error=logs/table2_backbones_%j.err
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
export DATA_DIR=/net/projects2/promega/2026_04_data

echo "=== Table 2: ViT + ResNet per-day classifiers ==="
"${PYTHON}" -u analysis/images/classifier/train_table2_backbones.py \
    --data_dir analysis/images/classifier/data/preprocessed/512x384/majority

echo "Done at $(date)"
