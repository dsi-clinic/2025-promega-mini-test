#!/bin/bash
#SBATCH --job-name=fig7_ensemble
#SBATCH --output=logs/fig7_ensemble_%j.out
#SBATCH --error=logs/fig7_ensemble_%j.err
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

echo "=== Fig7 deep ensemble per-day classifier ==="
"${PYTHON}" -u analysis/images/classifier/train_model_deep_ensemble.py \
    --data_dir analysis/images/classifier/data/preprocessed/512x384/majority

echo "Done at $(date)"
