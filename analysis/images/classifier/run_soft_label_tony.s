#!/bin/bash
#SBATCH --job-name=soft-label-tony
#SBATCH --partition=general
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

# ====== adjust these paths ======
PROJ_ROOT=/home/tonyluo/minitest
PY=${PROJ_ROOT}/analysis/images/classifier/train_model_soft_labels.py
DATA_DIR=${PROJ_ROOT}/analysis/images/classifier/data/preprocessed/512x384/raw_votes
OUT_DIR=/net/projects2/promega/tony_results/outputs_512x384_softlabels_tony
CONDA_PREFIX=/net/projects2/promega
# ================================

mkdir -p logs "${OUT_DIR}"

if command -v module >/dev/null 2>&1; then
  module purge
fi

echo "Conda prefix: ${CONDA_PREFIX}"
nvidia-smi || true
echo "Running ${PY}"
echo "DATA_DIR=${DATA_DIR}"

ARGS=(
  --data_dir "${DATA_DIR}"
  --batch-size 16
  --val-batch-size 16
  --min-votes 1
  --weight-by-votes
  --outdir "${OUT_DIR}"
)

cd "${PROJ_ROOT}"
export PYTHONPATH="${PROJ_ROOT}:$PYTHONPATH"
${CONDA_PREFIX}/bin/python3 -u "${PY}" "${ARGS[@]}"

echo "Done."
