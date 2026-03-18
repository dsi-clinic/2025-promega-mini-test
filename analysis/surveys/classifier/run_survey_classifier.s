#!/bin/bash
#SBATCH --job-name=survey-classifier
#SBATCH --partition=general
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=logs/survey_%j.out
#SBATCH --error=logs/survey_%j.err

set -euo pipefail

# ====== adjust these paths ======
PROJ_ROOT=/home/YOUR_GITHUB_NAME/YOUR_MINITEST_DIRECTORY
PY=${PROJ_ROOT}/analysis/surveys/classifier/simple_classifier.py
DATA_DIR=/path/to/data
CONDA_PREFIX=/net/projects2/promega
# ================================

cd "${PROJ_ROOT}"
mkdir -p analysis/surveys/classifier/logs

echo "Running survey classifier on compute node"
echo "Hostname: $(hostname)"
echo "Date: $(date)"
echo "Conda prefix: ${CONDA_PREFIX}"

# Check GPU availability
nvidia-smi || true

# Run the survey classifier
export PYTHONPATH=.
"${CONDA_PREFIX}/bin/python" -u "${PY}" --data-dir "${DATA_DIR}" --deterministic

echo "Survey classifier completed successfully"
