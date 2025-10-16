#!/bin/bash
#SBATCH --job-name=survey-classifier
#SBATCH --partition=general
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --output=logs/survey_%j.out
#SBATCH --error=logs/survey_%j.err

set -euo pipefail

# ====== adjust these paths ======
PROJ_ROOT=/home/tonyluo/minitest
PY=${PROJ_ROOT}/analysis/surveys/classifier/simple_classifier.py
CONDA_PREFIX=/net/projects2/promega
# ================================

cd "${PROJ_ROOT}"
mkdir -p analysis/surveys/classifier/logs

echo "Running survey classifier on compute node"
echo "Hostname: $(hostname)"
echo "Date: $(date)"

# Run the survey classifier
export PYTHONPATH=.
"${CONDA_PREFIX}/bin/python" "${PY}"

echo "Survey classifier completed successfully"
