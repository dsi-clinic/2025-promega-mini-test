#!/bin/bash
#SBATCH --job-name=survey-tony
#SBATCH --partition=general
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=logs/survey_tony_%j.out
#SBATCH --error=logs/survey_tony_%j.err

set -uo pipefail

# ====== adjust these paths ======
PROJ_ROOT=/home/tonyluo/minitest
PY=${PROJ_ROOT}/analysis/surveys/classifier/simple_classifier_tony.py
CONDA_PREFIX=/net/projects2/promega
# ================================

cd "${PROJ_ROOT}"
mkdir -p analysis/surveys/classifier/logs

echo "Running ENHANCED survey classifier (EfficientNetB7 + Focal Loss) on compute node"
echo "Hostname: $(hostname)"
echo "Date: $(date)"
echo "Conda prefix: ${CONDA_PREFIX}"

# Check GPU availability
nvidia-smi || true

# Run the enhanced survey classifier
export PYTHONPATH=.
"${CONDA_PREFIX}/bin/python" -u "${PY}"

echo "Enhanced survey classifier completed successfully"
