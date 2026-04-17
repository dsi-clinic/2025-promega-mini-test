#!/bin/bash
#SBATCH --job-name=perday-image-study
#SBATCH --partition=general
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -eo pipefail

PROJ_ROOT="/home/nickross/2025-promega-mini-test"
CONDA_PREFIX=/net/projects2/promega

mkdir -p "${PROJ_ROOT}/logs"

module purge 2>/dev/null || true

echo "Project root: ${PROJ_ROOT}"
echo "Conda prefix: ${CONDA_PREFIX}"
nvidia-smi || true

cd "${PROJ_ROOT}"
export PYTHONPATH=.
export ANALYSIS_OUTPUT_DIR=/net/projects2/promega/2026_04_15_data/analysis_output

echo "=== Running per-day image study ==="
"${CONDA_PREFIX}/bin/python" -u -m analysis.paper_2026_04.perday_image_study "$@"

echo "Done."
