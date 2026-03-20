#!/bin/bash
#SBATCH --job-name=perday-image-study
#SBATCH --partition=general
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

PROJ_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
CONDA_PREFIX=/net/projects2/promega

mkdir -p "${PROJ_ROOT}/logs"

if command -v module >/dev/null 2>&1; then
  module purge
fi

echo "Project root: ${PROJ_ROOT}"
echo "Conda prefix: ${CONDA_PREFIX}"
nvidia-smi || true

cd "${PROJ_ROOT}"
export PYTHONPATH=.

echo "=== Running per-day image study ==="
"${CONDA_PREFIX}/bin/python" -u -m analysis.images.classifier.run_study "$@"

echo "Done."
