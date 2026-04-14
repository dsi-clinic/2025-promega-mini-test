#!/bin/bash
#SBATCH --job-name=fig7_per_day
#SBATCH --output=logs/fig7_per_day_%A_%a.out
#SBATCH --error=logs/fig7_per_day_%A_%a.err
#SBATCH --time=8:00:00
#SBATCH --array=0-10
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
export ANALYSIS_OUTPUT_DIR=/net/projects2/promega/2026_04_data/analysis_output

DAYS=(6 8 10 13 15 17 20.5 24 26 28 30)
DAY=${DAYS[$SLURM_ARRAY_TASK_ID]}

echo "=== Fig7 per_day model, day=${DAY} (array task ${SLURM_ARRAY_TASK_ID}) ==="
"${PYTHON}" -u -m analysis.images.classifier.run_per_day_study \
    --model_type per_day \
    --day "${DAY}"

echo "Done at $(date)"
