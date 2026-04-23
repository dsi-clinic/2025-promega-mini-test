#!/bin/bash
#SBATCH --job-name=organoid_strips
#SBATCH --output=logs/organoid_strips_%j.out
#SBATCH --error=logs/organoid_strips_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=general

set -euo pipefail

PROJ_ROOT="${HOME}/2025-promega-mini-test"
CONDA_PREFIX="${HOME}/.conda/envs/core_env"
PYTHON="${CONDA_PREFIX}/bin/python3"

mkdir -p "${PROJ_ROOT}/logs"

module purge 2>/dev/null || true

cd "${PROJ_ROOT}"
export PYTHONPATH="${PROJ_ROOT}"

echo "=== Generating organoid strips ==="
"${PYTHON}" -u analysis/images/classifier/generate_organoid_strips.py \
    --overlay_dir /net/projects2/promega/2026_04_15_data/intermediate/overlays \
    --splits_csv data/2026_winter_student_splits.csv \
    --out_dir analysis/images/classifier/organoid_strips \
    --split all

echo "Done at $(date)"
