#!/bin/bash
#SBATCH --job-name=feature_corr
#SBATCH --partition=general
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

PROJ_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CONDA_PREFIX=/net/projects2/promega
SCRIPT="${PROJ_ROOT}/analysis/combined_model/compute_feature_correlation.py"
OUTDIR="${PROJ_ROOT}/analysis/combined_model/outputs/feature_correlation"

mkdir -p "${PROJ_ROOT}/analysis/combined_model/logs"
mkdir -p "${OUTDIR}"

echo "=============================================="
echo "  Feature Correlation Analysis"
echo "  Start  : $(date)"
echo "  Node   : ${SLURM_NODELIST}"
echo "  GPU    : ${CUDA_VISIBLE_DEVICES:-none}"
echo "  Output : ${OUTDIR}"
echo "=============================================="

nvidia-smi || true
cd "${PROJ_ROOT}"
${CONDA_PREFIX}/bin/python3 -u "${SCRIPT}" \
    --input-mode rgb \
    --batch-size 32 \
    --pca-dim 10 \
    --output-dir "${OUTDIR}"

echo "DONE — results in ${OUTDIR}"
echo "End: $(date)"
