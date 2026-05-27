#!/bin/bash
#SBATCH --job-name=adaptive_multimodal
#SBATCH --partition=general
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=/home/YOUR_USERNAME/2025-promega-mini-test/analysis/combined_model/logs/adaptive_%j.out
#SBATCH --error=/home/YOUR_USERNAME/2025-promega-mini-test/analysis/combined_model/logs/adaptive_%j.err

set -euo pipefail

PROJ_ROOT=/home/YOUR_USERNAME/2025-promega-mini-test
CONDA_PREFIX=/net/projects2/promega
SCRIPT="${PROJ_ROOT}/analysis/combined_model/train_adaptive_cv.py"
OUTDIR="${PROJ_ROOT}/analysis/combined_model/outputs/adaptive_multimodal"

mkdir -p "${PROJ_ROOT}/analysis/combined_model/logs"
mkdir -p "${OUTDIR}"

echo "=============================================="
echo "  Adaptive Multimodal CV Run"
echo "  Start  : $(date)"
echo "  Node   : ${SLURM_NODELIST}"
echo "  GPU    : ${CUDA_VISIBLE_DEVICES:-none}"
echo "  Output : ${OUTDIR}"
echo "=============================================="

nvidia-smi || true
cd "${PROJ_ROOT}"
${CONDA_PREFIX}/bin/python3 -u "${SCRIPT}" \
    --backbone efficientnet \
    --input-mode rgb \
    --cross-attn-proj-dim 128 \
    --cross-attn-heads 4 \
    --use-projection \
    --proj-dim 256 \
    --batch-size 16 \
    --learning-rate 1e-3 \
    --num-epochs 50 \
    --early-stopping-patience 20 \
    --use-augmentation \
    --output-dir "${OUTDIR}" 

EXIT_CODE=$?

echo "=============================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "DONE — results in ${OUTDIR}"
else
    echo "FAILED with exit code ${EXIT_CODE}"
fi
echo "End: $(date)"
echo "=============================================="

exit $EXIT_CODE
