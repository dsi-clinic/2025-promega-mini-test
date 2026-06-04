#!/bin/bash
#SBATCH --job-name=combined-v2
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=6:00:00
#SBATCH --output=logs/combined_v2_%j.out
#SBATCH --error=logs/combined_v2_%j.err

# Combined Image + Metabolite Classifier v2
# GPU is used for EfficientNet extraction, NN projector training, and MLP training.
#
# Configurable via env vars (override before sbatch) or pass extra CLI args at the end.
#
# Key env vars:
#   IMAGE_REDUCER   pca | nn_projector   (default: nn_projector)
#   CLASSIFIER      lgbm | mlp           (default: mlp)
#   N_IMG_FEATS     integer              (default: 12)
#   SCORING         f1_notaccept | f1_weighted | recall_notaccept | macro_f1
#                                        (default: f1_notaccept; ignored for mlp)
#   IMBALANCE       class_weight | scale_pos_weight | smote
#                                        (default: class_weight; ignored for mlp)
#   DAY_FILTER      e.g. Dy03            (default: unset = all days)
#
# Usage examples:
#   sbatch submit_combined.slurm                                   # default (nn_projector + mlp)
#   DAY_FILTER=Dy03 sbatch submit_combined.slurm                   # smoke test one day
#   IMAGE_REDUCER=pca CLASSIFIER=lgbm sbatch submit_combined.slurm # original approach
#   IMAGE_REDUCER=nn_projector CLASSIFIER=lgbm sbatch submit_combined.slurm

set -euo pipefail

# CONFIGURE: Set PROJ_ROOT to the repo root (parent of analysis/).
PROJ_ROOT="${PROJ_ROOT:-/home/liyading/gitcode/2025-promega-mini-test}"
DATA_ROOT="$PROJ_ROOT"

# Model configuration (override via env vars)
IMAGE_REDUCER="${IMAGE_REDUCER:-nn_projector}"
CLASSIFIER="${CLASSIFIER:-mlp}"
N_IMG_FEATS="${N_IMG_FEATS:-12}"
SCORING="${SCORING:-f1_notaccept}"
IMBALANCE="${IMBALANCE:-class_weight}"
DAY_FILTER="${DAY_FILTER:-}"

EXTRA_ARGS="${@}"

mkdir -p "${PROJ_ROOT}/logs"

if command -v module >/dev/null 2>&1; then
    module purge 2>/dev/null || true
fi

source /opt/conda/etc/profile.d/conda.sh
conda activate core_env

echo "=============================================="
echo "Combined Image+Metabolite Classifier v2"
echo "=============================================="
echo "Job ID:        ${SLURM_JOB_ID}"
echo "Node:          ${SLURM_NODELIST}"
echo "GPU:           ${CUDA_VISIBLE_DEVICES:-none}"
echo "Project:       ${PROJ_ROOT}"
echo "Data root:     ${DATA_ROOT}"
echo "Image reducer: ${IMAGE_REDUCER}"
echo "Classifier:    ${CLASSIFIER}"
echo "N img feats:   ${N_IMG_FEATS}"
echo "Scoring:       ${SCORING}"
echo "Imbalance:     ${IMBALANCE}"
echo "Day filter:    ${DAY_FILTER:-all}"
echo "Extra args:    ${EXTRA_ARGS}"
echo "Start:         $(date)"
echo ""

nvidia-smi || true
echo ""

cd "${PROJ_ROOT}/analysis/multimodal"

# Build optional --day_filter flag
DAY_FILTER_ARG=""
if [ -n "${DAY_FILTER}" ]; then
    DAY_FILTER_ARG="--day_filter ${DAY_FILTER}"
fi

python3 -u train_combined.py \
    --data_dir "${DATA_ROOT}" \
    --image_reducer "${IMAGE_REDUCER}" \
    --classifier "${CLASSIFIER}" \
    --n_image_features "${N_IMG_FEATS}" \
    --scoring "${SCORING}" \
    --imbalance "${IMBALANCE}" \
    --n_folds 3 \
    ${DAY_FILTER_ARG} \
    ${EXTRA_ARGS}

EXIT_CODE=$?

echo ""
echo "=============================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "DONE - Results in ${PROJ_ROOT}/analysis/multimodal/outputs/"
else
    echo "FAILED (exit $EXIT_CODE)"
fi
echo "End: $(date)"
echo "=============================================="

exit $EXIT_CODE
