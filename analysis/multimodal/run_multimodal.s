#!/bin/bash
#SBATCH --job-name=multimodal_sweep
#SBATCH --partition=general
#SBATCH --array=0-35                  # 36 jobs: 3 backbones × 2 fusions × 3 proj-dims × 2 growth
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --output=analysis/multimodal/logs/multimodal_%A_%a.out
#SBATCH --error=analysis/multimodal/logs/multimodal_%A_%a.err

set -e

PROJ_ROOT=/net/projects/CLS/lding/gitcode/2025-promega-mini-test

source /opt/conda/etc/profile.d/conda.sh
conda activate /net/projects2/promega

cd $PROJ_ROOT
mkdir -p analysis/multimodal/logs

# -------------------------------------------------------
# Parameter grid
# 3 backbones × 2 fusions × 3 proj-dims × 2 growth = 36
# -------------------------------------------------------
BACKBONES=(resnet efficientnet vit)   # index 0-2
FUSIONS=(concat gated)                # index 0-1
PROJ_DIMS=(32 64 128)                 # index 0-2
GROWTHS=(0 1)                         # 0 = off, 1 = on

# Decode task ID
# backbone  = task_id % 3
# fusion    = (task_id / 3) % 2
# proj_dim  = (task_id / 6) % 3
# growth    = (task_id / 18) % 2
BACKBONE=${BACKBONES[$((  $SLURM_ARRAY_TASK_ID        % 3 ))]}
FUSION=${FUSIONS[$((     ($SLURM_ARRAY_TASK_ID / 3)   % 2 ))]}
PROJ_DIM=${PROJ_DIMS[$((  ($SLURM_ARRAY_TASK_ID / 6)  % 3 ))]}
GROWTH=${GROWTHS[$((      ($SLURM_ARRAY_TASK_ID / 18) % 2 ))]}

# Build output dir name and optional flag
GROWTH_TAG="no_growth"
GROWTH_FLAG=""
if [ "$GROWTH" -eq 1 ]; then
    GROWTH_TAG="growth"
    GROWTH_FLAG="--use-growth-features"
fi

OUTDIR="analysis/multimodal/outputs_multimodal/${BACKBONE}_${FUSION}_proj${PROJ_DIM}_${GROWTH_TAG}"

echo "========================================="
echo "Multimodal Sweep"
echo "Job Array ID : $SLURM_ARRAY_JOB_ID"
echo "Task ID      : $SLURM_ARRAY_TASK_ID"
echo "Backbone     : $BACKBONE"
echo "Fusion       : $FUSION"
echo "Proj-dim     : $PROJ_DIM"
echo "Growth feats : $GROWTH_TAG"
echo "Output dir   : $OUTDIR"
echo "Date         : $(date)"
echo "Host         : $(hostname)"
python - <<'PY'
import torch
print("CUDA available:", torch.cuda.is_available())
print("GPUs:", torch.cuda.device_count())
print("Torch:", torch.__version__)
PY
echo "========================================="

python analysis/multimodal/train_multimodal.py \
    --backbone        "$BACKBONE" \
    --fusion-strategy "$FUSION" \
    --proj-dim        "$PROJ_DIM" \
    --use-metabolites \
    --use-augmentation \
    $GROWTH_FLAG \
    --batch-size              16 \
    --learning-rate           1e-3 \
    --num-epochs-phase1       50 \
    --early-stopping-patience 20 \
    --output-dir "$OUTDIR"

echo "========================================="
echo "Training Complete"
echo "Date: $(date)"
echo "========================================="
