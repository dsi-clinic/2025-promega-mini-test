#!/bin/bash
#SBATCH --job-name=multimodal_sweep
#SBATCH --partition=general           # correct partition
#SBATCH --array=0-15                  # 16 jobs: 2 backbones × 2 fusion × 4 input modes
#SBATCH --gres=gpu:a100:1             # request one A100 GPU
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --output=analysis/multimodal/logs/multimodal_%A_%a.out
#SBATCH --error=analysis/multimodal/logs/multimodal_%A_%a.err

set -e

# Project root
PROJ_ROOT=/home/waggonere/2025-promega-mini-test

# Proper conda activation
source /opt/conda/etc/profile.d/conda.sh
conda activate /net/projects2/promega

cd $PROJ_ROOT
mkdir -p analysis/multimodal/logs

# Define parameter arrays
BACKBONES=(resnet efficientnet)
FUSIONS=(concat gated)
INPUTS=(rgb overlay rgb_mask overlay_mask)

# Decode array task ID to parameters
# 2 backbones × 2 fusions × 4 inputs = 16 combinations
# b = (task_id / 8) % 2
# f = (task_id / 4) % 2
# i = task_id % 4
BACKBONE=${BACKBONES[$(( ($SLURM_ARRAY_TASK_ID / 8) % 2 ))]}
FUSION=${FUSIONS[$(( ($SLURM_ARRAY_TASK_ID / 4) % 2 ))]}
INPUT=${INPUTS[$(( $SLURM_ARRAY_TASK_ID % 4 ))]}

# Construct output directory
OUTDIR="analysis/multimodal/outputs_multimodal/${BACKBONE}_${INPUT}_${FUSION}"

echo "========================================="
echo "Starting Multimodal Training"
echo "Job Array ID: $SLURM_ARRAY_JOB_ID"
echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "Backbone: $BACKBONE"
echo "Input Mode: $INPUT"
echo "Fusion Strategy: $FUSION"
echo "Output Directory: $OUTDIR"
echo "Date: $(date)"
echo "Hostname: $(hostname)"
python - <<'PY'
import torch
print("CUDA available:", torch.cuda.is_available())
print("GPUs:", torch.cuda.device_count())
print("Torch:", torch.__version__)
PY
echo "========================================="

# Run multimodal training
python analysis/multimodal/train_multimodal.py \
    --backbone "$BACKBONE" \
    --input-mode "$INPUT" \
    --fusion-strategy "$FUSION" \
    --use-metabolites \
    --use-augmentation \
    --batch-size 16 \
    --learning-rate 1e-3 \
    --num-epochs-phase1 50 \
    --early-stopping-patience 20 \
    --output-dir "$OUTDIR"

echo "========================================="
echo "Training Complete"
echo "Date: $(date)"
echo "========================================="
