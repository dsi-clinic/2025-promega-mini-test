#!/bin/bash
#SBATCH --job-name=multimodal
#SBATCH --partition=general           # correct partition
#SBATCH --gres=gpu:a100:1             # request one A100 GPU
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=analysis/multimodal/logs/multimodal_%j.out
#SBATCH --error=analysis/multimodal/logs/multimodal_%j.err

set -e

# Project root
PROJ_ROOT=/home/waggonere/2025-promega-mini-test

# Proper conda activation
source /opt/conda/etc/profile.d/conda.sh
conda activate /net/projects2/promega

cd $PROJ_ROOT
mkdir -p analysis/multimodal/logs

echo "========================================="
echo "Starting Multimodal Training"
echo "Date: $(date)"
echo "Hostname: $(hostname)"
python - <<'PY'
import torch
print("CUDA available:", torch.cuda.is_available())
print("GPUs:", torch.cuda.device_count())
print("Torch:", torch.__version__)
PY
echo "========================================="

# ---- Run your multimodal experiment ----
python analysis/multimodal/train_multimodal.py \
    --backbone vit \
    --input-mode rgb \
    --use-metabolites \
    --fusion-strategy concat \
    --batch-size 16 \
    --learning-rate 1e-3 \
    --num-epochs-phase1 50 \
    --num-epochs-phase2 100 \
    --early-stopping-patience 20 \
    --use-augmentation \
    --output-dir analysis/multimodal/outputs_multimodal/vit_rgb_metabolites_concat

echo "========================================="
echo "Training Complete"
echo "Date: $(date)"
echo "========================================="
