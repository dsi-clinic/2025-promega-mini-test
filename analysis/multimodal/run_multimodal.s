#!/bin/bash
#SBATCH --job-name=multimodal_sweep
#SBATCH --partition=general
#SBATCH --array=0-15                                  # 2 backbones × 2 fusion × 4 input modes
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --output=analysis/multimodal/logs/multimodal_%A_%a.out
#SBATCH --error=analysis/multimodal/logs/multimodal_%A_%a.err

set -euo pipefail
cd "$(dirname "$0")/../.."   # repo root, auto-detected
mkdir -p analysis/multimodal/logs

# Decode SLURM_ARRAY_TASK_ID into (backbone, fusion, input)
BACKBONES=(resnet efficientnet)
FUSIONS=(concat gated)
INPUTS=(rgb overlay rgb_mask overlay_mask)
BACKBONE=${BACKBONES[$(( ($SLURM_ARRAY_TASK_ID / 8) % 2 ))]}
FUSION=${FUSIONS[$(( ($SLURM_ARRAY_TASK_ID / 4) % 2 ))]}
INPUT_MODE=${INPUTS[$(( $SLURM_ARRAY_TASK_ID % 4 ))]}

echo "Task $SLURM_ARRAY_TASK_ID  →  backbone=$BACKBONE  fusion=$FUSION  input=$INPUT_MODE"

make analysis-multimodal BACKBONE="$BACKBONE" FUSION="$FUSION" INPUT_MODE="$INPUT_MODE"
