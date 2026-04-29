#!/bin/bash
# Previously: run_tony_dinov2_fixed_splits.s
#SBATCH --job-name=dinov2-fixed-splits
#SBATCH --partition=general
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%A.out
#SBATCH --error=logs/%x_%A.err

set -euo pipefail
cd "$(dirname "$0")/../.."   # repo root, auto-detected
mkdir -p logs

make analysis-train-dinov2
