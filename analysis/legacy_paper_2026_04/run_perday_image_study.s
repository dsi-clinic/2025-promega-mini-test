#!/bin/bash
#SBATCH --job-name=perday-image-study
#SBATCH --partition=general
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd "$(dirname "$0")/../.."   # repo root, auto-detected
mkdir -p logs

make analysis-paper-perday ARGS="$*"
