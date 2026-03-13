#!/bin/bash
#SBATCH --job-name=kfold_study
#SBATCH --output=logs/per_day_study_kfold_%A_%a.out
#SBATCH --error=logs/per_day_study_kfold_%A_%a.err
#SBATCH --time=8:00:00
#SBATCH --array=0-494%40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=general

cd ${PROJ_ROOT:-/home/YOUR_GITHUB_USERNAME/promega-classifier}/image_classifier/comparison_runs
mkdir -p logs
cd ${PROJ_ROOT:-/home/YOUR_GITHUB_USERNAME/promega-classifier}
[ -f .env ] && while IFS= read -r line; do [[ "$line" =~ ^[[:space:]]*# ]] || [[ -z "${line// }" ]] && continue; export "$line"; done < .env
export BASE_PATH="${BASE_PATH:-/net/projects2/promega/data-analysis}"
export OUTPUT_FOLDER="${OUTPUT_FOLDER:-/net/projects2/promega/data-analysis/output}"
export PREDICTIONS_DIR="${PREDICTIONS_DIR:-/net/projects2/promega/data-analysis/predictions}"
export MANUAL_MASKS_DIR="${MANUAL_MASKS_DIR:-/net/projects2/promega/data-analysis/manual-masks}"

PYTHON=/net/projects2/promega/bin/python3
IDX=${SLURM_ARRAY_TASK_ID}

echo "5-fold study: array_index=$IDX (fold/setup/model/day decoded inside script)"
cd ${PROJ_ROOT:-/home/YOUR_GITHUB_USERNAME/promega-classifier}/image_classifier/comparison_runs
$PYTHON run_per_day_study_kfold.py --array_index $IDX
echo "Done at $(date)"
