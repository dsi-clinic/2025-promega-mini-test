#!/bin/bash
#SBATCH --job-name=pd_ov_pd
#SBATCH --output=comparison_runs/logs/per_day_study_overlay_per_day_%A_%a.out
#SBATCH --error=comparison_runs/logs/per_day_study_overlay_per_day_%A_%a.err
#SBATCH --time=8:00:00
#SBATCH --array=0-10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=general

# CONFIGURE: Set PROJ_ROOT or replace YOUR_GITHUB_USERNAME below.
PARENT="${PROJ_ROOT:-/home/YOUR_GITHUB_USERNAME/promega-classifier}"
REPO="$PARENT"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export PYTHONHASHSEED=42
cd $PARENT
mkdir -p comparison_runs/logs
[ -f "$REPO/.env" ] && set -a && source "$REPO/.env" && set +a
export BASE_PATH="${BASE_PATH:-/net/projects2/promega/data-analysis}"
export OUTPUT_FOLDER="${OUTPUT_FOLDER:-/net/projects2/promega/data-analysis/output}"
export RAW_IMAGE_DATA="${RAW_IMAGE_DATA:-/net/projects2/promega/data-analysis/raw-images}"
export IMAGE_VERIFICATION_FORM="${IMAGE_VERIFICATION_FORM:-/net/projects2/promega/data-analysis/image-verification-form.json}"
export PLOTS_FOLDER="${PLOTS_FOLDER:-/net/projects2/promega/data-analysis/plots}"
export LOGS_FOLDER="${LOGS_FOLDER:-/net/projects2/promega/data-analysis/logs}"
export NPY_OUTPUTS="${NPY_OUTPUTS:-/net/projects2/promega/data-analysis/npy-outputs}"
export PREDICTIONS_DIR="${PREDICTIONS_DIR:-/net/projects2/promega/data-analysis/predictions}"
export SURVEY_RESULTS="${SURVEY_RESULTS:-/net/projects2/promega/data-analysis/survey-results}"
export MANUAL_MASKS_DIR="${MANUAL_MASKS_DIR:-/net/projects2/promega/data-analysis/manual-masks}"
export META_FILE="${META_FILE:-/net/projects2/promega/data-analysis/metadata.json}"
export RAW_IMAGE_MAPPING_JSON="${RAW_IMAGE_MAPPING_JSON:-/net/projects2/promega/data-analysis/image-mapping.json}"
export TARGET_WIDTH="${TARGET_WIDTH:-512}"
export TARGET_HEIGHT="${TARGET_HEIGHT:-384}"
export TRAIN_RESIZED_DIR="${TRAIN_RESIZED_DIR:-/net/projects2/promega/data-analysis/train-resized}"
export TRAIN_MANUAL_MAPPING_DIR="${TRAIN_MANUAL_MAPPING_DIR:-/net/projects2/promega/data-analysis/train-mapping}"
export TRAIN_MANUAL_PROCESSED_DIR="${TRAIN_MANUAL_PROCESSED_DIR:-/net/projects2/promega/data-analysis/train-processed}"
export TRAIN_SPLITS_DIR="${TRAIN_SPLITS_DIR:-/net/projects2/promega/data-analysis/train-splits}"
export INFER_RESIZED_DIR="${INFER_RESIZED_DIR:-/net/projects2/promega/data-analysis/infer-resized}"
export INFER_MAPPING_TOTAL_JSON="${INFER_MAPPING_TOTAL_JSON:-/net/projects2/promega/data-analysis/infer-mapping.json}"
export MANUAL_THRESHOLD_MAPPING="${MANUAL_THRESHOLD_MAPPING:-/net/projects2/promega/data-analysis/threshold-mapping.json}"

DAYS=(6 8 10 13 15 17 20.5 24 26 28 30)
DAY=${DAYS[$SLURM_ARRAY_TASK_ID]}
PYTHON=/net/projects2/promega/bin/python3

echo "Per-day study OVERLAY: per_day model, day=$DAY (array $SLURM_ARRAY_TASK_ID)"
$PYTHON comparison_runs/run_per_day_study.py --model_type per_day --day $DAY --input_mode overlay
echo "Done at $(date)"
