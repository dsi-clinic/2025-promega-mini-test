#!/bin/bash
#SBATCH --job-name=kfold_2
#SBATCH --output=logs/per_day_study_kfold_2_%A_%a.out
#SBATCH --error=logs/per_day_study_kfold_2_%A_%a.err
#SBATCH --time=8:00:00
#SBATCH --array=198-296
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=general

cd ${PROJ_ROOT:-/home/YOUR_GITHUB_USERNAME/promega-classifier}/image_classifier/comparison_runs
mkdir -p logs
cd ${PROJ_ROOT:-/home/YOUR_GITHUB_USERNAME/promega-classifier}
if [[ -f .env ]]; then set -a; source .env; set +a; fi
export BASE_PATH="${BASE_PATH:-/net/projects2/promega/data-analysis}"
export OUTPUT_FOLDER="${OUTPUT_FOLDER:-/net/projects2/promega/data-analysis/output}"
export PREDICTIONS_DIR="${PREDICTIONS_DIR:-/net/projects2/promega/data-analysis/predictions}"
export MANUAL_MASKS_DIR="${MANUAL_MASKS_DIR:-/net/projects2/promega/data-analysis/manual-masks}"
export RAW_IMAGE_DATA="${RAW_IMAGE_DATA:-$BASE_PATH/raw-images}"
export IMAGE_VERIFICATION_FORM="${IMAGE_VERIFICATION_FORM:-$BASE_PATH/image_verification_form}"
export PLOTS_FOLDER="${PLOTS_FOLDER:-$OUTPUT_FOLDER/../plots}"
export LOGS_FOLDER="${LOGS_FOLDER:-$OUTPUT_FOLDER/../logs}"
export NPY_OUTPUTS="${NPY_OUTPUTS:-$OUTPUT_FOLDER/npy}"
export SURVEY_RESULTS="${SURVEY_RESULTS:-$BASE_PATH/survey_results}"
export META_FILE="${META_FILE:-$BASE_PATH/meta_file}"
export RAW_IMAGE_MAPPING_JSON="${RAW_IMAGE_MAPPING_JSON:-$BASE_PATH/raw_image_mapping.json}"
export TARGET_WIDTH="${TARGET_WIDTH:-224}"
export TARGET_HEIGHT="${TARGET_HEIGHT:-224}"
export TRAIN_RESIZED_DIR="${TRAIN_RESIZED_DIR:-$OUTPUT_FOLDER/train_resized}"
export TRAIN_MANUAL_MAPPING_DIR="${TRAIN_MANUAL_MAPPING_DIR:-$OUTPUT_FOLDER/train_manual_mapping}"
export TRAIN_MANUAL_PROCESSED_DIR="${TRAIN_MANUAL_PROCESSED_DIR:-$OUTPUT_FOLDER/train_manual_processed}"
export TRAIN_SPLITS_DIR="${TRAIN_SPLITS_DIR:-$OUTPUT_FOLDER/train_splits}"
export INFER_RESIZED_DIR="${INFER_RESIZED_DIR:-$OUTPUT_FOLDER/infer_resized}"
export INFER_MAPPING_TOTAL_JSON="${INFER_MAPPING_TOTAL_JSON:-$OUTPUT_FOLDER/infer_mapping_total.json}"
export MANUAL_THRESHOLD_MAPPING="${MANUAL_THRESHOLD_MAPPING:-$OUTPUT_FOLDER/manual_threshold_mapping}"

PYTHON=/net/projects2/promega/bin/python3
IDX=${SLURM_ARRAY_TASK_ID}

echo "5-fold study part 2: array_index=$IDX"
cd ${PROJ_ROOT:-/home/YOUR_GITHUB_USERNAME/promega-classifier}/image_classifier/comparison_runs
$PYTHON run_per_day_study_kfold.py --array_index $IDX
echo "Done at $(date)"
