#!/bin/bash
#SBATCH --job-name=calc_train_tnr
#SBATCH --output=logs/calc_train_tnr_%j.out
#SBATCH --error=logs/calc_train_tnr_%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --partition=general

# Set up environment
cd ${PROJ_ROOT:-/home/YOUR_GITHUB_USERNAME/promega-classifier}/image_classifier/comparison_runs
mkdir -p logs
cd ${PROJ_ROOT:-/home/YOUR_GITHUB_USERNAME/promega-classifier}

# Load environment variables from .env file
if [ -f .env ]; then
    while IFS= read -r line; do
        [[ "$line" =~ ^[[:space:]]*# ]] && continue
        [[ -z "${line// }" ]] && continue
        export "$line"
    done < .env
    echo "Loaded environment variables from .env"
fi

# Set required paths if not already set (fallback)
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

# Python environment
PYTHON=/net/projects2/promega/bin/python3

echo "========================================="
echo "Calculate Training TNR for CNN-LSTM"
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo ""

# Run calculation
cd ${PROJ_ROOT:-/home/YOUR_GITHUB_USERNAME/promega-classifier}/image_classifier/comparison_runs
$PYTHON calculate_training_tnr.py

echo ""
echo "Calculation completed at: $(date)"
