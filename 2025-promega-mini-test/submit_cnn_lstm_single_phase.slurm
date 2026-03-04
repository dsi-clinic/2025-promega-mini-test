#!/bin/bash
#SBATCH --job-name=cnn_lstm_single
#SBATCH --output=logs/cnn_lstm_single_%j.out
#SBATCH --error=logs/cnn_lstm_single_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8

# CNN-LSTM SINGLE-PHASE TRAINING
# All parameters trainable from start (no freezing)

echo "========================================="
echo "CNN-LSTM SINGLE-PHASE TRAINING"
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

# Setup
cd /home/your_name/image_classifier_ts/2025-promega-mini-test
mkdir -p logs

# Load environment variables from .env file (skip comments and empty lines)
if [ -f .env ]; then
    while IFS= read -r line; do
        # Skip comments and empty lines
        [[ "$line" =~ ^[[:space:]]*# ]] && continue
        [[ -z "${line// }" ]] && continue
        # Export the variable
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

echo "Python: $PYTHON"
echo "CUDA visible devices: $CUDA_VISIBLE_DEVICES"
echo ""

# Output directory
RUN_OUTDIR="/net/projects2/promega/data-analysis/output/cnn_lstm/models/run_${SLURM_JOB_ID}_single_phase"
echo "Run output dir: $RUN_OUTDIR"
echo ""

echo "========================================="
echo "Running train_organoid_lstm_single_phase.py"
echo "========================================="

$PYTHON analysis/images/cnn_lstm/train_organoid_lstm_single_phase.py \
    --epochs 20 \
    --batch-size 8 \
    --lr 0.0001 \
    --lstm-hidden 256 \
    --lstm-layers 2 \
    --output-dir "$RUN_OUTDIR"

TRAIN_EXIT=$?
echo ""
echo "Training exit code: $TRAIN_EXIT"

if [ $TRAIN_EXIT -ne 0 ]; then
    echo "ERROR: Training failed!"
    exit $TRAIN_EXIT
fi

echo ""
echo "========================================="
echo "JOB COMPLETE"
echo "========================================="
echo "End time: $(date)"
echo "Results directory: $RUN_OUTDIR"
echo ""

exit 0
