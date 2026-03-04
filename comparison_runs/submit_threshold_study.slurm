#!/bin/bash
#SBATCH --job-name=thr_study
#SBATCH --output=comparison_runs/logs/threshold_study_%j.out
#SBATCH --error=comparison_runs/logs/threshold_study_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=general
# Run with: sbatch submit_threshold_study.slurm  (executes on compute node)

# Ensure we run on a compute node (SLURM sets SLURM_JOB_ID when allocated)
if [ -z "${SLURM_JOB_ID}" ]; then
  echo "ERROR: Run with sbatch, not directly. Example: sbatch submit_threshold_study.slurm"
  exit 1
fi
echo "Running on compute node: $(hostname)"

# Overlay-only threshold study: per_day + effnet_ts, 11 days each. No checkpoints (small disk).
# Final result: comparison_runs/overlay_threshold_study_results.csv
# Run from amanda_temporal: sbatch comparison_runs/submit_threshold_study.slurm
PARENT="/home/tonyluo/amanda_temporal"
REPO="$PARENT/2025-promega-mini-test"
cd "$PARENT"
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

PYTHON=${PYTHON:-/usr/bin/python3}

echo "Threshold study: overlay, per_day + effnet_ts, 11 days each; then export CSV (no .pth saved)"
$PYTHON comparison_runs/run_threshold_study.py
echo "Done at $(date). Check overlay_threshold_study_results.csv"
