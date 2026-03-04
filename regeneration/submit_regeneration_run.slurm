#!/bin/bash
#SBATCH --job-name=regen_run
#SBATCH --output=regeneration/logs/regen_%j.out
#SBATCH --error=regeneration/logs/regen_%j.err
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=general
# Submit with: sbatch --export=RUN_NUMBER=1 submit_regeneration_run.slurm
# For all 3 in parallel: ./submit_all_regeneration_jobs.sh
# To re-run without overwriting (e.g. after seed fix): ./submit_all_regeneration_jobs_seed_fix.sh

if [ -z "${SLURM_JOB_ID}" ]; then
  echo "ERROR: Run with sbatch."
  exit 1
fi

RUN_NUMBER="${RUN_NUMBER:-1}"
RUN_SUFFIX="${RUN_SUFFIX:-}"   # e.g. _seed_fix → run_1_seed_fix, run_2_seed_fix, run_3_seed_fix
REPO="/home/tonyluo/amanda_temporal/2025-promega-mini-test"
REGEN="/home/tonyluo/amanda_temporal/regeneration"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export PYTHONHASHSEED=42
PARENT="/home/tonyluo/amanda_temporal"
OUT_DIR="$REGEN/run_${RUN_NUMBER}${RUN_SUFFIX}"

mkdir -p "$REGEN/logs"
echo "Regeneration run $RUN_NUMBER${RUN_SUFFIX} on $(hostname) -> $OUT_DIR"

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

# Use cluster Python (has torch, typing_extensions); fallback to python3
PYTHON="${PYTHON:-/net/projects2/promega/bin/python3}"
# comparison_runs is in PARENT (amanda_temporal)
cd "$PARENT"
mkdir -p "$OUT_DIR"
MODELS="${MODELS:-both}"
$PYTHON comparison_runs/run_threshold_study.py --output_dir "$OUT_DIR" --models "$MODELS"
$PYTHON comparison_runs/export_metrics_csv.py --setup overlay --overlay_dir "$OUT_DIR/per_day_study_overlay" -o "$OUT_DIR/metrics.csv"

echo "Done regeneration run $RUN_NUMBER${RUN_SUFFIX} at $(date)."
