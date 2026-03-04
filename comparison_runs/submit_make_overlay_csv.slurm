#!/bin/bash
#SBATCH --job-name=overlay_csv
#SBATCH --output=comparison_runs/logs/make_overlay_csv_%j.out
#SBATCH --error=comparison_runs/logs/make_overlay_csv_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=general
# Run with: sbatch submit_make_overlay_csv.slurm  (executes on compute node)

# Ensure we run on a compute node (SLURM sets SLURM_JOB_ID when allocated)
if [ -z "${SLURM_JOB_ID}" ]; then
  echo "ERROR: Run with sbatch, not directly. Example: sbatch submit_make_overlay_csv.slurm"
  exit 1
fi
echo "Running on compute node: $(hostname)"

# Final output: comparison_runs/overlay_threshold_study_results.csv
# Run from amanda_temporal: sbatch comparison_runs/submit_make_overlay_csv.slurm
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
export TARGET_WIDTH="${TARGET_WIDTH:-512}"
export TARGET_HEIGHT="${TARGET_HEIGHT:-384}"
for k in PREDICTIONS_DIR SURVEY_RESULTS MANUAL_MASKS_DIR META_FILE RAW_IMAGE_MAPPING_JSON TRAIN_RESIZED_DIR TRAIN_MANUAL_MAPPING_DIR TRAIN_MANUAL_PROCESSED_DIR TRAIN_SPLITS_DIR INFER_RESIZED_DIR INFER_MAPPING_TOTAL_JSON MANUAL_THRESHOLD_MAPPING; do
  export "${k}=${!k:-/net/projects2/promega/data-analysis}"
done
echo "Producing overlay_threshold_study_results.csv"
python3 comparison_runs/make_overlay_csv.py
echo "Done. CSV: $(ls -la comparison_runs/overlay_threshold_study_results.csv 2>/dev/null)"
