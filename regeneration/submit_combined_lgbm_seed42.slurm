#!/bin/bash
#SBATCH --job-name=lgb_s42
#SBATCH --output=regeneration/logs/combined_lgbm_seed42_%j.out
#SBATCH --error=regeneration/logs/combined_lgbm_seed42_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=general
#
# Run combined LGB with original seed-42 splits, save to a dedicated folder.
# Does NOT overwrite the existing combined_model/outputs/ contents.

set -e

PARENT="/home/tonyluo/amanda_temporal"
REPO="$PARENT/2025-promega-mini-test"
PYTHON="${PYTHON:-/net/projects2/promega/bin/python3}"

mkdir -p "$PARENT/regeneration/logs"
cd "$PARENT"

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

OUTDIR="combined_model/outputs/combined_lgbm_f1_notaccept_class_weight_pca12"
DEST="combined_model/outputs/original_seed42"

echo "=== Combined LGB original seed 42 ($(hostname), $(date)) ==="

# Back up current outputs (from seed-rotation run)
if [ -d "$OUTDIR" ]; then
    cp -r "$OUTDIR" "${OUTDIR}_backup_$$"
    echo "Backed up existing outputs -> ${OUTDIR}_backup_$$"
fi

# Run with default data_dir (uses seed-42 splits in 2025-promega-mini-test/data_splits/)
$PYTHON combined_model/train_combined_lgbm.py

# Copy results to dedicated folder
mkdir -p "$DEST"
cp -r "$OUTDIR"/* "$DEST/"
echo "Copied original seed-42 results -> $DEST/"

# Restore previous outputs
if [ -d "${OUTDIR}_backup_$$" ]; then
    rm -rf "$OUTDIR"
    mv "${OUTDIR}_backup_$$" "$OUTDIR"
    echo "Restored previous outputs."
fi

echo "=== Done at $(date) ==="
