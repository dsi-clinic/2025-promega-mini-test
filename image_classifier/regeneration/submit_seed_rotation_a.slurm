#!/bin/bash
#SBATCH --job-name=seed_rot_a
#SBATCH --output=regeneration/logs/seed_rotation_a_%j.out
#SBATCH --error=regeneration/logs/seed_rotation_a_%j.err
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=general
#
# Job A: seed 7 (per_day + effnet_ts + combined_lgbm), seed 13 + seed 99 combined_lgbm
# Estimated wall-clock: ~6-8 h
# -----------------------------------------------------------------------

set -e

# CONFIGURE: Set PROJ_ROOT or replace YOUR_GITHUB_USERNAME below.
PARENT="${PROJ_ROOT:-/home/YOUR_GITHUB_USERNAME/promega-classifier}"
REPO="$PARENT"
REGEN="$PARENT/image_classifier/regeneration"
PYTHON="${PYTHON:-/net/projects2/promega/bin/python3}"

export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export PYTHONHASHSEED=42

mkdir -p "$REGEN/logs"
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

echo "========================================"
echo "SEED ROTATION JOB A  ($(hostname), $(date))"
echo "========================================"

# ── Step 1: Generate splits for all 3 seeds ──
for S in 7 13 99; do
    echo "--- Generating splits for seed $S ---"
    $PYTHON "$REGEN/generate_seed_splits.py" --seed $S \
        --output_dir "$REGEN/seed_rotation_splits/s$S"
done

# ── Step 2: Seed 7 — per_day + effnet_ts ──
echo ""
echo "========================================"
echo "SEED 7: per_day + effnet_ts threshold study"
echo "========================================"
mkdir -p "$REGEN/seed_rotation_s7"
$PYTHON "$REGEN/run_seed_rotation_threshold.py" \
    --split_dir "$REGEN/seed_rotation_splits/s7/data_splits" \
    --output_dir "$REGEN/seed_rotation_s7" \
    --models both

$PYTHON image_classifier/comparison_runs/export_metrics_csv.py \
    --setup overlay \
    --overlay_dir "$REGEN/seed_rotation_s7/per_day_study_overlay" \
    -o "$REGEN/seed_rotation_s7/metrics.csv"

# ── Step 3: Seed 7 — combined LGB ──
echo ""
echo "========================================"
echo "SEED 7: combined LGB"
echo "========================================"
$PYTHON combined_model/train_combined_lgbm.py \
    --data_dir "$REGEN/seed_rotation_splits/s7"
mkdir -p "$REGEN/seed_rotation_s7/combined_lgbm"
cp -r combined_model/outputs/combined_lgbm_f1_notaccept_class_weight_pca12/* \
      "$REGEN/seed_rotation_s7/combined_lgbm/" 2>/dev/null || true

# ── Step 4: Seed 13 — combined LGB ──
echo ""
echo "========================================"
echo "SEED 13: combined LGB"
echo "========================================"
$PYTHON combined_model/train_combined_lgbm.py \
    --data_dir "$REGEN/seed_rotation_splits/s13"
mkdir -p "$REGEN/seed_rotation_s13/combined_lgbm"
cp -r combined_model/outputs/combined_lgbm_f1_notaccept_class_weight_pca12/* \
      "$REGEN/seed_rotation_s13/combined_lgbm/" 2>/dev/null || true

# ── Step 5: Seed 99 — combined LGB ──
echo ""
echo "========================================"
echo "SEED 99: combined LGB"
echo "========================================"
$PYTHON combined_model/train_combined_lgbm.py \
    --data_dir "$REGEN/seed_rotation_splits/s99"
mkdir -p "$REGEN/seed_rotation_s99/combined_lgbm"
cp -r combined_model/outputs/combined_lgbm_f1_notaccept_class_weight_pca12/* \
      "$REGEN/seed_rotation_s99/combined_lgbm/" 2>/dev/null || true

echo ""
echo "========================================"
echo "JOB A COMPLETE at $(date)"
echo "========================================"
