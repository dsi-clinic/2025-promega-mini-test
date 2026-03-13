#!/bin/bash
#SBATCH --partition=general
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Unified training script for all image classifier configurations
# Usage: sbatch --job-name=<name> run_training.s [--script train_model_accuracy|train_efficientnet_improved_tnr] [--input-path-key img_path|overlay_path] [--use-mask] [--outdir <path>] [--split-prefix <prefix>]
#
# Examples:
#   # Multi-backbone training (default)
#   sbatch --job-name=train-img run_training.s --input-path-key img_path
#   
#   # EfficientNet TNR-optimized training
#   sbatch --job-name=train-efficientnet-tnr run_training.s --script train_efficientnet_improved_tnr --input-path-key img_path --split-prefix both_train_exclude_stitch_only

set -euo pipefail

# ========== CONFIGURE FOR YOUR ENVIRONMENT ==========
# Set PROJ_ROOT to the absolute path of this repository.
# Example: export PROJ_ROOT=/home/YOUR_GITHUB_USERNAME/promega-classifier
PROJ_ROOT=${PROJ_ROOT:-/home/YOUR_GITHUB_USERNAME/promega-classifier}
CONDA_PREFIX=/net/projects2/promega
# =====================================================

# Parse arguments
SCRIPT="train_model_accuracy"  # Default to multi-backbone training
INPUT_KEY=""
USE_MASK=false
OUT_DIR=""
SPLIT_PREFIX="both_train_base"  # Default split prefix

while [[ $# -gt 0 ]]; do
    case $1 in
        --script)
            SCRIPT="$2"
            shift 2
            ;;
        --input-path-key)
            INPUT_KEY="$2"
            shift 2
            ;;
        --use-mask)
            USE_MASK=true
            shift
            ;;
        --outdir)
            OUT_DIR="$2"
            shift 2
            ;;
        --split-prefix)
            SPLIT_PREFIX="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: sbatch --job-name=<name> run_training.s [--script train_model_accuracy|train_efficientnet_improved_tnr] [--input-path-key img_path|overlay_path] [--use-mask] [--outdir <path>] [--split-prefix <prefix>]"
            exit 1
            ;;
    esac
done

# Set script path based on choice
if [[ "$SCRIPT" == "train_efficientnet_improved_tnr" ]]; then
    PY=${PROJ_ROOT}/image_classifier/training/train_efficientnet_improved_tnr.py
    # For EfficientNet TNR, build split file names from prefix
    # If prefix is "both_train_exclude_stitch_only", extract suffix "exclude_stitch_only"
    if [[ "$SPLIT_PREFIX" == both_train_* ]]; then
        SPLIT_SUFFIX=$(echo "$SPLIT_PREFIX" | sed 's/both_train_//')
    else
        SPLIT_SUFFIX="$SPLIT_PREFIX"
    fi
    TRAIN_SPLIT=${PROJ_ROOT}/data_splits/both_train_${SPLIT_SUFFIX}.json
    VAL_SPLIT=${PROJ_ROOT}/data_splits/both_val_${SPLIT_SUFFIX}.json
    TEST_SPLIT=${PROJ_ROOT}/data_splits/both_test_${SPLIT_SUFFIX}.json
else
    PY=${PROJ_ROOT}/image_classifier/training/train_model_accuracy.py
    # For multi-backbone, build split file names from prefix
    if [[ "$SPLIT_PREFIX" == both_train_* ]]; then
        SPLIT_SUFFIX=$(echo "$SPLIT_PREFIX" | sed 's/both_train_//')
    else
        SPLIT_SUFFIX="$SPLIT_PREFIX"
    fi
    TRAIN_SPLIT=${PROJ_ROOT}/data_splits/both_train_${SPLIT_SUFFIX}.json
    VAL_SPLIT=${PROJ_ROOT}/data_splits/both_val_${SPLIT_SUFFIX}.json
    TEST_SPLIT=${PROJ_ROOT}/data_splits/both_test_${SPLIT_SUFFIX}.json
fi

# Validate required arguments
if [[ "$SCRIPT" == "train_model_accuracy" ]]; then
    if [[ -z "$INPUT_KEY" ]]; then
        echo "Error: --input-path-key is required for train_model_accuracy (must be 'img_path' or 'overlay_path')"
        exit 1
    fi
    
    if [[ "$INPUT_KEY" != "img_path" && "$INPUT_KEY" != "overlay_path" ]]; then
        echo "Error: --input-path-key must be 'img_path' or 'overlay_path'"
        exit 1
    fi
fi

# Set default output directory if not provided
if [[ -z "$OUT_DIR" ]]; then
    if [[ "$SCRIPT" == "train_efficientnet_improved_tnr" ]]; then
        # For EfficientNet TNR, use a descriptive name based on split suffix
        # Extract suffix following metabolite classifier convention (both_train_<suffix>)
        if [[ "$SPLIT_PREFIX" == both_train_* ]]; then
            SPLIT_SUFFIX=$(echo "$SPLIT_PREFIX" | sed 's/both_train_//')
        else
            SPLIT_SUFFIX="$SPLIT_PREFIX"
        fi
        # Output directory naming: always include split suffix to match metabolite classifier pattern
        # This ensures output clearly indicates which split data was used (e.g., base, exclude_stitch_only)
        OUT_DIR=/net/projects2/promega/results/efficientnet_improved_tnr_${SPLIT_SUFFIX}
    else
        # For multi-backbone, use pattern: outputs_{nomask|mask}_{image|overlay}
        if [[ "$INPUT_KEY" == "img_path" ]]; then
            IMAGE_TYPE="image"
        else
            IMAGE_TYPE="overlay"
        fi
        
        if [[ "$USE_MASK" == true ]]; then
            OUT_DIR=/net/projects2/promega/results/outputs_mask_${IMAGE_TYPE}
        else
            OUT_DIR=/net/projects2/promega/results/outputs_nomask_${IMAGE_TYPE}
        fi
    fi
fi

mkdir -p logs "${OUT_DIR}"

if command -v module >/dev/null 2>&1; then
  module purge
fi

echo "Conda prefix: ${CONDA_PREFIX}"
nvidia-smi || true
echo "Running ${PY} with script: ${SCRIPT}"
echo "Train split: ${TRAIN_SPLIT}"
echo "Val split: ${VAL_SPLIT}"
echo "Test split: ${TEST_SPLIT}"
echo "Output directory: ${OUT_DIR}"

# Build arguments based on script type
if [[ "$SCRIPT" == "train_efficientnet_improved_tnr" ]]; then
    ARGS=(
        --outdir "${OUT_DIR}"
        --train-split "${TRAIN_SPLIT}"
        --val-split "${VAL_SPLIT}"
        --test-split "${TEST_SPLIT}"
        --batch-size 16
    )
else
    echo "Combination: input_key=${INPUT_KEY}, use_mask=${USE_MASK}"
    ARGS=(
        --train-split "${TRAIN_SPLIT}"
        --val-split "${VAL_SPLIT}"
        --test-split "${TEST_SPLIT}"
        --batch-size 16
        --val-batch-size 16
        --input-path-key "${INPUT_KEY}"
        --outdir "${OUT_DIR}"
    )
    
    if [[ "${USE_MASK}" == true ]]; then
        ARGS+=(--use-mask)
    fi
fi

cd "${PROJ_ROOT}"
export PYTHONPATH="${PROJ_ROOT}:$PYTHONPATH"
${CONDA_PREFIX}/bin/python3 -u "${PY}" "${ARGS[@]}"

TRAIN_EXIT=$?

if [ $TRAIN_EXIT -ne 0 ]; then
    echo "ERROR: Training failed with exit code $TRAIN_EXIT"
    exit $TRAIN_EXIT
fi

# If EfficientNet TNR training, automatically generate summary table
if [[ "$SCRIPT" == "train_efficientnet_improved_tnr" ]]; then
    echo ""
    echo "Training completed successfully. Generating summary table..."
    
    SUMMARY_SCRIPT=${PROJ_ROOT}/image_classifier/training/generate_efficientnet_summary.py
    # Extract suffix for output name
    if [[ "$SPLIT_PREFIX" == both_train_* ]]; then
        SUMMARY_SUFFIX=$(echo "$SPLIT_PREFIX" | sed 's/both_train_//')
    else
        SUMMARY_SUFFIX="$SPLIT_PREFIX"
    fi
    SUMMARY_OUTPUT_NAME="efficientnet_summary_$(echo "$SUMMARY_SUFFIX" | tr '[:lower:]' '[:upper:]')"
    
    ${CONDA_PREFIX}/bin/python3 -u "${SUMMARY_SCRIPT}" \
        --results-dir "${OUT_DIR}" \
        --split-prefix "${SPLIT_PREFIX}" \
        --output-name "${SUMMARY_OUTPUT_NAME}"
    
    SUMMARY_EXIT=$?
    
    if [ $SUMMARY_EXIT -ne 0 ]; then
        echo "WARNING: Summary table generation failed with exit code $SUMMARY_EXIT"
        echo "Training completed successfully, but summary generation failed."
    else
        echo "Summary table generated successfully."
    fi
fi

echo "Done."

