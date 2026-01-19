#!/bin/bash
#SBATCH --job-name=temporal_transformer
#SBATCH --output=logs/temporal_transformer_%j.out
#SBATCH --error=logs/temporal_transformer_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

# Temporal Transformer Image Classifier Training
# Processes image sequences (Dy03→Dy06→...→Dy30) with EfficientNet + Transformer

echo "========================================="
echo "TEMPORAL TRANSFORMER IMAGE CLASSIFIER TRAINING"
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

# Setup
cd /home/tonyluo/minitest
mkdir -p logs

# Python environment
PYTHON=/net/projects2/promega/bin/python3

echo "Python: $PYTHON"
echo "CUDA visible devices: $CUDA_VISIBLE_DEVICES"
echo ""

# Output directory
OUTDIR="analysis/images/classifier/temporal_transformer_NO_STITCH"

# Training phase
echo "========================================="
echo "Training Temporal Transformer Model"
echo "========================================="

$PYTHON analysis/images/image_classifier/train_temporal_transformer.py \
    --outdir "$OUTDIR" \
    --train-split data_splits/both_train_base_no_stitch.json \
    --val-split data_splits/both_val_base_no_stitch.json \
    --test-split data_splits/both_test_base_no_stitch.json \
    --batch-size 8

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
echo "Results directory: $OUTDIR"
echo ""
echo "Generated files:"
echo "  - Model: $OUTDIR/model.pth"
echo "  - Metrics: $OUTDIR/metrics_test.json"
echo "  - Predictions: $OUTDIR/predictions.json"
echo "  - Training curves: $OUTDIR/training_curves.png"
echo ""

exit 0
