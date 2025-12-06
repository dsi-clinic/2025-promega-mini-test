#!/bin/bash
# Monitor script for multi-backbone training (DINOv2, ResNet, EfficientNet) with fixed splits
# Usage: ./monitor_accuracy.sh [JOB_ID]
# If JOB_ID not provided, will search for the most recent job

JOB_ID=${1:-""}

# If no job ID provided, try to find the most recent one
if [ -z "$JOB_ID" ]; then
    echo "No job ID provided, searching for most recent job..."
    JOB_ID=$(squeue -u ${USER:-$USERNAME} -o "%.18i %.9P %.20j" -h | grep "train-img\|train-overlay" | head -1 | awk '{print $1}')
    if [ -z "$JOB_ID" ]; then
        echo "No running job found. Please provide job ID: ./monitor_accuracy.sh <JOB_ID>"
        exit 1
    fi
    echo "Found job ID: $JOB_ID"
fi

cd "${PROJ_ROOT:-/home/tonyluo/minitest}"
PROJ_ROOT="${PROJ_ROOT:-/home/tonyluo/minitest}"

while true; do
    clear
    echo "========================================"
    echo "Multi-Backbone Training Monitor - $(date)"
    echo "========================================"
    echo ""
    echo "Job Status:"
    squeue -u ${USER:-$USERNAME} -j $JOB_ID 2>/dev/null | head -3 || echo "  Job not in queue"
    echo ""
    
    LOG_FILE="${PROJ_ROOT}/analysis/images/classifier/logs/train-img_${JOB_ID}.out"
    # Try alternative log file names if the above doesn't exist
    if [ ! -f "$LOG_FILE" ]; then
        LOG_FILE="${PROJ_ROOT}/analysis/images/classifier/logs/train-overlay_${JOB_ID}.out"
    fi
    if [ ! -f "$LOG_FILE" ]; then
        LOG_FILE="${PROJ_ROOT}/analysis/images/classifier/logs/train-overlay-mask_${JOB_ID}.out"
    fi
    if [ -f "$LOG_FILE" ]; then
        echo "Recent Output (last 10 lines):"
        tail -15 "$LOG_FILE" | tail -10
    fi
    echo ""
    
    echo "Completed Days:"
    COUNT=0
    for day in Dy03 Dy06 Dy08 Dy10 Dy13 Dy15 Dy17 Dy21 Dy24 Dy28 Dy30; do
        COMPLETE=0
        RESULTS_DIR="${RESULTS_DIR:-/net/projects2/promega/results}"
        [ -f ${RESULTS_DIR}/outputs_512x384_fixed_splits/dinov2/$day/metrics_test.json ] && \
        [ -f ${RESULTS_DIR}/outputs_512x384_fixed_splits/resnet/$day/metrics_test.json ] && \
        [ -f ${RESULTS_DIR}/outputs_512x384_fixed_splits/efficientnet/$day/metrics_test.json ] && \
        COMPLETE=1 && COUNT=$((COUNT+1))
        
        if [ $COMPLETE -eq 1 ]; then
            echo "  [OK] $day"
        else
            echo "  [PENDING] $day"
        fi
    done
    echo ""
    echo "Progress: $COUNT/11 days complete"
    echo ""
    
    # Check if all days complete and job is done
    if [ $COUNT -eq 11 ] && [ -z "$(squeue -u ${USER:-$USERNAME} -j $JOB_ID 2>/dev/null | tail -n +2)" ]; then
        echo "[SUCCESS] ALL DAYS COMPLETE! Job finished successfully."
        break
    fi
    
    echo "Sleeping 60 seconds... (Press Ctrl+C to stop)"
    sleep 60
done






