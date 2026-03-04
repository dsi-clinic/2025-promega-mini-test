#!/usr/bin/env bash
# Run full Day 13/15 pipeline on GPU. Steps run one after another; any failure stops the run.
# Usage: ./run_all_gpu.sh   or   CUDA_VISIBLE_DEVICES=0 ./run_all_gpu.sh
set -e
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
echo "Using GPU: $CUDA_VISIBLE_DEVICES"
python run_all_gpu.py "$@"
