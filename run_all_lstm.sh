#!/bin/bash
# Run from project root with conda env active:
# conda activate core_env
# bash run_all_models.sh

OUT=/net/projects2/promega/project_data/model_tests/lstm_runs
mkdir -p "$OUT"

echo "========================================"
echo "1/4  BASE MODEL"
echo "========================================"
python analysis/images/cnn_lstm/train_base_model.py \
    --output-dir "$OUT/base_effnet" \
    --image-type clipped

echo "========================================"
echo "2/4  CNN-LSTM"
echo "========================================"
python analysis/images/cnn_lstm/train_organoid_lstm.py \
    --output-dir "$OUT/cnn_lstm"  \
    --image-type clipped

echo "========================================"
echo "3/4  TEMPORAL ABLATION (ATTENTION)"
echo "========================================"
python analysis/images/cnn_lstm/train_temporal_ablation_attn.py \
    --output-dir "$OUT/temporal_ablation_attn" \
    --image-type clipped

echo "========================================"
echo "4/4  TEMPORAL ABLATION (LSTM)"
echo "========================================"
python analysis/images/cnn_lstm/train_temporal_ablation_lstm.py \
    --output-dir "$OUT/temporal_ablation_lstm" \
    --image-type clipped