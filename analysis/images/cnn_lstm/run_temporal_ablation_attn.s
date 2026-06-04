#!/bin/bash
#SBATCH --job-name=cnn-lstm-attn
#SBATCH --partition=general
#SBATCH --qos=general
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/home/wenxu/2025-promega-mini-test/logs/%x_%j.out
#SBATCH --error=/home/wenxu/2025-promega-mini-test/logs/%x_%j.err
#SBATCH --signal=B:USR1@300
#SBATCH --requeue

set -euo pipefail
cd /home/wenxu/2025-promega-mini-test

make run ARGS="analysis/images/cnn_lstm/train_temporal_ablation_attn.py $*"
