#!/bin/bash
#SBATCH --job-name=promega-images-classifier
#SBATCH --partition=general
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00   
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

module purge

# Activate your environment
source /net/scratch/jiaweizhang/promega_my_old_version/venv/bin/activate

# Run your Python script
python analysis/images/classifier/train_model_accuracy.py
