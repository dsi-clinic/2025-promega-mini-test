#!/bin/bash
#SBATCH --job-name=day1315_challenge
#SBATCH --output=slurm_%x_%j.out
#SBATCH --error=slurm_%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --partition=general
## Uncomment if your cluster requires an account:
# #SBATCH --account=your_account

# Run full Day 13/15 challenge pipeline on a GPU node.
# Submit: sbatch run_all_gpu.slurm  (from day13_15_challenge, or set CHALLENGE_DIR below)
# Logs:   slurm_day1315_challenge_<jobid>.out, slurm_day1315_challenge_<jobid>.err

set -e
# Run from the directory where sbatch was run (submit from day13_15_challenge)
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"

## Optional: load modules / activate env (customize for your cluster)
# module load cuda/11.8
# module load anaconda3
# source activate your_pytorch_env

# Slurm sets CUDA_VISIBLE_DEVICES automatically when --gres=gpu:1 is used
python run_all_gpu.py
