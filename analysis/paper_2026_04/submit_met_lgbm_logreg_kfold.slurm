#!/bin/bash
#SBATCH --job-name=met-lgbm-lr
#SBATCH --partition=general
#SBATCH --mem=32G
#SBATCH --cpus-per-task=16
#SBATCH --time=6:00:00
#SBATCH --output=logs/met_lgbm_logreg_kfold_%j.out
#SBATCH --error=logs/met_lgbm_logreg_kfold_%j.err

# Met-only 10×4-fold repeated CV: LGBM + LogReg, mirroring combined_kfold fold structure.
# SEED=1, all_org_ids upfront, inner_tr_oids — directly comparable to combined results.
# No GPU required. 16 CPUs feed n_jobs=-1 in GridSearchCV.

cd /net/projects/CLS/lding/gitcode/2025-promega-mini-test
mkdir -p logs

conda run --no-capture-output -n core_env python3 \
    -m analysis.paper_2026_04.met_lgbm_logreg_kfold \
    --n-repeats 10 \
    --n-folds 4
