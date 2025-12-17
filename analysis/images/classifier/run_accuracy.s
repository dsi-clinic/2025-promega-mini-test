#!/bin/bash
#SBATCH --job-name=soft-label
#SBATCH --partition=general
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

# ====== adjust these paths ======
PROJ_ROOT=/home/YOUR_GITHUB_NAME/YOUR_MINITEST_DIRECTORY
PY=${PROJ_ROOT}/analysis/images/classifier/train_model_accuracy.py
DATA_DIR=/path/to/data/images
ALL_DATA_JSON=/path/to/all_data.json
CONDA_PREFIX=/net/projects2/promega                           # conda env path (same you used before)
# ================================

mkdir -p logs


# Optional: purge modules if your cluster uses them
if command -v module >/dev/null 2>&1; then
  module purge
fi

echo "Conda prefix: ${CONDA_PREFIX}"
nvidia-smi || true
echo "Running train_model_accuracy.py"
echo "DATA_DIR=${DATA_DIR}"

# Run the script (uses simple defaults from the python file)
cd "${PROJ_ROOT}"
echo "Current directory: $(pwd)"
echo "Python script: ${PY}"
echo "Data dir: ${DATA_DIR}"
ls -la "${PY}"

# Use python directly from conda environment
export PYTHONPATH=.
"${CONDA_PREFIX}/bin/python" -u "${PY}" \
  --data-dir "${DATA_DIR}" \
  --all-data-json "${ALL_DATA_JSON}" \
  --batch-size 16 \
  --val-frac 0.10 \
  --test-frac 0.10

echo "Done."
