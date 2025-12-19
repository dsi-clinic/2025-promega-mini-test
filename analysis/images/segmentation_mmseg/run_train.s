#!/bin/bash
#SBATCH --job-name=mmseg-train
#SBATCH --partition=general
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/net/projects2/promega/data_reorg/logs/%x_%j.out
#SBATCH --error=/net/projects2/promega/data_reorg/logs/%x_%j.err

set -euo pipefail

# ====== adjust these paths ======
PROJ_ROOT=/home/YOUR_GITHUB_NAME/YOUR_MINITEST_DIRECTORY
PY=${PROJ_ROOT}/analysis/images/segmentation_mmseg/train.py
SPLITS_DIR=/path/to/data/split/mask/json
WORK_DIR=/path/to/save/models
CONDA_PREFIX=/path/to/conda/environment                           # conda env path (same you used before)
# ================================

mkdir -p logs


# Optional: purge modules if your cluster uses them
if command -v module >/dev/null 2>&1; then
  module purge
fi

echo "Conda prefix: ${CONDA_PREFIX}"
nvidia-smi || true
echo "Running train.py"
echo "Splits dir: ${SPLITS_DIR}"
echo "Work dir: ${WORK_DIR}"

# Run the script (uses simple defaults from the python file)
cd "${PROJ_ROOT}"
echo "Current directory: $(pwd)"
echo "Python script: ${PY}"
ls -la "${PY}"

# Use python directly from conda environment
export PYTHONPATH=.
"${CONDA_PREFIX}/bin/python" -u "${PY}" \
  --splits-dir "${SPLITS_DIR}" \
  --work-dir "${WORK_DIR}"

echo "Done."
