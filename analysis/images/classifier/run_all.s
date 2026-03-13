#!/bin/bash

# Submit the four train_model_accuracy sweeps covering
#   1) image-only (no mask)
#   2) overlay-only (no mask)
#   3) image + mask
#   4) overlay + mask

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

submit() {
  local script="$1"
  if [[ ! -f "${SCRIPT_DIR}/${script}" ]]; then
    echo "Missing script: ${SCRIPT_DIR}/${script}" >&2
    exit 1
  fi
  sbatch "${SCRIPT_DIR}/${script}"
}

submit run_nomask_image.s
submit run_nomask_overlay.s
submit run_mask_image.s
submit run_mask_overlay.s
