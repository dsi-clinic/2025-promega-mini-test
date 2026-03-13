#!/usr/bin/env python3
"""
Automated pipeline: run all Day 13/15 challenge tasks sequentially on GPU.
- Step 1: Compute grayscale mean/std (no GPU)
- Step 2: Audit transforms (no GPU)
- Step 3: Train grayscale-norm models (GPU)
- Step 4: Train filled-mask models (GPU)
On any failure, errors are printed and the process exits so you can fix and re-run.
Run: python run_all_gpu.py
     or: CUDA_VISIBLE_DEVICES=0 python run_all_gpu.py
"""

import os
import sys
import subprocess
from pathlib import Path

CHALLENGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = CHALLENGE_DIR.parent.parent


def run_step(name, cmd, cwd=None, env=None):
    """Run a command; return True if success, False otherwise. Print stdout/stderr on failure."""
    cwd = cwd or str(CHALLENGE_DIR)
    env = env or os.environ.copy()
    print(f"\n{'=' * 60}\n>>> {name}\n{'=' * 60}")
    sys.stdout.flush()
    sys.stderr.flush()
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            env=env,
            shell=False,
            timeout=3600 * 6,  # 6h max per step
            capture_output=False,  # let output stream to terminal/log
        )
        if result.returncode != 0:
            print(f"\n[FAIL] {name} exited with code {result.returncode}")
            return False
        print(f"\n[OK] {name} finished.")
        return True
    except subprocess.TimeoutExpired as e:
        print(f"\n[FAIL] {name} timed out after {e.timeout}s.")
        return False
    except Exception as e:
        print(f"\n[FAIL] {name} raised: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    # Prefer GPU; use first visible device or default
    env = os.environ.copy()
    if "CUDA_VISIBLE_DEVICES" not in env:
        env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    python = sys.executable
    steps = [
        (
            "1. Compute grayscale mean/std",
            [python, str(CHALLENGE_DIR / "01_compute_grayscale_mean_std.py")],
        ),
        (
            "2. Audit transforms",
            [python, str(CHALLENGE_DIR / "02_audit_transforms.py")],
        ),
        (
            "3. Train grayscale-norm (effnet_ts + per_day, days 13 & 15)",
            [python, str(CHALLENGE_DIR / "run_1_grayscale_norm.py")],
        ),
        (
            "4. Train filled-mask (effnet_ts + per_day, days 13 & 15)",
            [python, str(CHALLENGE_DIR / "run_4_filled_mask.py")],
        ),
    ]
    for name, cmd in steps:
        if not run_step(name, cmd, cwd=str(CHALLENGE_DIR), env=env):
            print(
                "\nPipeline stopped. Fix the error above and re-run: python run_all_gpu.py"
            )
            sys.exit(1)
    print("\n" + "=" * 60)
    print("All steps completed successfully.")
    print(
        "Outputs: runs_grayscale_norm/, runs_filled_mask/, audit_transforms_summary.json, grayscale_mean_std_*.npy"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
