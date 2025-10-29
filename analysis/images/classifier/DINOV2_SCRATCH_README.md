# DINOv2 Training on Scratch Space

## Problem
Your `/net/projects2/promega` directory is **27GB over quota (227GB used / 200GB limit)**, which was causing the dinov2 training to fail with "Disk quota exceeded" errors.

## Solution
This setup runs dinov2 training and saves **all outputs to scratch2 space** to avoid quota issues.

## Files Created

### 1. `train_model_dinov2.py`
- Full training script using DINOv2 (facebook/dinov2-base) from HuggingFace
- Two-phase training: frozen backbone → fine-tuned backbone
- Generates:
  - Model checkpoints (`model.pth`)
  - Training curves (`training_curves.png`)
  - Metrics JSONs (`metrics_val.json`, `metrics_test.json`)
  - Summary CSV (`day_summary.csv`)
  - Accuracy/F1/ROC-AUC charts
  - Final summary JSON (`final_test_summary.json`)

### 2. `run_dinov2_scratch.s`
- SLURM batch script for GPU cluster
- Allocated: 12 hours, A100 GPU, 32GB RAM
- **Saves to `/scratch2/${USER}/dinov2_outputs_TIMESTAMP/`**
- Includes instructions to copy results back after completion

## Usage

### Submit the job:
```bash
cd /home/tonyluo/minitest
sbatch analysis/images/classifier/run_dinov2_scratch.s
```

### Check job status:
```bash
squeue -u $USER
```

### View logs (while running or after):
```bash
# Check output
tail -f analysis/images/classifier/logs/dino-scratch_JOBID.out

# Check errors
tail -f analysis/images/classifier/logs/dino-scratch_JOBID.err
```

### After completion, copy results to permanent storage:
```bash
# The script will print the exact command, but it will look like:
rsync -avz /scratch2/${USER}/dinov2_outputs_TIMESTAMP/ \
           /home/tonyluo/minitest/analysis/images/classifier/outputs_dinov2_final/
```

## Important Notes

⚠️ **Scratch2 is temporary storage!** Files may be auto-purged after some time. Always copy important results to permanent storage.

## Expected Output Structure

```
/scratch2/${USER}/dinov2_outputs_TIMESTAMP/
├── dinov2/
│   ├── Dy3/
│   │   ├── model.pth
│   │   ├── training_curves.png
│   │   ├── metrics_val.json
│   │   └── metrics_test.json
│   ├── Dy6/
│   ├── Dy8/
│   ├── Dy10/
│   ├── Dy13/
│   ├── Dy15/
│   ├── Dy17/
│   ├── Dy20_5/
│   ├── Dy24/
│   ├── Dy28/
│   └── Dy30/
├── day_summary.csv
├── accuracy_by_day.png
├── f1_by_day.png
├── rocauc_by_day.png
└── final_test_summary.json
```

## Days to be Trained
All available days from your dataset:
- Dy3, Dy6, Dy8, Dy10, Dy13, Dy15, Dy17, Dy20_5, Dy24, Dy28, Dy30

## Disk Space Status
- **Home**: 49.8/50 GB (99.6% used) ✅
- **Project (promega)**: 227/200 GB (**OVER QUOTA by 27GB**) 🔴
- **Scratch**: 0/50 GB (empty) ✅
- **Scratch2**: 0/50 GB (empty) ✅

Using scratch2 for this training avoids the quota issue entirely!





