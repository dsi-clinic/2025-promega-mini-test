# Image Quality Classification (Step 17)

Per-day image classifier that predicts organoid acceptability from processed images. Reads `data/all_data.json` directly; no separate metadata generation step.

## Run

```bash
make step17                                  # default sweep via Makefile
# or
make run ARGS="-m analysis.imagequality_classification.train_model_accuracy --help"
```

The Makefile target invokes `train_model_accuracy.py` with default flags (no mask, image input, deterministic). The `INPUT_KEY` and `USE_MASK` Makefile vars cover the four sweep variants:

| Variant | Command |
|---|---|
| RGB only | `make step17` (default) |
| Mask overlay only | `make step17 INPUT_KEY=overlay_path` |
| RGB + mask branch | `make step17 USE_MASK=1` |
| Overlay + mask branch | `make step17 USE_MASK=1 INPUT_KEY=overlay_path` |

For the DINOv2 backbone experiment (different script + fixed-splits replication):

```bash
make analysis-train-dinov2          # local (CPU/GPU on current node)
sbatch analysis/imagequality_classification/run_dinov2_fixed_splits.s   # submit to SLURM
```

The SLURM script auto-detects the repo root and dispatches to the same Makefile target — no per-user editing required.

## Scripts

| Script | Purpose |
|---|---|
| `train_model_accuracy.py` | Primary trainer. ViT / ResNet / EfficientNet / custom CNN. Phase 1 (frozen backbone) → Phase 2 (full fine-tune). Optional mask branch. |
| `train_model_dinov2.py` | DINOv2-based variant using `OrganoidDataset` + `filters_for_mode` from `pipeline.data_loader`. |

## Outputs

Models, metrics, and plots land in `$DATA_ROOT/models/imagequality_classification/` (default `/net/projects2/promega/2026_04_15_data/models/imagequality_classification/`).
