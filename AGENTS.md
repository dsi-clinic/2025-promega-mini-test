# Agent Instructions

This project uses **bd** (beads) for issue tracking. Run `bd onboard` to get started.

## Quick Reference

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --status in_progress  # Claim work
bd close <id>         # Complete work
bd sync               # Sync with git
```

## Environment

### Conda (MANDATORY)

All Python code **must** run inside the project conda environment. **Never `pip install`** into the system Python.

```bash
# PREFERRED: use make run for any Python command
make run ARGS="-m analysis.generate_splits --dry-run"
make run ARGS="scripts/my_script.py --flag value"

# Interactive shell (when needed)
conda activate core_env

# Direct invocation (avoid when make run works)
conda run --no-capture-output -n core_env python3 <script.py>
```

**`make run`** handles conda env, `PYTHONPATH`, and all configuration. Use it for all Python execution. Named Makefile targets (e.g. `make step16`) are preferred when they exist.

### Environment spec

`core_env.yaml` is the canonical environment definition (Python 3.11, PyTorch 2.8, scikit-learn 1.7, etc.). If you need a new package, add it to `core_env.yaml` and rebuild — do not `pip install` ad-hoc.

### PYTHONPATH

The repo root must be on `PYTHONPATH` for imports like `from pipeline.common.organoid_patterns import ...` and `from pipeline.data_loader import ...`. The Makefile sets this automatically. In interactive use:

```bash
export PYTHONPATH=$(pwd)
```

### Data directory

Remote data lives at `/net/projects2/promega/2026_04_15_data` (the `DATA_ROOT` in the Makefile), organized as `raw/` (inputs — never written to), `intermediate/` (regenerable pipeline outputs), `models/` (checkpoints), and `analysis_output/` (manual figures). Code should never hard-code these paths — use environment variables or Makefile variables.

## Project Rules

### 1. No pip install in system Python

All dependencies are managed via the `core_env` conda environment (defined by `core_env.yaml` in the repo). If a package is missing, add it to `core_env.yaml` and rebuild the environment. Never run `pip install <pkg>` outside the conda env.

### 2. Splits are organoid-level, not record-level

Organoids span multiple days (Dy03–Dy30). The **same organoid must stay in the same split** (train/val/test) across all days to prevent data leakage. The canonical split assignment lives in `data/splits/canonical_2026_winter.csv` and keys on `organoid_id` (e.g. `BA1 96_1 A1` — no day component). Code loads it via `Splits.canonical()` from `pipeline.splits`. Multiple named splits can coexist under `data/splits/` — see `data/splits/README.md`.

### 3. all_data.json is the single source of truth

Labels, features, filtering — everything is derived at runtime from `data/all_data.json`. Split CSVs under `data/splits/` contain only `organoid_id` and `split` (extra columns are ignored by `Splits.from_csv`). Do not materialize filtered/transformed data into separate JSON files for downstream models; use `pipeline/data_loader.py` + `pipeline/splits.py` instead.

### 4. Paper filter defaults

When reproducing paper results, apply these filters (already the defaults in `data_loader.py`):
- **Batches**: BA1 + BA2 only
- **Labels**: 4/5 vote consensus at Dy30
- **Metabolites**: All 5 required metabolites present (GlucoseGlo, GlutamateGlo, LactateGlo, PyruvateGlo, BCAAGlo)
- **Conditional metabolites**: MalateGlo included only for days > 10 (early-day values not assayed); BCAAGlo required all days
- **Images**: Valid processed `img_path` + `mask_path` on every day

### 5. Seed = 42 everywhere

All random operations (splits, model training, cross-validation) use `random_state=42` for reproducibility.

### 6. Day aliasing: Dy20/Dy21 → Dy20_5

Dy20 and Dy21 in the raw data represent the same biological timepoint. Canonicalize them to `Dy20_5` in analysis code.

### 7. Output locations

| What | Where | Git tracked? |
|------|-------|--------------|
| Split CSVs | `data/splits/*.csv` (canonical_2026_winter.csv + alternates) | Yes |
| Analysis code | `analysis/` | Yes |
| Generated figures | `$ANALYSIS_OUTPUT_DIR/figures/` (default: `$DATA_ROOT/analysis_output/figures/`) | No |
| Model checkpoints | `$DATA_ROOT/models/` | No |

### 8. Running analysis scripts

```bash
# Generate splits (one-time, already checked in)
make run ARGS="-m analysis.paper_2026_04.generate_splits"

# Run any analysis module
make run ARGS="-m analysis.<module_name> --flag value"
```

### 9. Positive class convention

In binary classification:
- Internal model training: `1` = Not Acceptable (positive/minority class)
- `0` = Acceptable
- `scale_pos_weight` = n_negative / n_positive for class imbalance

### 10. Metabolite feature convention

Per metabolite per day:
- `{MetaboliteName}_concentration_uM` — assay concentration
- `{MetaboliteName}_initial_concentration` — baseline
- `{MetaboliteName}_growth` — delta from previous day (optional)

Growth features require a previous timepoint and are unavailable at Dy03.

## Landing the Plane (Session Completion)

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **PUSH TO REMOTE** - This is MANDATORY:
   ```bash
   git pull --rebase
   bd sync
   git push
   git status  # MUST show "up to date with origin"
   ```
5. **Clean up** - Clear stashes, prune remote branches
6. **Verify** - All changes committed AND pushed
7. **Hand off** - Provide context for next session

**CRITICAL RULES:**
- Work is NOT complete until `git push` succeeds
- NEVER stop before pushing - that leaves work stranded locally
- NEVER say "ready to push when you are" - YOU must push
- If push fails, resolve and retry until it succeeds
