# Temporal Transformer Setup

## Configuration Required

Before running, update these paths in `submit_temporal_transformer.slurm`:

1. **Line 24**: Change `/path/to/minitest` to your minitest home directory
   ```bash
   cd /home/yourusername/minitest
   ```

2. **Line 29**: Change Python path if different
   ```bash
   PYTHON=/your/python/path/bin/python3
   ```

## Running

After updating paths, run from minitest home directory:
```bash
sbatch submit_temporal_transformer.slurm
```
