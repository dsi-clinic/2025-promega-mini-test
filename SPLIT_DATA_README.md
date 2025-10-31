# Data Split Script

## Purpose

Creates reproducible train/validation splits for comparing image and metabolite models.

**Key feature:** Splits by **organoid**, not individual samples. This prevents data leakage when training on early timepoints (Dy03-Dy28) to predict Dy30 outcomes.

## How It Works

- Labels come from Dy30 survey data (Acceptable/Not Acceptable)
- Each organoid appears in ONLY train or ONLY validation (never both)
- All timepoints for an organoid stay together
- Fixed random seed (42) ensures same split every time
- 80/20 train/val split with stratification

## Modes

### Base Mode (default)
```bash
python3 split_data_reproducible.py --mode base
```
- Only BA1+BA2 batches
- Only organoids with BOTH image and complete metabolite data
- Output: `data_splits/both_train_base.json`, `data_splits/both_val_base.json`

### Switch 1: Extra Image Samples
```bash
python3 split_data_reproducible.py --mode switch1
```
- Image model gets all BA1+BA2 organoids (with or without metabolites)
- Metabolite model still uses base mode intersection
- Output: `data_splits/image_train_switch1.json`, `data_splits/image_val_switch1.json`

### Switch 2: Include BA3+BA4
```bash
python3 split_data_reproducible.py --mode switch2
```
- Both models use BA1+BA2+BA3+BA4 organoids (intersection only)
- ⚠️ Note: BA3/BA4 have known issues per IDOR/Promega
- Output: `data_splits/both_train_switch2.json`, `data_splits/both_val_switch2.json`

### Switch 3: All Image Data
```bash
python3 split_data_reproducible.py --mode switch3
```
- Image model gets ALL organoids from all batches
- Metabolite still uses BA1+BA2 only
- Output: `data_splits/image_train_switch3.json`, `data_splits/image_val_switch3.json`

### All Modes
```bash
python3 split_data_reproducible.py --mode all
```
Generates all splits at once.

## Output Format

```json
{
  "BA1 96_1 A1": {
    "label": "Acceptable",
    "batch": "BA1",
    "timepoints": {
      "Dy03": {
        "img_path": "/path/to/image.png",
        "mask_path": "/path/to/mask.png",
        "day": "Dy03",
        "metabolites": {
          "GlucoseGlo": 9.827,
          "GlutamateGlo": 2.418,
          "LactateGlo": 7.247,
          "PyruvateGlo": 2.971
        }
      },
      "Dy06": { ... },
      ...
    }
  }
}
```

## Metabolite Restrictions Applied

Based on IDOR/Promega guidance:
- ✓ GlucoseGlo
- ✓ GlutamateGlo  
- ✓ LactateGlo
- ✓ PyruvateGlo
- ✗ MalateGlo (excluded - unreliable)
- ✗ BCAAGlo (excluded - unreliable for days ≤10)

## Example Results

**Base Mode (BA1+BA2):**
- 176 organoids training, 44 validation
- ~1,739 training samples across 11 timepoints
- ~415 validation samples across 11 timepoints

**Switch 2 (All batches):**
- 234 organoids training, 59 validation
- ~2,377 training samples across 11 timepoints
- ~580 validation samples across 11 timepoints

## Use Cases

1. **Train on early days, predict Dy30:**
   - Use Dy03-Dy10 data to predict final organoid quality
   
2. **Time-series analysis:**
   - Track how image/metabolite features evolve over time

3. **Multi-day training:**
   - Train on multiple timepoints to improve robustness

4. **Fair model comparison:**
   - Both models see exact same organoids, just different modalities



