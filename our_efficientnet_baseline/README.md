# Our EfficientNet Baseline Results

This directory contains the baseline EfficientNet (single timepoint) model results for comparison with CNN-LSTM temporal models.

## Contents

- **`baseline_results.json`**: Complete results for all day ranges (3, 6, 8, 10, 13, 15, 17, 20.5, 24, 30)
- **`day_*/`**: Individual model checkpoints and results for each day range

## Model Details

- **Architecture**: EfficientNet-B0 (single timepoint, not temporal)
- **Training**: Trained on each day separately (picks closest day to target)
- **Data Splits**: Same as CNN-LSTM models (uses `load_data_and_create_splits`)
- **Purpose**: Baseline comparison to show improvement from temporal modeling

## Key Results (Day 30 - Final Prediction)

From `baseline_results.json`, the day 30 results are:
- **Test Accuracy**: Check JSON for exact value
- **Test F1**: Check JSON for exact value
- **Best Validation Accuracy**: Check JSON for exact value

## Comparison with CNN-LSTM

To compare:
1. Load `baseline_results.json` (this directory)
2. Load CNN-LSTM results (from Amanda's or our runs)
3. Compare metrics at day 30 (final prediction)

## Source

Copied from: `/net/projects2/promega/data-analysis/output/base_models/base_effnet/`
