# Day 13 / 15 Challenge – Reporting Narrative (Task 3)

## Fixed reporting setup

- **Threshold:** Fixed at **0.5** (no day-specific thresholding) for a clean narrative.
- **Primary metric:** **Balanced accuracy** = (Sensitivity + Specificity) / 2.
- **Setup:** Overlay (RGB + organoid outline) for both per-day EfficientNet and EfficientNet time-series.

## Day 13 / 15 collapse (effnet_ts)

We **explicitly note** the following as a **robustness / pipeline issue** (not a threshold choice issue):

- **Day 13** (and similarly **Day 15**): The overlay effnet_ts model outputs **collapsed predicted probabilities** (e.g. validation ~0.21, range ~0.207–0.222). At threshold 0.5 the model predicts all “not acceptable” (e.g. test TN/FP/FN/TP = 6/0/31/0 → recall = 0, balanced acc = 0.5). Even at the “optimal” threshold ~0.22 the model barely recovers any TPs.
- **Interpretation:** The time-series model is effectively an almost-constant all-negative classifier on these days; the issue is **collapsed outputs / pipeline**, to be diagnosed and fixed (e.g. normalization, transforms, or input representation), not tuned away by threshold.

For reporting we **keep the fixed t = 0.5 narrative** and **document the Day 13/15 collapse** as a known limitation to address in follow-up work.
