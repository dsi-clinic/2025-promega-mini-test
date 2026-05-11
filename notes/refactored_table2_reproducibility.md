# Refactored Table 2 — Reproducibility

- **Reference document:** `notes/refactored_table2.md`
- **Code:** `analysis/imagequality_classification/train_model_dinov2.py`

---

## 1. Setup

- Split is fixed: based on previous JSON split file (`data/splits.csv`)
- In `notes/refactored_table2.md`, on the JSON split, the current version (DINOv2 only normalised) was compared with the variation (all backbones normalised); the current version was closer to the paper and was selected as the final model
- Image data: `/net/projects2/promega/2026_04_15_data/intermediate/resized_512x384/`

---

## 2. Patches?

- **Print metrics:** Paper Table 2 metric printout was added to `train_model_dinov2.py` (Avg. TNR, Early TNR (Dy3-10), Bal. Acc., Days TNR=0, F1 (NA)); confusion matrix counts (`tn`, `fp`, `fn`, `tp`) computed directly from raw predictions and saved to `metrics_test.json` per day; no impact on training
- **Deterministic:** Set deterministic condition using `set_deterministic(True)` from `analysis/imagequality_classification/train.py`, which sets `torch.backends.cudnn.deterministic = True`, `torch.backends.cudnn.benchmark = False`, and attempts `torch.use_deterministic_algorithms(True)` (with `try/except` for unsupported ops)

---

## 3. 4 Runs, Same Config — Results

| Run | Determinism | Notes |
|---|:-:|---|
| Run 1 | non-det | Original `train_model_dinov2.py`, no patches |
| Run 2 | non-det | Patch version |
| Run 3 | det | Patch version with `set_deterministic(True)` enabled |
| Run 4 | non-det | Patch version |

### DINOv2 (ViT)

| Metric | Run 1 (non-det) | Run 2 (non-det) | Run 3 (det) | Run 4 (non-det) |
|---|:-:|:-:|:-:|:-:|
| Avg. TNR | 48.5% | 45.2% | 45.7% | **37.7%** |
| Early TNR (Dy3-10) | 25.0% | 25.0% | 25.0% | 25.0% |
| Bal. Acc. | 62.4% | 64.8% | 64.6% | **61.9%** |
| Days TNR=0 | 3/11 | 3/11 | 3/11 | **4/11** |
| F1 (NA) | 35.8% | 37.7% | 36.6% | **31.8%** |

### ResNet50

| Metric | Run 1 (non-det) | Run 2 (non-det) | Run 3 (det) | Run 4 (non-det) |
|---|:-:|:-:|:-:|:-:|
| Avg. TNR | 43.5% | 43.9% | **42.4%** | 46.5% |
| Early TNR (Dy3-10) | 25.0% | **20.8%** | **20.8%** | 25.0% |
| Bal. Acc. | 61.5% | **58.6%** | 59.0% | 63.4% |
| Days TNR=0 | 1/11 | **3/11** | **3/11** | 1/11 |
| F1 (NA) | 33.6% | **28.7%** | 29.8% | 37.0% |

### EfficientNet-B0

| Metric | Run 1 (non-det) | Run 2 (non-det) | Run 3 (det) | Run 4 (non-det) |
|---|:-:|:-:|:-:|:-:|
| Avg. TNR | 44.8% | 39.2% | **37.9%** | 43.5% |
| Early TNR (Dy3-10) | 16.7% | **12.5%** | **12.5%** | 25.0% |
| Bal. Acc. | 65.3% | **62.8%** | 63.9% | 64.1% |
| Days TNR=0 | 1/11 | **2/11** | **2/11** | **2/11** |
| F1 (NA) | 38.3% | **33.7%** | 36.7% | 32.5% |

---

## 4. Variance Analysis

- Range = max − min across the four runs

| Metric | DINOv2 Range | ResNet50 Range | EffNet-B0 Range |
|---|:-:|:-:|:-:|
| Avg. TNR | 10.8% | 4.1% | 6.9% |
| Early TNR (Dy3-10) | 0% | 4.2% | 12.5% |
| Bal. Acc. | 2.9% | 4.8% | 2.5% |
| Days TNR=0 | 1 day | 2 days | 1 day |
| F1 (NA) | 5.9% | 8.3% | 5.8% |

---

## 5. Implications

- Single-seed variance reaches ~10% on Avg. TNR (DINOv2) and 12.5% on Early TNR (EfficientNet-B0), comparable to or larger than the reproduced-vs-paper gap
- `set_deterministic(True)` does not reduce variance. It fixes future runs of the same code to reproduce the same sample of the noise distribution, not the underlying spread
- With NA = 6 in most test-day splits, a single binary-prediction flip ≈ 17% swing in per-day TNR, which propagates to aggregate metrics
- Exact reproduction of paper's Table 2 numbers from a single-seed run is not feasible with this codebase
