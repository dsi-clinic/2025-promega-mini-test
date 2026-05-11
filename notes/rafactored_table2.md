# Table 2 Reproduction — Findings

**Code references:**
- New (refactored) code: `analysis/imagequality_classification/train_model_dinov2.py`
- Old code: `analysis/images/classifier/train_model_accuracy_tony_dinov2.py`

---

## 1. Paper Reference Values

- Label convention: Acceptable=1, Not Acceptable=0
- TNR is the key metric (minority-class focus)

| Metric | ViT (DINOv2) | ResNet50 | EfficientNet-B0 |
|---|:-:|:-:|:-:|
| Avg. TNR | 23.7% | 20.5% | 29.1% |
| Early TNR (Dy3-10) | 4.2% | 0.0% | 12.5% |
| Bal. Acc. | 58.0% | 57.2% | 59.0% |
| Days TNR=0 | 4/11 | 5/11 | 2/11 |
| F1 (NA) | 24.7% | 21.4% | 29.2% |

---

## 2. DINOv2 (ViT) — Configuration Sweep

| Metric | Paper | J + new | J + new + norm-all | W + new | W + new + norm-all |
|---|:-:|:-:|:-:|:-:|:-:|
| Avg. TNR | 23.7% | **48.5%** | 45.7% | 56.1% | 59.1% |
| Early TNR (Dy3-10) | 4.2% | 25.0% | **20.8%** | 50.0% | 45.8% |
| Bal. Acc. | 58.0% | **62.4%** | 62.2% | 64.6% | 66.3% |
| Days TNR=0 | 4/11 | **3/11** | 2/11 | 2/11 | 2/11 |
| F1 (NA) | 24.7% | **35.8%** | 35.2% | 32.9% | 36.2% |

---

## 3. ResNet50 — Configuration Sweep

| Metric | Paper | J + new | J + new + norm-all | W + new | W + new + norm-all |
|---|:-:|:-:|:-:|:-:|:-:|
| Avg. TNR | 20.5% | **43.5%** | 46.8% | 51.5% | 50.0% |
| Early TNR (Dy3-10) | 0.0% | 25.0% | **16.7%** | 62.5% | 33.3% |
| Bal. Acc. | 57.2% | **61.5%** | 62.1% | 60.8% | 63.1% |
| Days TNR=0 | 5/11 | **1/11** | **1/11** | 2/11 | 0/11 |
| F1 (NA) | 21.4% | **33.6%** | 36.1% | 30.8% | 36.7% |

---

## 4. EfficientNet-B0 — Configuration Sweep

| Metric | Paper | J + new | J + new + norm-all | W + new | W + new + norm-all |
|---|:-:|:-:|:-:|:-:|:-:|
| Avg. TNR | 29.1% | **44.8%** | 49.6% | 54.5% | 47.0% |
| Early TNR (Dy3-10) | 12.5% | **16.7%** | 29.2% | 25.0% | 45.8% |
| Bal. Acc. | 59.0% | **65.3%** | 65.6% | 65.7% | 62.5% |
| Days TNR=0 | 2/11 | **1/11** | **1/11** | 0/11 | 0/11 |
| F1 (NA) | 29.2% | **38.3%** | 41.3% | 40.0% | 35.2% |

---

## 5. Sources of new code Divergence (test set)

| Day | Tony original n | new code n | Diff | Note |
|---|:-:|:-:|:-:|---|
| Dy03–Dy17 | 37 | 36 | −1 | Consistent across all 7 early/mid days |
| Dy20.5 | 25 | 37 | +12 | new code includes 12 more samples |
| Dy24 | 43 | 40 | −3 | |
| Dy28 | 44 | 40 | −4 | |
| Dy30 | 44 | 40 | −4 | |

- Pattern holds identically across all three backbones → data-loading-stage difference, not model-specific

---

## 6. Summary

| Quantity | DINOv2 (ViT) | ResNet50 | EfficientNet-B0 |
|---|:-:|:-:|:-:|
| Closest configuration | J + new | J + new | J + new |
| Paper Avg. TNR | 23.7% | 20.5% | 29.1% |
| Reproduced Avg. TNR | 48.5% | 43.5% | 44.8% |
| Paper Days TNR=0 | 4/11 | 5/11 | 2/11 |
| Reproduced Days TNR=0 | 3/11 | 1/11 | 1/11 |
