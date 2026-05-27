# paper_2026_04 common.py label convention bug — diagnosis & fix

This note documents a label-key mismatch in `analysis/paper_2026_04/common.py:compute_classification_metrics` and the corrected version in `common_renamed.py` used by all paper-replication scripts.

## Background

- `data_loader.py` encodes labels via `LABEL_TO_INT`:
  - `0 = Acceptable`
  - `1 = Not Acceptable`
- Paper reports metrics with **Acceptable as the positive class** (sensitivity, TPR, recall_acceptable, etc.).
- sklearn's binary `confusion_matrix(..., labels=[0, 1])` treats the **second** class as positive — so under the internal encoding, sklearn's `tp/(tp+fn)` is the Not Acceptable recall.
- Mismatch between these two conventions is the source of the bug.

## Worked example

```python
y_true = [0, 1, 0, 0, 0]   # 4 Acceptable, 1 Not Acceptable
y_pred = [0, 0, 0, 0, 1]
```

**Ground-truth values (computed by hand):**

| Metric | Value | Derivation |
|---|---:|---|
| `recall_acceptable` | 0.75 | 3 of 4 Acceptable predicted correctly |
| `recall_not_acceptable` | 0.00 | 0 of 1 Not Acceptable predicted correctly |
| `precision_acceptable` | 0.75 | 3 of 4 Acceptable predictions correct |
| `precision_not_acceptable` | 0.00 | 0 of 1 Not Acceptable predictions correct |
| `f1_acceptable` | 0.75 | 2·0.75·0.75/(0.75+0.75) |
| `f1_not_acceptable` | 0.00 | 2·0·0/(0+0) → 0 |
| `accuracy` | 0.60 | 3 of 5 |
| `balanced_accuracy` | 0.375 | (0.75 + 0.00)/2 |
| `sensitivity` (= TPR, paper) | 0.75 | = recall_acceptable |
| `specificity` (= TNR, paper) | 0.00 | = recall_not_acceptable |

## `common.py` (buggy) — what it returns vs what each key claims

| Key (claimed meaning) | Returned value | Actually represents | Correct? |
|---|---:|---|---|
| `sensitivity` | 0.00 | NA recall (claims = Acceptable recall) | ✗ |
| `specificity` | 0.75 | Acceptable recall (claims = NA recall) | ✗ |
| `tpr_acceptable` | 0.00 | NA recall | ✗ |
| `tnr_not_acceptable` | 0.75 | Acceptable recall | ✗ |
| `recall_acceptable` | 0.00 | NA recall (sklearn rec[1]) | ✗ |
| `recall_not_acceptable` | 0.75 | Acceptable recall (sklearn rec[0]) | ✗ |
| `precision_acceptable` | 0.00 | NA precision | ✗ |
| `precision_not_acceptable` | 0.75 | Acceptable precision | ✗ |
| `f1_acceptable` | 0.00 | NA F1 | ✗ |
| `f1_not_acceptable` | 0.75 | Acceptable F1 | ✗ |
| `accuracy` | 0.60 | accuracy | ✓ |
| `balanced_accuracy` | 0.375 | balanced accuracy | ✓ |
| `confusion_matrix` | `{tn:3, fp:1, fn:1, tp:0}` | raw counts (NA=1=positive) | ✓ (raw) |
| `n_positive` | 1 | count of label 1 = NA | ✓ (internal naming) |
| `n_negative` | 4 | count of label 0 = Acceptable | ✓ (internal naming) |

**10 class-specific keys are inverted.** All Acceptable / Not Acceptable labels are swapped.

### Root cause

`common.py` runs sklearn under internal convention (NA = 1 = positive) but stores results under **paper key names** that assume the opposite mapping.

```python
# common.py
tp = NA correctly predicted    # because labels=[0,1] makes 1 the positive class
tn = Acceptable correctly predicted

tpr = tp / (tp + fn)   # = NA recall under internal convention
tnr = tn / (tn + fp)   # = Acceptable recall

# but then stored as:
"tpr_acceptable":     round(tpr, 4),   # mismatch: this is NA recall
"tnr_not_acceptable": round(tnr, 4),   # mismatch: this is Acceptable recall
```

Same swap happens in `precision_recall_fscore_support(..., labels=[0, 1])`:
- `rec[0]` = label 0 = Acceptable recall
- `rec[1]` = label 1 = NA recall

But `common.py` does:
```python
"recall_not_acceptable": round(rec[0], 4),   # mismatch: stores Acceptable recall
"recall_acceptable":     round(rec[1], 4),   # mismatch: stores NA recall
```

Six index-swapped keys (`precision_*`, `recall_*`, `f1_*` × Acceptable/NA) + four name-swapped keys (`sensitivity`, `specificity`, `tpr_acceptable`, `tnr_not_acceptable`) = 10 inverted keys.

## `common_renamed.py` (fixed) — same example

`common_renamed.py` runs sklearn under the **same** internal convention but explicitly translates the results into paper-convention key names before returning. Same `y_true` / `y_pred`:

| Key | Returned value | Represents | Correct? |
|---|---:|---|---|
| `sensitivity` | 0.75 | Acceptable recall (paper TPR) | ✓ |
| `specificity` | 0.00 | NA recall (paper TNR) | ✓ |
| `tpr_acceptable` | 0.75 | Acceptable recall | ✓ |
| `tnr_not_acceptable` | 0.00 | NA recall | ✓ |
| `recall_acceptable` | 0.75 | Acceptable recall | ✓ |
| `recall_not_acceptable` | 0.00 | NA recall | ✓ |
| `precision_acceptable` | 0.75 | Acceptable precision | ✓ |
| `precision_not_acceptable` | 0.00 | NA precision | ✓ |
| `f1_acceptable` | 0.75 | Acceptable F1 | ✓ |
| `f1_not_acceptable` | 0.00 | NA F1 | ✓ |
| `accuracy` | 0.60 | accuracy | ✓ |
| `balanced_accuracy` | 0.375 | balanced accuracy | ✓ |
| `confusion_matrix` | `{tn:3, fp:1, fn:1, tp:0}` | raw counts (NA=1=positive) | ✓ |
| `n_positive` | 1 | count of label 1 = NA (internal) | ✓ |
| `n_negative` | 4 | count of label 0 = Acceptable (internal) | ✓ |

### Fix mechanics (`common_renamed.py`)

```python
# Compute under internal convention (NA = 1 = positive).
tn, fp, fn, tp = cm.ravel()

# Re-label to paper convention (Acceptable = positive).
acceptable_recall = tn / (tn + fp) if (tn + fp) > 0 else 0.0   # paper TPR / sensitivity
na_recall         = tp / (tp + fn) if (tp + fn) > 0 else 0.0   # paper TNR / specificity

# Store with paper-consistent names.
"sensitivity":         round(acceptable_recall, 4),
"specificity":         round(na_recall, 4),
"tpr_acceptable":      round(acceptable_recall, 4),
"tnr_not_acceptable":  round(na_recall, 4),

# precision_recall_fscore_support returns index 0 = label 0 = Acceptable.
# Store using *correct* index → paper-consistent names.
"precision_acceptable":     round(prec[0], 4),
"recall_acceptable":        round(rec[0], 4),
"f1_acceptable":            round(f1[0], 4),
"precision_not_acceptable": round(prec[1], 4),
"recall_not_acceptable":    round(rec[1], 4),
"f1_not_acceptable":        round(f1[1], 4),
```

`confusion_matrix`, `n_positive`, `n_negative` kept under internal convention (raw counts; renaming them would be more confusing than helpful).

## Why `common.py` was not patched directly

- Other repo code may already consume `common.py` keys under the (wrong) interpretation — patching in place would silently change those downstream values.
- Safer to introduce a side-by-side corrected version (`common_renamed.py`) that paper-replication scripts opt into explicitly.

## Scope

- Currently `common_renamed.py` is applied to `analysis/paper_2026_04/metabolites_train.py` only.
