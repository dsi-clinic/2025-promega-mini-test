# 2026_06 metabolite prediction (our sample)

Metabolite-only classifier (LightGBM + Logistic Regression, per day) ported from
[`analysis/paper_2026_04/metabolites_train.py`](../paper_2026_04/metabolites_train.py)
but **scoped to our sample** — the IDOR col2 set (the 248 BA1+BA2 organoids
actually classified at Dy30) — and evaluated by **stratified cross-validation**
instead of a single held-out test split (the cohorts are small with a small
minority class, so a ~20-organoid test fold would carry only a handful of
Not-Acceptable cases).

The paper modeling config (`MODEL_SPECS`: estimator factory, hyperparameter
grid, CV scoring, threshold grid/scoring, scaler flag) is **imported**, not
copied, so it can't drift from the paper.

## Run

The package name starts with a digit, so it is run **by path**, not via `-m`:

```bash
make run ARGS="analysis/2026_06_metabolite_pred/run.py"
# or
PYTHONPATH=. python analysis/2026_06_metabolite_pred/run.py
```

Flags: `--cohort {strong,full,all}` (default `all`), `--days Dy30 Dy24 ...`
(default all days), `--skip-lgbm`, `--skip-lr`, `--folds N` (default 5),
`--seed N` (default 42).

## Two cohorts

Both are restricted to col2 (248) via `col2_membership_filter`; they differ only
in how the Dy30 survey vote becomes a binary label.

| Cohort | N | Acceptable | Not Acceptable | Labeling |
|---|---|---|---|---|
| `strong-consensus` | 198 | 165 | 33 | supermajority (≥4 of 5 regular votes); `paper_label_fn`. The 50 no-consensus (3-2/2-3) organoids are dropped. |
| `full` | 248 | 191 | 57 | simple majority of the 5 regular votes; `simple_majority_label_fn`. Resolves the 3-2/2-3 splits. |

Vote counts use the **regular-image** bucket only (`get_survey_vote_counts`),
matching the consensus rule in the merge step. Cohort sizes and label splits are
asserted in `build_cohort` and fail loudly on upstream data drift.

## Method

`cv.run_cv_for_day` does nested CV per (cohort, day, model):

- Outer `StratifiedGroupKFold` (default 5 folds, group = organoid id). Each
  organoid is predicted exactly once on a held-out fold (asserted).
- Inner `GridSearchCV` (`StratifiedGroupKFold`-3) on each outer-train fold to
  pick hyperparameters; LightGBM `scale_pos_weight` recomputed per train fold;
  `StandardScaler` (logreg) fit on the train fold only.
- Decision threshold tuned on the outer-train via inner `cross_val_predict`,
  maximizing the model's `threshold_scoring`.

Reported per (cohort, day, model):

- **pooled out-of-fold** metrics via `common.compute_classification_metrics`
  (this is the `balanced_accuracy` the by-day plot uses), plus
- **per-fold** `balanced_accuracy_cv_mean/std` and
  `recall_not_acceptable_cv_mean/std`, and `n` / `n_pos` / `n_neg` / `n_folds`.

Edge cases degrade rather than crash: if the minority class is too small to
stratify the requested folds, the fold count is reduced (or the day skipped),
logged each time.

## Outputs

- `$ANALYSIS_OUTPUT_DIR/metabolite_pred/results_strong-consensus.json`
- `$ANALYSIS_OUTPUT_DIR/metabolite_pred/results_full.json`
  (schema `results[model_display][day] = metrics_dict`; a distinct subdir so the
  paper's `metabolites/results.json` is never clobbered)
- `$ANALYSIS_OUTPUT_DIR/figures/metabolite_pred_<cohort>_LightGBM_vs_LogReg.png`

## Notes

- `recall_not_acceptable` from `common.compute_classification_metrics` is a
  pre-existing naming quirk (it indexes the Acceptable class); it is reused
  verbatim for comparability with the paper results.
- Labels follow AGENTS.md rule #9 (1 = Not Acceptable, 0 = Acceptable) and all
  data is read through `pipeline.data_loader` (rule #3).
