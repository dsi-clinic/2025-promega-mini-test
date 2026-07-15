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

## Metabolite data: exchange rate vs. concentration (what we use)

There are **two kinds** of metabolite quantity for this dataset:

1. **Exchange rate (flux)** — the lab's paper metric: a *size-normalized
   exchange rate* in **nmol/day/µm³**, positive = release into medium, negative =
   clearance/uptake. Derived (Methods): `(conditioned − unconditioned media
   conc) × well volume ÷ days-between-media-changes ÷ V_eff`, with `V = Area^1.32`
   and `V_eff = (V_media_change + V_sampling)/2`. **We do not ingest this.** The
   precomputed result ships unused in `data/normalized/EXCH_*.csv` ("reserved").
2. **Concentration (µM)** — the raw assay concentration in the media. This is
   what our models use: `concentration_uM` (primary) + `initial_concentration`
   (secondary).

**`concentration_uM` and `initial_concentration` are the *same* measurement at
two dilution scales** — `initial_concentration = concentration_uM ×
dilution_factor` *exactly* (Glucose ×2000, Lactate/BCAA ×400, Glutamate/Pyruvate
×100; matching the assay dilutions, CoV≈0). So `concentration_uM` is the in-assay
(diluted) reading and `initial_concentration` is the back-corrected (undiluted)
media concentration — **"initial" is NOT the unconditioned baseline**, and using
both as features is redundant (perfectly collinear within a metabolite).

**Our `scaled`+`delta` dials approximate, but do not equal, the exchange rate:**
we divide `concentration_uM / mask_area_um2` (area, current day) and take a
day-over-day concentration delta; the lab divides by `V_eff` (2-day-avg
**volume**, `Area^1.32`) and computes `(Δconc) × volume ÷ days`. Reproducing
their exchange rate would additionally require the **unconditioned-media
baseline**, **well volume**, and **days between media changes** (none are in our
concentration fields). Also note an internal inconsistency in the lab pipeline:
the Methods normalize by **volume** (`Area^1.32`), but the MMM notebook
normalizes metabolites by **area** (`Average_area_win`, /µm²), and their stored
`Volume` empirically scales as `Area^1.26`.

## Run

The package name starts with a digit, so it is run **by path**, not via `-m`:

```bash
make run ARGS="analysis/2026_06_metabolite_pred/run.py"
# or
PYTHONPATH=. python analysis/2026_06_metabolite_pred/run.py
```

Flags: `--cohort {strong,full,all}` (default `all`), `--days Dy30 Dy24 ...`
(default all days), `--configs {nominal_nodelta,nominal_delta,scaled_nodelta,scaled_delta,nominal_nodelta_win,nominal_delta_win,scaled_nodelta_win,scaled_delta_win} ...`
(default all eight), `--skip-lgbm`, `--skip-lr`, `--folds N` (default 5),
`--seed N` (default 42).

## Feature configurations (size x delta x winsorize)

Three independent dials are swept — **size** (nominal vs `/mask_area_um2`),
**delta** (levels only vs + day-over-day delta), and **winsorize** (raw vs
per-day 1/99 clip). The 4 base configs each get a winsorized (`_win`) twin, so
the default run is **8 configs x 2 cohorts = 16 figures**:

| Config key | Size | Delta | Winsorize |
|---|---|---|---|
| `nominal_nodelta` | nominal | no | no |
| `nominal_delta` | nominal | yes | no |
| `scaled_nodelta` | `/mask_area_um2` | no | no |
| `scaled_delta` | `/mask_area_um2` | yes | no |
| `nominal_nodelta_win` | nominal | no | yes |
| `nominal_delta_win` | nominal | yes | yes |
| `scaled_nodelta_win` | `/mask_area_um2` | no | yes |
| `scaled_delta_win` | `/mask_area_um2` | yes | yes |

- **Size** divides each metabolite measurement (and, with `+delta`, the delta)
  by the organoid's segmentation area `mask_area_um2` — our own mask-derived size
  (foreground px x per-axis um/px), which tracks Promega's `win_vol_norm` volume
  at R^2~=0.98. Nominal vs scaled share the same base, so any difference is
  attributable to size normalization alone.
- **Winsorize** reads the persisted per-day 1/99 columns (`concentration_uM_win`
  / `initial_concentration_win`) from `all_data.json` — see
  `pipeline/metabolites/winsorize.py` and `make winsorize-write`.

All three dials pass through to `get_metabolite_features` (`normalize_by_size`,
`include_growth`, `winsorize`).

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

- `$ANALYSIS_OUTPUT_DIR/metabolite_pred/results_<cohort>_<config>.json`
  (16 files: 2 cohorts x 8 configs; schema `results[model_display][day] =
  metrics_dict`; a distinct subdir so the paper's `metabolites/results.json` is
  never clobbered)
- `$ANALYSIS_OUTPUT_DIR/figures/metabolite_pred_<cohort>_<config>_LightGBM_vs_LogReg.png`
  (16 figures)
- `$ANALYSIS_OUTPUT_DIR/metabolite_pred/metabolite_pred_<cohort>_<config>_<model>_shap.txt`
  — out-of-fold SHAP importance per day (`shap_importance.py`, headline configs)
- `$ANALYSIS_OUTPUT_DIR/figures/metabolite_summary_<cohort>.png` +
  `metabolite_summary_table.csv` (`metabolite_summary_panel.py`)

## Notes

- `recall_not_acceptable` from `common.compute_classification_metrics` is a
  pre-existing naming quirk (it indexes the Acceptable class); it is reused
  verbatim for comparability with the paper results.
- Labels follow AGENTS.md rule #9 (1 = Not Acceptable, 0 = Acceptable) and all
  data is read through `pipeline.data_loader` (rule #3).

## Comparison to the RehenLab MMM good/bad prediction

**Source (the IDOR good/bad analysis we compare against):**
<https://github.com/RehenLab/MMM/blob/main/main_data_analysis.ipynb>
— the lab's own per-day Good/Bad classifier (cells 18–24 and 36; Figs 6D/6E).
This note covers **only the good/bad prediction**, not the descriptive figures
(violin/PCA/boxplots) that make up most of that notebook.

**TL;DR — same family, not identical** (so similar-range results, not an exact
match):
- **Same:** per-day **Logistic Regression** (`StandardScaler` + balanced
  weights), per-day **1/99 winsorization** (we reproduce their `_win`,
  `make verify-winsorize`), **metabolites normalized by organoid area**
  (`make verify-mask-area` ~0.4%), and **"Uncertain" excluded ≈ our
  strong-consensus** cohort.
- **Different:** we add **LightGBM** and **day-over-day deltas**, validate with
  **nested group K-fold CV** (vs their repeated 70/30 holdout), and model a
  **fixed metabolite-only** set — whereas they do an **exhaustive
  morphometry+metabolite feature search** and size-normalize by a **2-day
  averaged** `Average_area_win` (vs our current-day `mask_area_um2`).
- **Closest apples-to-apples:** our **LogReg · scaled · no-delta ·
  strong-consensus** config at a matched day (e.g. Dy30).

Detail below.

### Shared design (why results should be close)
- **Per-day modeling.** Each day modeled independently (their day list 3..30;
  ours `DAY_ORDER`).
- **Logistic Regression** with `StandardScaler` + `class_weight='balanced'` — we
  run exactly this as one of our two models (`MODEL_SPECS["logreg"]`).
- **Per-day 1st/99th-percentile winsorization** of features — identical recipe;
  our `pipeline/metabolites/winsorize.py` reproduces their `_win` columns and is
  asserted to match (`make verify-winsorize`).
- **Metabolites normalized by organoid AREA** — they divide by `Average_area_win`;
  we divide by `mask_area_um2` (the `scaled` dial). Both express metabolites as
  per-area exchange rates. (`make verify-mask-area` shows our area reproduces
  their `Area_win` to ~0.4%.)
- **Balanced accuracy** is the headline metric; they exclude the ambiguous
  middle class (`Uncertain`), which corresponds to our no-consensus 3-2/2-3
  organoids — so their Good/Bad set ≈ our **strong-consensus** cohort.

### Differences (why they won't match exactly)
| Aspect | RehenLab MMM | Ours (`2026_06_metabolite_pred`) |
|---|---|---|
| Models | Logistic Regression only | LightGBM **and** Logistic Regression |
| Validation | repeated stratified **70/30 holdout × 5 seeds**, fixed threshold 0.5 | nested stratified **group K-fold CV** (out-of-fold), inner GridSearch + threshold tuning |
| Feature search | **exhaustive** over morphometry + metabolite combos (1000s of sets), report best | fixed 6-metabolite set; **no morphometry** |
| Morphometry | included (Area_log, Feret, Circ, Solidity, AR, Complexity) | none (metabolite-only; morphometry lives in the image analyses) |
| Deltas | none in the classifier (growth rate is descriptive only) | day-over-day metabolite **delta** is a feature dial |
| Metabolite value | winsorized, area-normalized (`*_win_area_norm`) | raw `concentration_uM` / `concentration_uM_win`, optionally `/mask_area_um2` |
| Size metric | `Average_area_win` = winsorized mean of **current + previous** day area | `mask_area_um2` = our segmentation area, **current** day only |
| Per-day aggregation | per-organoid **mean** of that day's rows | one record per organoid-day |
| Positive class | Good = 1 | Not Acceptable = 1 (opposite encoding; same task) |
| Cohorts | Good/Bad (Uncertain excluded) | strong-consensus (≥4/5) and full (simple majority) |

### Closest apples-to-apples
The most comparable of our configs to their metabolite-driven result is
**Logistic Regression, `scaled` (size-normalized), no-delta, strong-consensus
cohort**, at a matched day (e.g. Dy30). That isolates the shared choices
(per-day LR, winsorized, area-normalized metabolites, Uncertain excluded); the
remaining gaps are our group-aware CV vs their repeated holdout, and their
inclusion of morphometry + exhaustive feature selection. A true match would also
require dropping the day-over-day delta and using their `Average_area_win`
(2-day) size rather than our current-day `mask_area_um2`.

## Winsorization scope: per-day vs whole-dataset (beads qp7)

We were told MalateGLO's stored `win` was winsorized over the **whole dataset**
(all days pooled), whereas the other five metabolites were winsorized **per-day**.
`verify_winsorize_scope.py` tests this empirically: for each metabolite it
winsorizes the raw `concentration_uM` per-day (1st/99th) *and* whole-dataset
(1st/99th), then measures the fraction of records where the lab `win` matches
`k * winsorized_raw` (`k` = the per-metabolite units constant, fit on the bulk
exactly as `pipeline.metabolites.verify_winsorization`). Both `win` and
`concentration_uM` are read straight from `all_data.json` (rules 3 & 16).

```bash
make run ARGS="analysis/2026_06_metabolite_pred/verify_winsorize_scope.py"
```

Match rate = fraction with `|win - k*raw_win| / |win| < 0.03`. A scope only
"reproduces" `win` if its match rate ≥ 0.50.

| Metabolite | per-day | whole-dataset | Scope that fits |
|---|---|---|---|
| GlucoseGlo | 0.967 | 0.955 | **per-day** |
| GlutamateGlo | 0.689 | 0.675 | **per-day** |
| LactateGlo | 0.972 | 0.946 | **per-day** |
| PyruvateGlo | 0.955 | 0.942 | **per-day** |
| BCAAGlo | 0.964 | 0.949 | **per-day** |
| MalateGlo | 0.041 | 0.040 | **neither** |

**Finding — the whole-dataset hypothesis for Malate is refuted.** All five
well-behaved metabolites fit **per-day** (per-day match rate consistently beats
whole-dataset — the difference is the ~1–2% of tail points that per-day and
pooled clipping bound differently). For **MalateGlo, neither scope reproduces
`win`**: both match rates collapse to ~0.04. This is not a scope problem —
Malate's raw `concentration_uM` runs −5662…+27 µM (26% negative, at the assay
noise floor) while its stored `win` is bounded in ~0.002…0.015 and always
positive, and `win` is **non-monotonic** in raw concentration. Since any
winsorization is monotonic non-decreasing in its input, no clip of
`concentration_uM` at any scope can produce Malate's `win`. Malate's `win` is a
**separately-cleaned signal**, consistent with the existing note in
`pipeline/metabolites/winsorize.py` (`MALATE` is the documented exception to the
provenance check). The determination is pinned in
`tests/test_winsorize_scope.py`.
