# 2026_07 EDA

Basic counting stats for the **IDOR sample (BA1 + BA2)**, computed entirely
through `pipeline.data_loader` (no raw `json.load` of `all_data.json`).

## Run

The package name starts with a digit, so it is run **by path**, not via `-m`:

```bash
make run ARGS="analysis/2026_07_EDA/eda.py"
# or
PYTHONPATH=. python analysis/2026_07_EDA/eda.py
```

Output: console tables + `$ANALYSIS_OUTPUT_DIR/figures/idor_eda_vote_splits.csv`
and `idor_eda_counts.csv`.

A companion script measures regular-vs-inverted image agreement:

```bash
make run ARGS="analysis/2026_07_EDA/inverse_regular_vote_correlation.py"
```

Output: console summary + `inverse_regular_vote_pairs.csv` and
`inverse_regular_vote_correlation.png`. See
[Regular vs inverted votes](#regular-vs-inverted-image-votes) below.

## What it reports

### Table 1 — Dy30 vote-split (col2, N=248)

Each classified-at-Dy30 organoid grouped by its survey tally
`Acceptable-NotAcceptable`, with the canonical consensus, counts, and percents.

This table counts the **regular-image vote bucket only** (`get_survey_vote_counts`),
which caps at 5 and is the bucket that actually decides the consensus label.
Each organoid is also re-shown on an inverted ("INV") copy of the image, so the
*combined* tally goes up to 10 — that full view is
`get_complete_survey_vote_counts` and is intentionally not used here. The
consensus column follows the merge rule (`compute_survey_majority`): a label
needs ≥ `MIN_VOTES` (=4) in the regular bucket, so 3-2 / 2-3 are **no
consensus**, not bare-majority wins. Example (numbers track the current
`all_data.json`):

| split | votes | consensus | count | pct |
|---|---|---|---|---|
| 5-0 | 5 | Acceptable | 107 | 43.1 |
| 4-1 | 5 | Acceptable | 58 | 23.4 |
| 3-2 | 5 | no consensus | 26 | 10.5 |
| 2-3 | 5 | no consensus | 24 | 9.7 |
| 1-4 | 5 | Not Acceptable | 21 | 8.5 |
| 0-5 | 5 | Not Acceptable | 12 | 4.8 |

This reconciles exactly with Table 2: Acceptable 107+58 = **165**, Not
Acceptable 21+12 = **33**, no consensus 26+24 = **50**.

### Table 2 — IDOR cohort cascade

| Metric | Count |
|---|---|
| BA1+BA2 organoids (total) | 309 |
| IDOR evaluated (col1) | 266 |
| **IDOR classified at Dy30 (col2)** | **248** |
| &nbsp;&nbsp;with good/bad consensus label | 198 |
| &nbsp;&nbsp;&nbsp;&nbsp;good (Acceptable) | 165 |
| &nbsp;&nbsp;&nbsp;&nbsp;bad (Not Acceptable) | 33 |
| &nbsp;&nbsp;reviewed, no consensus | 50 |
| col1 never reached Dy30 | 13 |
| col1 intro-survey (Dy30, excluded) | 5 |

Definitions (see `pipeline/data_loader.py` and
`analysis/verify_ba1_ba2_idor_list/`):

- **col1 (266)** — the IDOR partner's full evaluated list.
- **col2 (248)** — the subset with a Dy30 record carrying an assigned `main_id`
  (i.e. actually classified at Day 30). This is `266 − 13 never-reached-Dy30 −
  5 intro-survey`.
- **good/bad (198)** — col2 organoids with a Dy30 majority consensus
  ('Acceptable' / 'Not Acceptable'); the other 50 were reviewed without a clear
  majority. Derived via `OrganoidDataset(filters=idor_ba1_ba2_filters())`.

Counts are recomputed from `all_data.json` on every run, so they will track any
upstream data changes.

## Regular vs inverted-image votes

`inverse_regular_vote_correlation.py` checks whether re-showing an organoid on
an inverted ("INV") copy of its image changes how reviewers vote. It pairs the
per-bucket "Acceptable" counts (each out of 5) for every organoid carrying
**both** a `regular_votes` and `inverted_votes` Dy30 bucket.

Only 26 organoids were re-shown (BA1: 7, BA2: 12, BA4: 7) — a small set, so read
the correlation as indicative, not definitive. On the current `all_data.json`:

| Metric | Value |
|---|---|
| Organoids with both buckets | 26 |
| Pearson r (reg-acc vs inv-acc) | 0.92 |
| Spearman r | 0.90 |
| Mean \|reg − inv\| | 0.54 votes |
| Same ≥`MIN_VOTES` consensus | 21/26 (80.8%) |

Regular and inverted votes track each other closely; the five consensus
disagreements are all near the 4-of-5 threshold (e.g. a `4-1` bucket vs a `3-2`
bucket), never a clean flip. This supports using the regular bucket alone for
the canonical label (the inverted pass would rarely overturn it). The scatter
(`inverse_regular_vote_correlation.png`) plots reg-acc vs inv-acc against `y=x`.

## Note: merge label-conflict fix

While building this EDA, `BA1 96_1 E10` showed up with **0 votes** even though it
had 5 valid Dy30 reviews (4-1 Acceptable). Root cause was in the merge step
(`pipeline/merge/normalized_records.py`): organoid-level split-conflict
detection keyed on `organoid_id` while ignoring the day, so E10's Dy28 review
(Not Acceptable) and Dy30 review (Acceptable) were mistaken for a split conflict
and its label was wiped everywhere. The fix scopes conflict detection to the
canonical survey day (`SURVEY_LABEL_DAY = "Dy30"`); `all_data.json` was
regenerated. Net effect: E10 is now correctly Acceptable (consensus 197→198,
good 164→165), and propagated labels now consistently carry the Dy30 vote tally.
