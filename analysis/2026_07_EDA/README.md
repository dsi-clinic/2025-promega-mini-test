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

## What it reports

### Table 1 — Dy30 vote-split (col2, N=248)

Each classified-at-Dy30 organoid grouped by its survey tally
`Acceptable-NotAcceptable`, with the implied majority consensus, counts, and
percents. Most organoids got 5 reviews; a re-reviewed subset (19 organoids) got
10. Example (numbers track the current `all_data.json`):

| split | votes | consensus | count | pct |
|---|---|---|---|---|
| 5-0 | 5 | Acceptable | 96 | 38.7 |
| 4-1 | 5 | Acceptable | 56 | 22.6 |
| 3-2 | 5 | Acceptable | 26 | 10.5 |
| 2-3 | 5 | Not Acceptable | 22 | 8.9 |
| 1-4 | 5 | Not Acceptable | 19 | 7.7 |
| 0-5 | 5 | Not Acceptable | 10 | 4.0 |
| 10-0 … 0-10 | 10 | (re-reviews) | 19 | 7.6 |

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
