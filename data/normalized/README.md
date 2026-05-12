# Promega Normalized Metabolite Data

Residualized, winsorized, and (optionally) volume-normalized metabolite values produced by the Promega-side analysis. These are an alternative numeric representation of the same per-organoid-per-day assay readings already present in `data/all_data.json` as `concentration_uM` and `initial_concentration`.

## Files

| File | Used by pipeline? |
|---|---|
| `CONC_data_organoides_residualized_final.csv` | Yes — read by Step 2 (`pipeline/metabolites/metabolite_mapper.py --normalized-csv ...`). Adds `win` and `win_vol_norm` to each per-assay block in `metabolite_map.json`. |
| `EXCH_data_organoides_residualized_final.csv` | No — reserved for future analyses. Same row schema as CONC; ships alongside for provenance. |

## Fields written to `all_data.json`

For each `record["metabolite"][<assay>]`:

| Field | Source | Notes |
|---|---|---|
| `concentration_uM` | Raw assay (xlsx) | Existing, unchanged. |
| `initial_concentration` | Raw assay (xlsx) | Existing, unchanged. |
| `win` | CONC CSV | Winsorized + per-metabolite scaled. **Not on the same numeric scale as `concentration_uM`** (see below). |
| `win_vol_norm` | CONC CSV | `win` divided by organoid volume. Magnitudes are tiny (1e-7 to 1e-10). |

## Scaling caveat

`win` is *not* a simple winsorized copy of any raw field. For 5 of 6 metabolites the ratio `win / initial_concentration` sits at a tight ~0.001 (Promega applies a 1000× scaling on top of the winsorization), but the per-metabolite ratio to `concentration_uM` varies (2.0 for GlucoseGlo, 0.4 for LactateGlo/BCAAGlo, 0.1 for GlutamateGlo/PyruvateGlo).

MalateGlo is the exception: its raw `initial_concentration` sits at the assay noise floor (~26% of records have negative raw values), so the clean `/1000` relationship breaks down for that metabolite. Treat MalateGlo's `win` as a per-metabolite-scaled cleaned signal, not a unit conversion of anything raw.

`win` and `win_vol_norm` preserve within-metabolite ordering, so they're valid drop-in features for any model that consumes one metabolite column at a time. They are **not** interchangeable across metabolites without further per-feature standardization.

## How to use

```python
from pipeline.data_loader import OrganoidDataset, filters_for_mode
ds = OrganoidDataset("data/all_data.json", filters=filters_for_mode("base"))
X, y, feat_names, ids = ds.get_metabolite_features("train", day="Dy30", field="win")
```

Valid `field` values: `"concentration_uM"` (default, raw), `"initial_concentration"`, `"win"`, `"win_vol_norm"`.

## Regeneration

If the CONC CSV is updated, rerun the pipeline from Step 2 onwards:

```bash
make step2            # rebuilds metabolite_map.json with new normalized values
make pipeline-merge   # rebuilds all_data.json
```
