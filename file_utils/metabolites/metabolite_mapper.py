import pandas as pd, json
from config import METABOLITE_SOURCE_XLSX, METABOLITE_MAP_JSON
excel_path = METABOLITE_SOURCE_XLSX
output_json_path = METABOLITE_MAP_JSON

# Read Excel sheet
df = pd.read_excel(METABOLITE_SOURCE_XLSX, sheet_name="Experimental Values")

# Normalize column names (strip and lowercase for consistency)
df.columns = [col.strip().lower() for col in df.columns]

# Initialize output dict
metabolite_map = {}

for _, row in df.iterrows():
    try:
        batch = str(int(row["batch"]))  # e.g. 1
        plate = str(int(row["starting plate"]))  # e.g. 2
        ba = f"BA{batch} 96_{plate}"  # e.g. "BA2 96_1"
        day = f'Dy{int(row["day"]):02d}'  # e.g. "Dy28"
        well = row["96 well"].strip().upper()  # e.g. "A5"
        organoid_id = f"{ba} {day} {well}"

        assay = row["assay"].strip()
        conc = row.get("concentration um")
        init_conc = row.get("initial  concentration")
        is_outlier = str(row.get("rlu outside 3 stdev")).strip().lower() == "outlier"
        well_384 = row.get("384 well", "").strip().upper()

        if organoid_id not in metabolite_map:
            metabolite_map[organoid_id] = {}

        # Store under assay name
        metabolite_map[organoid_id][assay] = {
            "concentration_uM": conc,
            "initial_concentration": init_conc,
            "is_outlier": is_outlier,
            "well_384": well_384
        }

    except Exception as e:
        print(f"Skipping row due to error: {e}")


# Save the JSON
output_json_path.parent.mkdir(parents=True, exist_ok=True)
with open(output_json_path, "w") as f:
    json.dump(metabolite_map, f, indent=2)

print(f"Metabolite map saved to: {output_json_path} ({len(metabolite_map)} entries)")

