import json
import csv
from pathlib import Path

# --- File paths ---
INPUT_JSON = Path("data/raw/final_combined_metadata.json")
OUTPUT_CSV = Path("data/summary/vote_counts.csv")
OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

# --- Load JSON ---
with INPUT_JSON.open() as f:
    data = json.load(f)

# --- Collect vote counts ---
rows = []
for key, record in data.items():
    evals = record.get("survey", {}).get("evaluations", [])
    n_votes = len(evals)

    # Extract metadata if available
    dayID = record.get("dayID", "")
    BA = record.get("BA", "")
    wellID = record.get("wellID", "")

    rows.append(
        {"id": key, "dayID": dayID, "BA": BA, "wellID": wellID, "num_votes": n_votes}
    )

# --- Write CSV ---
with OUTPUT_CSV.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["id", "dayID", "BA", "wellID", "num_votes"])
    writer.writeheader()
    writer.writerows(rows)

print(f"✅ Vote count summary written to {OUTPUT_CSV}")

# --- Optional: quick stats ---
from collections import Counter

dist = Counter(r["num_votes"] for r in rows)
print("\nVote count distribution:")
for k in sorted(dist):
    print(f"  {k} votes: {dist[k]} samples")
