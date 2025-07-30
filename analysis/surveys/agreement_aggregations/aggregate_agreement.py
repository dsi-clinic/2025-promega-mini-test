import json
from collections import Counter

# Load the new structured JSON
with open('organoid_surveys_aggregated.json') as f:
    data = json.load(f)

labeled = {}
for organoid, values in data.items():
    evaluations = values.get("evaluations", [])
    eval_votes = [entry["evaluation"] for entry in evaluations if "evaluation" in entry]
    
    if not eval_votes:
        continue

    vote_count = Counter(eval_votes)
    top_eval, count = vote_count.most_common(1)[0]
    total_votes = sum(vote_count.values())

    if count / total_votes >= 0.99:
        labeled[organoid] = {
            "label": top_eval,
            "votes": dict(vote_count),
            "n_votes": total_votes
        }

# Save labeled organoids
with open('labeled_organoid_complete_agreement.json', 'w') as f:
    json.dump(labeled, f, indent=2)

print(f"Labeled organoids saved. Total: {len(labeled)}")