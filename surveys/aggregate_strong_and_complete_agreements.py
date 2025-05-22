import json
import sys

# Load organoid analysis results
try:
    with open('organoid_analysis_results.json') as f:
        analysis_data = json.load(f)
except FileNotFoundError:
    print("Error: 'organoid_analysis_results.json' not found.")
    sys.exit(1)
except json.JSONDecodeError:
    print("Error: Could not decode JSON from 'organoid_analysis_results.json'.")
    sys.exit(1)

# Collect organoid names with strong or complete agreement and their labels
labeled = {}
for organoid_name, analysis in analysis_data.get('analysis', {}).items():
    if analysis.get('agreement_level') in ('complete', 'strong'):
        for label, data in analysis.get('classifications', {}).items():
            if data.get('count', 0) > 0:
                labeled[organoid_name] = {"label": label}
                break

# Save to JSON
output_file = 'labeled_organoid_strong_agreement.json'
with open(output_file, 'w') as f:
    json.dump(labeled, f, indent=2)

print(f"Labeled organoid mapping saved to '{output_file}'. Total: {len(labeled)}")