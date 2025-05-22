import json

# Load the organoid analysis results
try:
    with open('organoid_analysis_results.json') as f:
        analysis_data = json.load(f)
except FileNotFoundError:
    print("Error: 'organoid_analysis_results.json' not found.")
    exit()
except json.JSONDecodeError:
    print("Error: Could not decode JSON from 'organoid_analysis_results.json'.")
    exit()

# Initialize the new mapping dictionary
labeled_organoids = {}

# Iterate through the organoid analysis results
for organoid_name, analysis in analysis_data['analysis'].items():
    if analysis['agreement_level'] == 'complete':
        # Get the classification label
        classifications = analysis['classifications']
        if classifications:
            # Assuming complete agreement means only one classification exists with a non-zero count
            for label, data in classifications.items():
                if data['count'] > 0:
                    label_to_use = label
                    break
            else:
                print(f"Warning: Organoid '{organoid_name}' has complete agreement but no classification label found. Skipping.")
                continue

            labeled_organoids[organoid_name] = {
                'label': label_to_use
            }
        else:
            print(f"Warning: Organoid '{organoid_name}' has complete agreement but no classifications found. Skipping.")

# Save the new labeled organoid mapping to a JSON file
output_file = 'complete_agreement_organoids.json'
with open(output_file, 'w') as f:
    json.dump(labeled_organoids, f, indent=2)

print(f"\nLabeled organoid mapping saved to '{output_file}'.")
print(f"Total number of labeled organoids: {len(labeled_organoids)}")