#!/usr/bin/env python3
"""
Verify the counts of organoids with image data vs metabolite data
"""
import json

ALL_DATA_JSON = 'all_data.json'
TARGET_DAY = 'Dy30'

# Load all_data.json
print("Loading all_data.json...")
with open(ALL_DATA_JSON) as f:
    all_data = json.load(f)

print(f"Total records in all_data.json: {len(all_data)}")

# Helper function to compute majority label from evaluations
def compute_majority_label(evaluations, min_votes=4):
    """Compute majority label from survey evaluations."""
    if not evaluations or len(evaluations) != 5:
        return None
    
    votes = {}
    for eval_data in evaluations:
        evaluation = eval_data.get('evaluation', '')
        if evaluation:
            votes[evaluation] = votes.get(evaluation, 0) + 1
    
    acceptable = votes.get('Acceptable', 0)
    not_acceptable = votes.get('Not Acceptable', 0)
    
    # Use majority threshold (at least 4 out of 5)
    if acceptable >= min_votes:
        return 'Acceptable'
    elif not_acceptable >= min_votes:
        return 'Not Acceptable'
    else:
        return None  # Skip ambiguous cases

# Count organoids with image data (Dy30 + survey + clear labels)
image_organoids = []
image_with_metabolites = []
image_without_metabolites = []

for key, value in all_data.items():
    # Filter for Dy30 records with survey data
    if value.get('dayID') != TARGET_DAY:
        continue
    
    if 'survey' not in value:
        continue
    
    # Check if processed image data exists
    if 'processed' not in value:
        continue
    
    # Get evaluations from survey data
    evaluations = value['survey'].get('evaluations', [])
    if not evaluations:
        continue
    
    # Compute label from evaluations
    label = compute_majority_label(evaluations, min_votes=4)
    if label is None:
        continue
    
    # This organoid has valid image data
    image_organoids.append(key)
    
    # Check if it also has metabolite data
    if 'metabolites' in value and value['metabolites']:
        image_with_metabolites.append(key)
    else:
        image_without_metabolites.append(key)

print(f"\n{'='*60}")
print("VERIFICATION RESULTS:")
print(f"{'='*60}")
print(f"Organoids with valid image data (Dy30, survey, clear labels): {len(image_organoids)}")
print(f"  - With metabolite data: {len(image_with_metabolites)}")
print(f"  - WITHOUT metabolite data: {len(image_without_metabolites)}")
print(f"{'='*60}")

if image_without_metabolites:
    print(f"\nOrganoids WITHOUT metabolite data ({len(image_without_metabolites)}):")
    for org_id in sorted(image_without_metabolites)[:10]:  # Show first 10
        print(f"  - {org_id}")
    if len(image_without_metabolites) > 10:
        print(f"  ... and {len(image_without_metabolites) - 10} more")

print(f"\n{'='*60}")
print("SUMMARY:")
print(f"{'='*60}")
print(f"Image organoids: {len(image_organoids)}")
print(f"Overlap with metabolites: {len(image_with_metabolites)}")
print(f"Percentage with metabolites: {len(image_with_metabolites)/len(image_organoids)*100:.1f}%")
print(f"{'='*60}")

