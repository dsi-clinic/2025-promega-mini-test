#!/usr/bin/env python3
"""
Count ALL organoids with image data vs metabolite data (not just training subset)
"""
import json

ALL_DATA_JSON = 'all_data.json'

# Load all_data.json
print("Loading all_data.json...")
with open(ALL_DATA_JSON) as f:
    all_data = json.load(f)

print(f"Total records in all_data.json: {len(all_data)}")

# Count ALL organoids with processed image data
all_with_images = []
all_with_images_and_metabolites = []
all_with_images_no_metabolites = []

for key, value in all_data.items():
    # Check if processed image data exists
    if 'processed' in value and value['processed']:
        all_with_images.append(key)
        
        # Check if it also has metabolite data
        if 'metabolites' in value and value['metabolites']:
            all_with_images_and_metabolites.append(key)
        else:
            all_with_images_no_metabolites.append(key)

print(f"\n{'='*60}")
print("ALL ORGANOIDS WITH PROCESSED IMAGES:")
print(f"{'='*60}")
print(f"Total with processed images: {len(all_with_images)}")
print(f"  - With metabolite data: {len(all_with_images_and_metabolites)}")
print(f"  - WITHOUT metabolite data: {len(all_with_images_no_metabolites)}")
print(f"Percentage with metabolites: {len(all_with_images_and_metabolites)/len(all_with_images)*100:.1f}%")
print(f"{'='*60}")

# Now count only Dy30 with survey labels (training subset)
dy30_with_survey = []
dy30_with_survey_and_metabolites = []

def compute_majority_label(evaluations, min_votes=4):
    if not evaluations or len(evaluations) != 5:
        return None
    votes = {}
    for eval_data in evaluations:
        evaluation = eval_data.get('evaluation', '')
        if evaluation:
            votes[evaluation] = votes.get(evaluation, 0) + 1
    acceptable = votes.get('Acceptable', 0)
    not_acceptable = votes.get('Not Acceptable', 0)
    if acceptable >= min_votes:
        return 'Acceptable'
    elif not_acceptable >= min_votes:
        return 'Not Acceptable'
    else:
        return None

for key, value in all_data.items():
    if value.get('dayID') != 'Dy30':
        continue
    if 'survey' not in value:
        continue
    if 'processed' not in value:
        continue
    evaluations = value['survey'].get('evaluations', [])
    if not evaluations:
        continue
    label = compute_majority_label(evaluations, min_votes=4)
    if label is None:
        continue
    
    dy30_with_survey.append(key)
    if 'metabolites' in value and value['metabolites']:
        dy30_with_survey_and_metabolites.append(key)

print(f"\nDy30 WITH SURVEY LABELS (training subset):")
print(f"{'='*60}")
print(f"Total Dy30 with clear survey labels: {len(dy30_with_survey)}")
print(f"  - With metabolite data: {len(dy30_with_survey_and_metabolites)}")
print(f"Percentage with metabolites: {len(dy30_with_survey_and_metabolites)/len(dy30_with_survey)*100:.1f}%")
print(f"{'='*60}")

# Show breakdown by day
print(f"\nBREAKDOWN BY DAY:")
print(f"{'='*60}")
day_counts = {}
for key, value in all_data.items():
    day = value.get('dayID', 'Unknown')
    if 'processed' in value and value['processed']:
        if day not in day_counts:
            day_counts[day] = {'total': 0, 'with_metabolites': 0}
        day_counts[day]['total'] += 1
        if 'metabolites' in value and value['metabolites']:
            day_counts[day]['with_metabolites'] += 1

for day in sorted(day_counts.keys()):
    counts = day_counts[day]
    pct = counts['with_metabolites']/counts['total']*100 if counts['total'] > 0 else 0
    print(f"{day}: {counts['total']} images, {counts['with_metabolites']} with metabolites ({pct:.1f}%)")
print(f"{'='*60}")

