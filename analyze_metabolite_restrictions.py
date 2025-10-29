#!/usr/bin/env python3
"""
Analyze metabolite data based on IDOR/Promega restrictions
"""
import json
from collections import Counter

ALL_DATA_JSON = 'all_data.json'

# Load all_data.json
print("Loading all_data.json...")
with open(ALL_DATA_JSON) as f:
    all_data = json.load(f)

print(f"Total records: {len(all_data)}")

# Check what metabolites are available
metabolite_types = set()
batch_ids = set()
day_ids = set()

for key, value in all_data.items():
    if 'metabolites' in value and value['metabolites']:
        for metabolite_name in value['metabolites'].keys():
            metabolite_types.add(metabolite_name)
    
    if 'BA' in value:
        batch_ids.add(value['BA'])
    
    if 'dayID' in value:
        day_ids.add(value['dayID'])

print(f"\n{'='*60}")
print("AVAILABLE METABOLITES:")
print(f"{'='*60}")
for met in sorted(metabolite_types):
    print(f"  - {met}")

print(f"\n{'='*60}")
print("AVAILABLE BATCHES:")
print(f"{'='*60}")
for batch in sorted(batch_ids):
    print(f"  - {batch}")

print(f"\n{'='*60}")
print("AVAILABLE DAYS:")
print(f"{'='*60}")
for day in sorted(day_ids):
    print(f"  - {day}")

# Show sample metabolite data structure
print(f"\n{'='*60}")
print("SAMPLE METABOLITE DATA STRUCTURE:")
print(f"{'='*60}")
for key, value in all_data.items():
    if 'metabolites' in value and value['metabolites']:
        print(f"Key: {key}")
        print(f"Day: {value.get('dayID')}")
        print(f"Batch: {value.get('BA')}")
        print(f"Metabolites:")
        for met_name, met_data in value['metabolites'].items():
            print(f"  {met_name}: {met_data}")
        break

# Count samples by batch
print(f"\n{'='*60}")
print("SAMPLE COUNTS BY BATCH (ALL DATA):")
print(f"{'='*60}")
batch_counts = Counter()
for key, value in all_data.items():
    if 'BA' in value:
        batch = value['BA'].split()[0] if ' ' in value['BA'] else value['BA']
        batch_counts[batch] += 1

for batch in sorted(batch_counts.keys()):
    restriction = "✓ USE THIS" if batch in ['BA1', 'BA2'] else "⚠️  AVOID (issues reported)"
    print(f"{batch}: {batch_counts[batch]:4d} samples - {restriction}")

# Count samples with metabolites by batch
print(f"\n{'='*60}")
print("SAMPLES WITH METABOLITES BY BATCH:")
print(f"{'='*60}")
batch_metabolite_counts = Counter()
for key, value in all_data.items():
    if 'BA' in value and 'metabolites' in value and value['metabolites']:
        batch = value['BA'].split()[0] if ' ' in value['BA'] else value['BA']
        batch_metabolite_counts[batch] += 1

for batch in sorted(batch_metabolite_counts.keys()):
    restriction = "✓ USE THIS" if batch in ['BA1', 'BA2'] else "⚠️  AVOID"
    print(f"{batch}: {batch_metabolite_counts[batch]:4d} samples with metabolites - {restriction}")

# Count by day with numeric extraction
print(f"\n{'='*60}")
print("METABOLITE AVAILABILITY BY DAY:")
print(f"{'='*60}")
day_met_counts = {}
for key, value in all_data.items():
    day = value.get('dayID')
    if day and 'metabolites' in value and value['metabolites']:
        if day not in day_met_counts:
            day_met_counts[day] = {
                'total': 0,
                'MalateGlo': 0,
                'BCAAGlo': 0
            }
        day_met_counts[day]['total'] += 1
        
        if 'MalateGlo' in value['metabolites']:
            day_met_counts[day]['MalateGlo'] += 1
        if 'BCAAGlo' in value['metabolites']:
            day_met_counts[day]['BCAAGlo'] += 1

# Sort by day number
def extract_day_num(day_str):
    import re
    match = re.search(r'\d+', day_str)
    return int(match.group()) if match else 0

for day in sorted(day_met_counts.keys(), key=extract_day_num):
    counts = day_met_counts[day]
    day_num = extract_day_num(day)
    restriction = ""
    if day_num <= 10:
        restriction = " ⚠️  RESTRICTED: Don't use MalateGlo/BCAAGlo"
    print(f"{day} (day {day_num:2d}): {counts['total']:3d} total, "
          f"{counts['MalateGlo']:3d} with MalateGlo, "
          f"{counts['BCAAGlo']:3d} with BCAAGlo{restriction}")

print(f"\n{'='*60}")
print("INTERPRETATION OF RESTRICTIONS:")
print(f"{'='*60}")
print("Based on IDOR/Promega call:")
print()
print("1. MalateGlo: DO NOT USE (unreliable data)")
print("   → Exclude from all metabolite features")
print()
print("2. BCAAGlo: DO NOT USE for days ≤ 10")
print("   → Days 03, 06, 08, 10: Exclude BCAAGlo")
print("   → Days 13+: Can use BCAAGlo (or maybe exclude entirely to be safe)")
print()
print("3. Batches BA3 and BA4: ISSUES REPORTED")
print("   → Only use BA1 and BA2 data")
print("   → Filter out any BA3/BA4 samples")
print()
print("4. RECOMMENDED METABOLITE FEATURES:")
print("   - GlucoseGlo ✓")
print("   - GlutamateGlo ✓")
print("   - LactateGlo ✓")
print("   - PyruvateGlo ✓")
print("   - BCAAGlo (only for days > 10, or exclude entirely)")
print("   - MalateGlo ✗ (DO NOT USE)")
print(f"{'='*60}")

# Calculate impact on dataset
print(f"\n{'='*60}")
print("IMPACT ON DATASET IF WE APPLY ALL RESTRICTIONS:")
print(f"{'='*60}")

# Count BA1/BA2 samples only
ba1_ba2_count = 0
ba1_ba2_with_met = 0
for key, value in all_data.items():
    if 'BA' in value:
        batch = value['BA'].split()[0] if ' ' in value['BA'] else value['BA']
        if batch in ['BA1', 'BA2']:
            ba1_ba2_count += 1
            if 'metabolites' in value and value['metabolites']:
                ba1_ba2_with_met += 1

print(f"Total BA1/BA2 samples: {ba1_ba2_count}")
print(f"BA1/BA2 with metabolites: {ba1_ba2_with_met}")
print(f"This is the usable dataset for metabolite analysis")
print(f"{'='*60}")

