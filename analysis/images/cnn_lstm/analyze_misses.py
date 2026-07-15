"""Inspect test-set misclassifications from a trained CNN-LSTM checkpoint.

Run from project root:
    make run ARGS="analysis/images/cnn_lstm/analyze_misses.py"
"""
import sys
from pathlib import Path

# Add project root to path so imports work
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import json

import torch

from analysis.images.cnn_lstm.organoid_dataset import (
    OrganoidTimeSeriesDataset,
    make_idor_series_splits,
)
from analysis.images.cnn_lstm.organoid_model import OrganoidCNN_LSTM

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
output_dir = Path('outputs/cnn_lstm')

ds, _train_ids, _val_ids, test_ids = make_idor_series_splits()

model = OrganoidCNN_LSTM(hidden_size=256, num_layers=2).to(device)
checkpoint = torch.load(output_dir / 'best_model_clipblur.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print("\n" + "=" * 80)
print("MISCLASSIFIED ORGANOIDS (label 1 = Not Acceptable per AGENTS.md rule #9)")
print("=" * 80)

# False positives = predicted Not Acceptable (1) but actually Acceptable (0).
# False negatives = predicted Acceptable (0) but actually Not Acceptable (1).
misclassified = {'false_positives': [], 'false_negatives': []}

with torch.no_grad():
    for org_id in test_ids:
        dataset = OrganoidTimeSeriesDataset([org_id], ds)
        seq, days_norm, label, _weight, _oid = dataset[0]
        seqs = seq.unsqueeze(0).to(device)
        days = days_norm.unsqueeze(0).to(device).float()

        logit = model(seqs, days)
        prob_pos = torch.sigmoid(logit).item()
        pred = 1 if prob_pos > 0.5 else 0
        true_label = int(label.item())

        if pred == true_label:
            continue

        info = {
            'organoid_id': org_id,
            'true_label': 'Not Acceptable' if true_label == 1 else 'Acceptable',
            'predicted_label': 'Not Acceptable' if pred == 1 else 'Acceptable',
            'prob_not_acceptable': prob_pos,
            'prob_acceptable': 1.0 - prob_pos,
        }
        if true_label == 0 and pred == 1:
            misclassified['false_positives'].append(info)
        else:
            misclassified['false_negatives'].append(info)

print(
    f"\nFALSE POSITIVES "
    f"(Acceptable predicted Not Acceptable): {len(misclassified['false_positives'])}"
)
print("-" * 80)
for item in misclassified['false_positives']:
    print(f"Organoid {item['organoid_id']}: P(NotAcc)={item['prob_not_acceptable']:.1%}")

print(
    f"\nFALSE NEGATIVES "
    f"(Not Acceptable predicted Acceptable): {len(misclassified['false_negatives'])}"
)
print("-" * 80)
for item in misclassified['false_negatives']:
    print(f"Organoid {item['organoid_id']}: P(NotAcc)={item['prob_not_acceptable']:.1%}")

out_path = output_dir / 'misclassified_analysis_clipblur.json'
with open(out_path, 'w') as f:
    json.dump(misclassified, f, indent=2)

print(f"\nSaved to {out_path}")
