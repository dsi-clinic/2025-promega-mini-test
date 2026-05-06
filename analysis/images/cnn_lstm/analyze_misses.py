import torch
import json
from pathlib import Path
from analysis.images.cnn_lstm.organoid_dataset import OrganoidTimeSeriesDataset, make_idor_series_splits
from analysis.images.cnn_lstm.organoid_model import OrganoidCNN_LSTM

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
output_dir = Path('outputs/cnn_lstm')

# Load data splits — runtime-computed from data/all_data.json (no JSON files)
ds, _train_ids, _val_ids, test_ids = make_idor_series_splits()

# Load model
model = OrganoidCNN_LSTM(num_classes=2, lstm_hidden=256, lstm_layers=2).to(device)
checkpoint = torch.load(output_dir / 'best_model_clipblur.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Analyze test set
print("\n" + "="*80)
print("MISCLASSIFIED ORGANOIDS")
print("="*80)

misclassified = {
    'false_positives': [],  # Bad organoids called Good
    'false_negatives': []   # Good organoids called Bad
}

with torch.no_grad():
    for org_id in test_ids:
        dataset = OrganoidTimeSeriesDataset([org_id], ds)
        seq, _days, label, _weight, _oid = dataset[0]
        images = seq.unsqueeze(0).to(device)

        output = model(images)
        probs = torch.softmax(output, dim=1)[0]
        pred = torch.argmax(output, dim=1).item()

        true_label = int(label.item())
        
        if pred != true_label:
            info = {
                'organoid_id': org_id,
                'true_label': 'Good' if true_label == 1 else 'Bad',
                'predicted_label': 'Good' if pred == 1 else 'Bad',
                'confidence': probs[pred].item(),
                'prob_bad': probs[0].item(),
                'prob_good': probs[1].item()
            }
            
            if true_label == 0 and pred == 1:
                misclassified['false_positives'].append(info)  # Bad called Good
            elif true_label == 1 and pred == 0:
                misclassified['false_negatives'].append(info)  # Good called Bad

# Print results
print(f"\nFALSE POSITIVES (Bad organoids called Good): {len(misclassified['false_positives'])}")
print("-" * 80)
for item in misclassified['false_positives']:
    print(f"Organoid {item['organoid_id']}: Confidence {item['confidence']:.1%}")
    print(f"  Prob(Bad)={item['prob_bad']:.1%}, Prob(Good)={item['prob_good']:.1%}\n")

print(f"\nFALSE NEGATIVES (Good organoids called Bad): {len(misclassified['false_negatives'])}")
print("-" * 80)
for item in misclassified['false_negatives']:
    print(f"Organoid {item['organoid_id']}: Confidence {item['confidence']:.1%}")
    print(f"  Prob(Bad)={item['prob_bad']:.1%}, Prob(Good)={item['prob_good']:.1%}\n")

# Save
with open(output_dir / 'misclassified_analysis_clipblur.json', 'w') as f:
    json.dump(misclassified, f, indent=2)

print(f"\nSaved to {output_dir / 'misclassified_analysis_clipblur.json'}")