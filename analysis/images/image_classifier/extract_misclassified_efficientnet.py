#!/usr/bin/env python3
"""
Extract misclassified samples from EfficientNet training results.
Creates CSV files with image paths and metadata for misclassified samples per day.
"""

import json
import csv
from pathlib import Path
import argparse
import torch
import numpy as np
from PIL import Image
from torchvision import transforms as T
from torch.utils.data import Dataset, DataLoader
import sys
from pathlib import Path

# Import ImageOnlyClassifier from the training script
# NOTE: This assumes the script is run from the repository root
# where 'analysis/images/image_classifier' exists
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

from train_efficientnet_improved_tnr import ImageOnlyClassifier

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TARGET_SIZE = (384, 512)
BACKBONE_NAME = "efficientnet_b0"
BACKBONE_KEY = "efficientnet"

class SimpleDataset(Dataset):
    """Simple dataset for loading images and labels for misclassification analysis."""
    
    def __init__(self, img_paths, labels):
        """Initialize simple dataset.
        
        Args:
            img_paths: List of image file paths.
            labels: List of binary labels (0 or 1).
        """
        self.img_paths = img_paths
        self.labels = labels
        self.transform = T.Compose([
            T.Resize(TARGET_SIZE),
            T.ToTensor(),
        ])
    
    def __len__(self):
        """Return dataset size."""
        return len(self.labels)
    
    def __getitem__(self, idx):
        """Get item by index.
        
        Args:
            idx: Index of item to retrieve.
        
        Returns:
            Tuple of (image tensor, label tensor, image path).
        """
        img = Image.open(self.img_paths[idx]).convert("RGB")
        img = self.transform(img)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return img, label, self.img_paths[idx]

def load_day_data(split_file, day_str):
    """Load data for a specific day with full metadata.
    
    Args:
        split_file: Path to JSON split file containing organoid data.
        day_str: Day string to extract (e.g., 'Dy03', 'Dy28').
    
    Returns:
        List of dictionaries with sample metadata (organoid_id, img_path, label, etc.).
    """
    with open(split_file) as f:
        data = json.load(f)
    
    samples = []
    label_map = {"Acceptable": 1, "Not Acceptable": 0}
    
    for org_id, org_data in data.items():
        label_str = org_data.get('label')
        if label_str not in label_map:
            continue
        
        timepoints = org_data.get('timepoints', {})
        if day_str not in timepoints:
            continue
        
        tp_data = timepoints[day_str]
        img_path = tp_data.get('img_path')
        if not img_path or not Path(img_path).exists():
            continue
        
        samples.append({
            'organoid_id': org_id,
            'img_path': img_path,
            'label': label_map[label_str],
            'label_str': label_str,
            'batch': org_data.get('batch', 'Unknown')
        })
    
    return samples

def extract_misclassified(model, samples, day_str, output_dir):
    """Extract misclassified samples for a given day.
    
    Runs model inference on samples and saves misclassified examples to CSV.
    
    Args:
        model: Trained classifier model.
        samples: List of sample dictionaries with img_path, label, etc.
        day_str: Day string (e.g., 'Dy03', 'Dy28').
        output_dir: Directory to save misclassified samples CSV files.
    """
    if len(samples) == 0:
        return
    
    img_paths = [s['img_path'] for s in samples]
    labels = np.array([s['label'] for s in samples])
    
    dataset = SimpleDataset(img_paths, labels)
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)
    
    model.eval()
    predictions = []
    probabilities = []
    
    with torch.no_grad():
        for img, label, _ in loader:
            img = img.to(DEVICE)
            logit = model(img)
            prob = torch.sigmoid(logit).cpu().numpy()
            pred = (prob > 0.5).astype(int)
            predictions.extend(pred)
            probabilities.extend(prob)
    
    predictions = np.array(predictions)
    probabilities = np.array(probabilities)
    
    # Find misclassified samples
    misclassified = []
    for i, sample in enumerate(samples):
        pred = predictions[i]
        prob = probabilities[i]
        label = sample['label']
        
        if pred != label:
            misclassified.append({
                'organoid_id': sample['organoid_id'],
                'img_path': sample['img_path'],
                'true_label': sample['label_str'],
                'predicted_label': 'Acceptable' if pred == 1 else 'Not Acceptable',
                'probability': float(prob),
                'batch': sample['batch'],
                'error_type': 'False Positive' if (pred == 1 and label == 0) else 'False Negative'
            })
    
    if len(misclassified) == 0:
        print(f"  {day_str}: No misclassified samples (perfect accuracy!)")
        return
    
    # Save to CSV
    day_output_dir = output_dir / day_str
    day_output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = day_output_dir / "misclassified_samples.csv"
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['organoid_id', 'img_path', 'true_label', 
                                                'predicted_label', 'probability', 'batch', 'error_type'])
        writer.writeheader()
        writer.writerows(misclassified)
    
    print(f"  {day_str}: {len(misclassified)}/{len(samples)} misclassified → {output_file}")
    
    # Summary by error type
    fp_count = sum(1 for m in misclassified if m['error_type'] == 'False Positive')
    fn_count = sum(1 for m in misclassified if m['error_type'] == 'False Negative')
    print(f"    False Positives: {fp_count}, False Negatives: {fn_count}")

def main():
    """Entry point for extracting misclassified samples.
    
    Loads trained models, runs inference on test data, and saves
    misclassified samples to CSV files organized by day.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True, help="Directory with training results")
    parser.add_argument("--test-split", required=True, help="Test split JSON file")
    parser.add_argument("--output-dir", required=True, help="Output directory for misclassified samples")
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    test_split = Path(args.test_split)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("EXTRACTING MISCLASSIFIED SAMPLES")
    print("="*80)
    print(f"Results directory: {results_dir}")
    print(f"Test split: {test_split}")
    print(f"Output directory: {output_dir}")
    print(f"Device: {DEVICE}\n")
    
    efficientnet_dir = results_dir / BACKBONE_KEY
    if not efficientnet_dir.exists():
        print(f"ERROR: EfficientNet results not found: {efficientnet_dir}")
        return
    
    # Process each day
    for day_dir in sorted(efficientnet_dir.iterdir()):
        if not day_dir.is_dir():
            continue
        
        day_str = day_dir.name
        model_path = day_dir / "model.pth"
        
        if not model_path.exists():
            print(f"  {day_str}: Model not found, skipping")
            continue
        
        # Load model
        model = ImageOnlyClassifier(BACKBONE_NAME, TARGET_SIZE).to(DEVICE)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval()
        
        # Load test data
        samples = load_day_data(test_split, day_str)
        if len(samples) == 0:
            print(f"  {day_str}: No test samples, skipping")
            continue
        
        # Extract misclassified
        extract_misclassified(model, samples, day_str, output_dir)
    
    print(f"\n{'='*80}")
    print("[OK] MISCLASSIFIED EXTRACTION COMPLETE")
    print(f"{'='*80}")
    print(f"Results saved to: {output_dir}")

if __name__ == "__main__":
    main()









