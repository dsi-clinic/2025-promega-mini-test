#!/usr/bin/env python3
"""
Threshold tuning for existing EfficientNet models.
Find optimal threshold on validation set to maximize balanced accuracy.
NO RETRAINING NEEDED - uses existing trained models.
"""

import json
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score
import sys
sys.path.append('analysis/images/classifier')

# Import model class from training script
from train_model_accuracy_tony_dinov2 import ImageOnlyClassifier

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TARGET_SIZE = (384, 512)

class SimpleDataset(Dataset):
    def __init__(self, img_paths, labels):
        self.img_paths = img_paths
        self.labels = labels
        self.transform = T.Compose([
            T.Resize(TARGET_SIZE),
            T.ToTensor(),
        ])
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert("RGB")
        img = self.transform(img)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return img, label

def find_optimal_threshold(model, loader, criterion='balanced_accuracy'):
    """Find threshold that maximizes criterion on validation set."""
    model.eval()
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for img, label in loader:
            img = img.to(DEVICE)
            logit = model(img)
            prob = torch.sigmoid(logit).cpu().numpy()
            all_probs.extend(prob)
            all_labels.extend(label.numpy())
    
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    best_threshold = 0.5
    best_score = -1
    results = []
    
    for threshold in np.arange(0.05, 0.95, 0.05):
        preds = (all_probs > threshold).astype(int)
        
        tn = ((preds == 0) & (all_labels == 0)).sum()
        fp = ((preds == 1) & (all_labels == 0)).sum()
        fn = ((preds == 0) & (all_labels == 1)).sum()
        tp = ((preds == 1) & (all_labels == 1)).sum()
        
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        acc = (tp + tn) / len(all_labels) if len(all_labels) > 0 else 0.0
        
        if criterion == 'balanced_accuracy':
            score = (tpr + tnr) / 2.0
        elif criterion == 'tnr':
            score = tnr
        elif criterion == 'f1_tnr_harmonic':
            f1 = 2*tp / (2*tp + fp + fn) if (2*tp + fp + fn) > 0 else 0.0
            score = 2 * (f1 * tnr) / (f1 + tnr) if (f1 + tnr) > 0 else 0.0
        
        results.append({
            'threshold': threshold,
            'score': score,
            'acc': acc,
            'tpr': tpr,
            'tnr': tnr,
            'tp': int(tp),
            'fp': int(fp),
            'tn': int(tn),
            'fn': int(fn)
        })
        
        if score > best_score:
            best_score = score
            best_threshold = threshold
    
    return best_threshold, best_score, results

def evaluate_with_threshold(model, loader, threshold=0.5):
    """Evaluate model on test set with given threshold."""
    model.eval()
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for img, label in loader:
            img = img.to(DEVICE)
            logit = model(img)
            prob = torch.sigmoid(logit).cpu().numpy()
            all_probs.extend(prob)
            all_labels.extend(label.numpy())
    
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    preds = (all_probs > threshold).astype(int)
    
    # Calculate metrics
    tn = ((preds == 0) & (all_labels == 0)).sum()
    fp = ((preds == 1) & (all_labels == 0)).sum()
    fn = ((preds == 0) & (all_labels == 1)).sum()
    tp = ((preds == 1) & (all_labels == 1)).sum()
    
    acc = accuracy_score(all_labels, preds)
    f1 = f1_score(all_labels, preds)
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    balanced_acc = (tpr + tnr) / 2.0
    
    try:
        roc_auc = roc_auc_score(all_labels, all_probs)
    except:
        roc_auc = None
    
    try:
        pr_auc = average_precision_score(all_labels, all_probs)
    except:
        pr_auc = None
    
    return {
        'accuracy': float(acc),
        'f1': float(f1),
        'tpr': float(tpr),
        'tnr': float(tnr),
        'balanced_accuracy': float(balanced_acc),
        'roc_auc': float(roc_auc) if roc_auc is not None else None,
        'pr_auc': float(pr_auc) if pr_auc is not None else None,
        'tp': int(tp),
        'fp': int(fp),
        'tn': int(tn),
        'fn': int(fn),
        'threshold': float(threshold)
    }

def load_day_data(split_file, day_str):
    """Load data for a specific day from split file."""
    with open(split_file) as f:
        data = json.load(f)
    
    img_paths = []
    labels = []
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
        
        img_paths.append(img_path)
        labels.append(label_map[label_str])
    
    return np.array(img_paths), np.array(labels)

def main():
    # Configurations
    model_base_dir = Path("analysis/images/classifier/outputs_512x384_tony_dinov2_fixed_splits_NO_STITCH")
    val_split = Path("data_splits/both_val_base_no_stitch.json")
    test_split = Path("data_splits/both_test_base_no_stitch.json")
    
    output_dir = Path("analysis/images/classifier/threshold_tuned_efficientnet_NO_STITCH")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    backbone_key = "efficientnet"
    backbone_name = "efficientnet_b0"
    
    print("="*80)
    print("THRESHOLD TUNING FOR EFFICIENTNET (NO RETRAINING)")
    print("="*80)
    print(f"\nModel: {backbone_key}")
    print(f"Base model directory: {model_base_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Device: {DEVICE}\n")
    
    # Get all days
    all_days = []
    for day_dir in (model_base_dir / backbone_key).iterdir():
        if day_dir.is_dir():
            all_days.append(day_dir.name)
    
    def day_sort_key(day_str):
        day_num_str = day_str.replace("Dy", "").replace("_", ".")
        return float(day_num_str)
    
    all_days = sorted(all_days, key=day_sort_key)
    
    print(f"Found {len(all_days)} days: {', '.join(all_days)}\n")
    
    summary_results = []
    
    for day_str in all_days:
        print(f"\n{'='*80}")
        print(f"Processing {day_str}")
        print('='*80)
        
        # Load model
        model_path = model_base_dir / backbone_key / day_str / "model.pth"
        if not model_path.exists():
            print(f"[WARNING] Model not found: {model_path}")
            continue
        
        model = ImageOnlyClassifier(backbone_key, backbone_name, TARGET_SIZE, use_mask=False).to(DEVICE)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval()
        print(f"[OK] Loaded model from {model_path}")
        
        # Load validation data
        val_imgs, val_labels = load_day_data(val_split, day_str)
        if len(val_imgs) == 0:
            print(f"[WARNING] No validation data for {day_str}")
            continue
        
        val_loader = DataLoader(
            SimpleDataset(val_imgs, val_labels),
            batch_size=16,
            shuffle=False,
            num_workers=0
        )
        print(f"[OK] Loaded {len(val_imgs)} validation samples")
        
        # Find optimal threshold
        print("\nFinding optimal threshold on validation set...")
        optimal_thresh, best_bal_acc, threshold_results = find_optimal_threshold(
            model, val_loader, criterion='balanced_accuracy'
        )
        
        print(f"\n[OK] Optimal threshold: {optimal_thresh:.3f}")
        print(f"  Best balanced accuracy: {best_bal_acc:.4f}")
        
        # Show baseline (threshold=0.5) for comparison
        baseline_result = [r for r in threshold_results if abs(r['threshold'] - 0.5) < 0.01][0]
        print(f"\n  Baseline (threshold=0.5):")
        print(f"    Balanced Acc: {baseline_result['score']:.4f}, TNR: {baseline_result['tnr']:.4f}, TPR: {baseline_result['tpr']:.4f}")
        print(f"  Optimized (threshold={optimal_thresh:.2f}):")
        optimal_result = [r for r in threshold_results if abs(r['threshold'] - optimal_thresh) < 0.01][0]
        print(f"    Balanced Acc: {optimal_result['score']:.4f}, TNR: {optimal_result['tnr']:.4f}, TPR: {optimal_result['tpr']:.4f}")
        print(f"  Improvement: TNR +{(optimal_result['tnr'] - baseline_result['tnr'])*100:.1f}%")
        
        # Load test data
        test_imgs, test_labels = load_day_data(test_split, day_str)
        if len(test_imgs) == 0:
            print(f"[WARNING] No test data for {day_str}")
            continue
        
        test_loader = DataLoader(
            SimpleDataset(test_imgs, test_labels),
            batch_size=16,
            shuffle=False,
            num_workers=0
        )
        print(f"\n[OK] Loaded {len(test_imgs)} test samples")
        
        # Evaluate on test set with baseline and optimal thresholds
        print("\nEvaluating on test set...")
        baseline_test = evaluate_with_threshold(model, test_loader, threshold=0.5)
        optimal_test = evaluate_with_threshold(model, test_loader, threshold=optimal_thresh)
        
        print(f"\n  Test Results (threshold=0.5):")
        print(f"    Accuracy: {baseline_test['accuracy']:.4f}, F1: {baseline_test['f1']:.4f}")
        print(f"    TNR: {baseline_test['tnr']:.4f}, TPR: {baseline_test['tpr']:.4f}")
        print(f"    Balanced Acc: {baseline_test['balanced_accuracy']:.4f}")
        
        print(f"\n  Test Results (threshold={optimal_thresh:.2f}):")
        print(f"    Accuracy: {optimal_test['accuracy']:.4f}, F1: {optimal_test['f1']:.4f}")
        print(f"    TNR: {optimal_test['tnr']:.4f}, TPR: {optimal_test['tpr']:.4f}")
        print(f"    Balanced Acc: {optimal_test['balanced_accuracy']:.4f}")
        
        print(f"\n  Improvements on Test Set:")
        print(f"    TNR: {baseline_test['tnr']:.4f} → {optimal_test['tnr']:.4f} (+{(optimal_test['tnr']-baseline_test['tnr'])*100:.1f}%)")
        print(f"    Balanced Acc: {baseline_test['balanced_accuracy']:.4f} → {optimal_test['balanced_accuracy']:.4f} (+{(optimal_test['balanced_accuracy']-baseline_test['balanced_accuracy'])*100:.1f}%)")
        
        # Save results
        day_output_dir = output_dir / day_str
        day_output_dir.mkdir(parents=True, exist_ok=True)
        
        results = {
            'day': day_str,
            'backbone': backbone_key,
            'optimal_threshold': optimal_thresh,
            'baseline_val': baseline_result,
            'optimal_val': optimal_result,
            'baseline_test': baseline_test,
            'optimal_test': optimal_test,
            'threshold_search_results': threshold_results
        }
        
        with open(day_output_dir / "threshold_tuning_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n[OK] Saved results to {day_output_dir / 'threshold_tuning_results.json'}")
        
        summary_results.append({
            'day': day_str,
            'optimal_threshold': optimal_thresh,
            'baseline_tnr': baseline_test['tnr'],
            'optimal_tnr': optimal_test['tnr'],
            'tnr_improvement': optimal_test['tnr'] - baseline_test['tnr'],
            'baseline_balanced_acc': baseline_test['balanced_accuracy'],
            'optimal_balanced_acc': optimal_test['balanced_accuracy']
        })
    
    # Print overall summary
    print("\n" + "="*80)
    print("OVERALL SUMMARY")
    print("="*80)
    
    if summary_results:
        avg_baseline_tnr = np.mean([r['baseline_tnr'] for r in summary_results])
        avg_optimal_tnr = np.mean([r['optimal_tnr'] for r in summary_results])
        avg_improvement = np.mean([r['tnr_improvement'] for r in summary_results])
        
        print(f"\nAverage TNR (Test Set):")
        print(f"  Baseline (threshold=0.5): {avg_baseline_tnr:.4f}")
        print(f"  Optimized thresholds:     {avg_optimal_tnr:.4f}")
        print(f"  Average improvement:      +{avg_improvement:.4f} (+{avg_improvement*100:.1f}%)")
        
        print(f"\nPer-Day Summary:")
        print(f"{'Day':<8} {'Opt Thresh':>12} {'Baseline TNR':>14} {'Optimal TNR':>13} {'Improvement':>13}")
        print("-"*80)
        for r in summary_results:
            print(f"{r['day']:<8} {r['optimal_threshold']:>12.3f} {r['baseline_tnr']:>14.4f} "
                  f"{r['optimal_tnr']:>13.4f} {r['tnr_improvement']:>+13.4f}")
        
        # Save summary
        with open(output_dir / "summary_all_days.json", 'w') as f:
            json.dump(summary_results, f, indent=2)
        
        print(f"\n[OK] Saved overall summary to {output_dir / 'summary_all_days.json'}")
    
    print("\n" + "="*80)
    print("[OK] THRESHOLD TUNING COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {output_dir}")
    print("\nNext steps:")
    print("1. Review threshold_tuning_results.json for each day")
    print("2. Use optimal thresholds for inference")
    print("3. Proceed with improved training script for even better results")

if __name__ == "__main__":
    main()

