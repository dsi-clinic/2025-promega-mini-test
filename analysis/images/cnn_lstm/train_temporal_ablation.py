"""
Temporal ablation study - train models with different day ranges
RUN FROM PROJECT ROOT: python analysis/images/cnn_lstm/train_temporal_ablation.py
"""
import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from config import OUTPUT_FOLDER
from analysis.images.cnn_lstm.organoid_dataset import (
    OrganoidTimeSeriesDataset, 
    load_data_and_create_splits
)
from analysis.images.cnn_lstm.organoid_model import OrganoidCNN_LSTM

# Import training functions from existing script
from analysis.images.cnn_lstm.train_organoid_lstm import train_one_epoch, evaluate

# Define augmentation for training
train_transform = transforms.Compose([
    transforms.RandomRotation(degrees=15),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
])

# Define day ranges to test
DAY_RANGES = [
    8,      # Early (3 timepoints)
    10,     # Early-mid (4 timepoints)
    13,     # Mid (5 timepoints)
    15,     # Mid-late WITH THE DIP (6 timepoints)
    17,     # Post-dip (7 timepoints)
    20.5,   # Late (8 timepoints)
    24,     # Later (9 timepoints)
    30,     # Full baseline (11 timepoints)
]

def train_for_day_range(max_day, train_ids, val_ids, test_ids, 
                        series_metadata, data, global_mean, device,
                        output_dir):
    """Train a model using only days up to max_day with simple end-to-end training"""
    
    print(f"\n{'='*70}")
    print(f"TRAINING WITH DAYS 3-{max_day}")
    print(f"{'='*70}")
    
    # Create datasets with max_day filter 
    train_dataset = OrganoidTimeSeriesDataset(
        train_ids, series_metadata, data,
        global_mean=global_mean,
        max_day=max_day,
        transform=train_transform
    )

    val_dataset = OrganoidTimeSeriesDataset(
        val_ids, series_metadata, data,
        global_mean=global_mean,
        max_day=max_day,
        transform=None
    )

    test_dataset = OrganoidTimeSeriesDataset(
        test_ids, series_metadata, data,
        global_mean=global_mean,
        max_day=max_day,
        transform=None
    )
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)
    
    # Calculate class weights for imbalanced data
    train_labels = []
    for org_id in train_ids:
        entry_keys = series_metadata[org_id]['entry_keys']
        final_entry = data[entry_keys[-1]]
        survey = final_entry.get('survey', {})
        if 'evaluations' in survey:
            votes = [ev.get('evaluation') for ev in survey['evaluations']]
            if votes.count('Acceptable') > votes.count('Not Acceptable'):
                train_labels.append(1)
            else:
                train_labels.append(0)
    
    n_good = sum(train_labels)
    n_bad = len(train_labels) - n_good
    weight_for_0 = len(train_labels) / (2 * n_bad)
    weight_for_1 = len(train_labels) / (2 * n_good)
    class_weights = torch.FloatTensor([weight_for_0, weight_for_1]).to(device)
    
    print(f"Class weights: Bad={weight_for_0:.3f}, Good={weight_for_1:.3f}")
    
    # Create model
    model = OrganoidCNN_LSTM(
        num_classes=2,
        lstm_hidden=256,
        lstm_layers=2
    ).to(device)
    
    # ========== SIMPLE END-TO-END TRAINING ==========
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=7e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5
    )
    
    best_val_acc = 0
    best_model_state = None
    patience_counter = 0
    
    for epoch in range(100):
        # Training with gradient clipping
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for sequences, labels in tqdm(train_loader, desc="Training"):
            sequences, labels = sequences.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Clip gradients!
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        
        train_acc = train_correct / train_total
        
        # Validation
        val_loss, val_acc, val_prec, val_rec, val_f1, _, _ = evaluate(
            model, val_loader, criterion, device
        )
        
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1:2d}: Train {train_acc:.3f} | Val {val_acc:.3f} | F1 {val_f1:.3f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            print(f"  *** New best!")
        else:
            patience_counter += 1
            if patience_counter >= 12:
                print(f"  Early stopping at epoch {epoch+1}")
                break
    # ================================================
    
    print(f"✅ Best val acc: {best_val_acc:.3f}")
    
    # Load best model and evaluate on test
    model.load_state_dict(best_model_state)
    test_loss, test_acc, test_prec, test_rec, test_f1, _, _ = evaluate(
        model, test_loader, criterion, device
    )
    
    print(f"\n📊 FINAL TEST RESULTS:")
    print(f"   Accuracy:  {test_acc:.3f}")
    print(f"   Precision: {test_prec:.3f}")
    print(f"   Recall:    {test_rec:.3f}")
    print(f"   F1 Score:  {test_f1:.3f}")
    
    # Save model
    model_path = output_dir / f'model_days_3-{max_day}.pth'
    torch.save({
        'model_state_dict': best_model_state,
        'max_day': max_day,
        'best_val_acc': best_val_acc,
        'test_acc': test_acc,
        'test_precision': test_prec,
        'test_recall': test_rec,
        'test_f1': test_f1,
    }, model_path)
    print(f"✅ Saved model to {model_path}")
    
    # Clear GPU memory before next run
    del model
    del train_loader
    del val_loader
    del test_loader
    del train_dataset
    del val_dataset
    del test_dataset
    torch.cuda.empty_cache()

    return {
        'max_day': max_day,
        'best_val_acc': best_val_acc,
        'test_acc': test_acc,
        'test_precision': test_prec,
        'test_recall': test_rec,
        'test_f1': test_f1,
        'model_path': str(model_path)
    }


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = OUTPUT_FOLDER / 'cnn_lstm' / 'temporal_ablation_efnet_simple'
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving outputs to: {output_dir}")
    
    # Load data
    print("\n" + "="*70)
    print("LOADING DATA")
    print("="*70)
    
    series_metadata_path = OUTPUT_FOLDER / 'complete_series_metadata_no_blanks.json'
    data_path = OUTPUT_FOLDER / 'complete_series_data_no_blanks.json'
    
    train_ids, val_ids, test_ids, series_metadata, data = load_data_and_create_splits(
        series_metadata_path, data_path, random_seed=42
    )
    
    # Load global mean (already computed during main training)
    global_mean_path = OUTPUT_FOLDER / 'cnn_lstm' / 'global_mean.npy'
    if not global_mean_path.exists():
        raise FileNotFoundError(
            f"Global mean not found at {global_mean_path}. "
            "Please run train_organoid_lstm.py first!"
        )
    
    global_mean = np.load(global_mean_path)
    print(f"✅ Loaded global mean: {global_mean}")
    
    # Run ablation study for each day range
    print("\n" + "="*70)
    print("STARTING TEMPORAL ABLATION STUDY (SIMPLE TRAINING)")
    print("="*70)
    
    results = []
    
    for max_day in DAY_RANGES:
        result = train_for_day_range(
            max_day, train_ids, val_ids, test_ids,
            series_metadata, data, global_mean, device,
            output_dir
        )
        results.append(result)
    
    # Save summary results
    results_path = output_dir / 'temporal_ablation_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary table
    print("\n" + "="*70)
    print("TEMPORAL ABLATION SUMMARY")
    print("="*70)
    print(f"{'Day Range':<15} {'Val Acc':<12} {'Test Acc':<12} {'Test F1':<12}")
    print("-"*70)
    
    for r in results:
        print(f"Days 3-{r['max_day']:<7} {r['best_val_acc']:<12.3f} {r['test_acc']:<12.3f} {r['test_f1']:<12.3f}")
    
    print("\n" + "="*70)
    print("KEY FINDINGS:")
    
    # Find best performing range
    best = max(results, key=lambda x: x['test_acc'])
    print(f"  • Best performance: Days 3-{best['max_day']} ({best['test_acc']:.1%} test accuracy)")
    
    # Check if adding more days helps
    sorted_results = sorted(results, key=lambda x: x['max_day'])
    improvements = []
    for i in range(1, len(sorted_results)):
        diff = sorted_results[i]['test_acc'] - sorted_results[i-1]['test_acc']
        improvements.append((sorted_results[i]['max_day'], diff))
    
    print(f"  • Accuracy gains by adding days:")
    for day, gain in improvements:
        direction = "↑" if gain > 0 else "↓"
        print(f"    - Adding up to Day {day}: {direction} {abs(gain):.1%}")
    
    print(f"\nResults saved to {results_path}")
    print("="*70)


if __name__ == '__main__':
    main()