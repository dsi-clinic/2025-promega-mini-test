"""
Training script for organoid CNN-LSTM model - SINGLE-PHASE VERSION
All parameters trainable from start (no freezing)
RUN FROM PROJECT ROOT: python analysis/images/cnn_lstm/train_organoid_lstm_single_phase.py
"""
import sys
from pathlib import Path

# Add project root to path so imports work
ROOT = Path(__file__).resolve().parents[3]  # Go up 3 levels to root
sys.path.insert(0, str(ROOT))

import argparse
import json
import random
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from config import OUTPUT_FOLDER

def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
from analysis.images.cnn_lstm.organoid_dataset import (
    OrganoidTimeSeriesDataset, 
    load_data_and_create_splits,
    compute_global_mean_from_ids
)
from analysis.images.cnn_lstm.organoid_model import OrganoidCNN_LSTM

def collate_variable_length(batch):
    """
    Custom collate function to handle variable-length sequences.
    Pads sequences to the maximum length in the batch.
    """
    # Unpack batch - each item is (seq, days_norm, label, weight, organoid_id)
    sequences, days_norms, labels, weights, organoid_ids = zip(*batch)
    
    # Find max sequence length
    max_len = max(seq.shape[0] for seq in sequences)
    C, H, W = sequences[0].shape[1:]
    
    # Pad sequences and days_norms
    padded_sequences = []
    padded_days = []
    for seq, days in zip(sequences, days_norms):
        seq_len = seq.shape[0]
        if seq_len < max_len:
            # Pad sequence with zeros (no device specified, will be moved later)
            padding = torch.zeros(max_len - seq_len, C, H, W, dtype=seq.dtype)
            seq = torch.cat([seq, padding], dim=0)
            # Pad days_norm with zeros
            days_padding = torch.zeros(max_len - seq_len, dtype=days.dtype)
            days = torch.cat([days, days_padding], dim=0)
        padded_sequences.append(seq)
        padded_days.append(days)
    
    # Stack into batches
    sequences_batch = torch.stack(padded_sequences, dim=0)  # (batch, T, C, H, W)
    days_batch = torch.stack(padded_days, dim=0)  # (batch, T)
    labels_batch = torch.stack(labels, dim=0)
    weights_batch = torch.stack(weights, dim=0)
    
    return sequences_batch, days_batch, labels_batch, weights_batch, list(organoid_ids)

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for batch in tqdm(dataloader, desc="Training"):
        # Handle dataset returning (seq, days_norm, label, weight, id) or (images, labels)
        if len(batch) == 5:
            images, days_norm, labels, weights, ids = batch
        else:
            images, labels = batch
        
        # Move to device
        images = images.to(device)
        labels = labels.to(device).long()  # CrossEntropyLoss requires Long dtype
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    
    return avg_loss, accuracy


def evaluate(model, dataloader, criterion, device):
    """Evaluate model"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Handle dataset returning (seq, days_norm, label, weight, id) or (images, labels)
            if len(batch) == 5:
                images, days_norm, labels, weights, ids = batch
            else:
                images, labels = batch
            
            images = images.to(device)
            labels = labels.to(device).long()  # CrossEntropyLoss requires Long dtype
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary', zero_division=0
    )
    
    return avg_loss, accuracy, precision, recall, f1, all_preds, all_labels


def main():
    parser = argparse.ArgumentParser(description='Train CNN-LSTM for organoid classification (Single-Phase)')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--lstm-hidden', type=int, default=256, help='LSTM hidden size')
    parser.add_argument('--lstm-layers', type=int, default=2, help='Number of LSTM layers')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory (default: OUTPUT_FOLDER/cnn_lstm)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    set_seed(args.seed)
    print(f"Random seed set to: {args.seed}")
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = OUTPUT_FOLDER / 'cnn_lstm'
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving outputs to: {output_dir}")
    
    # Load data
    print("\n" + "="*70)
    print("LOADING DATA")
    print("="*70)
    
    series_metadata_path = OUTPUT_FOLDER / 'complete_series_metadata_no_blanks.json'
    data_path = OUTPUT_FOLDER / 'complete_series_data_no_blanks.json'
    
    # After load_data_and_create_splits:
    # Using Amanda's split ratios: 0.66/0.17/0.17 (gives 48 test samples like Amanda's results)
    train_ids, val_ids, test_ids, series_metadata, data = load_data_and_create_splits(
        series_metadata_path, data_path,
        train_ratio=0.66,
        val_ratio=0.17,
        test_ratio=0.17
    )

    # Compute global mean from training set
    print("\nComputing global mean from training set...")
    global_mean = compute_global_mean_from_ids(train_ids, series_metadata, data)

    # Save
    np.save(output_dir / 'global_mean.npy', global_mean)
    print(f"Saved global mean to {output_dir / 'global_mean.npy'}")

    # Create datasets with the saved global_mean
    train_dataset = OrganoidTimeSeriesDataset(
        train_ids, series_metadata, data, 
        global_mean=global_mean
    )

    val_dataset = OrganoidTimeSeriesDataset(
        val_ids, series_metadata, data,
        global_mean=global_mean
    )

    test_dataset = OrganoidTimeSeriesDataset(
        test_ids, series_metadata, data,
        global_mean=global_mean
    )

    # Create dataloaders with custom collate function for variable-length sequences
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=collate_variable_length)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=collate_variable_length)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=collate_variable_length)
    
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
    
    print(f"\nClass weights: Bad={weight_for_0:.3f}, Good={weight_for_1:.3f}")
    
    # Create model
    print("\n" + "="*70)
    print("CREATING MODEL")
    print("="*70)
    
    model = OrganoidCNN_LSTM(
        num_classes=2,
        lstm_hidden=args.lstm_hidden,
        lstm_layers=args.lstm_layers
    ).to(device)
    
    # SINGLE-PHASE: Unfreeze all parameters from start (model starts with CNN frozen, so unfreeze it)
    for param in model.cnn.parameters():
        param.requires_grad = True
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print("✅ SINGLE-PHASE: All parameters trainable from start")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5
    )
    
    # ========== SINGLE-PHASE TRAINING ==========
    print("\n" + "="*70)
    print("SINGLE-PHASE TRAINING: All Parameters Trainable")
    print("="*70)
    
    best_val_acc = 0
    best_val_loss = float('inf')
    patience_counter = 0
    train_history = []
    
    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc, val_prec, val_rec, val_f1, _, _ = evaluate(
            model, val_loader, criterion, device
        )
        
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch+1:2d}/{args.epochs}: Train {train_acc:.3f} | Val {val_acc:.3f} | F1 {val_f1:.3f} | LR {current_lr:.6f}")
        
        # No 'phase' field in history (single-phase)
        train_history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'val_precision': val_prec,
            'val_recall': val_rec,
            'val_f1': val_f1
        })
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, output_dir / 'best_model.pth')
            print(f"  *** Best: {val_acc:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= 10:
                print(f"  Early stopping at epoch {epoch+1}")
                break
    
    print(f"\n✅ Training Complete! Final Best Val Acc: {best_val_acc:.4f}")
    
    # Final evaluation on test set
    print("\n" + "="*70)
    print("FINAL EVALUATION ON TEST SET")
    print("="*70)
    
    # Load best model
    checkpoint = torch.load(output_dir / 'best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_acc, test_precision, test_recall, test_f1, test_preds, test_labels = evaluate(
        model, test_loader, criterion, device
    )
    
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall: {test_recall:.4f}")
    print(f"Test F1: {test_f1:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(test_labels, test_preds)
    print(f"\nConfusion Matrix:")
    print(f"                Predicted")
    print(f"              Bad    Good")
    print(f"Actual Bad   {cm[0,0]:4d}   {cm[0,1]:4d}")
    print(f"       Good  {cm[1,0]:4d}   {cm[1,1]:4d}")
    
    # Save results
    results = {
        'args': vars(args),
        'best_val_acc': best_val_acc,
        'best_val_loss': best_val_loss,
        'test_acc': test_acc,
        'test_precision': test_precision,
        'test_recall': test_recall,
        'test_f1': test_f1,
        'confusion_matrix': cm.tolist(),
        'train_history': train_history
    }
    
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_dir / 'results.json'}")
    print(f"Best model saved to {output_dir / 'best_model.pth'}")


if __name__ == '__main__':
    main()
