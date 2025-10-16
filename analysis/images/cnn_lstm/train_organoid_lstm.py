"""
Training script for organoid CNN-LSTM model
RUN FROM PROJECT ROOT: python analysis/images/cnn_lstm/train_organoid_lstm.py
"""
import sys
from pathlib import Path

# Add project root to path so imports work
ROOT = Path(__file__).resolve().parents[3]  # Go up 3 levels to root
sys.path.insert(0, str(ROOT))

import argparse
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from config import OUTPUT_FOLDER
from analysis.images.cnn_lstm.organoid_dataset import (
    OrganoidTimeSeriesDataset, 
    load_data_and_create_splits,
    compute_global_mean_from_ids
)
from analysis.images.cnn_lstm.organoid_model import OrganoidCNN_LSTM

# Rest of the file stays the same...

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for images, labels in tqdm(dataloader, desc="Training"):
        # Move to device
        images = images.to(device)
        labels = labels.to(device)
        
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
        for images, labels in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            labels = labels.to(device)
            
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
    parser = argparse.ArgumentParser(description='Train CNN-LSTM for organoid classification')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--lstm-hidden', type=int, default=256, help='LSTM hidden size')
    parser.add_argument('--lstm-layers', type=int, default=2, help='Number of LSTM layers')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory (default: OUTPUT_FOLDER/cnn_lstm)')
    args = parser.parse_args()
    
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
    train_ids, val_ids, test_ids, series_metadata, data = load_data_and_create_splits(
        series_metadata_path, data_path
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

    # Create dataloaders (no changes)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    # Calculate class weights for imbalanced data
    # Count labels in training set
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
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
    
    # Training loop
    print("\n" + "="*70)
    print("TRAINING")
    print("="*70)
    
    best_val_loss = float('inf')
    best_val_acc = 0
    train_history = []
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 70)
        
        # Train
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc, val_precision, val_recall, val_f1, _, _ = evaluate(
            model, val_loader, criterion, device
        )
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Print metrics
        print(f"\nTrain Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
        print(f"Val Precision: {val_precision:.4f} | Val Recall: {val_recall:.4f} | Val F1: {val_f1:.4f}")
        print(f"Learning Rate: {current_lr:.6f}")
        
        # Save history
        train_history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'val_precision': val_precision,
            'val_recall': val_recall,
            'val_f1': val_f1
        })
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, output_dir / 'best_model.pth')
            print(f"*** Saved best model (Val Acc: {val_acc:.4f}) ***")
    
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