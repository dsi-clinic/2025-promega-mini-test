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
from analysis.cnn_lstm.organoid_dataset import (
    OrganoidTimeSeriesDataset, 
    load_data_and_create_splits,
    compute_global_mean_from_ids
)
from analysis.cnn_lstm.organoid_model import OrganoidCNN_LSTM

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

    # ========== TWO-PHASE TRAINING FUNCTION ==========
    def train_two_phase(model, train_loader, val_loader, criterion, device, output_dir):
        """Two-phase training with freeze/unfreeze"""
        
        # PHASE 1: Frozen CNN
        print("\n" + "="*70)
        print("PHASE 1: Training LSTM Only (CNN Frozen)")
        print("="*70)
        
        for param in model.cnn.parameters():
            param.requires_grad = False
        
        optimizer_phase1 = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=1e-3
        )
        scheduler_phase1 = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_phase1, mode='min', patience=5, factor=0.5
        )
        
        best_val_acc = 0
        best_val_loss = float('inf')
        patience_counter = 0
        phase1_history = []
        
        for epoch in range(50):
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer_phase1, device
            )
            val_loss, val_acc, val_prec, val_rec, val_f1, _, _ = evaluate(
                model, val_loader, criterion, device
            )
            
            scheduler_phase1.step(val_loss)
            current_lr = optimizer_phase1.param_groups[0]['lr']
            
            print(f"[P1] Epoch {epoch+1:2d}: Train {train_acc:.3f} | Val {val_acc:.3f} | F1 {val_f1:.3f} | LR {current_lr:.6f}")
            
            phase1_history.append({
                'epoch': epoch + 1,
                'phase': 1,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'val_f1': val_f1
            })
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_val_loss = val_loss
                patience_counter = 0
                torch.save({
                    'epoch': epoch,
                    'phase': 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer_phase1.state_dict(),
                    'val_acc': val_acc,
                    'val_loss': val_loss,
                }, output_dir / 'phase1_best.pth')
                print(f"  *** Phase 1 Best: {val_acc:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= 10:
                    print(f"  Early stopping Phase 1 at epoch {epoch+1}")
                    break
        
        # Load best phase 1
        print(f"\nLoading best Phase 1 model (Val Acc: {best_val_acc:.4f})")
        checkpoint = torch.load(output_dir / 'phase1_best.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # PHASE 2: Unfreeze CNN
        print("\n" + "="*70)
        print("PHASE 2: Fine-Tuning Entire Network (CNN Unfrozen)")
        print("="*70)
        
        for param in model.cnn.parameters():
            param.requires_grad = True
        
        optimizer_phase2 = optim.Adam(model.parameters(), lr=1e-4)
        scheduler_phase2 = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_phase2, mode='min', patience=7, factor=0.5
        )
        
        best_val_acc = 0
        best_val_loss = float('inf')
        patience_counter = 0
        phase2_history = []
        
        for epoch in range(100):
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer_phase2, device
            )
            val_loss, val_acc, val_prec, val_rec, val_f1, _, _ = evaluate(
                model, val_loader, criterion, device
            )
            
            scheduler_phase2.step(val_loss)
            current_lr = optimizer_phase2.param_groups[0]['lr']
            
            print(f"[P2] Epoch {epoch+1:3d}: Train {train_acc:.3f} | Val {val_acc:.3f} | F1 {val_f1:.3f} | LR {current_lr:.6f}")
            
            phase2_history.append({
                'epoch': epoch + 1,
                'phase': 2,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'val_f1': val_f1
            })
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_val_loss = val_loss
                patience_counter = 0
                torch.save({
                    'epoch': epoch,
                    'phase': 2,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer_phase2.state_dict(),
                    'val_acc': val_acc,
                    'val_loss': val_loss,
                }, output_dir / 'best_model.pth')
                print(f"  *** Phase 2 Best: {val_acc:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= 15:
                    print(f"  Early stopping Phase 2 at epoch {epoch+1}")
                    break
        
        print(f"\n✅ Training Complete! Final Best Val Acc: {best_val_acc:.4f}")
        
        return phase1_history + phase2_history, best_val_acc, best_val_loss
    
    # ========== CALL THE FUNCTION ==========
    train_history, best_val_acc, best_val_loss = train_two_phase(
        model, train_loader, val_loader, criterion, device, output_dir
    )
    # =======================================
    
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