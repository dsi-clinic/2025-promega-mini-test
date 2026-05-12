"""
Training script for organoid CNN-LSTM model
RUN FROM PROJECT ROOT: python analysis/images/cnn_lstm/train_organoid_lstm.py
"""
import os
import random
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add project root to path so imports work
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from analysis.images.cnn_lstm.organoid_dataset import (
    OrganoidTimeSeriesDataset,
    make_idor_series_splits,
)
from analysis.images.cnn_lstm.organoid_model import OrganoidCNN_LSTM

SEED = 42


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_epoch(model, dataloader, criterion, optimizer, device, n_pos, n_neg):
    """Train for one epoch with BCE + per-sample class weighting.

    Labels follow rule #9: 1 = Not Acceptable (positive/minority), 0 = Acceptable.
    n_pos/n_neg are used to upweight the minority class symmetrically.
    """
    model.train()
    total_loss = 0.0
    total_n = 0
    all_preds = []
    all_labels = []

    for seqs, days, labels, weights, _ids in tqdm(dataloader, desc="Training"):
        seqs = seqs.to(device)
        days = days.to(device).float()
        labels = labels.float().to(device)
        weights = weights.to(device).float()

        optimizer.zero_grad()
        logits = model(seqs, days)

        loss_raw = criterion(logits, labels)
        cls_w = labels * (n_neg / n_pos) + (1 - labels) * (n_pos / n_neg)
        loss = (loss_raw * weights * cls_w).mean()

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        total_n += labels.size(0)
        preds = (torch.sigmoid(logits) > 0.5).long()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.long().cpu().numpy())

    avg_loss = total_loss / max(1, total_n)
    accuracy = accuracy_score(all_labels, all_preds)
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    """Evaluate model. Mirrors train_one_epoch's BCE contract (no class weighting)."""
    model.eval()
    total_loss = 0.0
    total_n = 0
    all_preds = []
    all_labels = []

    for seqs, days, labels, _weights, _ids in tqdm(dataloader, desc="Evaluating"):
        seqs = seqs.to(device)
        days = days.to(device).float()
        labels = labels.float().to(device)

        logits = model(seqs, days)
        loss_raw = criterion(logits, labels)
        total_loss += loss_raw.mean().item() * labels.size(0)
        total_n += labels.size(0)

        preds = (torch.sigmoid(logits) > 0.5).long()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.long().cpu().numpy())

    avg_loss = total_loss / max(1, total_n)
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary', zero_division=0
    )
    return avg_loss, accuracy, precision, recall, f1, all_preds, all_labels


def main():
    parser = argparse.ArgumentParser(description='Train CNN-LSTM for organoid classification')
    parser.add_argument('--epochs', type=int, default=20, help='Max epochs per phase (phase1=50, phase2=100 by default)')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--lstm-hidden', type=int, default=256, help='LSTM hidden size')
    parser.add_argument('--lstm-layers', type=int, default=2, help='Number of LSTM layers')
    parser.add_argument('--output-dir', type=str, default='outputs/cnn_lstm', help='Output directory')
    parser.add_argument('--image-type', type=str, default='clipped', choices=['clipped', 'std'],
                        help='Image variant to use: clipped (575x575 AR meanfill) or std (512x384)')
    args = parser.parse_args()

    set_seed(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving outputs to: {output_dir}")

    print("\n" + "="*70)
    print("LOADING DATA")
    print("="*70)

    ds, train_ids, val_ids, test_ids = make_idor_series_splits()

    print(f"Using image type: {args.image_type}")
    train_dataset = OrganoidTimeSeriesDataset(train_ids, ds, image_type=args.image_type)
    val_dataset = OrganoidTimeSeriesDataset(val_ids, ds, image_type=args.image_type)
    test_dataset = OrganoidTimeSeriesDataset(test_ids, ds, image_type=args.image_type)

    pin = (device.type == "cuda")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=pin)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=pin)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=pin)

    # Class balance per rule #9 (1 = Not Acceptable / positive minority class).
    train_labels = [
        1 if ds.organoid_label(oid) == 'Not Acceptable' else 0
        for oid in train_ids
    ]
    n_pos = max(1, int(np.sum(train_labels)))           # Not Acceptable
    n_neg = max(1, int(len(train_labels) - n_pos))      # Acceptable
    print(f"\nClass balance (train): Not Acceptable={n_pos}, Acceptable={n_neg}")

    print("\n" + "="*70)
    print("CREATING MODEL")
    print("="*70)

    model = OrganoidCNN_LSTM(
        hidden_size=args.lstm_hidden,
        num_layers=args.lstm_layers,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    pos_weight = torch.tensor([n_neg / n_pos], device=device, dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='none')

    def train_two_phase(model, train_loader, val_loader, criterion, device, output_dir):
        """Two-phase training: phase 1 freezes CNN, phase 2 unfreezes for fine-tuning."""
        # PHASE 1: frozen CNN
        print("\n" + "="*70)
        print("PHASE 1: Training LSTM Only (CNN Frozen)")
        print("="*70)
        for param in model.cnn.parameters():
            param.requires_grad = False

        optimizer_phase1 = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=1e-3,
        )
        scheduler_phase1 = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_phase1, mode='min', patience=5, factor=0.5
        )

        best_val_acc = 0.0
        best_val_loss = float('inf')
        patience_counter = 0
        phase1_history = []

        for epoch in range(50):
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer_phase1, device, n_pos, n_neg
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
                'val_f1': val_f1,
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

        print(f"\nLoading best Phase 1 model (Val Acc: {best_val_acc:.4f})")
        checkpoint = torch.load(output_dir / 'phase1_best.pth')
        model.load_state_dict(checkpoint['model_state_dict'])

        # PHASE 2: unfreeze CNN
        print("\n" + "="*70)
        print("PHASE 2: Fine-Tuning Entire Network (CNN Unfrozen)")
        print("="*70)
        model.unfreeze_last_blocks(n_blocks=2)

        optimizer_phase2 = optim.Adam(model.parameters(), lr=1e-4)
        scheduler_phase2 = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_phase2, mode='min', patience=7, factor=0.5
        )

        best_val_acc = 0.0
        best_val_loss = float('inf')
        patience_counter = 0
        phase2_history = []

        for epoch in range(100):
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer_phase2, device, n_pos, n_neg
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
                'val_f1': val_f1,
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

        print(f"\nTraining Complete. Final Best Val Acc: {best_val_acc:.4f}")
        return phase1_history + phase2_history, best_val_acc, best_val_loss

    train_history, best_val_acc, best_val_loss = train_two_phase(
        model, train_loader, val_loader, criterion, device, output_dir
    )

    print("\n" + "="*70)
    print("FINAL EVALUATION ON TEST SET")
    print("="*70)
    checkpoint = torch.load(output_dir / 'best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])

    test_loss, test_acc, test_precision, test_recall, test_f1, test_preds, test_labels = evaluate(
        model, test_loader, criterion, device
    )
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Precision (Not Acceptable): {test_precision:.4f}")
    print(f"Test Recall (Not Acceptable): {test_recall:.4f}")
    print(f"Test F1 (Not Acceptable): {test_f1:.4f}")

    cm = confusion_matrix(test_labels, test_preds)
    print(f"\nConfusion Matrix:")
    print(f"                       Predicted")
    print(f"                Acceptable   Not Acceptable")
    print(f"Acceptable        {cm[0,0]:4d}            {cm[0,1]:4d}")
    print(f"Not Acceptable    {cm[1,0]:4d}            {cm[1,1]:4d}")

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)
    classes = ['Acceptable (0)', 'Not Acceptable (1)']
    ax.set(xticks=[0, 1], yticks=[0, 1],
           xticklabels=classes, yticklabels=classes,
           xlabel='Predicted', ylabel='Actual',
           title='Confusion Matrix (Test Set)')
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black')
    plt.tight_layout()
    cm_path = output_dir / 'confusion_matrix.png'
    plt.savefig(cm_path, dpi=150)
    plt.close(fig)
    print(f"Confusion matrix saved -> {cm_path}")

    if train_history:
        epochs = [h['epoch'] for h in train_history]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        ax1.plot(epochs, [h['train_acc'] for h in train_history], label='Train Acc')
        ax1.plot(epochs, [h['val_acc'] for h in train_history], label='Val Acc')
        phase2_start = next((h['epoch'] for h in train_history if h.get('phase') == 2), None)
        if phase2_start:
            ax1.axvline(x=phase2_start, color='gray', linestyle='--', alpha=0.6, label='Phase 2 start')
        ax1.set(xlabel='Epoch', ylabel='Accuracy', title='Accuracy over Training')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.plot(epochs, [h['train_loss'] for h in train_history], label='Train Loss')
        ax2.plot(epochs, [h['val_loss'] for h in train_history], label='Val Loss')
        if phase2_start:
            ax2.axvline(x=phase2_start, color='gray', linestyle='--', alpha=0.6, label='Phase 2 start')
        ax2.set(xlabel='Epoch', ylabel='Loss', title='Loss over Training')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = output_dir / 'training_curves.png'
        plt.savefig(plot_path, dpi=150)
        plt.close(fig)
        print(f"Training curves saved -> {plot_path}")

    results = {
        'args': vars(args),
        'best_val_acc': best_val_acc,
        'best_val_loss': best_val_loss,
        'test_acc': test_acc,
        'test_precision': test_precision,
        'test_recall': test_recall,
        'test_f1': test_f1,
        'confusion_matrix': cm.tolist(),
        'train_history': train_history,
    }
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_dir / 'results.json'}")
    print(f"Best model saved to {output_dir / 'best_model.pth'}")


if __name__ == '__main__':
    main()
