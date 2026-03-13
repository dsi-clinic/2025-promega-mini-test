#!/usr/bin/env python3
"""
Run OUR time-series EfficientNet (EfficientNet + Temporal Attention) with same data/splits as Amanda's CNN-LSTM.
Uses both_*_base.json, full time series (no single-day). Saves to our_effnet_timeseries_all_data/ for comparison.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from image_classifier.cnn_lstm.load_split_data import load_split_data
from image_classifier.cnn_lstm.organoid_dataset import (
    OrganoidTimeSeriesDataset,
    compute_global_mean_from_ids,
)
from image_classifier.cnn_lstm.train_organoid_lstm_single_phase import (
    collate_variable_length,
)
from image_classifier.cnn_lstm.train_temporal_ablation_attn import (
    OrganoidCNN_TAtt,
    set_seed,
    evaluate_binary,
    BATCH_SIZE,
    NUM_WORKERS,
    MAX_EPOCHS,
    WARMUP_EPOCHS,
    LR_HEAD,
    LR_CNN_UNFREEZE,
    GRAD_CLIP,
    PATIENCE,
    ATTN_DROPOUT,
)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import json
from sklearn.metrics import confusion_matrix

# Same seed as CNN-LSTM for fair comparison
SEED = 42


def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    output_dir = Path(__file__).parent / "our_effnet_timeseries_all_data"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Same data as Amanda's CNN-LSTM
    split_dir = REPO_ROOT / "data_splits"
    train_ids, val_ids, test_ids, series_metadata, data = load_split_data(
        split_dir / "both_train_base.json",
        split_dir / "both_val_base.json",
        split_dir / "both_test_base.json",
    )
    print(f"Splits: train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}")

    # Global mean from training set (same as CNN-LSTM)
    print("Computing global mean from training set...")
    global_mean = compute_global_mean_from_ids(train_ids, series_metadata, data)
    np.save(output_dir / "global_mean.npy", global_mean)
    print(f"Saved {output_dir / 'global_mean.npy'}")

    # Full time series: no max_day, no transform (same data as Amanda)
    train_dataset = OrganoidTimeSeriesDataset(
        train_ids, series_metadata, data, global_mean=global_mean
    )
    val_dataset = OrganoidTimeSeriesDataset(
        val_ids, series_metadata, data, global_mean=global_mean
    )
    test_dataset = OrganoidTimeSeriesDataset(
        test_ids, series_metadata, data, global_mean=global_mean
    )

    pin = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=pin,
        collate_fn=collate_variable_length,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin,
        collate_fn=collate_variable_length,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin,
        collate_fn=collate_variable_length,
    )

    # Class balance (same idea as CNN-LSTM / single-day EfficientNet)
    train_labels = []
    for org_id in train_ids:
        s = str(series_metadata[org_id].get("label", "")).strip().lower()
        lab = 1 if s in ("good", "acceptable", "accepted") else 0
        train_labels.append(lab)
    n_good = int(np.sum(train_labels))
    n_bad = int(len(train_labels) - n_good)
    if n_good == 0:
        n_good = 1
    if n_bad == 0:
        n_bad = 1
    pos_weight = torch.tensor([n_bad / n_good], device=device, dtype=torch.float32)
    print(
        f"Class balance: good={n_good}, bad={n_bad}, pos_weight={pos_weight.item():.3f}"
    )

    model = OrganoidCNN_TAtt(attn_dropout=ATTN_DROPOUT).to(device)

    def make_optimizer(lr_cnn, lr_head):
        params_cnn = [p for n, p in model.cnn.named_parameters() if p.requires_grad]
        params_head = [
            p
            for n, p in model.named_parameters()
            if not n.startswith("cnn.") and p.requires_grad
        ]
        groups = []
        if params_cnn:
            groups.append({"params": params_cnn, "lr": lr_cnn})
        if params_head:
            groups.append({"params": params_head, "lr": lr_head})
        return optim.Adam(groups)

    optimizer = make_optimizer(lr_cnn=0.0, lr_head=LR_HEAD)
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    w_pos = n_bad / n_good
    w_neg = n_good / n_bad
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    best_val_acc = -1.0
    best_state = None
    bad_epochs = 0

    print("\n" + "=" * 70)
    print("TRAINING EFFICIENTNET TIME-SERIES (Temporal Attention)")
    print("=" * 70)

    for epoch in range(1, MAX_EPOCHS + 1):
        if epoch == WARMUP_EPOCHS + 1:
            model.unfreeze_last_blocks()
            optimizer = make_optimizer(lr_cnn=LR_CNN_UNFREEZE, lr_head=LR_HEAD)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=5
            )
            print("→ Unfroze last CNN blocks.")

        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for seqs, days, labels, weights, ids in tqdm(
            train_loader, desc=f"Epoch {epoch:02d}", leave=False
        ):
            seqs = seqs.to(device)
            days = days.to(device).float()
            labels = labels.float().to(device)
            weights = weights.to(device).float()

            optimizer.zero_grad()
            logits, _ = model(seqs, days)
            loss_raw = criterion(logits, labels)
            cls_w = labels * w_pos + (1 - labels) * w_neg
            loss = (loss_raw * weights * cls_w).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            preds = (torch.sigmoid(logits).view(-1) > 0.5).long()
            correct += (preds == labels.long()).sum().item()
            total += labels.size(0)

        train_loss = running_loss / max(1, total)
        train_acc = correct / max(1, total)

        val_loss, val_acc, val_prec, val_rec, val_f1, val_auc, val_ap, _, _ = (
            evaluate_binary(model, val_loader, criterion, device)
        )
        scheduler.step(val_loss)

        print(
            f"Epoch {epoch:02d} | Train {train_acc:.3f} / {train_loss:.4f} | "
            f"Val {val_acc:.3f} / {val_loss:.4f} (P {val_prec:.3f} R {val_rec:.3f} F1 {val_f1:.3f})"
        )

        if val_acc > best_val_acc + 1e-4:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
            print("  * new best")
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                print(f"  Early stopping at epoch {epoch}")
                break

    if best_state is None:
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state, strict=True)
    (
        test_loss,
        test_acc,
        test_prec,
        test_rec,
        test_f1,
        test_auc,
        test_ap,
        test_fp,
        test_fn,
    ) = evaluate_binary(model, test_loader, criterion, device)

    best_model_path = output_dir / "best_model.pth"
    torch.save(
        {"state_dict": best_state, "best_val_acc": float(best_val_acc)},
        best_model_path,
    )
    print(f"\nSaved {best_model_path}")

    # Get test preds/labels for confusion matrix (same format as CNN-LSTM and single-day EfficientNet)
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for seqs, days_norm, labels, weights, ids in test_loader:
            seqs = seqs.to(device)
            days_norm = days_norm.to(device).float()
            logits, _ = model(seqs, days_norm)
            preds = (torch.sigmoid(logits) > 0.5).long().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.long().cpu().numpy())
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])

    # Same result keys as CNN-LSTM / single-day EfficientNet for direct comparison
    results = {
        "best_val_acc": float(best_val_acc),
        "test_acc": float(test_acc),
        "test_precision": float(test_prec),
        "test_recall": float(test_rec),
        "test_f1": float(test_f1),
        "confusion_matrix": cm.tolist(),
        "model_path": str(best_model_path.resolve()),
        "test_auc": float(test_auc),
        "test_ap": float(test_ap),
        "test_false_positives": list(test_fp),
        "test_false_negatives": list(test_fn),
    }
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_dir / 'results.json'}")
    print(
        f"Test Acc {test_acc:.4f} | F1 {test_f1:.4f} | P {test_prec:.4f} | R {test_rec:.4f}"
    )
    print(f"Confusion matrix: TN={cm[0, 0]} FP={cm[0, 1]} FN={cm[1, 0]} TP={cm[1, 1]}")


if __name__ == "__main__":
    main()
