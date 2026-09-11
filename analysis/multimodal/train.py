#!/usr/bin/env python3
"""Training + evaluation for the multimodal classifier.

eval_epoch returns both metrics and (preds, labels). Per-organoid CSVs are
built by callers from those arrays — no separate eval_epoch_detailed.
Batch unpacking is uniform: every modality combo emits ``*inputs, label``,
so we don't branch on input_mode.
"""

from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader

from .data import MultimodalRowDataset, get_transforms
from .models import EarlyStopping, MultimodalClassifier


def _run_batches(model, loader, config: dict,
                 *, optimizer=None, criterion=None, weights=None) -> tuple[list, np.ndarray, np.ndarray]:
    """Single batch loop. If optimizer is given, train; else just predict.

    Returns (losses_list, preds_array, labels_array).
    """
    train = optimizer is not None
    model.train() if train else model.eval()
    losses, preds, labels = [], [], []

    with torch.set_grad_enabled(train):
        for batch in loader:
            *inputs, y = batch  # uniform — see module docstring
            inputs = [x.to(config["device"]) for x in inputs]
            y_dev = y.to(config["device"])

            logits = model(*inputs)

            if train:
                loss = criterion(logits, y_dev)
                w = torch.tensor([weights[int(l)] for l in y_dev], device=y_dev.device)
                loss = (loss * w).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

            preds.extend(torch.sigmoid(logits).detach().cpu().numpy())
            labels.extend(y.numpy())

    return losses, np.array(preds), np.array(labels)


def train_epoch(model, loader, optimizer, criterion, weights, config):
    losses, preds, labels = _run_batches(model, loader, config,
                                         optimizer=optimizer, criterion=criterion, weights=weights)
    acc = accuracy_score(labels, (preds > 0.5).astype(int))
    return float(np.mean(losses)), float(acc)


def eval_epoch(model, loader, config) -> dict:
    """Evaluate; return metrics + raw (preds, labels) so callers can build CSVs."""
    _, preds, labels = _run_batches(model, loader, config)
    preds_bin = (preds > 0.5).astype(int)

    cm = confusion_matrix(labels, preds_bin, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    opt_thresh, acc_opt, f1_opt = 0.5, accuracy_score(labels, preds_bin), f1_score(labels, preds_bin, zero_division=0)
    if len(np.unique(labels)) > 1:
        fpr, tpr, thresholds = roc_curve(labels, preds)
        idx = np.argmax(tpr - fpr)
        opt_thresh = thresholds[idx]
        preds_opt = (preds >= opt_thresh).astype(int)
        acc_opt = accuracy_score(labels, preds_opt)
        f1_opt = f1_score(labels, preds_opt, zero_division=0)

    return {
        "acc": accuracy_score(labels, preds_bin),
        "f1": f1_score(labels, preds_bin, zero_division=0),
        "recall": recall_score(labels, preds_bin, zero_division=0),
        "precision": precision_score(labels, preds_bin, zero_division=0),
        "specificity": specificity,
        "auc": roc_auc_score(labels, preds) if len(np.unique(labels)) > 1 else None,
        "pr_auc": average_precision_score(labels, preds) if len(np.unique(labels)) > 1 else None,
        "acc_opt": acc_opt,
        "f1_opt": f1_opt,
        "opt_thresh": opt_thresh,
        "preds": preds,
        "labels": labels,
        "confusion_matrix": {"TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn)},
    }


def _make_loaders(train_df, val_df, test_df, config, *, scaler):
    t_train = get_transforms(config, augment=True) if config["use_images"] else None
    t_eval = get_transforms(config, augment=False) if config["use_images"] else None

    if scaler is None:
        train_ds = MultimodalRowDataset(train_df, config, t_train, fit_scaler=True)
        scaler = train_ds.scaler
    else:
        train_ds = MultimodalRowDataset(train_df, config, t_train, scaler=scaler)
    val_ds = MultimodalRowDataset(val_df, config, t_eval, scaler=scaler)
    test_ds = MultimodalRowDataset(test_df, config, t_eval, scaler=scaler) if test_df is not None else None

    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"], num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=config["batch_size"], num_workers=4) if test_ds else None
    return train_ds, val_ds, test_ds, train_loader, val_loader, test_loader, scaler


def _class_weights_from(train_ds: MultimodalRowDataset) -> dict:
    labels = [train_ds.label_map.get(train_ds.df.iloc[i]["label"], 0) for i in range(len(train_ds))]
    weights_arr = compute_class_weight("balanced", classes=np.unique(labels), y=labels)
    return {int(c): float(w) for c, w in zip(np.unique(labels), weights_arr)}


def pretrain_shared_backbone(train_df, val_df, config):
    """Pretrain a shared backbone on all training samples across all days.

    Returns (best_state_dict, scaler) for downstream per-day models to inherit.
    """
    print(f"\n{'=' * 60}\nPretraining Shared Backbone (All Days)\n{'=' * 60}")
    train_ds, val_ds, _, train_loader, val_loader, _, scaler = _make_loaders(
        train_df, val_df, None, config, scaler=None,
    )
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    class_weights = _class_weights_from(train_ds)
    model = MultimodalClassifier(config).to(config["device"])
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=config["learning_rate"], weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss(reduction="none")

    best_val_acc, best_state = -np.inf, None
    es = EarlyStopping(config["early_stopping_patience"])

    print("Training shared backbone (all parameters trainable)")
    for epoch in range(config["num_epochs_phase1"]):
        tl, ta = train_epoch(model, train_loader, optimizer, criterion, class_weights, config)
        vr = eval_epoch(model, val_loader, config)
        if vr["acc"] > best_val_acc:
            best_val_acc, best_state = vr["acc"], model.state_dict().copy()
        if epoch % 10 == 0:
            print(f"Ep {epoch}: loss={tl:.4f}, tr_acc={ta:.3f}, val_acc={vr['acc']:.3f}")
        if es(vr["acc"]):
            print(f"Early stop at epoch {epoch}")
            break

    print(f"Shared backbone pretraining complete. Best val acc: {best_val_acc:.3f}\n")
    return best_state, scaler


def train_for_day(day, train_df, val_df, test_df, config, output_dir,
                  shared_backbone_state=None, shared_scaler=None):
    """Train a frozen-backbone head for one day."""
    print(f"\n{'=' * 60}\nTraining for {day}\n{'=' * 60}")

    train_day = train_df[train_df["day"] == day]
    val_day = val_df[val_df["day"] == day]
    test_day = test_df[test_df["day"] == day]

    if len(train_day) == 0:
        print(f"No training data for {day}")
        return None

    print(f"{day} label counts:")
    print(f"  Train: {train_day['label'].value_counts().to_dict()}")
    print(f"  Val  : {val_day['label'].value_counts().to_dict()}")
    print(f"  Test : {test_day['label'].value_counts().to_dict()}")

    train_ds, _, test_ds, train_loader, val_loader, test_loader, _ = _make_loaders(
        train_day, val_day, test_day, config, scaler=shared_scaler,
    )
    print(f"Train: {len(train_ds)}, Val: {len(val_day)}, Test: {len(test_ds)}")

    class_weights = _class_weights_from(train_ds)
    model = MultimodalClassifier(config).to(config["device"])

    if shared_backbone_state is not None:
        print("Loading shared backbone weights...")
        model_dict = model.state_dict()
        pretrained = {
            k: v for k, v in shared_backbone_state.items()
            if k in model_dict and (k.startswith("backbone.") or k.startswith("meta_branch."))
        }
        model_dict.update(pretrained)
        model.load_state_dict(model_dict, strict=False)
        print(f"Loaded {len(pretrained)} pretrained parameters")

    if config["use_images"]:
        print("Freezing backbone...")
        for param in model.backbone.parameters():
            param.requires_grad = False

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=config["learning_rate"], weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss(reduction="none")

    history = defaultdict(list)
    best_val_acc, best_state = -np.inf, None
    es = EarlyStopping(config["early_stopping_patience"])

    print("Training day-specific head (backbone frozen)")
    for epoch in range(config["num_epochs_phase1"]):
        tl, ta = train_epoch(model, train_loader, optimizer, criterion, class_weights, config)
        vr = eval_epoch(model, val_loader, config)
        history["train_loss"].append(tl)
        history["train_acc"].append(ta)
        history["val_acc"].append(vr["acc"])
        if vr["acc"] > best_val_acc:
            best_val_acc, best_state = vr["acc"], model.state_dict().copy()
        if epoch % 10 == 0:
            print(f"Ep {epoch}: loss={tl:.4f}, tr_acc={ta:.3f}, val_acc={vr['acc']:.3f}")
        if es(vr["acc"]):
            print(f"Early stop at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    test_res = eval_epoch(model, test_loader, config)
    test_res["best_val_acc"] = best_val_acc
    test_res["history"] = dict(history)
    test_res["model_state"] = best_state
    test_res["test_df"] = test_ds.df
    return test_res
