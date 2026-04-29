#!/usr/bin/env python3
"""Validation/test metric computation for the image quality classifier."""

import json

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             roc_auc_score)

from .models import DEVICE


def evaluate_on_loader(model, loader, use_mask=False):
    """Inference (no grad). Returns (preds_bin, trues, accuracy, f1, probs)."""
    model.eval()
    preds_bin, trues, probs = [], [], []
    with torch.no_grad():
        for batch in loader:
            if use_mask:
                img, mask, lbl = batch
                prob = torch.sigmoid(model(img.to(DEVICE), mask.to(DEVICE))).cpu().numpy()
            else:
                img, lbl = batch
                prob = torch.sigmoid(model(img.to(DEVICE))).cpu().numpy()
            probs.extend(prob)
            preds_bin.extend((prob > 0.5).astype(int))
            trues.extend(lbl.numpy())
    preds_bin, trues, probs = np.array(preds_bin), np.array(trues), np.array(probs)
    return preds_bin, trues, float(accuracy_score(trues, preds_bin)), float(f1_score(trues, preds_bin)), probs


def safe_roc_auc(trues, probs):
    try:
        return float(roc_auc_score(trues, probs))
    except Exception:
        return None


def get_validation_metrics(model, y_val, model_dir, val_loader, day, cfg,
                           backbone_key, backbone_name, val_img_paths):
    preds_bin, trues, acc, f1, probs = evaluate_on_loader(model, val_loader, use_mask=cfg.use_mask)
    metrics = {
        "metrics": {
            "day": day, "split": "val",
            "accuracy": float(acc), "f1": float(f1),
            "roc_auc": safe_roc_auc(trues, probs),
            "pr_auc": float(average_precision_score(trues, probs)) if len(trues) > 0 else None,
            "n": int(len(y_val)),
            "batch_size": int(cfg.val_batch_size),
            "input_key": cfg.input_path_key,
            "use_mask": cfg.use_mask,
        },
        "model": {
            "backbone_key": backbone_key, "backbone_name": backbone_name,
            "target_size": cfg.target_size, "use_mask": cfg.use_mask,
            "deterministic": cfg.deterministic,
        },
        "results": {
            "image_paths": val_img_paths,
            "true_labels": trues.tolist(),
            "predicted_probabilities": probs.tolist(),
            "predicted_binary": preds_bin.tolist(),
        },
    }
    with (model_dir / "metrics_val.json").open("w") as f:
        json.dump(metrics, f, indent=2)
    return metrics


def get_test_metrics(model, y_val, model_dir, test_loader, day, best_acc, cfg,
                     backbone_key, backbone_name, test_img_paths):
    from .cli import day_to_int  # avoid import cycle

    preds_bin, trues, acc, f1, probs = evaluate_on_loader(model, test_loader, use_mask=cfg.use_mask)
    metrics = {
        "metrics": {
            "day": day, "day_no": day_to_int(day), "split": "test",
            "accuracy": float(acc), "f1": float(f1),
            "roc_auc": safe_roc_auc(trues, probs),
            "pr_auc": float(average_precision_score(trues, probs)) if len(trues) > 0 else None,
            "val_accuracy_for_selection": float(best_acc),
            "val_n": int(len(y_val)),
            "test_n": int(len(trues)),
            "actual_good": int(trues.sum()),
            "predicted_good": int(preds_bin.sum()),
            "batch_size_train": int(cfg.batch_size),
            "batch_size_valtest": int(cfg.val_batch_size),
        },
        "model": {
            "backbone_key": backbone_key, "backbone_name": backbone_name,
            "target_size": cfg.target_size, "input_key": cfg.input_path_key,
            "use_mask": cfg.use_mask, "deterministic": cfg.deterministic,
        },
        "results": {
            "image_paths": test_img_paths,
            "true_labels": trues.tolist(),
            "predicted_probabilities": probs.tolist(),
            "predicted_binary": preds_bin.tolist(),
        },
    }
    with (model_dir / "metrics_test.json").open("w") as f:
        json.dump(metrics, f, indent=2)
    print(f"📝 Saved metrics → {model_dir / 'metrics_val.json'} and {model_dir / 'metrics_test.json'}")
    return metrics
