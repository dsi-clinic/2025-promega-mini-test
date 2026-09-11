#!/usr/bin/env python3
"""Training loop + per-day orchestration for the image quality classifier."""

import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from .data import filter_missing_files, make_loader
from .eval import get_test_metrics, get_validation_metrics
from .models import BACKBONES, DEVICE, EarlyStopping, ImageOnlyClassifier
from .plots import plot_training_curve


def set_deterministic(deterministic: bool) -> None:
    if torch.cuda.is_available() and deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True)
    except RuntimeError as e:
        print(f"Warning: Could not enable deterministic algorithms: {e}")


def set_seed(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class FocalLoss(nn.Module):
    """Binary focal loss with alpha weighting (operates on raw logits).

    Reduces the relative loss for well-classified examples, focusing
    optimization on hard ones. Defaults match the dinov2 trainer's settings
    (gamma=2.0, alpha=0.25). Returns per-element loss when reduction='none'.
    """

    def __init__(self, gamma=2.0, alpha=0.25, reduction="none"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = targets * probs + (1 - targets) * (1 - probs)
        modulating = torch.pow(1 - p_t, self.gamma)
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        loss = alpha_t * modulating * bce
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def epoch_loop(model, loader, optimizer, class_weights, train=True, use_mask=False, loss_fn=None):
    model.train() if train else model.eval()
    if loss_fn is None:
        loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    losses, preds, trues = [], [], []

    for batch in loader:
        if use_mask:
            img, mask, label = batch
            img, mask, label = img.to(DEVICE), mask.to(DEVICE), label.to(DEVICE)
            logit = model(img, mask)
        else:
            img, label = batch
            img, label = img.to(DEVICE), label.to(DEVICE)
            logit = model(img)
        loss = loss_fn(logit, label)
        weight = torch.tensor([class_weights[int(l.item())] for l in label], device=label.device)
        loss = (loss * weight).mean()

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        losses.append(loss.item())
        preds.extend(torch.sigmoid(logit).detach().cpu().numpy())
        trues.extend(label.cpu().numpy())

    preds_bin = (np.array(preds) > 0.5).astype(int)
    return float(np.mean(losses)), accuracy_score(trues, preds_bin), preds_bin, np.array(trues)


def run_phases(model, model_path, backbone_key, backbone_name, day,
               train_loader, val_loader, class_weights, cfg,
               *, loss_fn=None, scheduler_factory=None):
    """Phase 1 (frozen backbone) → Phase 2 (partial unfreeze). Save best by val acc.

    loss_fn: optional override for epoch_loop's loss (None → BCE).
    scheduler_factory: optional callable (opt) → torch.optim scheduler.
        If given, scheduler.step(vacc) is called after each epoch.
    """
    phases = [(cfg.epoch1, 1e-3, 20), (cfg.epoch2, 1e-4, 30)]
    history = defaultdict(list)
    best_acc = -np.inf

    opt = None
    for phase, (n_epochs, lr, patience) in enumerate(phases, start=1):
        if phase == 2:
            model.unfreeze_backbone()
        opt = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
        scheduler = scheduler_factory(opt) if scheduler_factory else None
        es = EarlyStopping(patience=patience)

        for epoch in range(n_epochs):
            tl, tacc, _, _ = epoch_loop(model, train_loader, opt, class_weights,
                                        train=True, use_mask=cfg.use_mask, loss_fn=loss_fn)
            vl, vacc, _, _ = epoch_loop(model, val_loader, opt, class_weights,
                                        train=False, use_mask=cfg.use_mask, loss_fn=loss_fn)
            history["train_loss"].append(tl)
            history["val_loss"].append(vl)
            history["train_acc"].append(tacc)
            history["val_acc"].append(vacc)
            print(f"[{day}][{backbone_key}][P{phase}][{epoch:03d}][bs={cfg.batch_size}/{cfg.val_batch_size}] "
                  f"loss {tl:.4f}/{vl:.4f} acc {tacc:.3f}/{vacc:.3f}")
            if vacc > best_acc:
                best_acc = vacc
                torch.save(model.state_dict(), model_path)
            if scheduler is not None:
                scheduler.step(vacc)
            if es.step(vacc):
                break

    return history, best_acc


def run_training_for_day(day: str, data: dict, backbone_key: str,
                         backbone_name: str, cfg) -> dict:
    """Train + validate on a single day. Select by VAL acc, report on TEST."""
    labels = data.get("label", [])
    imgs = data.get(cfg.input_path_key, [])
    masks = data.get("mask_path", [])

    filtered = filter_missing_files(day, labels, imgs, masks, backbone_key, cfg)
    if filtered is None:
        return None
    imgs, labels, masks = filtered

    if cfg.use_mask:
        X_tmp, X_test, M_tmp, M_test, y_tmp, y_test = train_test_split(
            imgs, masks, labels, test_size=cfg.test_frac, stratify=labels, random_state=cfg.seed
        )
    else:
        X_tmp, X_test, y_tmp, y_test = train_test_split(
            imgs, labels, test_size=cfg.test_frac, stratify=labels, random_state=cfg.seed
        )

    val_frac_cond = cfg.val_frac / (1.0 - cfg.test_frac)
    if cfg.use_mask:
        X_tr, X_val, M_tr, M_val, y_tr, y_val = train_test_split(
            X_tmp, M_tmp, y_tmp, test_size=val_frac_cond, stratify=y_tmp, random_state=cfg.seed
        )
    else:
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_tmp, y_tmp, test_size=val_frac_cond, stratify=y_tmp, random_state=cfg.seed
        )

    weights = compute_class_weight("balanced", classes=np.unique(y_tr), y=y_tr)
    class_weights = {int(k): float(w) for k, w in zip(np.unique(y_tr), weights)}

    train_mask = M_tr if cfg.use_mask else None
    train_loader = make_loader(X_tr, y_tr, mask_paths=train_mask, augment=False,
                               batch_size=cfg.batch_size, cfg=cfg)
    val_loader = make_loader(X_val, y_val, mask_paths=train_mask, augment=False,
                             batch_size=cfg.val_batch_size, cfg=cfg)
    test_loader = make_loader(X_test, y_test, mask_paths=train_mask, augment=False,
                              batch_size=cfg.val_batch_size, cfg=cfg)

    set_seed(cfg.seed, cfg.deterministic)
    model = ImageOnlyClassifier(
        backbone_key, backbone_name, cfg.target_size,
        use_mask=cfg.use_mask, deterministic=cfg.deterministic,
    ).to(DEVICE)
    model_dir = cfg.out_dir / backbone_key / day
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "model.pth"

    history, best_acc = run_phases(model, model_path, backbone_key, backbone_name,
                                   day, train_loader, val_loader, class_weights, cfg)
    plot_training_curve(history, model_dir)

    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    val_metrics = get_validation_metrics(model, y_val, model_dir, val_loader,
                                         day, cfg, backbone_key, backbone_name, X_val)
    test_metrics = get_test_metrics(model, y_val, model_dir, test_loader,
                                    day, best_acc, cfg, backbone_key, backbone_name, X_test)

    return {
        "day": day,
        "day_no": test_metrics["metrics"]["day_no"],
        "backbone_key": backbone_key,
        "val_accuracy": float(best_acc),
        "test_accuracy": test_metrics["metrics"]["accuracy"],
        "test_f1": test_metrics["metrics"]["f1"],
        "val_roc_auc": val_metrics["metrics"]["roc_auc"],
        "test_roc_auc": test_metrics["metrics"]["roc_auc"],
        "val_num": int(len(y_val)),
        "test_num": test_metrics["metrics"]["test_n"],
        "test_actual_good": test_metrics["metrics"]["actual_good"],
        "test_pred_good": test_metrics["metrics"]["predicted_good"],
    }


def collect_results(json_data, cfg):
    """Train all backbones for each day; pick best per day by validation accuracy."""
    per_day_best = {}
    per_model_results = {bk: {} for bk in BACKBONES}

    for day, data in json_data["records"].items():
        if not data:
            print(f"⚠ Skipping {day} — no records")
            continue

        best = None
        for backbone_key, backbone_name in BACKBONES.items():
            res = run_training_for_day(day, data, backbone_key, backbone_name, cfg=cfg)
            if res is None:
                continue
            per_model_results[backbone_key][day] = res
            if (best is None) or (res["val_accuracy"] > best["val_accuracy"]):
                best = res

        if best:
            per_day_best[day] = best
            print(f"✅ Best for {day} (by VAL): {best['backbone_key']} | "
                  f"val acc={best['val_accuracy']:.3f} | "
                  f"TEST acc={best['test_accuracy']:.3f}, f1={best['test_f1']:.3f}")
        else:
            print(f"⚠ No valid result for {day}")

    if not per_day_best:
        print("❌ No days produced results; aborting summary.")
        return None
    return per_day_best, per_model_results
