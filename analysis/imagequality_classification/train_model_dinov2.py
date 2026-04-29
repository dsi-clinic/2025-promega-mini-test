#!/usr/bin/env python3
"""DINOv2 / ResNet / EfficientNet image-quality classifier with fixed splits.

Reads directly from data/all_data.json via pipeline.data_loader.OrganoidDataset
(paper-default filter preset: BA1+BA2, complete metabolites, valid images,
≥4/5 vote consensus at Dy30). Splits come from data/2026_winter_student_splits.csv.

Focal loss + ReduceLROnPlateau scheduler. Shares backbone/head/dataset/loop
plumbing with train_model_accuracy.py via the sibling modules:
    models.py — ImageOnlyClassifier (with DINOv2 branch), EarlyStopping
    data.py   — ImagePathDataset (with ImageNet-norm flag)
    train.py  — set_seed, set_deterministic, epoch_loop, FocalLoss
    eval.py   — evaluate_on_loader
    plots.py  — plot_training_curve, plot_metric
    cli.py    — day_to_int

Invoked via ``make analysis-train-dinov2``.

(Previously: train_model_accuracy_tony_dinov2.py)
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from pipeline.data_loader import IMAGE_MODE_TO_PATH_KEY, OrganoidDataset, filters_for_mode

from .cli import day_to_int
from .data import ImagePathDataset
from .eval import evaluate_on_loader, safe_roc_auc
from .models import BACKBONES_DINOV2, DEVICE, EarlyStopping, ImageOnlyClassifier
from .plots import plot_metric, plot_training_curve
from .train import FocalLoss, epoch_loop, set_seed

ALL_DATA_PATH = Path("data/all_data.json")
SPLITS_CSV = Path("data/2026_winter_student_splits.csv")
OUT_ROOT = Path("analysis/outputs/imagequality_classification/dinov2_fixed_splits")
TARGET_SIZE = (384, 512)  # (H, W) — torchvision Resize convention
BATCH_SIZE = 16
NUM_WORKERS = 0
SEED = 1

LABEL_MAP = {"Acceptable": 1, "Not Acceptable": 0}
PATH_KEY_TO_IMAGE_MODE = {v: k for k, v in IMAGE_MODE_TO_PATH_KEY.items()}


def make_loader(imgs, labels, augment, batch_size, mask_paths=None, use_mask=False, normalize=False):
    ds = ImagePathDataset(
        imgs, labels, target_size=TARGET_SIZE,
        mask_paths=mask_paths, augment=augment, use_mask=use_mask, normalize=normalize,
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=augment, num_workers=NUM_WORKERS)


def extract_samples_by_day(ds: OrganoidDataset, split: str, day_str: str,
                           input_key: str = "img_path", use_mask: bool = False):
    """Pull (img_paths, labels, mask_paths) for one split+day from an OrganoidDataset."""
    img_mode = PATH_KEY_TO_IMAGE_MODE.get(input_key, input_key)
    img_triples = ds.get_image_paths(split, day_str, mode=img_mode)
    mask_lookup = {
        org_id: path for org_id, _, path in ds.get_image_paths(split, day_str, mode="mask")
    } if use_mask else None

    img_paths, labels, mask_paths = [], [], ([] if use_mask else None)
    for org_id, label_str, img_path in img_triples:
        if label_str not in LABEL_MAP:
            continue
        if not img_path or not Path(img_path).exists():
            continue
        if use_mask:
            mpath = mask_lookup.get(org_id)
            if not mpath or not Path(mpath).exists():
                continue
            mask_paths.append(mpath)
        img_paths.append(img_path)
        labels.append(LABEL_MAP[label_str])

    return (
        np.array(img_paths),
        np.array(labels, dtype=int),
        np.array(mask_paths) if use_mask else None,
    )


def run_phases(model, model_path, backbone_key, day, train_loader, val_loader,
               class_weights, train_bs, val_bs, use_mask):
    """Phase 1 (frozen) → Phase 2 (partial unfreeze) with focal loss + ReduceLROnPlateau."""
    focal = FocalLoss(gamma=2.0, alpha=0.25)
    history = defaultdict(list)
    best_acc = -np.inf

    phases = [(1, 100, 1e-3, 20), (2, 300, 1e-4, 30)]
    for phase, n_epochs, lr, patience in phases:
        if phase == 2:
            model.unfreeze_backbone()
        opt = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
        scheduler = ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=10, min_lr=1e-7)
        es = EarlyStopping(patience=patience)

        for epoch in range(n_epochs):
            tl, tacc, _, _ = epoch_loop(model, train_loader, opt, class_weights,
                                        train=True, use_mask=use_mask, loss_fn=focal)
            vl, vacc, _, _ = epoch_loop(model, val_loader, opt, class_weights,
                                        train=False, use_mask=use_mask, loss_fn=focal)
            history["train_loss"].append(tl)
            history["val_loss"].append(vl)
            history["train_acc"].append(tacc)
            history["val_acc"].append(vacc)
            print(f"[{day}][{backbone_key}][P{phase}][{epoch:03d}][bs={train_bs}/{val_bs}] "
                  f"loss {tl:.4f}/{vl:.4f} acc {tacc:.3f}/{vacc:.3f}")
            if vacc > best_acc:
                best_acc = vacc
                torch.save(model.state_dict(), model_path)
            scheduler.step(vacc)
            if es.step(vacc):
                break
    return history, best_acc


def run_training_for_day(day_str, backbone_key, backbone_name, ds, train_bs, val_bs,
                         out_root, input_key, use_mask):
    """Train on fixed splits; select by VAL acc, report on TEST."""
    train_imgs, train_labels, train_masks = extract_samples_by_day(ds, "train", day_str, input_key, use_mask)
    val_imgs,   val_labels,   val_masks   = extract_samples_by_day(ds, "val",   day_str, input_key, use_mask)
    test_imgs,  test_labels,  test_masks  = extract_samples_by_day(ds, "test",  day_str, input_key, use_mask)

    if len(train_imgs) == 0 or len(val_imgs) == 0 or len(test_imgs) == 0:
        print(f"Skipping {day_str}: empty split (train={len(train_imgs)}, val={len(val_imgs)}, test={len(test_imgs)})")
        return None

    weights = compute_class_weight("balanced", classes=np.unique(train_labels), y=train_labels)
    class_weights = {int(k): float(w) for k, w in zip(np.unique(train_labels), weights)}

    use_imagenet_norm = backbone_key == "dinov2"
    train_loader = make_loader(train_imgs, train_labels, mask_paths=train_masks, augment=True,
                               batch_size=train_bs, use_mask=use_mask, normalize=use_imagenet_norm)
    val_loader   = make_loader(val_imgs,   val_labels,   mask_paths=val_masks,   augment=False,
                               batch_size=val_bs,   use_mask=use_mask, normalize=use_imagenet_norm)
    test_loader  = make_loader(test_imgs,  test_labels,  mask_paths=test_masks,  augment=False,
                               batch_size=val_bs,   use_mask=use_mask, normalize=use_imagenet_norm)

    model = ImageOnlyClassifier(backbone_key, backbone_name, TARGET_SIZE, use_mask=use_mask).to(DEVICE)
    model_dir = out_root / backbone_key / day_str
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "model.pth"

    history, best_acc = run_phases(model, model_path, backbone_key, day_str,
                                   train_loader, val_loader, class_weights,
                                   train_bs, val_bs, use_mask)
    plot_training_curve(history, model_dir)

    model.load_state_dict(torch.load(model_path, map_location=DEVICE))

    _, val_trues, val_acc, val_f1, val_probs = evaluate_on_loader(model, val_loader, use_mask=use_mask)
    val_pr_auc = float(average_precision_score(val_trues, val_probs)) if len(val_trues) else None
    val_roc_auc = safe_roc_auc(val_trues, val_probs)

    preds_bin, trues, test_acc, test_f1, test_probs = evaluate_on_loader(model, test_loader, use_mask=use_mask)
    test_pr_auc = float(average_precision_score(trues, test_probs)) if len(trues) else None
    test_roc_auc = safe_roc_auc(trues, test_probs)

    day_no = day_to_int(day_str)
    actual_good = int(trues.sum())
    predicted_good = int(preds_bin.sum())

    val_metrics = {
        "day": day_str, "split": "val",
        "accuracy": float(val_acc), "f1": float(val_f1),
        "roc_auc": val_roc_auc, "pr_auc": val_pr_auc,
        "n": int(len(val_labels)), "batch_size": int(val_bs),
        "input_key": input_key, "use_mask": use_mask,
    }
    (model_dir / "metrics_val.json").write_text(json.dumps(val_metrics, indent=2))

    test_metrics = {
        "day": day_str, "day_no": day_no, "split": "test",
        "accuracy": float(test_acc), "f1": float(test_f1),
        "roc_auc": test_roc_auc, "pr_auc": test_pr_auc,
        "val_accuracy_for_selection": float(best_acc),
        "val_n": int(len(val_labels)), "test_n": int(len(trues)),
        "actual_good": actual_good, "predicted_good": predicted_good,
        "batch_size_train": int(train_bs), "batch_size_valtest": int(val_bs),
        "backbone_key": backbone_key, "input_key": input_key, "use_mask": use_mask,
    }
    (model_dir / "metrics_test.json").write_text(json.dumps(test_metrics, indent=2))
    print(f"Saved metrics → {model_dir / 'metrics_val.json'} and {model_dir / 'metrics_test.json'}")

    return {
        "day": day_str, "day_no": day_no, "backbone_key": backbone_key,
        "val_accuracy": float(best_acc),
        "test_accuracy": float(test_acc), "test_f1": float(test_f1),
        "val_roc_auc": val_roc_auc, "test_roc_auc": test_roc_auc,
        "val_num": int(len(val_labels)), "test_num": int(len(trues)),
        "test_actual_good": actual_good, "test_pred_good": predicted_good,
    }


def write_summary(per_day_best, per_model_results, out_dir, train_bs, val_bs, mode):
    """4-column day_summary.csv + 3 metric plots + final_test_summary.json."""
    rows = [
        {
            "Day No": per_day_best[d]["day_no"],
            "Num in Sample": per_day_best[d]["test_num"],
            "Actual Good": per_day_best[d]["test_actual_good"],
            "Predicted Good": per_day_best[d]["test_pred_good"],
        }
        for d in sorted(per_day_best.keys(), key=day_to_int)
    ]
    table_path = out_dir / "day_summary.csv"
    with table_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Day No", "Num in Sample", "Actual Good", "Predicted Good"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved per-day summary table → {table_path}")

    day_numbers = {day: res["day_no"] for day_res in per_model_results.values() for day, res in day_res.items()}
    if day_numbers:
        unique_day_nos = sorted(set(day_numbers.values()))
        for metric, ylabel, title, filename in [
            ("test_accuracy", "Accuracy (test)", "Per-day Test Accuracy by Backbone", "accuracy_by_model.png"),
            ("test_f1",       "F1 score (test)", "Per-day Test F1 by Backbone",       "f1_by_model.png"),
            ("test_roc_auc",  "ROC AUC (test)",  "Per-day Test ROC AUC by Backbone",  "rocauc_by_model.png"),
        ]:
            plot_metric(metric, ylabel, title, filename, per_model_results,
                        day_numbers, unique_day_nos, out_dir)

    per_model_summary = {
        bk: {
            "per_day": {
                day: {
                    "day_no": int(day_numbers.get(day, res["day_no"])),
                    "test_accuracy": float(res["test_accuracy"]),
                    "test_f1": float(res["test_f1"]),
                    "test_roc_auc": (None if res["test_roc_auc"] is None else float(res["test_roc_auc"])),
                    "val_accuracy": float(res["val_accuracy"]),
                    "val_roc_auc": (None if res["val_roc_auc"] is None else float(res["val_roc_auc"])),
                    "test_num": int(res["test_num"]),
                }
                for day, res in day_res.items()
            }
        }
        for bk, day_res in per_model_results.items()
    }
    summary = {
        "per_model": per_model_summary,
        "batch_size_train": int(train_bs),
        "batch_size_valtest": int(val_bs),
        "used_fixed_splits": True,
        "mode": mode,
    }
    (out_dir / "final_test_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Saved final test summary → {out_dir / 'final_test_summary.json'}")

    print("\n=== Summary Table (TEST) ===")
    print(f"{'Day No':>6} | {'Num in Sample':>13} | {'Actual Good':>11} | {'Predicted Good':>14}")
    print("-" * 54)
    for row in rows:
        print(f"{row['Day No']:>6} | {row['Num in Sample']:>13} | {row['Actual Good']:>11} | {row['Predicted Good']:>14}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", default=OUT_ROOT, help="Where to save outputs")
    p.add_argument("--all-data", default=ALL_DATA_PATH, help="Path to all_data.json")
    p.add_argument("--splits-csv", default=SPLITS_CSV, help="Path to organoid splits CSV")
    p.add_argument("--mode", default="base", choices=["base", "switch1", "switch2", "switch3"],
                   help="Split mode preset (see filters_for_mode)")
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Train batch size")
    p.add_argument("--val-batch-size", type=int, default=None, help="Val/Test batch size (defaults to train)")
    p.add_argument("--use-mask", action="store_true", help="Include mask tensors and a mask branch")
    p.add_argument("--input-path-key", choices=["img_path", "overlay_path"], default="img_path",
                   help="Which JSON field to use as the primary image input")
    return p.parse_args()


def main():
    set_seed(SEED, deterministic=False)
    args = parse_args()

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_data_path = Path(args.all_data)
    splits_csv = Path(args.splits_csv)
    if not all_data_path.exists():
        raise FileNotFoundError(f"all_data.json not found: {all_data_path}")
    if not splits_csv.exists():
        raise FileNotFoundError(f"splits CSV not found: {splits_csv}")

    ds = OrganoidDataset(
        str(all_data_path),
        splits_csv=str(splits_csv),
        filters=filters_for_mode(args.mode, modality="image"),
    )
    print(ds.summary())

    train_bs = int(args.batch_size)
    val_bs = int(args.val_batch_size) if args.val_batch_size is not None else train_bs
    use_mask = bool(args.use_mask)
    input_key = str(args.input_path_key)

    print(f"Using batch sizes — train: {train_bs}, val/test: {val_bs}")
    print(f"Target size (HxW): {TARGET_SIZE}")
    print(f"Input field: {input_key}; masks enabled: {use_mask}")
    print(f"Mode preset: {args.mode} (modality=image)")

    all_days = ds.days
    print(f"Found {len(all_days)} days: {', '.join(all_days)}")

    per_day_best = {}
    per_model_results = {bk: {} for bk in BACKBONES_DINOV2}
    for day_str in all_days:
        best = None
        for backbone_key, backbone_name in BACKBONES_DINOV2.items():
            res = run_training_for_day(day_str, backbone_key, backbone_name, ds,
                                       train_bs, val_bs, out_dir, input_key, use_mask)
            if res is None:
                continue
            per_model_results[backbone_key][day_str] = res
            if (best is None) or (res["val_accuracy"] > best["val_accuracy"]):
                best = res
        if best:
            per_day_best[day_str] = best
            print(f"Best for {day_str} (by VAL): {best['backbone_key']} | "
                  f"val acc={best['val_accuracy']:.3f} | "
                  f"TEST acc={best['test_accuracy']:.3f}, f1={best['test_f1']:.3f}")
        else:
            print(f"No valid result for {day_str}")

    if not per_day_best:
        print("No days produced results; aborting summary.")
        return
    write_summary(per_day_best, per_model_results, out_dir, train_bs, val_bs, args.mode)


if __name__ == "__main__":
    main()
