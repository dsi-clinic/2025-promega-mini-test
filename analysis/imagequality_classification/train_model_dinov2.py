#!/usr/bin/env python3
"""DINOv2 / ResNet / EfficientNet image-quality classifier with fixed splits.

Reads directly from data/all_data.json via pipeline.data_loader.OrganoidDataset
(paper-default filter preset: BA1+BA2, complete metabolites, valid images,
≥4/5 vote consensus at Dy30). Splits come from data/splits/canonical_2026_winter.csv via Splits.from_csv.

Focal loss + ReduceLROnPlateau scheduler — both threaded through the shared
``train.run_phases`` via its ``loss_fn`` / ``scheduler_factory`` parameters.
The rest of the plumbing (extract_samples_by_day, build_results_table,
create_summary, plot_training_curve, ImagePathDataset, ImageOnlyClassifier,
EarlyStopping) lives in the sibling modules.

Invoked via ``make analysis-train-dinov2``.
"""

import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from sklearn.utils.class_weight import compute_class_weight
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from pipeline.data_loader import OrganoidDataset, filters_for_mode
from pipeline.splits import CANONICAL_PATH, Splits

from .cli import build_results_table, create_summary, day_to_int
from .data import ImagePathDataset, extract_samples_by_day
from .eval import evaluate_on_loader, safe_roc_auc
from .models import BACKBONES_DINOV2, DEVICE, ImageOnlyClassifier
from .plots import plot_training_curve
from .train import FocalLoss, run_phases, set_seed

ALL_DATA_PATH = Path("data/all_data.json")
SPLITS_CSV = CANONICAL_PATH
OUT_ROOT = Path("analysis/outputs/imagequality_classification/dinov2_fixed_splits")
TARGET_SIZE = (384, 512)  # (H, W) — torchvision Resize convention
BATCH_SIZE = 16
NUM_WORKERS = 0
SEED = 1


@dataclasses.dataclass
class _PhasesCfg:
    """Minimal config shape for train.run_phases (uses these attrs only)."""

    epoch1: int = 100
    epoch2: int = 300
    use_mask: bool = False
    batch_size: int = BATCH_SIZE
    val_batch_size: int = BATCH_SIZE


def _make_loader(imgs, labels, *, augment, batch_size,
                 mask_paths=None, use_mask=False, normalize=False):
    ds = ImagePathDataset(
        imgs, labels, target_size=TARGET_SIZE,
        mask_paths=mask_paths, augment=augment, use_mask=use_mask, normalize=normalize,
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=augment, num_workers=NUM_WORKERS)


def _scheduler_factory(opt):
    return ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=10, min_lr=1e-7)


def run_training_for_day(day_str, backbone_key, backbone_name, ds, train_bs, val_bs,
                         out_root, input_key, use_mask):
    """Train on fixed splits; select by VAL acc, report on TEST."""
    train_imgs, train_labels, train_masks = extract_samples_by_day(
        ds, day_str, split="train", input_key=input_key, use_mask=use_mask)
    val_imgs, val_labels, val_masks = extract_samples_by_day(
        ds, day_str, split="val", input_key=input_key, use_mask=use_mask)
    test_imgs, test_labels, test_masks = extract_samples_by_day(
        ds, day_str, split="test", input_key=input_key, use_mask=use_mask)

    if len(train_imgs) == 0 or len(val_imgs) == 0 or len(test_imgs) == 0:
        print(f"Skipping {day_str}: empty split "
              f"(train={len(train_imgs)}, val={len(val_imgs)}, test={len(test_imgs)})")
        return None

    weights = compute_class_weight("balanced", classes=np.unique(train_labels), y=train_labels)
    class_weights = {int(k): float(w) for k, w in zip(np.unique(train_labels), weights)}

    use_imagenet_norm = backbone_key == "dinov2"
    train_loader = _make_loader(train_imgs, train_labels, mask_paths=train_masks,
                                augment=True, batch_size=train_bs,
                                use_mask=use_mask, normalize=use_imagenet_norm)
    val_loader = _make_loader(val_imgs, val_labels, mask_paths=val_masks,
                              augment=False, batch_size=val_bs,
                              use_mask=use_mask, normalize=use_imagenet_norm)
    test_loader = _make_loader(test_imgs, test_labels, mask_paths=test_masks,
                               augment=False, batch_size=val_bs,
                               use_mask=use_mask, normalize=use_imagenet_norm)

    model = ImageOnlyClassifier(backbone_key, backbone_name, TARGET_SIZE, use_mask=use_mask).to(DEVICE)
    model_dir = out_root / backbone_key / day_str
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "model.pth"

    phases_cfg = _PhasesCfg(use_mask=use_mask, batch_size=train_bs, val_batch_size=val_bs)
    history, best_acc = run_phases(
        model, model_path, backbone_key, backbone_name, day_str,
        train_loader, val_loader, class_weights, phases_cfg,
        loss_fn=FocalLoss(gamma=2.0, alpha=0.25),
        scheduler_factory=_scheduler_factory,
    )
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


@dataclasses.dataclass
class _SummaryCfg:
    """Adapter so the shared cli.create_summary's cfg.* attrs work for fixed splits."""

    out_dir: Path
    batch_size: int
    val_batch_size: int
    test_frac: float = 0.0  # not meaningful for fixed splits
    val_frac: float = 0.0


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
        splits=Splits.from_csv(splits_csv),
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

    summary_cfg = _SummaryCfg(out_dir=out_dir, batch_size=train_bs, val_batch_size=val_bs)
    rows = build_results_table(per_day_best, summary_cfg)
    create_summary(
        per_model_results, rows, summary_cfg,
        used_fixed_splits=True,
        mode=args.mode,
    )


if __name__ == "__main__":
    main()
