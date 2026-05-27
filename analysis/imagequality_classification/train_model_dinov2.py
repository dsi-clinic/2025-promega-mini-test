#!/usr/bin/env python3
"""DINOv2 / ResNet / EfficientNet image-quality classifier with fixed splits.

Reads directly from data/all_data.json via pipeline.data_loader.OrganoidDataset
(paper-default filter preset: BA1+BA2, complete metabolites, valid images,
≥4/5 vote consensus at Dy30). Splits come from data/splits/canonical_2026_winter.csv via Splits.from_csv.

Image set up aligned with `analysis/paper_2026_04/perday_image_study.py`:
    - Default image source: cm_source_image_abs (aspect-ratio-conserved 575x575
      from resized_575_square/), not the legacy img_path (mean-filled 512x384).
    - Model input size: (384, 512) —  resizes the 575x575 source down
    - No filter applied 
 
Label convention (matches pipeline.data_loader.LABEL_TO_INT):
    Not Acceptable = 1 (positive class in our code)
    Acceptable     = 0 (negative class in our code)
 
Paper Table 2 metric convention (FLIPPED from above):
    The paper treats Acceptable as positive and Not Acceptable as negative,
    so paper's "TNR" = recall on Not Acceptable, paper's "F1(NA)" = F1 of
    the (paper's) negative class = F1 of Not Acceptable. The confusion
    matrix counts saved to metrics_test.json (tn/fp/fn/tp) follow the
    PAPER convention so the printed table matches the paper directly.
 
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

from pipeline.data_loader import OrganoidDataset
from pipeline.splits import Splits

from .cli import build_results_table, create_summary, day_to_int
from .data import ImagePathDataset, extract_samples_by_day
from .eval import evaluate_on_loader, safe_roc_auc
from .models import BACKBONES_DINOV2, DEVICE, ImageOnlyClassifier
from .plots import plot_training_curve
from .train import FocalLoss, run_phases, set_seed

ALL_DATA_PATH = Path("data/all_data.json")
SPLITS_CSV = Path("data/splits/canonical_2026_winter.csv")
OUT_ROOT = Path("analysis/outputs/imagequality_classification/dinov2_fixed_splits")
TARGET_SIZE = (384, 512)  # (H, W) — model input size 
BATCH_SIZE = 16
NUM_WORKERS = 0
SEED = 1

# Paper Table 2 reports "Early TNR" averaged over these early-timepoint days.
EARLY_DAYS = {"Dy03", "Dy06", "Dy08", "Dy10"}


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

    # Confusion matrix in PAPER convention (Acc=positive, NA=negative).
    # Our internal LABEL_TO_INT is NA=1/Acc=0, so we flip when counting:
    #   paper TN = NA correctly identified  -> trues==1 & preds==1
    #   paper FP = NA missed (called Acc)   -> trues==1 & preds==0
    #   paper FN = Acc called NA            -> trues==0 & preds==1
    #   paper TP = Acc correctly identified -> trues==0 & preds==0
    # With these: paper_TNR = TN/(TN+FP) = recall on NA, which is what the
    # paper's Table 2 reports as "TNR". Same for F1(NA).
    TN = int(((trues == 1) & (preds_bin == 1)).sum())
    FP = int(((trues == 1) & (preds_bin == 0)).sum())
    FN = int(((trues == 0) & (preds_bin == 1)).sum())
    TP = int(((trues == 0) & (preds_bin == 0)).sum())

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
        "tn": TN, "fp": FP, "fn": FN, "tp": TP,
        "confusion_convention": "paper (Acc=positive, NA=negative)",
        "batch_size_train": int(train_bs), "batch_size_valtest": int(val_bs),
        "backbone_key": backbone_key, "input_key": input_key, "use_mask": use_mask,
    }
    (model_dir / "metrics_test.json").write_text(json.dumps(test_metrics, indent=2))
    print(f"Saved metrics -> {model_dir / 'metrics_val.json'} and {model_dir / 'metrics_test.json'}")

    return {
        "day": day_str, "day_no": day_no, "backbone_key": backbone_key,
        "val_accuracy": float(best_acc),
        "test_accuracy": float(test_acc), "test_f1": float(test_f1),
        "val_roc_auc": val_roc_auc, "test_roc_auc": test_roc_auc,
        "val_num": int(len(val_labels)), "test_num": int(len(trues)),
        "test_actual_good": actual_good, "test_pred_good": predicted_good,
        "tn": TN, "fp": FP, "fn": FN, "tp": TP,
    }


def _per_day_metrics(TN: int, FP: int, FN: int, TP: int) -> dict:
    """Compute paper Table 2 per-day metrics from PAPER-convention counts.

    Inputs (TN/FP/FN/TP) follow the paper's convention:
        Acc = positive (paper "TP" = Acc correctly identified)
        NA  = negative (paper "TN" = NA  correctly identified)

    Computes:
      - TNR = recall on NA  = TN / (TN + FP)
      - TPR = recall on Acc = TP / (TP + FN)
      - Bal.Acc = (TNR + TPR) / 2
      - F1(NA)  = F1 of the (paper's) negative class = F1 of Not Acceptable
                prec_NA = TN / (TN + FN)
                rec_NA  = TN / (TN + FP)
    """
    tnr = TN / (TN + FP) if (TN + FP) > 0 else 0.0  # recall(NA)
    tpr = TP / (TP + FN) if (TP + FN) > 0 else 0.0  # recall(Acc)
    bal_acc = (tnr + tpr) / 2
    prec_na = TN / (TN + FN) if (TN + FN) > 0 else 0.0
    rec_na = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    f1_na = (2 * prec_na * rec_na / (prec_na + rec_na)
             if (prec_na + rec_na) > 0 else 0.0)
    return {"tnr": tnr, "tpr": tpr, "bal_acc": bal_acc, "f1_na": f1_na}


def _compute_paper_metrics(per_model_results: dict) -> dict:
    """Compute paper Table 2 aggregate metrics per backbone."""
    aggregated = {}
    for bk, day_res in per_model_results.items():
        if not day_res:
            continue
        per_day = []
        for day, r in day_res.items():
            m = _per_day_metrics(r["tn"], r["fp"], r["fn"], r["tp"])
            m["day"] = day
            m["is_early"] = day in EARLY_DAYS
            per_day.append(m)
        early = [m["tnr"] for m in per_day if m["is_early"]]
        aggregated[bk] = {
            "avg_tnr": float(np.mean([m["tnr"] for m in per_day])),
            "early_tnr": float(np.mean(early)) if early else None,
            "bal_acc": float(np.mean([m["bal_acc"] for m in per_day])),
            "days_tnr_zero": sum(1 for m in per_day if m["tnr"] == 0.0),
            "f1_na": float(np.mean([m["f1_na"] for m in per_day])),
            "n_days": len(per_day),
            "n_early": len(early),
        }
    return aggregated


def _print_paper_metrics_table(metrics: dict) -> None:
    """Print paper Table 2-style aggregate per backbone."""
    print("\n--- Paper Table 2 metrics (aggregated per backbone) ---")
    print("Convention: Acc=positive, NA=negative (matches paper)")
    print("  TNR    = recall on Not Acceptable")
    print("  F1(NA) = F1 of Not Acceptable class")
    print()
    print(f"{'Backbone':>14} | {'Avg.TNR':>8} | {'EarlyTNR':>9} | {'Bal.Acc':>8} | "
          f"{'DaysTNR=0':>10} | {'F1(NA)':>7}")
    print("-" * 75)
    for bk, m in metrics.items():
        early_s = f"{m['early_tnr']*100:>7.1f}%" if m["early_tnr"] is not None else "    N/A "
        print(f"{bk:>14} | {m['avg_tnr']*100:>6.1f}% | {early_s:>9} | "
              f"{m['bal_acc']*100:>6.1f}% | {m['days_tnr_zero']:>3}/{m['n_days']:<5} | "
              f"{m['f1_na']*100:>5.1f}%")
    print(f"\nEarly days = {sorted(EARLY_DAYS)} (n_early per backbone may be < 4 "
          f"if some early days were skipped)")


@dataclasses.dataclass
class _SummaryCfg:
    """Adapter so the shared cli.create_summary's cfg.* attrs work for fixed splits."""

    out_dir: Path
    batch_size: int
    val_batch_size: int
    test_frac: float = 0.0
    val_frac: float = 0.0


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", default=OUT_ROOT, help="Where to save outputs")
    p.add_argument("--all-data", default=ALL_DATA_PATH, help="Path to all_data.json")
    p.add_argument("--splits-csv", default=SPLITS_CSV, help="Path to organoid splits CSV")
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Train batch size")
    p.add_argument("--val-batch-size", type=int, default=None, help="Val/Test batch size (defaults to train)")
    p.add_argument("--use-mask", action="store_true", help="Include mask tensors and a mask branch")
    p.add_argument("--input-path-key",
                   choices=["cm_source_image", "img_path", "overlay_path"],
                   default="cm_source_image",
                   help="Image source. cm_source_image = AR-conserved 575x575. "
                        "img_path = legacy mean-filled 512x384. overlay_path = QC overlays.")
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

    # No filter — matches perday_image_study.py.
    ds = OrganoidDataset(
        str(all_data_path),
        splits=Splits.from_csv(splits_csv),
    )
    print(ds.summary())

    train_bs = int(args.batch_size)
    val_bs = int(args.val_batch_size) if args.val_batch_size is not None else train_bs
    use_mask = bool(args.use_mask)
    input_key = str(args.input_path_key)

    print(f"Using batch sizes - train: {train_bs}, val/test: {val_bs}")
    print(f"Target size (HxW): {TARGET_SIZE}")
    print(f"Input field: {input_key}; masks enabled: {use_mask}")

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
    )

    paper_metrics = _compute_paper_metrics(per_model_results)
    _print_paper_metrics_table(paper_metrics)


if __name__ == "__main__":
    main()

