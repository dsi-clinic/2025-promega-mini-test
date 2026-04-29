#!/usr/bin/env python3
"""Multimodal Organoid Quality Classification — top-level orchestrator.

The work lives in sibling modules:
    cli.py    — argparse + config dict
    data.py   — MultimodalRowDataset + load_and_prepare_data + transforms
    models.py — MaskBranch, MetaboliteBranch, MultimodalClassifier, EarlyStopping
    train.py  — train_epoch, eval_epoch, pretrain_shared_backbone, train_for_day
    plots.py  — per-day curves + cross-day metric panels

Invoked via ``make analysis-multimodal``.
"""

import json

import numpy as np
import pandas as pd
import torch

from .cli import build_config, parse_args, print_config
from .data import day_to_int, load_and_prepare_data
from .plots import plot_metrics_by_day, plot_training_curves
from .train import pretrain_shared_backbone, train_for_day

SEED = 1


def set_seed(seed: int = SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_organoid_predictions(test_res, test_df) -> list:
    """Build per-organoid CSV rows from eval_epoch's preds/labels arrays."""
    preds = test_res["preds"]
    labels = test_res["labels"].astype(int)
    preds_bin = (preds > 0.5).astype(int)
    rows = []
    for idx in range(len(test_df)):
        true = int(labels[idx])
        pred = int(preds_bin[idx])
        if true == 1 and pred == 1:
            cm_cat = "TP"
        elif true == 0 and pred == 1:
            cm_cat = "FP"
        elif true == 1 and pred == 0:
            cm_cat = "FN"
        else:
            cm_cat = "TN"
        rows.append({
            "Organoid_ID": test_df.iloc[idx]["org_id"],
            "True_Label": true,
            "Predicted_Probability": float(preds[idx]),
            "Predicted_Label": pred,
            "Correct": pred == true,
            "CM_Category": cm_cat,
        })
    return rows


def _save_day_artifacts(day, test_res, day_dir):
    """Write per-day model.pth, organoid_predictions.csv, metrics_test.json, training_curves.png."""
    day_dir.mkdir(parents=True, exist_ok=True)
    torch.save(test_res["model_state"], day_dir / "model.pth")

    org_rows = _build_organoid_predictions(test_res, test_res["test_df"])
    pd.DataFrame(org_rows).to_csv(day_dir / "organoid_predictions.csv", index=False)
    print(f"Saved per-organoid predictions to {day_dir / 'organoid_predictions.csv'}")

    cm = test_res["confusion_matrix"]
    metrics = {
        "day": day,
        "test_acc": float(test_res["acc"]),
        "test_f1": float(test_res["f1"]),
        "test_recall": float(test_res["recall"]),
        "test_precision": float(test_res["precision"]),
        "test_specificity": float(test_res["specificity"]),
        "test_auc": float(test_res["auc"]) if test_res["auc"] else None,
        "test_pr_auc": float(test_res["pr_auc"]) if test_res["pr_auc"] else None,
        "test_acc_opt": float(test_res["acc_opt"]),
        "test_f1_opt": float(test_res["f1_opt"]),
        "opt_thresh": float(test_res["opt_thresh"]),
        "val_acc": float(test_res["best_val_acc"]),
        "confusion_matrix": cm,
    }
    (day_dir / "metrics_test.json").write_text(json.dumps(metrics, indent=2))
    plot_training_curves(test_res["history"], day, day_dir)


def _build_summary_row(day, test_res, config) -> dict:
    cm = test_res["confusion_matrix"]
    return {
        "Day": day,
        "Day_No": day_to_int(day),
        "Backbone": config["backbone"],
        "Test_Accuracy": test_res["acc"],
        "Test_F1": test_res["f1"],
        "Test_Recall": test_res["recall"],
        "Test_Precision": test_res["precision"],
        "Test_Specificity": test_res["specificity"],
        "Test_ROC_AUC": test_res["auc"] if test_res["auc"] else None,
        "TP": cm["TP"], "FP": cm["FP"], "TN": cm["TN"], "FN": cm["FN"],
    }


def _update_master_csv(summary: pd.DataFrame, config: dict, output_dir):
    """Append/replace this run's rows in outputs_multimodal/overall/master_results.csv."""
    overall_dir = output_dir.parent / "overall"
    overall_dir.mkdir(parents=True, exist_ok=True)
    master_csv = overall_dir / "master_results.csv"

    model_id = f"{config['backbone']}_{config['input_mode']}_{config['fusion_strategy']}"
    summary["Model_ID"] = model_id
    summary["Input_Mode"] = config["input_mode"]
    summary["Fusion_Strategy"] = config["fusion_strategy"]
    summary["Use_Metabolites"] = config["use_metabolites"]

    cols = ["Model_ID", "Backbone", "Input_Mode", "Fusion_Strategy", "Use_Metabolites",
            "Day", "Day_No", "Test_Accuracy", "Test_F1", "Test_Recall", "Test_Precision",
            "Test_Specificity", "Test_ROC_AUC", "TP", "FP", "TN", "FN"]
    summary = summary[cols]

    if master_csv.exists():
        existing = pd.read_csv(master_csv)
        existing = existing[existing["Model_ID"] != model_id]
        master_df = pd.concat([existing, summary], ignore_index=True)
    else:
        master_df = summary
    master_df = master_df.sort_values(["Model_ID", "Day_No"])
    master_df.to_csv(master_csv, index=False)
    print(f"\nMaster results updated at {master_csv}")


def main():
    from pathlib import Path

    args = parse_args()
    config = build_config(args)
    set_seed()
    print_config(config)

    print("Loading data splits...")
    train_df, val_df, test_df = load_and_prepare_data(config)
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    shared_state, shared_scaler = pretrain_shared_backbone(train_df, val_df, config)

    days_to_train = args.days if args.days else sorted(train_df["day"].unique(), key=day_to_int)
    print(f"Days to train: {days_to_train}\n")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for day in days_to_train:
        res = train_for_day(day, train_df, val_df, test_df, config, output_dir,
                            shared_state, shared_scaler)
        if res is None:
            continue
        _save_day_artifacts(day, res, output_dir / day)
        summary_rows.append(_build_summary_row(day, res, config))

    if not summary_rows:
        return

    summary = pd.DataFrame(summary_rows).sort_values("Day_No")
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(summary.to_string(index=False))

    summary.to_csv(output_dir / "results_summary.csv", index=False)
    _update_master_csv(summary.copy(), config, output_dir)
    plot_metrics_by_day(summary, output_dir)
    print(f"\nResults saved to {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
