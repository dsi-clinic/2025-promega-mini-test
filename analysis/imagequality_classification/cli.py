#!/usr/bin/env python3
"""Config dataclass, CLI parser, summary builders, and main() for image quality classifier."""

import argparse
import csv
import dataclasses
import json
from pathlib import Path

from pipeline.data_loader import get_day_int_floor

from .plots import plot_metric


@dataclasses.dataclass
class Config:
    data_dir: Path = dataclasses.field(metadata={
        "help": "Path to base data directory (root) containing identifiers/, images/, classifiers/, etc."
    })
    all_data_json: Path = dataclasses.field(default=None, metadata={
        "help": "Path to all data JSON file (defaults to data_dir/identifiers/all_data.json)"
    })
    image_classifier_json: Path = dataclasses.field(default=None, metadata={
        "help": "Path to image classifier JSON file (defaults to data_dir/identifiers/image_classifier.json)"
    })
    epoch1: int = dataclasses.field(default=100, metadata={
        "help": "Number of training epochs for phase 1 (frozen backbone)"
    })
    epoch2: int = dataclasses.field(default=300, metadata={
        "help": "Number of training epochs for phase 2 (unfrozen backbone)"
    })
    batch_size: int = dataclasses.field(default=16, metadata={"help": "Training batch size"})
    val_batch_size: int = dataclasses.field(default=None, metadata={
        "help": "Validation/Test batch size (defaults to training batch size)"
    })
    test_frac: float = dataclasses.field(default=0.1, metadata={"help": "Fraction of data used for testing"})
    val_frac: float = dataclasses.field(default=0.1, metadata={"help": "Fraction of data used for validation"})
    use_mask: bool = dataclasses.field(default=False, metadata={
        "help": "Include mask tensors and a mask branch in the classifier"
    })
    input_path_key: str = dataclasses.field(default="img_path", metadata={
        "help": "Which JSON field to use as the primary image input ('img_path' or 'overlay_path')"
    })
    target_width: int = dataclasses.field(default=512, metadata={"help": "Target input image width (pixels)"})
    target_height: int = dataclasses.field(default=384, metadata={"help": "Target input image height (pixels)"})
    num_workers: int = dataclasses.field(default=0, metadata={
        "help": "Number of subprocesses for data loading (0 = main process)"
    })
    seed: int = dataclasses.field(default=1, metadata={"help": "Random seed for reproducibility"})
    deterministic: bool = dataclasses.field(default=False, metadata={
        "help": "Use deterministic operations for reproducibility"
    })

    def __post_init__(self):
        if not (0.0 < self.test_frac < 0.5):
            raise ValueError("test-frac must be in (0, 0.5)")
        if not (0.0 < self.val_frac < 0.5):
            raise ValueError("val-frac must be in (0, 0.5)")
        if not (self.val_frac + self.test_frac < 0.9):
            raise ValueError("Sum of val-frac and test-frac too large.")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0")

        self.out_dir = self.data_dir / "models" / "imagequality_classification"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.val_batch_size = int(self.val_batch_size) if self.val_batch_size is not None else self.batch_size
        self.target_size: tuple = (self.target_height, self.target_width)  # (H, W)

        self.all_data_json = (
            Path(self.all_data_json) if self.all_data_json
            else self.data_dir / "identifiers" / "all_data.json"
        )
        self.image_classifier_json = (
            Path(self.image_classifier_json) if self.image_classifier_json
            else self.data_dir / "identifiers" / "image_classifier.json"
        )


def create_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run image classifier on organoid images")
    for field in dataclasses.fields(Config):
        flags = [f"--{field.name.replace('_', '-')}"]
        kwargs = {"help": field.metadata.get("help", ""), "default": field.default}
        if field.type == bool:
            kwargs["action"] = "store_true" if field.default is False else "store_false"
        else:
            kwargs["type"] = field.type
        parser.add_argument(*flags, **kwargs)
    return parser


def get_args() -> Config:
    return Config(**vars(create_args().parse_args()))


def print_config_stats(cfg: Config) -> None:
    print(f"🧪 Using batch sizes — train: {cfg.batch_size}, val/test: {cfg.val_batch_size}")
    print(f"🔀 Split fractions — train: {1.0 - cfg.test_frac - cfg.val_frac:.2f}, "
          f"val: {cfg.val_frac:.2f}, test: {cfg.test_frac:.2f}")
    print(f"🖼️ Target size (HxW): {cfg.target_size}")
    print(f"🗂️ Input field: {cfg.input_path_key}; masks enabled: {cfg.use_mask}")


def day_to_int(day_str: str) -> int:
    """Day → int for sorting/CSV display. Wraps pipeline.data_loader.get_day_int_floor with a -1 fallback."""
    n = get_day_int_floor(day_str)
    return -1 if n is None else n


def build_results_table(per_day_best, cfg: Config):
    """4-column per-day TEST summary CSV."""
    rows = []
    for d in sorted(per_day_best.keys(), key=day_to_int):
        r = per_day_best[d]
        rows.append({
            "Day No": r["day_no"],
            "Num in Sample": r["test_num"],
            "Actual Good": r["test_actual_good"],
            "Predicted Good": r["test_pred_good"],
        })
    table_path = cfg.out_dir / "day_summary.csv"
    with table_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["Day No", "Num in Sample", "Actual Good", "Predicted Good"]
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"🧾 Saved table → {table_path}")
    return rows


def create_summary(per_model_results, rows, cfg: Config, **extra_metadata) -> None:
    """Per-model JSON summary + 3 metric plots + stdout table.

    ``extra_metadata`` keys are merged into the top-level summary dict so
    callers (e.g. the dinov2 trainer) can record fixed-splits / mode markers
    without subclassing this function.
    """
    day_numbers = {
        day: res["day_no"]
        for day_res in per_model_results.values()
        for day, res in day_res.items()
    }

    if day_numbers:
        unique_day_nos = sorted(set(day_numbers.values()))
        for metric, ylabel, title, filename in [
            ("test_accuracy", "Accuracy (test)", "Per-day Test Accuracy by Backbone", "accuracy_by_model.png"),
            ("test_f1",       "F1 score (test)", "Per-day Test F1 by Backbone",       "f1_by_model.png"),
            ("test_roc_auc",  "ROC AUC (test)",  "Per-day Test ROC AUC by Backbone",  "rocauc_by_model.png"),
        ]:
            plot_metric(metric, ylabel, title, filename, per_model_results,
                        day_numbers, unique_day_nos, cfg.out_dir)

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
        "batch_size_train": int(cfg.batch_size),
        "batch_size_valtest": int(cfg.val_batch_size),
        "split_fractions": {
            "train": float(1.0 - cfg.test_frac - cfg.val_frac),
            "val": float(cfg.val_frac),
            "test": float(cfg.test_frac),
        },
        **extra_metadata,
    }
    summary_path = cfg.out_dir / "final_test_summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"✅ Saved final test summary → {summary_path}")

    print("\n=== Summary Table (TEST) ===")
    print(f"{'Day No':>6} | {'Num in Sample':>13} | {'Actual Good':>11} | {'Predicted Good':>14}")
    print("-" * 54)
    for row in rows:
        print(f"{row['Day No']:>6} | {row['Num in Sample']:>13} | {row['Actual Good']:>11} | {row['Predicted Good']:>14}")
