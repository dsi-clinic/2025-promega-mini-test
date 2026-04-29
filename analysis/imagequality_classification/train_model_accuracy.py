#!/usr/bin/env python3
"""Image-quality classifier: top-level orchestrator.

The work lives in sibling modules:
    cli.py     — Config dataclass, arg parser, summary builders
    data.py    — ImageClassifierEmitter, ImagePathDataset, file-existence filter
    models.py  — SmallCNNBackbone, MaskBranch, ImageOnlyClassifier, BACKBONES
    train.py   — set_seed, run_phases, run_training_for_day, collect_results
    eval.py    — evaluate_on_loader, get_validation_metrics, get_test_metrics
    plots.py   — plot_training_curve, plot_metric

Invoked via ``make step17`` (or ``analysis-train-dinov2`` for the DINOv2 variant).
"""

import datetime
import json

from .cli import build_results_table, create_summary, get_args, print_config_stats
from .data import load_image_classifier_views
from .train import collect_results, set_deterministic, set_seed


def main() -> None:
    start = datetime.datetime.now()
    cfg = get_args()
    for k, v in vars(cfg).items():
        print(f"{k}: {v}")

    set_deterministic(cfg.deterministic)
    set_seed(cfg.seed, cfg.deterministic)
    print_config_stats(cfg)

    views = load_image_classifier_views(cfg.all_data_json)
    with open(cfg.image_classifier_json, "w") as f:
        json.dump(views, f, indent=2)
    print(f"Saved image classifier views to {cfg.image_classifier_json}")

    total = sum(len(d.get("img_path", [])) for d in views.get("records", {}).values())
    skipped = views.get("metadata", {}).get("total_skipped")
    print(f"Loaded {total} image classifier views from image_classifier.json")
    print(f"Skipped {skipped} records without processed image paths, evaluations, or labels")

    result = collect_results(views, cfg)
    if result is None:
        return
    per_day_best, per_model_results = result

    rows = build_results_table(per_day_best, cfg)
    create_summary(per_model_results, rows, cfg)

    print(f"Execution time: {datetime.datetime.now() - start}")


if __name__ == "__main__":
    main()
