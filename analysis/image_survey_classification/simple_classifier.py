#!/usr/bin/env python3
"""Image-survey classifier (Step 18): top-level orchestrator.

The work lives in sibling modules:
    cli.py     — Config dataclass + arg parser
    data.py    — SurveyClassifierEmitter, load_survey_classifier_views,
                 extract_day_data, create_dataset, augment_data
    models.py  — initialize_model (ResNet50V2 + mask branch)
    metrics.py — weighted_f1_score_keras, macro_f1_score_keras
    train.py   — set_seed, set_deterministic, train_model
    eval.py    — evaluate_model, plot_confusion_matrix
    plots.py   — plot_model_metrics

Invoked via ``make step18``.
"""

import datetime
import json
from collections import Counter

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

from pipeline.data_loader import INT_TO_LABEL

from .cli import get_args
from .data import create_dataset, extract_day_data, load_survey_classifier_views
from .eval import evaluate_model, plot_confusion_matrix
from .models import initialize_model
from .plots import plot_model_metrics
from .train import set_deterministic, set_seed, train_model


def print_class_distribution(indexed_labels):
    print("\n--- Class Distribution (Before Split) ---")
    counts = Counter(indexed_labels)
    for class_idx, count in sorted(counts.items()):
        print(f"Class {class_idx} ('{INT_TO_LABEL[class_idx]}'): {count} samples")
    print("------------------------------------------")


def main():
    start_time = datetime.datetime.now()
    cfg = get_args()
    for k, v in vars(cfg).items():
        print(f"{k}: {v}")

    set_seed(cfg.seed)
    set_deterministic(cfg.deterministic)

    gpus = tf.config.list_physical_devices("GPU")
    print(f"TensorFlow is using the following GPUs: {gpus}" if gpus else "TensorFlow is using the CPU.")

    print(f"--- Loading data from: {cfg.all_data_json} ---")
    survey_classifier_views = load_survey_classifier_views(cfg.all_data_json)
    with open(cfg.survey_classifier_json, "w") as f:
        json.dump(survey_classifier_views, f, indent=2)
    print(f"Saved survey classifier views to {cfg.survey_classifier_json}")

    image_paths, indexed_labels, mask_paths = extract_day_data(survey_classifier_views, cfg.target_day)
    print_class_distribution(indexed_labels)

    indexed_labels = np.array(indexed_labels)
    X_img_train, X_img_val, X_mask_train, X_mask_val, y_train, y_val = train_test_split(
        image_paths, mask_paths, indexed_labels,
        test_size=0.2, stratify=indexed_labels, random_state=cfg.seed,
    )

    print("\n--- Calculating Class Weights ---")
    weights_arr = class_weight.compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train.flatten()),
        y=y_train.flatten(),
    )
    class_weights = {i: w for i, w in enumerate(weights_arr)}
    print(f"Class Weights: {class_weights}")

    train_dataset = create_dataset(X_img_train, X_mask_train, y_train,
                                   cfg.batch_size, cfg.target_size,
                                   augment=True, shuffle=True, seed=cfg.seed)
    val_dataset = create_dataset(X_img_val, X_mask_val, y_val,
                                 cfg.batch_size, cfg.target_size,
                                 augment=False, shuffle=False, seed=cfg.seed)

    model, base_model = initialize_model(train_dataset)
    print("\n--- Initial Model Summary (Base Frozen) ---")
    model.summary()

    history, history_fine_tune = train_model(
        model, base_model, train_dataset, val_dataset, class_weights,
        cfg.epoch1, cfg.epoch2,
    )

    evaluate_model(history, history_fine_tune, model, val_dataset, cfg.out_dir)
    plot_model_metrics(history, history_fine_tune, cfg.out_dir)
    plot_confusion_matrix(model, val_dataset, X_img_val, cfg.out_dir)

    print(f"Execution time: {datetime.datetime.now() - start_time}")


if __name__ == "__main__":
    main()
