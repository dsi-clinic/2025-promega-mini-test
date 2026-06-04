#!/usr/bin/env python3
"""Two-phase training loop: frozen backbone → unfrozen fine-tune."""

import os
import random

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping


def set_deterministic(deterministic: bool) -> None:
    """Best-effort deterministic ops + cuDNN. No-op if TF version too old."""
    if not deterministic:
        return
    os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
    os.environ.setdefault("TF_CUDNN_DETERMINISTIC", "1")
    try:
        tf.config.experimental.enable_op_determinism()
    except AttributeError:
        print("Warning: deterministic ops not available in this TensorFlow version. Using seeds only.")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.keras.utils.set_random_seed(seed)


def train_model(model, base_model, train_dataset, val_dataset, class_weights,
                epoch1: int, epoch2: int):
    """Phase 1 (frozen) → phase 2 (last 10 layers unfrozen, lr=1e-3).

    Returns ``(history_phase1, history_phase2)`` so callers can stitch metrics.
    """
    early_stopping = EarlyStopping(monitor="val_auc", patience=20, verbose=1,
                                   mode="max", restore_best_weights=True)

    print(f"\n--- Training Phase 1: Frozen Base Model ({epoch1} epochs) ---")
    history = model.fit(
        train_dataset, epochs=epoch1, validation_data=val_dataset,
        callbacks=[early_stopping], class_weight=class_weights,
    )
    print("----------------------------------------------------------")

    print("\n--- Training Phase 2: Unfreezing and Fine-tuning ---")
    base_model.trainable = True
    for layer in base_model.layers[-10:]:
        layer.trainable = True

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    print("\n--- Model Summary (Base Unfrozen, Fine-tuning LR) ---")
    model.summary()
    print("----------------------------------------------------")

    early_stopping_fine_tune = EarlyStopping(monitor="val_auc", patience=30, verbose=1,
                                             mode="max", restore_best_weights=True)
    history_fine_tune = model.fit(
        train_dataset, epochs=epoch2, validation_data=val_dataset,
        callbacks=[early_stopping_fine_tune], class_weight=class_weights,
    )
    print("----------------------------------------------------------")
    return history, history_fine_tune
