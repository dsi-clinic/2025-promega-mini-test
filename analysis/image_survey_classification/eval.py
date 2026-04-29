#!/usr/bin/env python3
"""Validation evaluation + confusion-matrix export."""

import json

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from pipeline.data_loader import INT_TO_LABEL


def evaluate_model(history, history_fine_tune, model, val_dataset, out_dir):
    """Run model.evaluate on val_dataset, print results, save the .h5 model."""
    results = model.evaluate(val_dataset, verbose=0)
    print("\nValidation Results (Final Model):")
    print(f"  Loss: {results[0]:.4f}")
    print(f"  Accuracy: {results[1]:.4f}")
    print(f"  AUC: {results[2]:.4f}")
    print(f"  Precision: {results[3]:.4f}")
    print(f"  Recall: {results[4]:.4f}")

    model.save(out_dir / "organoid_classifier_final_model_with_augmentation.h5")
    print("\nFinal model saved as 'organoid_classifier_final_model_with_augmentation.h5'")


def plot_confusion_matrix(model, val_dataset, val_img_paths, out_dir):
    """Predict on val_dataset, save confusion-matrix PNG + per-sample metrics JSON."""
    print("\n--- Generating Confusion Matrix ---")
    y_true_all, y_pred_proba_all = [], []
    for (images_batch, masks_batch), labels_batch in val_dataset:
        y_true_all.extend(labels_batch.numpy().flatten())
        y_pred_proba_all.extend(model.predict([images_batch, masks_batch]).flatten())

    y_true = np.array(y_true_all)
    y_proba = np.array(y_pred_proba_all)
    y_pred = (y_proba > 0.5).astype(int)

    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(cm)

    labels = [INT_TO_LABEL[0], INT_TO_LABEL[1]]
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", values_format="d")
    disp.ax_.set_xlabel("Predicted label")
    disp.ax_.set_ylabel("Actual label")
    disp.ax_.set_title("Actual vs Predicted Confusion Matrix")
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix.png")
    plt.close()

    metrics = {
        "val_img_paths": list(val_img_paths),
        "val_true_labels": y_true.tolist(),
        "predicted_probabilities": y_proba.tolist(),
        "binary_predictions": y_pred.tolist(),
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
