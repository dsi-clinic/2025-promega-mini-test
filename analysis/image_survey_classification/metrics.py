#!/usr/bin/env python3
"""Custom Keras F1 metrics (weighted + macro) wrapped around sklearn."""

import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score as sk_f1_score
from tensorflow.keras import backend as K


def _f1_via_sklearn(y_true, y_pred, average):
    """Round to binary, evaluate via sklearn, with empty-batch guard."""
    y_pred_binary = K.round(y_pred)
    y_true_flat = K.flatten(y_true)
    y_pred_flat = K.flatten(y_pred_binary)

    def _py_func(y_true_tensor, y_pred_tensor):
        y_true_np = y_true_tensor.numpy().astype(int)
        y_pred_np = y_pred_tensor.numpy().astype(int)
        if not (np.any(y_true_np == 0) or np.any(y_true_np == 1)):
            return 0.0
        return sk_f1_score(y_true_np, y_pred_np, average=average)

    f1 = tf.py_function(_py_func, inp=[y_true_flat, y_pred_flat], Tout=tf.float32)
    f1.set_shape([])
    return f1


def weighted_f1_score_keras(y_true, y_pred):
    return _f1_via_sklearn(y_true, y_pred, average="weighted")


def macro_f1_score_keras(y_true, y_pred):
    return _f1_via_sklearn(y_true, y_pred, average="macro")
