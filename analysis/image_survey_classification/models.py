#!/usr/bin/env python3
"""Dual-branch ResNet50V2 + small mask CNN classifier."""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.layers import (Conv2D, Dense, Dropout, Flatten,
                                     GlobalAveragePooling2D, MaxPooling2D)


def initialize_model(train_dataset):
    """Build and compile the (ResNet50V2 + mask-branch) classifier.

    Inspects the first batch of ``train_dataset`` to infer img/mask shapes,
    then creates a model with ResNet50V2 (frozen for phase 1) + a small mask
    CNN. Returns ``(model, base_model)`` so callers can later unfreeze
    ``base_model.layers[-10:]`` for phase-2 fine-tuning.
    """
    for (img_batch, mask_batch), _ in train_dataset.take(1):
        img_shape = img_batch.shape[1:]
        mask_shape = mask_batch.shape[1:]
    print(f"Determined IMG_SHAPE: {img_shape}")
    print(f"Determined MASK_SHAPE: {mask_shape}")

    base_model = ResNet50V2(include_top=False, weights="imagenet", input_shape=img_shape)
    base_model.trainable = False

    input_image = keras.Input(shape=img_shape)
    input_mask = keras.Input(shape=mask_shape)

    base_output = base_model(input_image)
    pooled_output = GlobalAveragePooling2D()(base_output)

    mask_features = Conv2D(32, (3, 3), activation="relu", padding="same")(input_mask)
    mask_features = MaxPooling2D((2, 2))(mask_features)
    mask_features = Conv2D(64, (3, 3), activation="relu", padding="same")(mask_features)
    mask_features = MaxPooling2D((2, 2))(mask_features)
    mask_features = Flatten()(mask_features)
    mask_features = Dense(64, activation="relu")(mask_features)

    merged = keras.layers.concatenate([pooled_output, mask_features])
    dense_layer = Dense(128, activation="relu")(merged)
    dropout_layer = Dropout(0.5)(dense_layer)
    output_layer = Dense(1, activation="sigmoid")(dropout_layer)

    model = keras.Model(inputs=[input_image, input_mask], outputs=output_layer)
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model, base_model
