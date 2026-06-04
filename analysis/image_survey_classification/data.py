#!/usr/bin/env python3
"""Survey-classifier emitter + tf.data pipeline.

SurveyClassifierEmitter builds a per-day view from all_data.json restricted to
records that have survey evaluations and a paper-acceptable image. The
tf.data helpers (create_dataset / load_and_preprocess_tf / augment_data) load
images and masks lazily and feed the dual-branch ResNet50V2 classifier.
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping

import tensorflow as tf

from pipeline.common.json_views import BaseViewEmitter
from pipeline.data_loader import MIN_VOTES
from pipeline.merge.normalized_records import OrganoidRecord  # noqa: F401  (kept for typing reference)

SCHEMA_DICT = Dict[str, Any]


class SurveyClassifierEmitter(BaseViewEmitter):
    """Build view payload for the human-survey grounded classifier."""

    name = "survey_classifier"

    def __init__(self, survey_day: int = 30, min_votes: int = MIN_VOTES):
        self.survey_day = f"Dy{survey_day:02d}"
        self.min_votes = min_votes
        self._records_by_day: Dict[str, List[SCHEMA_DICT]] = defaultdict(list)
        self._skipped_records_by_day: Dict[str, List[str]] = defaultdict(list)

    def process(self, record: SCHEMA_DICT) -> None:
        if record.get("day", {}).get("id") != self.survey_day:
            return

        evals = record.get("survey", {}).get("evaluations")
        img_path = record.get("images", {}).get("img_path")
        mask_path = record.get("images", {}).get("mask_path")
        label = record.get("label", {}).get("acceptance_flag")

        if (
            not evals
            or not record.get("day", {}).get("id")
            or not img_path
            or not mask_path
            or label not in self.label_list
        ):
            self._skipped_records_by_day[record.get("day", {}).get("id")].append(record.get("id"))
            return

        self._records_by_day[record.get("day", {}).get("id")].append({
            "id": record.get("id"),
            "img_path": img_path,
            "mask_path": mask_path,
            "label": label,
        })


def load_survey_classifier_views(all_data_json: Path) -> Mapping[str, SCHEMA_DICT]:
    """Replay the emitter over all_data.json and return its finalized view."""
    with open(all_data_json) as f:
        records = json.load(f)
    emitter = SurveyClassifierEmitter()
    for record in records.values():
        emitter.process(record)
    return emitter.finalize()


def extract_day_data(all_data, target_day):
    """Pull (img_paths, labels, mask_paths) for one day out of the views payload."""
    day_data = all_data.get("records", {}).get(target_day, {})
    imgs = day_data.get("img_path", [])
    labels = day_data.get("label", [])
    masks = day_data.get("mask_path", [])

    assert len(imgs) == len(labels) == len(masks), (
        f"Day {target_day} has mismatched list lengths: "
        f"{len(imgs)} images, {len(labels)} labels, {len(masks)} masks"
    )

    if not imgs or not labels or not masks:
        print("\n❌ Error: No matching data found.")
        print("   - Check that survey_classifier.json has survey data for Dy30")
        print("   - Check that evaluations have clear majority votes")
        raise RuntimeError("Error: No survey data found.")

    print(f"✓ Loaded {len(imgs)} total records from survey_classifier.json")
    print(f"⚠ Skipped {all_data.get('metadata', {}).get('total_skipped')} records "
          f"without processed image paths, evaluations, or labels")
    return imgs, labels, masks


def load_and_preprocess_tf(img_path_tensor, mask_path_tensor, label_tensor, target_size):
    """py_function-friendly loader. Decodes JPEG/PNG, resizes, normalizes to [0,1]."""
    img_path = img_path_tensor.numpy().decode("utf-8")
    mask_path = mask_path_tensor.numpy().decode("utf-8")

    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, target_size)
    img = tf.cast(img, tf.float32) / 255.0

    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.image.resize(mask, target_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    mask = tf.cast(mask, tf.float32) / 255.0

    label = tf.cast(label_tensor, tf.float32)
    label = tf.reshape(label, (1,))
    return img, mask, label


def augment_data(img, mask, label):
    """Random-flip + photometric jitter on image only; mask flips with image."""
    if tf.random.uniform(()) > 0.5:
        img = tf.image.flip_left_right(img)
        mask = tf.image.flip_left_right(mask)
    img = tf.image.random_brightness(img, max_delta=0.2)
    img = tf.image.random_contrast(img, lower=0.8, upper=1.2)
    img = tf.image.random_hue(img, max_delta=0.1)
    img = tf.image.random_saturation(img, lower=0.8, upper=1.2)
    return (img, mask), label


def create_dataset(img_paths, mask_paths, labels, batch_size, target_size,
                   augment=False, shuffle=True, seed=None):
    """Build a tf.data.Dataset from path lists. Returns ((img, mask), label) batches."""
    img_path_tensor = tf.constant(img_paths)
    mask_path_tensor = tf.constant(mask_paths)
    label_tensor = tf.constant(labels, dtype=tf.int32)

    dataset = tf.data.Dataset.from_tensor_slices((img_path_tensor, mask_path_tensor, label_tensor))
    ts = tf.convert_to_tensor(target_size, dtype=tf.int32)

    dataset = dataset.map(
        lambda ip, mp, l: tf.py_function(
            load_and_preprocess_tf,
            inp=[ip, mp, l, ts],
            Tout=(tf.float32, tf.float32, tf.float32),
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    dataset = dataset.map(
        lambda img, mask, label: (
            tf.ensure_shape(img, (target_size[0], target_size[1], 3)),
            tf.ensure_shape(mask, (target_size[0], target_size[1], 1)),
            tf.ensure_shape(label, (1,)),
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(img_paths), seed=seed) if seed \
            else dataset.shuffle(buffer_size=len(img_paths))

    if augment:
        dataset = dataset.map(augment_data, num_parallel_calls=tf.data.AUTOTUNE)
    else:
        dataset = dataset.map(lambda img, mask, label: ((img, mask), label),
                              num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset
