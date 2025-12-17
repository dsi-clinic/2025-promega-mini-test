# Standard
import argparse
import dataclasses
import datetime
import json
from collections import Counter, defaultdict
from pathlib import Path
import random
from typing import Any, Dict, List, Mapping

# Third party
import dotenv
dotenv.load_dotenv()    # Load environment variables ahead of torch imports

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import f1_score as sk_f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from tensorflow import keras
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D

# Application
from file_utils.common.json_views import BaseViewEmitter
from file_utils.common.organoid_patterns import OrganoidPatterns, OrganoidNormalizer
from file_utils.merge.normalized_records import OrganoidRecord

# --- Constants ---
METRICS_TO_COMBINE = ['loss', 'val_loss', 'accuracy', 'val_accuracy', 'auc',
                          'val_auc', 'precision', 'val_precision', 'recall',
                          'val_recall']
SCHEMA_DICT = Dict[str, Any]

# --- Classes ---
@dataclasses.dataclass
class Config:
    data_dir: Path = dataclasses.field(metadata={
        "help": "Path to data directory containing organoid data"
    })
    all_data_json: Path = dataclasses.field(default=None, metadata={
        "help": "Path to all data JSON file"
    })
    survey_classifier_json: Path = dataclasses.field(default=None, metadata={
        "help": "Path to survey classifier JSON file (defaults to out_dir/../identifiers/survey_classifier.json)"
    })
    batch_size: int = dataclasses.field(default=8, metadata={
        "help": "Training batch size"
    })
    epoch1: int = dataclasses.field(default=50, metadata={
        "help": "Number of training epochs for phase 1 (frozen backbone)"
    })
    epoch2: int = dataclasses.field(default=150, metadata={
        "help": "Number of training epochs for phase 2 (unfrozen backbone)"
    })
    target_day: str = dataclasses.field(default="Dy30", metadata={
        "help": "Target day"
    })     # Focus on day 30 which has survey data
    target_width: int = dataclasses.field(default=224, metadata={
        "help": "Target input image width (pixels)"
    })
    target_height: int = dataclasses.field(default=224, metadata={
        "help": "Target input image height (pixels)"
    })
    deterministic: bool = dataclasses.field(default=False, metadata={
        "help": "Use deterministic operations"
    })
    seed: int = dataclasses.field(default=1, metadata={
        "help": "Random seed for reproducibility"
    })
    def __post_init__(self):
        self.target_size: tuple = (self.target_width, self.target_height)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.out_dir = self.data_dir / "results"
        self.out_dir.mkdir(parents=True, exist_ok=True)

        if self.survey_classifier_json is None:
            self.survey_classifier_json = self.data_dir.parent.joinpath("identifiers", "survey_classifier.json")
        else:
            self.survey_classifier_json = Path(self.survey_classifier_json)

        if self.all_data_json is None:
            self.all_data_json = self.data_dir.parent.joinpath("identifiers", "all_data.json")
        else:
            self.all_data_json = Path(self.all_data_json)

class SurveyClassifierEmitter(BaseViewEmitter):
    """Build view payload for the human-survey grounded classifier."""

    name = "survey_classifier"

    def __init__(self, survey_day: int = 30, min_votes: int = 4):
        self.survey_day = f"Dy{survey_day:02d}"
        self.min_votes = min_votes
        self._records_by_day: Dict[str, List[SCHEMA_DICT]] = defaultdict(list)
        self._skipped_records_by_day: Dict[str, List[str]] = defaultdict(list)

    def process(self, record: SCHEMA_DICT) -> None:
        if record.get("day", {}).get("id") != self.survey_day:
            return

        evals = record.get("survey", {}).get("evaluations")
        img_path = record.get("images", {}).get("processed", {}).get("img_path")
        mask_path = record.get("images", {}).get("processed", {}).get("mask_path")
        label = record.get("label", {}).get("acceptance_flag")

        if not evals or not record.get("day", {}).get("id") or not img_path or not mask_path \
            or label not in self.label_list:
            self._skipped_records_by_day[record.get("day", {}).get("id")].append(record.get("id"))
            return

        payload = {
            "id": record.get("id"),
            "img_path": img_path,
            "mask_path": mask_path,
            "label": label,
        }
        self._records_by_day[record.get("day", {}).get("id")].append(payload)

# --- Functions ---
def get_args():
    """Retrieve and return command line arguments via the Config class"""
    arg_parser = create_args()
    args = arg_parser.parse_args()
    cfg = Config(**vars(args))
    return cfg

def create_args() -> argparse.ArgumentParser:
    """Create an ArgumentParser from the Config dataclass."""
    parser = argparse.ArgumentParser(description="Run image classifier on organoid images")

    for field in dataclasses.fields(Config):
        # Build argument flag and help message
        flags = [f"--{field.name.replace('_', '-')}"]

        kwargs = {
            "help": field.metadata.get("help", ""),
            "default": field.default
        }

        # Determine argument type
        if field.type == bool:
            kwargs["action"] = "store_true" if field.default is False else "store_false"
        else:
            kwargs["type"] = field.type

        parser.add_argument(*flags, **kwargs)

    return parser

def set_deterministic(deterministic):
    try:
        tf.config.experimental.enable_op_determinism()
    except AttributeError:
        print("Warning: Deterministic operations not available in this TensorFlow version. Using seeds only.")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.keras.utils.set_random_seed(seed)

def load_survey_classifier_views(all_data_json: Path) -> Mapping[str, SCHEMA_DICT]:
    """Load survey classifier views from all data (records) JSON file."""
    with open(all_data_json) as f:
        records = json.load(f)

    emitter = SurveyClassifierEmitter()
    for record in records.values():
        emitter.process(record)
    return emitter.finalize()

def extract_day_data(all_data, target_day):
    """Extract survey data for a particular day."""

    day_data = all_data.get("records", {}).get(target_day, {})
    imgs = day_data.get("img_path", [])
    labels = day_data.get("label", [])
    masks = day_data.get("mask_path", [])

    assert len(imgs) == len(labels) == len(masks), (
        f"Day {target_day} has mismatched list lengths: "
        f"{len(imgs)} images, {len(labels)} labels, {len(masks)} masks"
    )

        # Check if we have any data
    if not imgs or not labels or not masks:
        print("\n❌ Error: No matching data found.")
        print("   - Check that survey_classifier.json has survey data for Dy30")
        print("   - Check that evaluations have clear majority votes")
        raise RuntimeError("Error: No survey data found.")

    print(f"✓ Loaded {len(imgs)} total records from survey_classifier.json")
    print(f"⚠ Skipped {all_data.get('metadata', '').get('total_skipped')} records without processed image paths, evalutations, or labels")

    return imgs, labels, masks

# --- Helper function to compute majority label from evaluations ---
def compute_majority_label(evaluations, min_votes=4):
    """Compute majority label from survey evaluations."""
    if not evaluations or len(evaluations) != 5:
        return None

    votes = {}
    for eval_data in evaluations:
        evaluation = eval_data.get('evaluation', '')
        if evaluation:
            votes[evaluation] = votes.get(evaluation, 0) + 1

    acceptable = votes.get('Acceptable', 0)
    not_acceptable = votes.get('Not Acceptable', 0)

    # Use majority threshold (at least 4 out of 5)
    if acceptable >= min_votes:
        return 'Acceptable'
    elif not_acceptable >= min_votes:
        return 'Not Acceptable'
    else:
        return None  # Skip ambiguous cases

def print_class_distribution(indexed_labels):
    """Calculate and print class distribution."""
    print("\n--- Class Distribution (Before Split) ---")
    label_to_index = {"Not Acceptable": 0, "Acceptable": 1}  # Explicitly map to 0 and 1
    class_counts = Counter(indexed_labels)
    for class_idx, count in sorted(class_counts.items()):
        label_name = [name for name, idx in label_to_index.items() if idx == class_idx][0]
        print(f"Class {class_idx} ('{label_name}'): {count} samples")
    print("------------------------------------------")

def create_dataset(img_paths, mask_paths, labels, batch_size, target_size, augment=False, shuffle=True, seed=None):
    # Convert lists to TensorFlow tensors
    img_path_tensor = tf.constant(img_paths)
    mask_path_tensor = tf.constant(mask_paths)
    label_tensor = tf.constant(labels, dtype=tf.int32) # Labels as int for now, cast later

    dataset = tf.data.Dataset.from_tensor_slices((img_path_tensor, mask_path_tensor, label_tensor))
    ts = tf.convert_to_tensor(target_size, dtype=tf.int32)

    # Use tf.py_function for loading and preprocessing to handle PIL/Numpy operations
    # The Tout argument matches the flat return of load_and_preprocess_tf
    dataset = dataset.map(
        lambda ip, mp, l: tf.py_function(
            load_and_preprocess_tf,
            inp=[ip, mp, l, ts],
            Tout=(tf.float32, tf.float32, tf.float32) # Corrected Tout: (img_dtype, mask_dtype, label_dtype)
        ),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # IMPORANT: Set shapes after py_function, as it often loses static shape info.
    # img: (target_size[0], target_size[1], 3)
    # mask: (target_size[0], target_size[1], 1)
    # label: (1,) (scalar label reshaped to 1-element vector)
    dataset = dataset.map(
        lambda img, mask, label: (
            tf.ensure_shape(img, (target_size[0], target_size[1], 3)),
            tf.ensure_shape(mask, (target_size[0], target_size[1], 1)),
            tf.ensure_shape(label, (1,)) # Set shape for the label
        ),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    if shuffle:
        if seed:
            dataset = dataset.shuffle(buffer_size=len(img_paths), seed=seed) # Shuffle the dataset with provided seed
        else:
            dataset = dataset.shuffle(buffer_size=len(img_paths)) # Shuffle the dataset with global seed

    if augment:
        # Augment data, then structure it for the model.
        # The map function receives img, mask, label from the previous step.
        dataset = dataset.map(augment_data, num_parallel_calls=tf.data.AUTOTUNE)
    else:
        # If not augmenting, still need to structure the output for the model.
        dataset = dataset.map(lambda img, mask, label: ((img, mask), label), num_parallel_calls=tf.data.AUTOTUNE)


    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset

def load_and_preprocess_tf(img_path_tensor, mask_path_tensor, label_tensor, target_size):

    # Decode string tensors to actual strings
    img_path = img_path_tensor.numpy().decode('utf-8')
    mask_path = mask_path_tensor.numpy().decode('utf-8')

    # Load image
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, target_size)
    img = tf.cast(img, tf.float32) / 255.0

    # Load mask
    mask = tf.io.read_file(mask_path)
    # Assuming masks are grayscale PNGs
    mask = tf.image.decode_png(mask, channels=1) # Decode as 1 channel
    mask = tf.image.resize(mask, target_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR) # Use NEAREST for masks
    mask = tf.cast(mask, tf.float32) / 255.0

    # The label needs to be explicitly reshaped to (1,) to give it a defined rank.
    label = tf.cast(label_tensor, tf.float32)
    label = tf.reshape(label, (1,)) # Ensure label has a defined shape, e.g., (1,) for a scalar

    return img, mask, label # Return flat tuple for py_function Tout

def augment_data(img, mask, label):
    # Data augmentation operations
    # Note: TensorFlow's global seed (set via tf.random.set_seed) ensures
    # deterministic behavior for random operations within each epoch

    # Apply random horizontal flip
    # Ensure the seed is passed consistently if using stateful ops, or rely on tf.random ops.
    # For simplicity, using conditional flip.
    if tf.random.uniform(()) > 0.5:
        img = tf.image.flip_left_right(img)
        mask = tf.image.flip_left_right(mask)

    # Apply random rotation
    #if tf.random.uniform(()) > 0.75: # Example: 25% chance of 90-degree rotation
    #    k = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int332) # 0, 90, 180, 270 degrees
    #    img = tf.image.rot90(img, k=k)
    #    mask = tf.image.rot90(mask, k=k)

    # Random brightness (only on image)
    img = tf.image.random_brightness(img, max_delta=0.2)
    # Random contrast (only on image)
    img = tf.image.random_contrast(img, lower=0.8, upper=1.2)
    # Random hue (only on image if 3 channels)
    img = tf.image.random_hue(img, max_delta=0.1)
    # Random saturation (only on image)
    img = tf.image.random_saturation(img, lower=0.8, upper=1.2)

    return (img, mask), label # Return in (inputs, label) format for model.fit

def initialize_model(train_dataset):
    """Define and initialize model."""
    # Get shapes from the first batch to define model input shapes
    # The shapes should now be well-defined due to tf.ensure_shape
    for (img_batch, mask_batch), _ in train_dataset.take(1):
        img_shape = img_batch.shape[1:]
        mask_shape = mask_batch.shape[1:]
    print(f"Determined IMG_SHAPE: {img_shape}")
    print(f"Determined MASK_SHAPE: {mask_shape}")

    # --- 7. Define a CNN model with a pre-trained base ---
    # Load a pre-trained model (e.g., ResNet50V2) without the top (classification) layer
    base_model = ResNet50V2(include_top=False, weights='imagenet', input_shape=img_shape)
    base_model.trainable = False # Freeze the base model's weights initially

    # Create separate input layers for the image and the mask
    input_image = keras.Input(shape=img_shape)
    input_mask = keras.Input(shape=mask_shape)

    # Pass the image input through the pre-trained base model
    base_output = base_model(input_image)
    pooled_output = GlobalAveragePooling2D()(base_output) # Reduce spatial dimensions

    # Process the mask input with a smaller CNN
    mask_features = Conv2D(32, (3, 3), activation='relu', padding='same')(input_mask)
    mask_features = MaxPooling2D((2, 2))(mask_features)
    mask_features = Conv2D(64, (3, 3), activation='relu', padding='same')(mask_features)
    mask_features = MaxPooling2D((2, 2))(mask_features)
    mask_features = Flatten()(mask_features)
    mask_features = Dense(64, activation='relu')(mask_features)

    # Combine the features from the pre-trained model and the mask processing CNN
    merged = keras.layers.concatenate([pooled_output, mask_features])

    # Add a classification head
    dense_layer = Dense(128, activation='relu')(merged)
    dropout_layer = Dropout(0.5)(dense_layer)
    output_layer = Dense(1, activation='sigmoid')(dropout_layer)

    # Create the final model with two inputs and one output
    model = keras.Model(inputs=[input_image, input_mask], outputs=output_layer)

    # Compile the model with GPU-compatible metrics
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )
    return model, base_model

def train_model(model, base_model, train_dataset, val_dataset, class_weights,
                epoch1, epoch2):
    """Train the model in two phases: (1) frozen backbone and (2) unfrozen backbone."""
    early_stopping = EarlyStopping(
        monitor='val_auc',
        patience=20, # Increased patience a bit
        verbose=1,
        mode='max',
        restore_best_weights=True
    )

    epochs_phase1 = epoch1 # Train for fewer epochs initially with frozen base

    print(f"\n--- Training Phase 1: Frozen Base Model with Augmentation ({epochs_phase1} epochs) ---")
    history = model.fit(
        train_dataset, # Use the TF Dataset here
        epochs=epochs_phase1,
        validation_data=val_dataset, # Use the TF Dataset here
        callbacks=[early_stopping],
        class_weight=class_weights # Apply class weights here
    )
    print("----------------------------------------------------------")

    # --- 10. Unfreeze and Fine-tune (Phase 2 with Data Augmentation) ---
    print("\n--- Training Phase 2: Unfreezing and Fine-tuning Base Model with Augmentation ---")
    # Unfreeze a portion of the base model
    base_model.trainable = True
    for layer in base_model.layers[-10:]: # Unfreeze last 10 layers, for example
        layer.trainable = True

    # It's crucial to re-compile the model after unfreezing layers for the changes to take effect.
    # Use a small learning rate for fine-tuning.
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )

    print("\n--- Model Summary (Base Unfrozen, Fine-tuning LR) ---")
    model.summary()
    print("----------------------------------------------------")

    epochs_phase2 = epoch2 # More epochs for fine-tuning, total epochs will be epochs_phase1 + epochs_phase2

    # Reset early stopping for the second phase to allow more training
    early_stopping_fine_tune = EarlyStopping(
        monitor='val_auc',
        patience=30, # Increased patience for fine-tuning
        verbose=1,
        mode='max',
        restore_best_weights=True
    )

    history_fine_tune = model.fit(
        train_dataset, # Use the TF Dataset here
        epochs=epochs_phase2,
        validation_data=val_dataset, # Use the TF Dataset here
        callbacks=[early_stopping_fine_tune],
        class_weight=class_weights # Apply class weights here as well
    )
    print("----------------------------------------------------------")
    return history, history_fine_tune

def evaluate_model(history, history_fine_tune, model, val_dataset, out_dir):
    """Evaluate the model using the validation dataset and save metrics."""

    # Evaluate the model
    results = model.evaluate(val_dataset, verbose=0)
    print(f"\nValidation Results (Final Model):")
    print(f"  Loss: {results[0]:.4f}")
    print(f"  Accuracy: {results[1]:.4f}")
    print(f"  AUC: {results[2]:.4f}")
    print(f"  Precision: {results[3]:.4f}")
    print(f"  Recall: {results[4]:.4f}")

    # Save the trained model ---
    model.save(out_dir.joinpath('organoid_classifier_final_model_with_augmentation.h5'))
    print("\nFinal model classifier saved as 'organoid_classifier_final_model_with_augmentation.h5'")

def plot_model_metrics(history, history_fine_tune, out_dir):
    """Plot AUC score and loss metrics."""
    # Combine histories for plotting
    for key in METRICS_TO_COMBINE:
        if key in history.history and key in history_fine_tune.history:
            history.history[key].extend(history_fine_tune.history[key])
        elif key in history_fine_tune.history: # In case a metric was only added in phase 2 (unlikely here, but good practice)
            history.history[key] = history_fine_tune.history[key]

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history.get('auc', []), label='Train AUC')
    plt.plot(history.history.get('val_auc', []), label='Validation AUC')
    plt.xlabel('Epoch')
    plt.ylabel('AUC Score')
    plt.legend()
    plt.title('Training and Validation AUC')
    plt.savefig(out_dir.joinpath('training_auc_final_model_with_augmentation.png'))

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.savefig(out_dir.joinpath('training_loss_final_model_with_augmentation.png'))

    print("\nTraining history plots saved as 'training_auc_final_model_with_augmentation.png' and 'training_loss_final_model_with_augmentation.png'")

def plot_confusion_matrix(model, val_dataset, val_img_paths, out_dir):
    """Generate a confusion matrix from validation dataset."""
    print("\n--- Generating Confusion Matrix ---")
    # To get predictions for the confusion matrix, iterate through the validation dataset
    y_true_all = []
    y_pred_proba_all = []

    for (images_batch, masks_batch), labels_batch in val_dataset:
        y_true_all.extend(labels_batch.numpy().flatten())
        y_pred_proba_all.extend(model.predict([images_batch, masks_batch]).flatten())

    y_true_all = np.array(y_true_all)
    y_pred_proba_all = np.array(y_pred_proba_all)

    y_pred = (y_pred_proba_all > 0.5).astype(int) # Convert probabilities to binary predictions

    cm = confusion_matrix(y_true_all, y_pred)
    print("Confusion Matrix:")
    print(cm)

    labels = ["Not Acceptable", "Acceptable"]
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", values_format="d")
    disp.ax_.set_xlabel("Predicted label")
    disp.ax_.set_ylabel("Actual label")
    disp.ax_.set_title("Actual vs Predicted Confusion Matrix")
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix.png")
    plt.close()

    # Save metrics to JSON file
    metrics = {
        "val_img_paths": val_img_paths,
        "val_true_labels": y_true_all.tolist(),
        "predicted_probabilities": y_pred_proba_all.tolist(),
        "binary_predictions": y_pred.tolist(),
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

# --- Helper function to get mapping paths ---
def get_mapping_paths(prepocessed_json_dir, batch_number, day_number=30):
    """Get zero-padded mapping JSON paths."""
    day_str = f"{day_number:02d}"
    if batch_number == 2:
        return [
            Path(prepocessed_json_dir) / f"BA2_96_1_Dy{day_str}" / f"image_mapping_BA2_96_1_Dy{day_str}_processed.json",
            Path(prepocessed_json_dir) / f"BA2_96_2_Dy{day_str}" / f"image_mapping_BA2_96_2_Dy{day_str}_processed.json"
        ]
    else:
        return [
            Path(prepocessed_json_dir) / f"BA{batch_number}_Dy{day_str}" / f"image_mapping_BA{batch_number}_Dy{day_str}_processed.json"
        ]

# --- Helper function to normalize keys for matching ---
def normalize_key(key):
    """Convert old-style keys to match new-style keys."""
    # Remove spaces and make uppercase
    key = key.replace(" ", "").upper()
    # Handle batch numbers (Ba1 -> BA1, Ba2 -> BA2)
    # Use centralized pattern for cleaning
    key = OrganoidNormalizer.clean_string(key)
    # Remove plate designators
    key = OrganoidPatterns.PLATE_REMOVE.sub('', key)
    # Standardize case
    key = key.upper().replace('DY', 'Dy')
    return key

# --- 6. Define F1 Score Metric for Keras ---
def weighted_f1_score_keras(y_true, y_pred):
    # Round predictions to binary (0 or 1)
    y_pred_binary = K.round(y_pred)

    # Flatten the tensors
    y_true_flat = K.flatten(y_true)
    y_pred_flat = K.flatten(y_pred_binary)

    # Define the Python function to be wrapped
    def _weighted_f1_py_func(y_true_tensor, y_pred_tensor):
        # Convert EagerTensors to NumPy arrays
        y_true_np = y_true_tensor.numpy()
        y_pred_np = y_pred_tensor.numpy()

        # Handle potential empty batches, or cases where one class has no true instances
        # It's better to check for presence of unique labels here
        if not (np.any(y_true_np == 0) or np.any(y_true_np == 1)):
             return 0.0 # Return a default if no relevant labels are present in the batch

        # Ensure y_true_np and y_pred_np are 1D arrays for sklearn
        y_true_np = y_true_np.astype(int) # Now .astype(int) works on NumPy array
        y_pred_np = y_pred_np.astype(int) # Now .astype(int) works on NumPy array

        # Calculate weighted F1 score using sklearn
        return sk_f1_score(y_true_np, y_pred_np, average='weighted')

    # Wrap the Python function with tf.py_function
    weighted_f1 = tf.py_function(
        _weighted_f1_py_func,
        inp=[y_true_flat, y_pred_flat],
        Tout=tf.float32 # Output is a single float
    )
    weighted_f1.set_shape([]) # Set shape to scalar (no dimensions)
    return weighted_f1

def macro_f1_score_keras(y_true, y_pred):
    y_pred_binary = K.round(y_pred)
    y_true_flat = K.flatten(y_true)
    y_pred_flat = K.flatten(y_pred_binary)

    def _macro_f1_py_func(y_true_tensor, y_pred_tensor):
        y_true_np = y_true_tensor.numpy()
        y_pred_np = y_pred_tensor.numpy()

        if not (np.any(y_true_np == 0) or np.any(y_true_np == 1)):
            return 0.0

        y_true_np = y_true_np.astype(int)
        y_pred_np = y_pred_np.astype(int)

        return sk_f1_score(y_true_np, y_pred_np, average='macro')

    macro_f1 = tf.py_function(
        _macro_f1_py_func,
        inp=[y_true_flat, y_pred_flat],
        Tout=tf.float32
    )
    macro_f1.set_shape([])
    return macro_f1

def main():
    start_time = datetime.datetime.now()
    # --- 1. command line arguments and setup ---
    cfg = get_args()
    for key,val in vars(cfg).items():
        print(f"{key}: {val}")

    # --- Set seeds and deterministic operations ---
    set_seed(cfg.seed)
    set_deterministic(cfg.deterministic)

    # --- Check for GPU availability ---
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"TensorFlow is using the following GPUs: {gpus}")
    else:
        print("TensorFlow is using the CPU.")

    # --- 2. Load data (unified data source) ---
    print(f"--- Loading data from: {cfg.all_data_json} ---")
    survey_classifier_views = load_survey_classifier_views(cfg.all_data_json)
    with open(cfg.survey_classifier_json, "w") as f:
        json.dump(survey_classifier_views, f, indent=2)
    print(f"Saved survey classifier views to {cfg.survey_classifier_json}")

    # --- 3. Extract Dy30 data with survey ---
    image_paths, indexed_labels, mask_paths = extract_day_data(survey_classifier_views, cfg.target_day)

    # --- 4. Prepare data for training ---
    print_class_distribution(indexed_labels)

    # --- 5. Split data into training and validation sets ---
    # We split paths and labels first, then load images on demand in the TF Dataset.
    indexed_labels = np.array(indexed_labels)
    X_img_path_train, X_img_path_val, X_mask_path_train, X_mask_path_val, y_train, y_val = train_test_split(
        image_paths, mask_paths, indexed_labels, test_size=0.2, stratify=indexed_labels, random_state=cfg.seed
    )

    # --- 6. Calculate and Apply Class Weights ---
    print("\n--- Calculating Class Weights ---")
    # Flatten y_train for class_weight.compute_class_weight as it expects 1D array
    class_weights_array = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train.flatten()),
        y=y_train.flatten()
    )
    class_weights = {i: weight for i, weight in enumerate(class_weights_array)}
    print(f"Class Weights: {class_weights}")
    print("-------------------------------")

    # --- 7. Data Loading and Augmentation with tf.data.Dataset ---
    batch_size = cfg.batch_size
    train_dataset = create_dataset(X_img_path_train, X_mask_path_train, y_train,
                                   batch_size, cfg.target_size, augment=True,
                                   shuffle=True, seed=cfg.seed)
    val_dataset = create_dataset(X_img_path_val, X_mask_path_val, y_val,
                                 batch_size, cfg.target_size, augment=False,
                                 shuffle=False, seed=cfg.seed)

    # --- 8. Initialize model with training dataset
    model, base_model = initialize_model(train_dataset)
    print("\n--- Initial Model Summary (Base Frozen) ---")
    model.summary()
    print("------------------------------------------")

    # --- 9. Define Early Stopping Callback ---
    history, history_fine_tune = train_model(model, base_model, train_dataset,
                                             val_dataset, class_weights,
                                             cfg.epoch1, cfg.epoch2)

    # --- 10. Evaluate the model ---
    # To evaluate with the dataset, we need to convert it to a format model.evaluate expects.
    # We'll use the validation dataset directly.
    evaluate_model(history, history_fine_tune, model, val_dataset, cfg.out_dir)

    # --- 11. Visualize training history ---
    plot_model_metrics(history, history_fine_tune, cfg.out_dir)

    # --- 12. Print Confusion Matrix ---
    plot_confusion_matrix(model, val_dataset, X_img_path_val, cfg.out_dir)

    end_time = datetime.datetime.now()
    print(f"Execution time: {end_time - start_time}")

if __name__ == "__main__":
    main()
