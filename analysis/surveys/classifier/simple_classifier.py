import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score as sk_f1_score
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
import tensorflow as tf
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.callbacks import EarlyStopping
from pathlib import Path
import re
from file_utils.common.organoid_patterns import OrganoidPatterns, OrganoidNormalizer
from sklearn.utils import class_weight
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# --- Check for GPU availability ---
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"TensorFlow is using the following GPUs: {gpus}")
else:
    print("TensorFlow is using the CPU.")

# --- Constants ---
ALL_DATA_JSON = 'all_data.json'
TARGET_SIZE = (224, 224) # Define target size as a constant
TARGET_DAY = 'Dy30'  # Focus on day 30 which has survey data
PREPROCESSED_JSON_DIR = '/net/projects2/promega/data-analysis/output/infer_resized_512x384/auto_processed'

# --- Helper function to get mapping paths ---
def get_mapping_paths(batch_number, day_number=30):
    """Get zero-padded mapping JSON paths."""
    day_str = f"{day_number:02d}"
    if batch_number == 2:
        return [
            Path(PREPROCESSED_JSON_DIR) / f"BA2_96_1_Dy{day_str}" / f"image_mapping_BA2_96_1_Dy{day_str}_processed.json",
            Path(PREPROCESSED_JSON_DIR) / f"BA2_96_2_Dy{day_str}" / f"image_mapping_BA2_96_2_Dy{day_str}_processed.json"
        ]
    else:
        return [
            Path(PREPROCESSED_JSON_DIR) / f"BA{batch_number}_Dy{day_str}" / f"image_mapping_BA{batch_number}_Dy{day_str}_processed.json"
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

# --- 1. Load the labeled data for labels ---

# FOR COMPLETE AGREEMENTS (more accurate): 'complete_agreement_organoids.json':

# FOR STRONG AGREEMENTS (more data): 'labeled_organoid_strong_agreement.json':

with open('analysis/surveys/agreement_aggregations/labeled_organoid_majority_agreement.json') as f:
    labeled_data = json.load(f)


modified_labeled_data = {}
for key, data in labeled_data.items():
    # Use regex to find and replace the day number (e.g., DyXX) in the key
    new_key = OrganoidPatterns.DAY_EXTRACT.sub(TARGET_DAY, key)
    modified_labeled_data[new_key] = data

labeled_data = modified_labeled_data # Overwrite with the modified data


# Create a dictionary to map normalized image keys to labels
key_to_label = {normalize_key(key): data['label'] for key, data in labeled_data.items()}

# --- 2. Load the new mapping data and combine with old labels ---
all_new_data = {}

# Extract day number from TARGET_DAY (e.g., 'Dy30' -> 30)
target_day_num = int(TARGET_DAY.replace('Dy', ''))

for batch_num in [1, 2, 3]:
    mapping_paths = get_mapping_paths(batch_num, target_day_num)
    for path in mapping_paths:
        try:
            with open(path) as f:
                new_data = json.load(f)
                # Process each entry in the new mapping
                for key, value in new_data.items():
                    # Normalize the new key for matching
                    normalized_new_key = normalize_key(key)
                    # Only keep entries that exist in the old labeled data
                    if normalized_new_key in key_to_label:
                        all_new_data[key] = {
                            'img_path': value['img_path'],
                            'seg_map_path': value['mask_path'],
                            'label': key_to_label[normalized_new_key]
                        }
        except FileNotFoundError:
            print(f"Warning: Mapping file not found: {path}")
            continue
        except json.JSONDecodeError:
            print(f"Warning: Could not decode JSON from: {path}")
            continue

# Check if we have any data
if not all_new_data:
    print("Error: No matching data found between old and new mappings.")
    exit()

# --- 3. Prepare data for training ---
image_paths = []
mask_paths = []
labels = []

for item in all_new_data.values():
    image_paths.append(item['img_path'])
    mask_paths.append(item['seg_map_path'])
    labels.append(item['label'])

unique_labels = sorted(list(set(labels)))
label_to_index = {"Not Acceptable": 0, "Acceptable": 1}  # Explicitly map to 0 and 1
indexed_labels = np.array([label_to_index[label] for label in labels])
num_classes = 1  # Binary classification uses 1 output unit with sigmoid

# --- Calculate and Print Class Distribution ---
print("\n--- Class Distribution (Before Split) ---")
class_counts = Counter(indexed_labels)
for class_idx, count in sorted(class_counts.items()):
    label_name = [name for name, idx in label_to_index.items() if idx == class_idx][0]
    print(f"Class {class_idx} ('{label_name}'): {count} samples")
print("------------------------------------------")

# --- 4. Split data into training and validation sets ---
# We split paths and labels first, then load images on demand in the TF Dataset.
X_img_path_train, X_img_path_val, X_mask_path_train, X_mask_path_val, y_train, y_val = train_test_split(
    image_paths, mask_paths, indexed_labels, test_size=0.2, stratify=indexed_labels
)

# --- Calculate and Apply Class Weights ---
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

# --- 5. Data Loading and Augmentation with tf.data.Dataset ---

def load_and_preprocess_tf(img_path_tensor, mask_path_tensor, label_tensor, target_size=TARGET_SIZE):
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
    # Ensure a consistent seed for transformations that apply to both image and mask
    seed = tf.random.uniform(shape=[], maxval=1000000, dtype=tf.int32)

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

def create_dataset(img_paths, mask_paths, labels, batch_size, augment=False, shuffle=True):
    # Convert lists to TensorFlow tensors
    img_path_tensor = tf.constant(img_paths)
    mask_path_tensor = tf.constant(mask_paths)
    label_tensor = tf.constant(labels, dtype=tf.int32) # Labels as int for now, cast later

    dataset = tf.data.Dataset.from_tensor_slices((img_path_tensor, mask_path_tensor, label_tensor))

    # Use tf.py_function for loading and preprocessing to handle PIL/Numpy operations
    # The Tout argument matches the flat return of load_and_preprocess_tf
    dataset = dataset.map(
        lambda ip, mp, l: tf.py_function(
            load_and_preprocess_tf,
            inp=[ip, mp, l],
            Tout=(tf.float32, tf.float32, tf.float32) # Corrected Tout: (img_dtype, mask_dtype, label_dtype)
        ),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # IMPORANT: Set shapes after py_function, as it often loses static shape info.
    # img: (TARGET_SIZE[0], TARGET_SIZE[1], 3)
    # mask: (TARGET_SIZE[0], TARGET_SIZE[1], 1)
    # label: (1,) (scalar label reshaped to 1-element vector)
    dataset = dataset.map(
        lambda img, mask, label: (
            tf.ensure_shape(img, (TARGET_SIZE[0], TARGET_SIZE[1], 3)),
            tf.ensure_shape(mask, (TARGET_SIZE[0], TARGET_SIZE[1], 1)),
            tf.ensure_shape(label, (1,)) # Set shape for the label
        ),
        num_parallel_calls=tf.data.AUTOTUNE
    )


    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(img_paths)) # Shuffle the dataset

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

batch_size = 8
train_dataset = create_dataset(X_img_path_train, X_mask_path_train, y_train, batch_size, augment=True, shuffle=True)
val_dataset = create_dataset(X_img_path_val, X_mask_path_val, y_val, batch_size, augment=False, shuffle=False)

# Get shapes from the first batch to define model input shapes
# The shapes should now be well-defined due to tf.ensure_shape
for (img_batch, mask_batch), _ in train_dataset.take(1):
    IMG_SHAPE = img_batch.shape[1:]
    MASK_SHAPE = mask_batch.shape[1:]
print(f"Determined IMG_SHAPE: {IMG_SHAPE}")
print(f"Determined MASK_SHAPE: {MASK_SHAPE}")

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

# --- 7. Define a CNN model with a pre-trained base ---
# Load a pre-trained model (e.g., ResNet50V2) without the top (classification) layer
base_model = ResNet50V2(include_top=False, weights='imagenet', input_shape=IMG_SHAPE)
base_model.trainable = False # Freeze the base model's weights initially

# Create separate input layers for the image and the mask
input_image = keras.Input(shape=IMG_SHAPE)
input_mask = keras.Input(shape=MASK_SHAPE)

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

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[weighted_f1_score_keras])

print("\n--- Initial Model Summary (Base Frozen) ---")
model.summary()
print("------------------------------------------")

# --- 8. Define Early Stopping Callback ---
early_stopping = EarlyStopping(
    monitor='val_weighted_f1_score_keras',
    patience=20, # Increased patience a bit
    verbose=1,
    mode='max',
    restore_best_weights=True
)

# --- 9. Train the model (Phase 1: Frozen Base with Data Augmentation) ---
epochs_phase1 = 50 # Train for fewer epochs initially with frozen base

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
    metrics=[weighted_f1_score_keras]
)

print("\n--- Model Summary (Base Unfrozen, Fine-tuning LR) ---")
model.summary()
print("----------------------------------------------------")

epochs_phase2 = 150 # More epochs for fine-tuning, total epochs will be epochs_phase1 + epochs_phase2
# Reset early stopping for the second phase to allow more training
early_stopping_fine_tune = EarlyStopping(
    monitor='val_weighted_f1_score_keras',
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

metrics_to_combine = ['loss', 'val_loss', 'weighted_f1_score_keras', 'val_weighted_f1_score_keras']

# Combine histories for plotting
for key in metrics_to_combine:
    if key in history.history and key in history_fine_tune.history:
        history.history[key].extend(history_fine_tune.history[key])
    elif key in history_fine_tune.history: # In case a metric was only added in phase 2 (unlikely here, but good practice)
        history.history[key] = history_fine_tune.history[key]

# --- 11. Evaluate the model ---
# To evaluate with the dataset, we need to convert it to a format model.evaluate expects.
# We'll use the validation dataset directly.
loss, f1 = model.evaluate(val_dataset, verbose=0)
print(f"\nValidation Loss (Final Model): {loss:.4f}")
print(f"Validation F1 Score (Final Model): {f1:.4f}")

# Save the trained model ---
model.save('organoid_classifier_final_model_with_augmentation.h5')
print("\nFinal model classifier saved as 'organoid_classifier_final_model_with_augmentation.h5'")

# --- 12. Visualize training history ---
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['weighted_f1_score_keras'], label='Train Weighted F1 Score')
plt.plot(history.history['val_weighted_f1_score_keras'], label='Validation Weighted F1 Score')
plt.xlabel('Epoch')
plt.ylabel('F1 Score')
plt.legend()
plt.title('Training and Validation F1 Score')
plt.savefig('training_f1_score_final_model_with_augmentation.png')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.savefig('training_loss_final_model_with_augmentation.png')

print("\nTraining history plots saved as 'training_f1_score_final_model_with_augmentation.png' and 'training_loss_final_model_with_augmentation.png'")

# --- 13. Print Confusion Matrix ---
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