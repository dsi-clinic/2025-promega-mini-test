import json
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import to_categorical
from PIL import Image
import os
import tensorflow as tf
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.callbacks import EarlyStopping

# --- Check for GPU availability ---
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"TensorFlow is using the following GPUs: {gpus}")
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs available")
    except RuntimeError as e:
        print(e)
else:
    print("TensorFlow is using the CPU.")

# --- 1. Load the labeled data ---
try:
    with open('labeled_organoid_mapping_for_classification.json') as f:
        labeled_data = json.load(f)
except FileNotFoundError:
    print("Error: 'labeled_organoid_mapping_for_classification.json' not found.")
    exit()
except json.JSONDecodeError:
    print("Error: Could not decode JSON from 'labeled_organoid_mapping_for_classification.json'.")
    exit()

# --- 2. Prepare data for training ---
image_paths = []
mask_paths = []
labels = []

for item in labeled_data.values():
    image_paths.append(item['img_path'])
    mask_paths.append(item['seg_map_path'])
    labels.append(item['label'])

unique_labels = sorted(list(set(labels)))
label_to_index = {label: i for i, label in enumerate(unique_labels)}
indexed_labels = np.array([label_to_index[label] for label in labels])
num_classes = len(unique_labels)
categorical_labels = to_categorical(indexed_labels, num_classes=num_classes)

# --- 3. Load and preprocess images and masks for the top model ---
def load_and_preprocess_top_model(img_path, mask_path, target_size=(224, 224)): # Standard input size for many top models
    try:
        print(f"Processing image: {img_path}")
        img = image.load_img(img_path, target_size=target_size)
        img_array = image.img_to_array(img) / 255.0  # Normalize to [0, 1]
        print(f"Image array shape after loading and resizing: {img_array.shape}")

        print(f"Processing mask: {mask_path}")
        mask = Image.open(mask_path).resize(target_size, Image.NEAREST) # Resize mask to the target size
        mask_array = np.array(mask) / 255.0 # Normalize mask

        # Expand mask dimensions to (height, width, 1)
        mask_array_expanded = np.expand_dims(mask_array, axis=-1)

        # Combine image and mask as separate inputs (more flexible)
        return img_array, mask_array_expanded
    except Exception as e:
        print(f"Error loading or preprocessing image/mask: {e}")
        return None, None

processed_image_data = []
processed_mask_data = []
corresponding_labels = []

for img_path, mask_path, label in zip(image_paths, mask_paths, categorical_labels):
    img, mask = load_and_preprocess_top_model(img_path, mask_path)
    if img is not None and mask is not None:
        processed_image_data.append(img)
        processed_mask_data.append(mask)
        corresponding_labels.append(label)

processed_image_data = np.array(processed_image_data)
processed_mask_data = np.array(processed_mask_data)
corresponding_labels = np.array(corresponding_labels)

# Ensure we have data after preprocessing
if not processed_image_data.shape[0] == len(corresponding_labels):
    print("Error: Number of processed images does not match the number of labels. Check loading and preprocessing.")
    exit()

# --- 4. Split data into training and validation sets ---
X_img_train, X_img_val, X_mask_train, X_mask_val, y_train, y_val = train_test_split(
    processed_image_data, processed_mask_data, corresponding_labels, test_size=0.2, stratify=corresponding_labels
)

# --- 5. Define F1 Score Metric for Keras ---
def f1_score(y_true, y_pred):
    def recall_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision, recall = precision_m(y_true, y_pred), recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

# --- 6. Define a CNN model with a pre-trained base ---
IMG_SHAPE = X_img_train.shape[1:]
MASK_SHAPE = X_mask_train.shape[1:]

# Load a pre-trained model (e.g., ResNet50V2) without the top (classification) layer
base_model = ResNet50V2(include_top=False, weights='imagenet', input_shape=IMG_SHAPE)
base_model.trainable = False # Freeze the base model's weights

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
output_layer = Dense(num_classes, activation='softmax')(dropout_layer)

# Create the final model with two inputs and one output
model = keras.Model(inputs=[input_image, input_mask], outputs=output_layer)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[f1_score])

# Print the model summary
model.summary()

# --- 7. Define Early Stopping Callback ---
early_stopping = EarlyStopping(
    monitor='val_f1_score',
    patience=20, # Number of epochs with no improvement after which training will be stopped
    verbose=1, # 0: silent, 1: show updates, 2: show more updates
    mode='max', # 'max' for accuracy and F1-score, 'min' for loss
    restore_best_weights=True # Restore model weights from the epoch with the best value of the monitored quantity.
)

# --- 8. Train the model with Early Stopping ---
epochs = 200 # Increased max epochs to allow early stopping to take effect
batch_size = 8 # Smaller batch size might help with small dataset

history = model.fit(
    [X_img_train, X_mask_train], y_train,
    epochs=epochs,
    batch_size=batch_size,
    validation_data=([X_img_val, X_mask_val], y_val),
    callbacks=[early_stopping] # Pass the early stopping callback here
)

# --- 9. Evaluate the model ---
loss, f1 = model.evaluate([X_img_val, X_mask_val], y_val, verbose=0)
print(f"\nValidation Loss: {loss:.4f}")
print(f"Validation F1 Score: {f1:.4f}")

# Save the trained model ---
model.save('organoid_classifier_top_model_f1_early_stopping.h5')
print("\nTop model classifier saved as 'organoid_classifier_top_model_f1_early_stopping.h5'")

# Visualize training history (if you have matplotlib) ---
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['f1_score'], label='Train F1 Score')
plt.plot(history.history['val_f1_score'], label='Validation F1 Score')
plt.xlabel('Epoch')
plt.ylabel('F1 Score')
plt.legend()
plt.title('Training and Validation F1 Score')
plt.savefig('training_f1_score_top_model_early_stopping.png') # Save the F1 score plot

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.savefig('training_loss_top_model_early_stopping.png') # Save the loss plot

print("\nTraining history plots saved as 'training_f1_score_top_model_early_stopping.png' and 'training_loss_top_model_early_stopping.png'")