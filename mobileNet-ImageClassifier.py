import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras import Sequential, layers, callbacks, optimizers, Input
from tensorflow.keras.layers import Dense,Flatten,GlobalAveragePooling2D,Conv2D,MaxPool2D,Dropout,BatchNormalization
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing import image
from keras.metrics import BinaryAccuracy, AUC
from tqdm import tqdm
from collections import Counter

with open('balanced_output_binary_vectors_no_snow.json', 'r') as f:
    labels_dict = json.load(f)

image_dir = 'static/images_balanced'
image_keys = list(labels_dict.keys())

label_order = ["Summit Visible", "Clear Sky"]

train_files, test_val_files = train_test_split(image_keys, test_size=0.2, random_state=30)
val_files, test_files = train_test_split(test_val_files, test_size=0.5, random_state=30)

print("Training label distribution:")
print(Counter([tuple(labels_dict[file]) for file in train_files]))
print("Validation label distribution:")
print(Counter([tuple(labels_dict[file]) for file in val_files]))
print("Test label distribution:")
print(Counter([tuple(labels_dict[file]) for file in test_files]))

def weighted_binary_crossentropy(pos_weights):
    def loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        bce = - (y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred))
        weighted_bce = bce * (y_true * pos_weights + (1 - y_true))
        return tf.reduce_mean(weighted_bce)
    return loss

# Give more weight to summit visible = 1 and clear sky = 0
pos_weights = tf.constant([1.0, 3.0], dtype=tf.float32)

data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomZoom(0.1)
])

def preprocess_image(img_path, label):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [400, 400])
    img = img / 255.0
    img = data_augmentation(img)
    return img, label

def create_dataset(image_files, labels_dict, batch_size=32):
    image_paths = []
    labels = []
    for img_file in image_files:
        img_path = os.path.join(image_dir, img_file)
        label_vector = labels_dict.get(img_file, [])
        image_paths.append(img_path)
        labels.append(label_vector)
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    
    dataset = dataset.map(preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset

train_dataset = create_dataset(train_files, labels_dict, batch_size=32)
val_dataset = create_dataset(val_files, labels_dict, batch_size=32)
test_dataset = create_dataset(test_files, labels_dict, batch_size=32)

for images, labels in val_dataset.take(1):
    print("Labels from dataset:", labels.numpy())
    for i in range(min(5, images.shape[0])):
        plt.imshow(images[i])
        plt.title(f"Label: {labels.numpy()[i]}")
        plt.axis('off')
        plt.show()
    break

# Load MobileNetV2 pretrained on ImageNet, excluding its top classification layers
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(400, 400, 3))
base_model.trainable = False  # Freeze the base model to retain learned features

inputs = Input(shape=(400, 400, 3))

x = base_model(inputs, training=False)

x = GlobalAveragePooling2D()(x)

x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)

outputs = Dense(2, activation='sigmoid')(x)

model = Model(inputs, outputs)

model.summary()

initial_lr = 0.0005
lr_schedule = optimizers.schedules.ExponentialDecay(
    initial_learning_rate = initial_lr,
    decay_steps=10000,
    decay_rate=0.9,
    staircase=True
)

def subset_accuracy(y_true, y_pred):
    y_pred = tf.round(y_pred)
    equal = tf.reduce_all(tf.equal(y_true, y_pred), axis=-1)
    return tf.reduce_mean(tf.cast(equal, tf.float32))

early_stopping=callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
checkpoint_callback=callbacks.ModelCheckpoint(
    "model_checkpoint.keras",
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=15,
    callbacks=[early_stopping, checkpoint_callback]
)

model.save('model_test.keras')

test_loss, test_acc = model.evaluate(test_dataset)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_acc}")

for images, labels in val_dataset.take(1):
    preds = model.predict(images)
    print("Predictions (rounded):\n", np.round(preds[:10]))
    print("True Labels:\n", labels.numpy()[:10])

for x, y in train_dataset.take(1):
    print("Shape of x:", x.shape)
    print("Shape of y:", y.shape)
    print("First 5 labels:\n", y[:5].numpy())

raw_preds = model.predict(test_dataset)
print("First 10 raw predictions:\n", raw_preds[:10])

correct_count = 0
total_count = 0

# Iterate over the entire validation dataset
for images, labels in val_dataset:
    # Get predictions for the current batch
    preds = model.predict(images)
    # Round the predictions to 0 or 1
    rounded_preds = np.round(preds)
    # Convert labels to numpy array for easier comparison
    true_labels = labels.numpy()
    
    # Compare each prediction with its true label
    for pred, true in zip(rounded_preds, true_labels):
        total_count += 1
        if np.array_equal(pred, true):
            correct_count += 1

print(f"{correct_count}/{total_count} right")

correct_count = 0
total_count = 0

# Iterate over the entire validation dataset
for images, labels in val_dataset:
    # Get predictions for the current batch
    preds = model.predict(images)
    # Round the predictions to 0 or 1
    rounded_preds = np.round(preds)
    # Convert labels to numpy array for easier comparison
    true_labels = labels.numpy()
    
    # Compare each prediction with its true label
    for pred, true in zip(rounded_preds, true_labels):
        total_count += 1
        if np.array_equal(pred, true):
            correct_count += 1

print(f"{correct_count}/{total_count} right")

plt.figure(figsize=(10, 4))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training vs. Validation Accuracy')
plt.legend()
plt.show()

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt

all_preds = []
all_labels = []
for images, labels in val_dataset:
    preds = model.predict(images)
    all_preds.extend(np.round(preds).astype(int))
    all_labels.extend(labels.numpy().astype(int))

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

# For each label (column) compute confusion matrix and metrics:
for i, label_name in enumerate(["Summit Visible", "Clear Sky"]):
    cm = confusion_matrix(all_labels[:, i], all_preds[:, i])
    
    # Unpack confusion matrix: 
    # [ [TN, FP],
    #   [FN, TP] ]
    TN, FP, FN, TP = cm.ravel()
    
    # Compute Recall and Precision:
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    
    print(f"--- {label_name} ---")
    print("Confusion Matrix:")
    print(cm)
    print(f"Recall: {recall*100:.1f}%")
    print(f"Precision: {precision*100:.1f}%\n")
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix for {label_name}')
    plt.show()