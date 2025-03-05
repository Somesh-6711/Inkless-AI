import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, LSTM, Dropout, Reshape, BatchNormalization, Input
from tensorflow.keras.models import Model
import numpy as np
import cv2
import os

# Check TensorFlow version (Ensure it's TensorFlow 2.x)
assert tf.__version__.startswith('2'), "This script requires TensorFlow 2.x!"

# Define image dimensions
IMG_HEIGHT, IMG_WIDTH = 28, 28
NUM_CLASSES = 26  # A-Z (26 output classes)

# Load dataset (Use IAM dataset or custom handwritten characters)
def load_dataset(data_path):
    images = []
    labels = []
    for filename in os.listdir(data_path):
        if filename.endswith(".png"):
            img = cv2.imread(os.path.join(data_path, filename), cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
            img = img.astype(np.float32) / 255.0
            images.append(img)
            labels.append(filename[0])  # Assuming filename starts with label (e.g., "A_123.png")
    return np.array(images).reshape(-1, IMG_WIDTH, IMG_HEIGHT, 1), np.array(labels)

# Load data
X_train, y_train = load_dataset("data/handwritten_chars/")

# Convert labels to categorical (A-Z => 26 classes)
y_train = np.array([ord(label) - ord('A') for label in y_train])  # Convert 'A'-'Z' to 0-25
y_train = tf.keras.utils.to_categorical(y_train, num_classes=NUM_CLASSES)

# Define CRNN Model
def build_model():
    inputs = Input(shape=(IMG_WIDTH, IMG_HEIGHT, 1))

    # Convolutional layers
    x = Conv2D(32, (3, 3), activation="relu", padding="same")(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Reshape for LSTM layers
    x = Reshape((7, 128))(x)  # 7 time-steps with 128 features each
    x = LSTM(256, return_sequences=True)(x)
    x = LSTM(256, return_sequences=False)(x)
    
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.5)(x)
    outputs = Dense(NUM_CLASSES, activation="softmax")(x)

    model = Model(inputs, outputs)
    return model

# Compile the model
model = build_model()
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

# Save model as .h5
model_save_path = "models/handwriting_recognition/handwriting_model.h5"
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
model.save(model_save_path)

print(f"✅ Model training complete! Saved to '{model_save_path}'")
