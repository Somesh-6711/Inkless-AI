import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, LSTM, Dropout
from tensorflow.keras.models import Sequential
import numpy as np
import cv2
import os

# Load dataset (Use IAM Handwriting Dataset or your own)
# Assume images are in 'data/handwritten_chars/' and labels in 'data/labels.txt'
IMG_SIZE = (28, 28)

def load_dataset(data_path):
    images = []
    labels = []
    for filename in os.listdir(data_path):
        if filename.endswith(".png"):
            img = cv2.imread(os.path.join(data_path, filename), cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, IMG_SIZE)
            img = img.astype(np.float32) / 255.0
            images.append(img)
            labels.append(filename[0])  # Assuming filename starts with label (e.g., "A_123.png")
    return np.array(images).reshape(-1, 28, 28, 1), np.array(labels)

X_train, y_train = load_dataset("data/handwritten_chars/")

# Convert labels to categorical (A-Z => 26 classes)
y_train = np.array([ord(label) - ord('A') for label in y_train])  # Convert 'A'-'Z' to 0-25
y_train = tf.keras.utils.to_categorical(y_train, num_classes=26)

# Define model (CNN + LSTM for sequential handwriting recognition)
model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.2),
    Dense(26, activation="softmax")  # 26 output classes for A-Z
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

# Save model
model.save("models/handwriting_recognition/handwriting_model.h5")

print("Model training complete! Saved to 'models/handwriting_recognition/handwriting_model.h5'")
