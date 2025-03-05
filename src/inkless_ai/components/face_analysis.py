import cv2
import numpy as np
from deepface import DeepFace

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Define gender labels
GENDER_LABELS = {"Man": "Male", "Woman": "Female"}

def preprocess_face(face_img):
    """Prepares the face image for DeepFace analysis."""
    face_img = cv2.resize(face_img, (224, 224))  # Resize to DeepFace-friendly dimensions
    return face_img

def predict_age_gender(face_img):
    """Predicts age & gender using DeepFace."""
    try:
        analysis = DeepFace.analyze(face_img, actions=["age", "gender"], enforce_detection=False)
        age = analysis[0]["age"]
        gender = analysis[0]["dominant_gender"].capitalize()

        # Map DeepFace gender output to standard Male/Female
        gender = GENDER_LABELS.get(gender, gender)

        return str(age), gender

    except Exception as e:
        print(f"DeepFace prediction error: {e}")
        return "Unknown", "Unknown"

def detect_faces(frame):
    """Detects faces and predicts age & gender."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]  # Crop detected face
        age, gender = predict_age_gender(face_img)

        # Draw bounding box and label
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        label = f"{gender}, Age: {age}"
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return frame
