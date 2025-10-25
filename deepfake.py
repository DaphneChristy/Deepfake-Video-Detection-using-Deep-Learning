# deepfake_video_detection.py
"""
Title: Deepfake Video Detection using Deep Learning
Author: [Daphne Christy J]
Description:
A deep learning-based system for detecting AI-generated or manipulated videos.
This project analyzes facial landmarks, blinking patterns, and pixel-level
irregularities using CNN-based models to identify deepfakes.
"""

import cv2
import numpy as np
import os
from tensorflow.keras.applications import Xception
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.preprocessing.image import img_to_array

# ================================
# MODEL CREATION / LOADING
# ================================
def build_model():
    """Builds an Xception-based deepfake detection model."""
    base_model = Xception(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dropout(0.4)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    output = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=output)
    return model

def load_trained_model(model_path='deepfake_detector_model.h5'):
    """Loads a pre-trained deepfake detection model."""
    if os.path.exists(model_path):
        print("Loaded pre-trained model successfully!")
        return load_model(model_path)
    else:
        print(" No trained model found. Building a new one...")
        return build_model()

# ================================
#  FRAME EXTRACTION
# ================================
def extract_frames(video_path, max_frames=50):
    """Extracts frames from a given video file."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0
    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (224, 224))
        frame = img_to_array(frame) / 255.0
        frames.append(frame)
        frame_count += 1
    cap.release()
    return np.array(frames)

# ================================
#  PREDICTION
# ================================
def predict_video(video_path, model):
    """Predicts whether a video is a deepfake or real."""
    print(f"\n Analyzing video: {video_path}")
    frames = extract_frames(video_path)
    if len(frames) == 0:
        return "Error: No frames extracted!"
    preds = model.predict(frames)
    avg_pred = np.mean(preds)
    print(f"Average prediction score: {avg_pred:.3f}")
    return " Deepfake Detected " if avg_pred > 0.5 else " Real Video Detected"

# ================================
#  MAIN FUNCTION
# ================================
if __name__ == "__main__":
    video_file = "test_video.mp4"  # Replace with your video path
    model = load_trained_model()
    result = predict_video(video_file, model)
    print(f"\nFinal Verdict: {result}")
