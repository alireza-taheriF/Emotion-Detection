# src/camera_emotion_custom.py
import os
import time
import cv2
import torch
import numpy as np
from torchvision import transforms
from model import EmotionCNN

# Configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'custom_emotion_cnn.pth')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model
def load_model():
    model = EmotionCNN()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

# Emotion mapping
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Preprocessing transform
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Run detection and measure FPS
def run_custom_emotion_detector():
    model = load_model()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam")
        return

    frame_count = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Prepare face ROI; here we use entire frame
        img = cv2.resize(frame, (48, 48))
        img_tensor = transform(img).unsqueeze(0).to(DEVICE)

        # Inference
        with torch.no_grad():
            outputs = model(img_tensor)
            _, pred = torch.max(outputs, 1)
            emotion = EMOTIONS[pred.item()]

        # Display
        cv2.putText(frame, emotion, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.imshow('Custom Model Emotion', frame)

        frame_count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Calculate FPS
    elapsed = time.time() - start_time
    fps = frame_count / elapsed if elapsed > 0 else 0
    print(f"Processed {frame_count} frames in {elapsed:.2f}s — FPS: {fps:.2f}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    run_custom_emotion_detector()
