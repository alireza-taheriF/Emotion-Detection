# src/camera_emotion.py

import cv2
from deepface import DeepFace
from emotion_utils import init_csv, log_emotion

def run_emotion_detector():
    init_csv()
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Cannot access webcam")
        return

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        try:
            result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            dominant_emotion = result[0]['dominant_emotion']
            # لاگ کردن
            log_emotion(frame_idx, dominant_emotion)
            # نمایش روی تصویر
            cv2.putText(frame, f"{dominant_emotion}", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        except Exception as e:
            print(f"Error at frame {frame_idx}:", e)

        cv2.imshow("Live Emotion Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
