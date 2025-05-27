# 🎭 Real-Time Emotion Detection from Facial Expressions

A deep learning–based emotion recognition system trained on the FER-2013 dataset and tested on real-time webcam input using PyTorch and OpenCV.

---

## 📂 Project Structure

```markdown
emotion_detector/
├── assets/
│ ├── emotion_log.csv
│ └── confusion_matrix.png
├── data/
│ └── fer2013/
│ ├── train.csv
│ └── test.csv
├── models/
│ └── custom_emotion_cnn.pth
├── src/
│ ├── main.py
│ ├── model.py
│ ├── train_model.py
│ ├── camera_emotion_custom.py
│ ├── emotion_utils.py
│ ├── evaluate_model.py
│ └── analyze_emotions.py
├── FERPlus/ (optional)
│ ├── src/
│ └── data/
├── requirements.txt
└── README.md
```

> ⚠️ Due to GitHub file size limits, some large files like `train.csv`, `test.csv`, and the trained model `custom_emotion_cnn.pth` are not included. Instructions for obtaining them are provided below.

---

## 🧠 Model Info

- **Architecture**: Custom CNN (built from scratch)
- **Dataset**: [FER-2013](https://www.kaggle.com/datasets/msambare/fer2013)
- **Framework**: PyTorch
- **Test Accuracy**: ~61%

---

## 📊 Evaluation Metrics

| Emotion  | Precision | Recall | F1-score | Support |
|----------|-----------|--------|----------|---------|
| Angry    | 0.46      | 0.54   | 0.50     | 467     |
| Disgust  | 0.68      | 0.48   | 0.56     | 56      |
| Fear     | 0.53      | 0.41   | 0.46     | 496     |
| Happy    | 0.80      | 0.80   | 0.80     | 895     |
| Sad      | 0.55      | 0.45   | 0.49     | 653     |
| Surprise | 0.79      | 0.76   | 0.77     | 415     |
| Neutral  | 0.47      | 0.60   | 0.53     | 607     |

- **Overall Accuracy**: 61% (on 3589 test samples)

---

## 🎥 Run Real-Time Emotion Detection

Make sure your webcam is connected, and then run:

```bash
python src/camera_emotion_custom.py
```
Predictions will appear live on your face in the webcam feed.

## ⚙️ Setup Instructions
1. Clone the repository
```bash
git clone https://github.com/alireza-taheriF/Emotion-Detection.git
cd Emotion-Detection
```

2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

. Download FER-2013 dataset: https://www.kaggle.com/datasets/msambare/fer2013
→ Place train.csv and test.csv inside data/fer2013/
. Either:
. Train the model using src/train_model.py
. Or use the pre-trained model by requesting custom_emotion_cnn.pth from the author

## 🪪 License
MIT License
Author: Alireza Taheri Fakhr
Purpose: Research experience and resume-building for a future multi-modal (audio-visual) emotion detection system

## 📫 Contact
Feel free to connect:
📧 alirezataherifakhr@gmail.com
💼 LinkedIn: https://www.linkedin.com/in/alireza-taheri-a34179164/


