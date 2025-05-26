# src/emotion_utils.py

import csv
import os
from datetime import datetime

CSV_PATH = os.path.join(os.path.dirname(__file__), '..', 'assets', 'emotion_log.csv')

def init_csv():
    """اگر فایل وجود نداره، هدر بنویس."""
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    if not os.path.isfile(CSV_PATH):
        with open(CSV_PATH, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'frame_idx', 'emotion'])

def log_emotion(frame_idx: int, emotion: str):
    """یک ردیف جدید با زمان، شماره فریم و احساس بنویس."""
    timestamp = datetime.utcnow().isoformat()
    with open(CSV_PATH, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, frame_idx, emotion])
