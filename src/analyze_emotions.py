# src/analyze_emotions.py

import pandas as pd
import matplotlib.pyplot as plt
import os

# مسیر فایل لاگ
csv_path = os.path.join(os.path.dirname(__file__), '..', 'assets', 'emotion_log.csv')

def load_data():
    df = pd.read_csv(csv_path, parse_dates=['timestamp'])
    return df

def plot_emotion_counts(df):
    # گروه‌بندی بر اساس احساس و شمارش
    counts = df['emotion'].value_counts()
    counts.plot(kind='bar')
    plt.title("Emotion Frequency")
    plt.xlabel("Emotion")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

def plot_emotion_timeline(df):
    # تعداد احساسات در پنجره‌های زمانی (مثلاً هر 10 ثانیه)
    df.set_index('timestamp', inplace=True)
    # گروه‌بندی هر 10 ثانیه
    resampled = df['emotion'].groupby(pd.Grouper(freq='10S')).value_counts().unstack(fill_value=0)
    resampled.plot()
    plt.title("Emotion Timeline (10s windows)")
    plt.xlabel("Time")
    plt.ylabel("Count per window")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    df = load_data()
    print("Loaded", len(df), "records")
    plot_emotion_counts(df)
    plot_emotion_timeline(df)
