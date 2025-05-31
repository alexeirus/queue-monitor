import streamlit as st
import pandas as pd
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
from datetime import datetime
from ultralytics import YOLO
import requests
import pytz
import os
import requests

MODEL_URL = "https://ultralytics.com/assets/yolov8s.pt"
MODEL_PATH = "yolov8s.pt"

# Download the model if not already present
if not os.path.exists(MODEL_PATH):
    print("Downloading YOLO model...")
    response = requests.get(MODEL_URL)
    with open(MODEL_PATH, "wb") as f:
        f.write(response.content)

# Load the model
model = YOLO(MODEL_PATH)

# Config
QUEUE_AREA = (390, 324, 1276, 595)
MIN_CONFIDENCE = 0.025
MIN_HEIGHT = 10
DENSITY_FACTOR = 0.95
HISTORY_FILE = "queue_history.csv"
TIMEZONE = 'Europe/Tallinn'

# Load model and history
tz = pytz.timezone(TIMEZONE)

if os.path.exists(HISTORY_FILE):
    history = pd.read_csv(HISTORY_FILE)
    history['timestamp'] = pd.to_datetime(history['timestamp'])
else:
    history = pd.DataFrame(columns=['timestamp', 'count', 'day_of_week', 'hour'])

# Functions
def fetch_image(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content))
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        return img[QUEUE_AREA[1]:QUEUE_AREA[3], QUEUE_AREA[0]:QUEUE_AREA[2]]
    except:
        return None

def detect_pedestrians(image):
    if image is None:
        return []

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    results = model(image, classes=0, conf=MIN_CONFIDENCE, imgsz=640)
    detections = []

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            height = y2 - y1
            if height > MIN_HEIGHT:
                roi = thresh[y1:y2, x1:x2]
                if np.sum(roi) > 0.05 * 255 * roi.size:
                    detections.append((x1, y1, x2, y2))

    return detections

def predict_trend(history):
    if len(history) < 4:
        return "Insufficient data for trend prediction"
    recent = history.tail(4)['count'].values
    diff = recent[-1] - recent[0]
    if diff > 5:
        return "Queue is growing quickly 📈"
    elif diff > 0:
        return "Queue is growing slowly ↗"
    elif diff < -5:
        return "Queue is shrinking quickly 📉"
    elif diff < 0:
        return "Queue is shrinking slowly ↘"
    return "Queue is stable →"

def best_hours_to_cross(history):
    if len(history) < 24:
        return "Need more data for recommendations"
    avg_by_hour = history.groupby('hour')['count'].mean()
    best_hours = avg_by_hour.sort_values().head(3).index.tolist()
    return ", ".join(f"{h}:00-{h+1}:00" for h in best_hours)

def update_history(count):
    now = datetime.now(tz)
    new_entry = {'timestamp': now, 'count': count, 'day_of_week': now.weekday(), 'hour': now.hour}
    global history
    history = pd.concat([history, pd.DataFrame([new_entry])], ignore_index=True)
    history.to_csv(HISTORY_FILE, index=False)

# Streamlit UI
st.set_page_config(page_title="Queue Monitor", layout="wide")
st.title("🔍 Live Queue Monitor")

image = fetch_image("https://thumbs.balticlivecam.com/blc/narva.jpg")
if image is not None:
    detections = detect_pedestrians(image)
    count = min(len(detections) * DENSITY_FACTOR, len(detections) * 3)
    update_history(count)

    trend = predict_trend(history)
    best_times = best_hours_to_cross(history)

    for (x1, y1, x2, y2) in detections:
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    st.image(image, caption=f"Live View ({datetime.now(tz).strftime('%Y-%m-%d %H:%M:%S')})", channels="BGR")
    st.metric("Current Queue Count", int(count))
    st.info(trend)
    st.success(f"Best Times to Cross: {best_times}")
else:
    st.error("Unable to fetch image from camera feed.")
