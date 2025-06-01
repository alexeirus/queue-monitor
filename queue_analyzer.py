import cv2
import numpy as np
import pandas as pd
from datetime import datetime
import pytz
import os

QUEUE_AREA = (390, 324, 1276, 595)
MIN_CONFIDENCE = 0.025
MIN_HEIGHT = 10
DENSITY_FACTOR = 0.95
HISTORY_FILE = "queue_history.csv"
TIMEZONE = 'Europe/Tallinn'

class QueueAnalyzer:
    def __init__(self, model):
        self.model = model
        self.tz = pytz.timezone(TIMEZONE)
        self.history = self.load_history()

    def load_history(self):
        if os.path.exists(HISTORY_FILE):
            df = pd.read_csv(HISTORY_FILE, parse_dates=["timestamp"])
        else:
            df = pd.DataFrame(columns=["timestamp", "count", "day_of_week", "hour"])
        return df

    def save_history(self):
        self.history.to_csv(HISTORY_FILE, index=False)

    def update_history(self, count):
        now = datetime.now(self.tz)
        new_entry = {
            "timestamp": now,
            "count": count,
            "day_of_week": now.weekday(),
            "hour": now.hour
        }
        self.history = pd.concat([self.history, pd.DataFrame([new_entry])], ignore_index=True)
        self.save_history()

    def fetch_image(self, url):
        import requests
        from PIL import Image
        from io import BytesIO

        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            img = Image.open(BytesIO(response.content))
            img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            return img[QUEUE_AREA[1]:QUEUE_AREA[3], QUEUE_AREA[0]:QUEUE_AREA[2]]
        except Exception as e:
            print(f"Image fetch failed: {e}")
            return None

    def detect_pedestrians(self, image):
        if image is None:
            return []

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 2)
        results = self.model(image, classes=0, conf=MIN_CONFIDENCE, imgsz=640)
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

    def predict_trend(self):
        if len(self.history) < 4:
            return "Insufficient data"
        recent = self.history.tail(4)["count"].values
        delta = recent[-1] - recent[0]
        if delta > 5:
            return "Queue is growing quickly 📈"
        elif delta > 0:
            return "Queue is growing slowly ↗"
        elif delta < -5:
            return "Queue is shrinking quickly 📉"
        elif delta < 0:
            return "Queue is shrinking slowly ↘"
        return "Queue is stable →"

    def best_hours_to_cross(self):
        if len(self.history) < 24:
            return "Need more data"
        avg_by_hour = self.history.groupby("hour")["count"].mean()
        best_hours = avg_by_hour.sort_values().head(3).index.tolist()
        return ", ".join(f"{h}:00-{h+1}:00" for h in best_hours)
