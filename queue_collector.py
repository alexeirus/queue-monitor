
from ultralytics import YOLO
from queue_analyzer import QueueAnalyzer
from datetime import datetime
import time
import pytz

MODEL_PATH = "yolov8s.pt"
CAMERA_URL = "https://thumbs.balticlivecam.com/blc/narva.jpg"
TIMEZONE = "Europe/Tallinn"
tz = pytz.timezone(TIMEZONE)

model = YOLO(MODEL_PATH)
analyzer = QueueAnalyzer(model)

def run_cycle():
    now = datetime.now(tz)
    if 5 <= now.hour <= 23:
        print(f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] Running cycle...")
        image = analyzer.fetch_image(CAMERA_URL)
        if image is not None:
            detections = analyzer.detect_pedestrians(image)
            base_count = len(detections)
            adjusted_count = base_count + 50 if base_count > 0 else 0
            analyzer.update_history(adjusted_count)
            print(f"Detected {base_count} (+50 offset → {adjusted_count}) people.")
        else:
            print("Image fetch failed.")
    else:
        print(f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] Outside monitoring hours.")

if __name__ == "__main__":
    while True:
        run_cycle()
        time.sleep(180)
