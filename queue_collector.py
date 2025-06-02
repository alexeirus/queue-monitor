import os
import requests
import time
from datetime import datetime
import pytz
import torch.serialization
import torch.nn.modules.container  # 👈 Required for PyTorch 2.6+
from ultralytics import YOLO
from queue_analyzer import QueueAnalyzer

MODEL_URL = "https://ultralytics.com/assets/yolov8s.pt"
MODEL_PATH = "yolov8s.pt"
CAMERA_URL = "https://thumbs.balticlivecam.com/blc/narva.jpg"
TIMEZONE = "Europe/Tallinn"
tz = pytz.timezone(TIMEZONE)

# 🧠 PyTorch 2.6+ compatibility: allow required globals
torch.serialization.add_safe_globals([
    __import__("ultralytics.nn.tasks").nn.tasks.DetectionModel,
    torch.nn.modules.container.Sequential
])

# 📥 Download YOLOv8s model if needed
if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 10000000:
    print("Downloading YOLOv8s model...")
    r = requests.get(MODEL_URL, stream=True)
    with open(MODEL_PATH, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)

# ✅ Load model + analyzer
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
