import os
import requests
import time
from datetime import datetime
import pytz
import torch.serialization
import torch.nn.modules.container
import ultralytics.nn.tasks
import ultralytics.nn.modules.conv
import ultralytics.nn.modules.block
import ultralytics.nn.modules.head
from ultralytics import YOLO
from queue_analyzer import QueueAnalyzer

MODEL_URL = "https://ultralytics.com/assets/yolov8s.pt"
MODEL_PATH = "yolov8s.pt"
CAMERA_URL = "https://thumbs.balticlivecam.com/blc/narva.jpg"
TIMEZONE = "Europe/Tallinn"
tz = pytz.timezone(TIMEZONE)

# ✅ PyTorch 2.6+ safe class registration
torch.serialization.add_safe_globals([
    ultralytics.nn.tasks.DetectionModel,
    ultralytics.nn.modules.conv.Conv,
    ultralytics.nn.modules.conv.C2f,
    ultralytics.nn.modules.block.Bottleneck,
    ultralytics.nn.modules.block.Concat,
    ultralytics.nn.modules.head.SPPF,  # ✅ now from correct location
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
