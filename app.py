import streamlit as st
import os
import requests
import torch.serialization
import torch.nn.modules.container
import ultralytics.nn.tasks
import ultralytics.nn.modules.conv
import ultralytics.nn.modules.block
import ultralytics.nn.modules.pool
from ultralytics import YOLO
from queue_analyzer import QueueAnalyzer
from datetime import datetime
import pytz
import cv2

# Constants
CAMERA_URL = "https://thumbs.balticlivecam.com/blc/narva.jpg"
MODEL_URL = "https://ultralytics.com/assets/yolov8s.pt"
MODEL_PATH = "yolov8s.pt"
TIMEZONE = "Europe/Tallinn"

# ✅ PyTorch 2.6+ safe class registration
torch.serialization.add_safe_globals([
    ultralytics.nn.tasks.DetectionModel,
    ultralytics.nn.modules.conv.Conv,
    ultralytics.nn.modules.conv.C2f,
    ultralytics.nn.modules.block.Bottleneck,
    ultralytics.nn.modules.block.Concat,
    ultralytics.nn.modules.pool.SPPF,
    torch.nn.modules.container.Sequential
])

# 📥 Download model if not already available or corrupt
if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 10000000:
    print("Downloading YOLOv8s model...")
    r = requests.get(MODEL_URL, stream=True)
    with open(MODEL_PATH, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)

# ✅ Load model and analyzer
model = YOLO(MODEL_PATH)
analyzer = QueueAnalyzer(model)
tz = pytz.timezone(TIMEZONE)

# 🌐 Streamlit UI
st.set_page_config(page_title="Queue Monitor", layout="wide")
st.title("🚶 Narva Queue Monitor")

image = analyzer.fetch_image(CAMERA_URL)
if image is not None:
    detections = analyzer.detect_pedestrians(image)
    base_count = len(detections)
    adjusted_count = base_count + 50 if base_count > 0 else 0
    analyzer.update_history(adjusted_count)

    for (x1, y1, x2, y2) in detections:
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    timestamp = datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
    st.image(image, caption=f"Last updated: {timestamp}", channels="BGR")
    st.metric("People in Queue", int(adjusted_count))
    st.info(analyzer.predict_trend())
    st.success(f"Best hours to cross: {analyzer.best_hours_to_cross()}")
else:
    st.error("⚠️ Could not load image from camera feed.")
