import streamlit as st
import os
import requests
from datetime import datetime
import pytz
import cv2
import torch.serialization
import torch.nn.modules.container
import ultralytics.nn.tasks
import ultralytics.nn.modules
import ultralytics.nn.modules.conv
import ultralytics.nn.modules.block
from ultralytics import YOLO
from queue_analyzer import QueueAnalyzer

# ✅ Optional SPPF registration
try:
    import ultralytics.nn.modules.common as ul_common
    sppf = ul_common.SPPF
except (ImportError, AttributeError):
    sppf = None

# ✅ Register required globals
safe_globals = [
    ultralytics.nn.tasks.DetectionModel,
    ultralytics.nn.modules.Conv,
    ultralytics.nn.modules.conv.Conv,
    ultralytics.nn.modules.conv.Concat,
    ultralytics.nn.modules.block.C2f,
    ultralytics.nn.modules.block.Bottleneck,
    torch.nn.modules.container.Sequential
]
if sppf:
    safe_globals.append(sppf)

torch.serialization.add_safe_globals(safe_globals)

# Config
CAMERA_URL = "https://thumbs.balticlivecam.com/blc/narva.jpg"
MODEL_URL = "https://ultralytics.com/assets/yolov8s.pt"
MODEL_PATH = "yolov8s.pt"
TIMEZONE = "Europe/Tallinn"
tz = pytz.timezone(TIMEZONE)

# ✅ Model download helper
def ensure_model_downloaded():
    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 10000000:
        st.warning("Downloading YOLOv8s model...")
        r = requests.get(MODEL_URL, stream=True)
        with open(MODEL_PATH, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

# ✅ Streamlit App
st.set_page_config(page_title="Queue Monitor", layout="wide")
st.title("🚶 Narva Queue Monitor")

# Make sure model is downloaded before loading
ensure_model_downloaded()

# Load YOLO and analyzer
model = YOLO(MODEL_PATH)
analyzer = QueueAnalyzer(model)

# Main logic
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
