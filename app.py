
import streamlit as st
from ultralytics import YOLO
from queue_analyzer import QueueAnalyzer
from datetime import datetime
import pytz
import cv2

# Constants
CAMERA_URL = "https://thumbs.balticlivecam.com/blc/narva.jpg"
MODEL_PATH = "yolov8s.pt"
TIMEZONE = "Europe/Tallinn"

# Load model and analyzer
model = YOLO(MODEL_PATH)
analyzer = QueueAnalyzer(model)
tz = pytz.timezone(TIMEZONE)

# Streamlit UI
st.set_page_config(page_title="Queue Monitor", layout="wide")
st.title("🚶 Narva Queue Monitor")

image = analyzer.fetch_image(CAMERA_URL)
if image is not None:
    detections = analyzer.detect_pedestrians(image)
    base_count = len(detections)
    adjusted_count = base_count + 50 if base_count > 0 else 0
    analyzer.update_history(adjusted_count)

    # Draw boxes
    for (x1, y1, x2, y2) in detections:
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    timestamp = datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
    st.image(image, caption=f"Last updated: {timestamp}", channels="BGR")
    st.metric("People in Queue", int(adjusted_count))
    st.info(analyzer.predict_trend())
    st.success(f"Best hours to cross: {analyzer.best_hours_to_cross()}")
else:
    st.error("⚠️ Could not load image from camera feed.")
