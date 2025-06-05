import os
import requests
import time
import streamlit as st
from datetime import datetime
import pytz
import pandas as pd
import cv2 # Make sure you have OpenCV installed if you're processing images directly in app.py

# It's good practice to ensure these imports are consistent if you need them for safe_globals in app.py too
import torch.serialization
import torch.nn.modules.container
import ultralytics.nn.tasks
import ultralytics.nn.modules
import ultralytics.nn.modules.conv
import ultralytics.nn.modules.block
from ultralytics import YOLO

# Import QueueAnalyzer - ensure queue_analyzer.py is accessible to app.py
from queue_analyzer import QueueAnalyzer

# --- Safe Globals for PyTorch (Necessary if app.py also loads the model) ---
# ✅ Optional SPPF registration
try:
    import ultralytics.nn.modules.common as ul_common
    sppf = ul_common.SPmonmonF
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
# --- End Safe Globals ---

# --- Configuration ---
MODEL_URL = "https://ultralytics.com/assets/yolov8s.pt"
MODEL_PATH = "yolov8s.pt"
CAMERA_URL = "https://thumbs.balticlivecam.com/blc/narva.jpg"
TIMEZONE = "Europe/Tallinn"
tz = pytz.timezone(TIMEZONE)
QUEUE_HISTORY_CSV = 'queue_history.csv' # Define the CSV file path

# --- Helper Functions ---

@st.cache_data
def download_yolo_model(model_path, model_url):
    """Downloads the YOLOv8s model if it doesn't exist or is incomplete.
       Cached with Streamlit to avoid re-downloading on every rerun.
    """
    if not os.path.exists(model_path) or os.path.getsize(model_path) < 10000000:
        st.warning(f"Downloading YOLOv8s model to {model_path}...")
        try:
            r = requests.get(model_url, stream=True)
            r.raise_for_status()
            with open(model_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            st.success("YOLOv8s model downloaded successfully.")
        except requests.exceptions.RequestException as e:
            st.error(f"Error downloading model: {e}. Please check the URL or network connection.")
            return False # Indicate download failure
    return True # Indicate download success

@st.cache_resource # Use st.cache_resource for models/large objects
def load_yolo_model(model_path):
    """Loads the YOLO model. Cached with Streamlit to load only once."""
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading YOLO model: {e}. Ensure the model file is valid.")
        return None

@st.cache_data(ttl=60) # Cache for 60 seconds to avoid constant re-reads
def load_queue_history(csv_path):
    """Loads and returns the queue history from CSV."""
    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
        try:
            # Ensure the timestamp column is parsed as datetime
            df = pd.read_csv(csv_path, parse_dates=['timestamp'], date_format='%Y-%m-%dT%H:%M:%S.%f')
            # Set timestamp as index for easier time-based operations
            df['timestamp'] = df['timestamp'].dt.tz_localize('UTC').dt.tz_convert(TIMEZONE)
            return df.set_index('timestamp')
        except pd.errors.EmptyDataError:
            st.warning("Queue history CSV is empty. Waiting for data from worker.")
            return pd.DataFrame(columns=['timestamp', 'person_count']).set_index('timestamp')
        except Exception as e:
            st.error(f"Error loading queue history CSV: {e}")
            return pd.DataFrame(columns=['timestamp', 'person_count']).set_index('timestamp')
    else:
        st.info("Queue history CSV not found or is empty. Data will appear once the worker starts.")
        return pd.DataFrame(columns=['timestamp', 'person_count']).set_index('timestamp')

@st.cache_data(ttl=5) # Cache for 5 seconds to avoid constant image re-fetches
def fetch_latest_image(url):
    """Fetches the latest image from the camera URL."""
    try:
        response = requests.get(url, timeout=10) # Added timeout for robustness
        response.raise_for_status()
        image_array = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)
        return image_array
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching camera image: {e}")
        return None
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None

# --- Streamlit App Layout ---
st.set_page_config(page_title="Narva Queue Monitor", layout="wide", initial_sidebar_state="collapsed")
st.title("🚶 Narva Queue Monitor")
st.markdown("---")

# Initialize Analyzer (only if model is loaded successfully)
analyzer = None

# Download model only once (cached)
if download_yolo_model(MODEL_PATH, MODEL_URL):
    # Load model only once (cached)
    model = load_yolo_model(MODEL_PATH)
    if model:
        analyzer = QueueAnalyzer(model)
        # Ensure analyzer's history is loaded from the common CSV
        analyzer.load_history_from_csv(QUEUE_HISTORY_CSV)
    else:
        st.warning("Model could not be loaded. Live detection features will be limited.")
else:
    st.error("Could not download YOLO model. Please check network connection.")

# --- Display Latest Camera Image ---
st.header("Live Camera Feed")
latest_image_placeholder = st.empty() # Placeholder for the image to allow dynamic updates

# Fetch and display the image (and potentially draw detections)
import numpy as np # Import numpy for cv2.imdecode

image = fetch_latest_image(CAMERA_URL)
if image is not None:
    timestamp = datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
    
    # If model and analyzer are available, perform detection for display
    if analyzer:
        detections = analyzer.detect_pedestrians(image)
        for (x1, y1, x2, y2) in detections:
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Get the latest adjusted count from the analyzer (which loaded from CSV)
        # Or, if you want to show the 'live' detection count from this image:
        base_count_live = len(detections)
        adjusted_count_live = base_count_live + 50 if base_count_live > 0 else 0
        latest_image_placeholder.image(image, caption=f"Last updated: {timestamp} (Live Detections Shown)", channels="BGR", use_column_width=True)
        st.metric("Detected People (Live View)", int(adjusted_count_live))
    else:
        latest_image_placeholder.image(image, caption=f"Last updated: {timestamp} (No Live Detections)", channels="BGR", use_column_width=True)
        st.info("Live detection is unavailable because the model could not be loaded.")
else:
    latest_image_placeholder.error("⚠️ Could not load image from camera feed.")
st.markdown("---")

# --- Display Historical Data and Predictions ---
st.header("Queue History and Predictions")

# Load queue history from the CSV
queue_df = load_queue_history(QUEUE_HISTORY_CSV)

if not queue_df.empty:
    # Display the latest recorded count from history, which is updated by queue_collector.py
    latest_history_count = queue_df['person_count'].iloc[-1] if not queue_df.empty else 0
    st.metric("Latest Recorded People (from History)", int(latest_history_count))

    st.subheader("Queue Trends Over Time")
    # Plotting the queue history
    st.line_chart(queue_df['person_count'])

    # Predictions (assuming QueueAnalyzer can predict based on loaded history)
    # Ensure analyzer is initialized before calling its methods
    if analyzer:
        # Before predicting, make sure the analyzer has the latest data.
        # It's better to ensure QueueAnalyzer loads its history from the CSV
        # when it's initialized or has a method to refresh it.
        # For this example, assuming analyzer.predict_trend() and best_hours_to_cross()
        # use the internal history which is now loaded from the CSV.
        st.info(analyzer.predict_trend())
        st.success(f"Best hours to cross: {analyzer.best_hours_to_cross()}")
    else:
        st.warning("Cannot provide predictions as QueueAnalyzer could not be initialized.")
else:
    st.info("No historical queue data available yet. Please wait for the `queue_collector.py` worker to generate data.")

# --- Auto-refresh feature for Streamlit ---
# This will rerun the script every few seconds to update the display
st.markdown("---")
refresh_interval_sec = st.slider("Auto-refresh interval (seconds)", 5, 30, 10)
time.sleep(refresh_interval_sec)
st.rerun()
