import os
import requests
import time
import streamlit as st
from datetime import datetime
import pytz
import pandas as pd
import cv2
import numpy as np
import base64 # Import base64 for decoding credentials
from google.cloud import storage # Import GCS client
from io import BytesIO # For reading CSV from bytes

# It's good practice to ensure these imports are consistent if you need them for safe_globals in app.py too
import torch.serialization
import torch.nn.modules.container
import ultralytics.nn.tasks
import ultralytics.nn.modules
import ultralytics.nn.modules.conv
import ultralytics.nn.modules.block
from ultralytics import YOLO

# Import QueueAnalyzer - ensure queue_analyzer.py is accessible to app.py
from queue_analyzer import QueueAnalyzer # Note: This QueueAnalyzer will still have its own _load_history_from_csv,
                                       # but for app.py's display, we'll use a separate GCS download function.

# --- Safe Globals for PyTorch ---
try:
    from ultralytics.nn.modules.block import SPPF
    sppf_module = SPPF
except (ImportError, AttributeError):
    sppf_module = None

safe_globals = [
    ultralytics.nn.tasks.DetectionModel,
    ultralytics.nn.modules.Conv,
    ultralytics.nn.modules.conv.Conv,
    ultralytics.nn.modules.conv.Concat,
    ultralytics.nn.modules.block.C2f,
    ultralytics.nn.modules.block.Bottleneck,
    torch.nn.modules.container.Sequential
]
if sppf_module:
    safe_globals.append(sppf_module)
torch.serialization.add_safe_globals(safe_globals)
# --- End Safe Globals ---

# --- Configuration ---
MODEL_URL = "https://ultralytics.com/assets/yolov8s.pt"
MODEL_PATH = "yolov8s.pt"
CAMERA_URL = "https://thumbs.balticlivecam.com/blc/narva.jpg"
TIMEZONE = "Europe/Tallinn"
tz = pytz.timezone(TIMEZONE)

# GCS Configuration for app.py
GCS_BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME", "narva-queue-history-data") # <--- REPLACE WITH YOUR BUCKET NAME
GCS_OBJECT_NAME = "queue_history.csv" # The name of the file in GCS


# --- Helper function for GCS client (same as in queue_collector.py) ---
def get_gcs_client_app():
    creds_base64 = os.environ.get("GCS_CREDENTIALS_BASE64")
    if creds_base64:
        creds_json_str = base64.b64decode(creds_base64).decode('utf-8')
        temp_creds_file = 'gcs_temp_creds_app.json' # Use a different temp file name
        with open(temp_creds_file, 'w') as f:
            f.write(creds_json_str)
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = temp_creds_file
        client = storage.Client()
        return client
    else:
        st.error("GCS_CREDENTIALS_BASE64 environment variable not found. GCS client cannot be initialized.")
        return None


# --- Helper Functions ---

@st.cache_data
def download_yolo_model(model_path, model_url):
    # ... (no changes needed here, copy the original content)
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

@st.cache_resource
def load_yolo_model(model_path):
    # ... (no changes needed here, copy the original content)
    """Loads the YOLO model. Cached with Streamlit to load only once."""
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading YOLO model: {e}. Ensure the model file is valid.")
        return None

@st.cache_data(ttl=60) # Cache for 60 seconds to avoid constant re-downloads from GCS
def load_queue_history_from_gcs_for_display():
    """
    Loads and returns the queue history from GCS for Streamlit display.
    """
    gcs_client = get_gcs_client_app()
    if not gcs_client:
        return pd.DataFrame(columns=['timestamp', 'person_count']).set_index('timestamp')

    bucket = gcs_client.bucket(GCS_BUCKET_NAME)
    blob = bucket.blob(GCS_OBJECT_NAME)

    if blob.exists():
        try:
            csv_bytes = blob.download_as_bytes()
            df = pd.read_csv(BytesIO(csv_bytes), dtype={'count': int}) # Ensure 'count' is read as int

            # Convert timestamp, handling timezone offset
            df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H:%M:%S.%f%z', errors='coerce')
            df.dropna(subset=['timestamp'], inplace=True)
            df['timestamp'] = df['timestamp'].dt.tz_convert(TIMEZONE)
            df.set_index('timestamp', inplace=True)

            # Ensure 'person_count' column exists, rename 'count' to 'person_count' if necessary
            if 'count' in df.columns:
                 df.rename(columns={'count': 'person_count'}, inplace=True)
            elif 'person_count' not in df.columns:
                 st.error("CSV must contain 'person_count' or 'count' column.")
                 return pd.DataFrame(columns=['timestamp', 'person_count']).set_index('timestamp')
            
            df['person_count'] = pd.to_numeric(df['person_count'], errors='coerce')
            df.dropna(subset=['person_count'], inplace=True)

            return df
        except pd.errors.EmptyDataError:
            st.warning("Queue history GCS object is empty. Waiting for data from worker.")
            return pd.DataFrame(columns=['timestamp', 'person_count']).set_index('timestamp')
        except Exception as e:
            st.error(f"Error loading queue history from GCS for display: {e}")
            return pd.DataFrame(columns=['timestamp', 'person_count']).set_index('timestamp')
    else:
        st.info("Queue history GCS object not found. Data will appear once the worker starts collecting.")
        return pd.DataFrame(columns=['timestamp', 'person_count']).set_index('timestamp')


@st.cache_data(ttl=5) # Cache for 5 seconds to avoid constant image re-fetches
def fetch_latest_image(url):
    # ... (no changes needed here, copy the original content)
    """Fetches the latest image from the camera URL."""
    try:
        response = requests.get(url, timeout=10)
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
analyzer = None # Initialize analyzer to None

# Download model only once (cached)
if download_yolo_model(MODEL_PATH, MODEL_URL):
    # Load model only once (cached)
    model = load_yolo_model(MODEL_PATH)
    if model:
        # NOTE: QueueAnalyzer's __init__ will load history from its local CSV by default
        # However, for prediction purposes, we will ensure it also uses the GCS data for its history_df
        # This will be done by explicitly passing the GCS-loaded dataframe if available.
        analyzer = QueueAnalyzer(model) # Initialize normally
        # If GCS data is loaded, replace analyzer's history_df with it
        # This ensures predictions use the *latest* data from GCS
        gcs_history_df = load_queue_history_from_gcs_for_display() # This loads for display, but can be reused
        if not gcs_history_df.empty:
            analyzer.history_df = gcs_history_df.rename(columns={'person_count': 'count'}) # Ensure column name matches analyzer's expectation
            # print(f"Analyzer history_df updated with {len(analyzer.history_df)} entries from GCS.") # For debugging
        else:
            st.info("Analyzer initialized, but no GCS history available for predictions yet.")
    else:
        st.warning("Model could not be loaded. Live detection features will be limited.")
else:
    st.error("Could not download YOLO model. Please check network connection.")

# --- Display Latest Camera Image ---
st.header("Live Camera Feed")
latest_image_placeholder = st.empty() # Placeholder for the image to allow dynamic updates

# Fetch and display the image (and potentially draw detections)
image = fetch_latest_image(CAMERA_URL)
if image is not None:
    timestamp = datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
    
    # If model and analyzer are available, perform detection for display
    if analyzer:
        detections = analyzer.detect_pedestrians(image)
        for (x1, y1, x2, y2) in detections:
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
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

# Load queue history for Streamlit display using the dedicated GCS function
queue_df = load_queue_history_from_gcs_for_display()

if not queue_df.empty:
    # Display the latest recorded count from history
    latest_history_count = queue_df['person_count'].iloc[-1] if not queue_df.empty else 0
    st.metric("Latest Recorded People (from History)", int(latest_history_count))

    st.subheader("Queue Trends Over Time")
    # Plotting the queue history
    st.line_chart(queue_df['person_count'])

    # Predictions (assuming QueueAnalyzer can predict based on loaded history)
    if analyzer:
        # Ensure the analyzer's history_df is up-to-date with GCS data for predictions
        # (This was handled above during analyzer initialization now)
        st.info(analyzer.predict_trend())
        st.success(f"Best hours to cross: {analyzer.best_hours_to_cross()}")
    else:
        st.warning("Cannot provide predictions as QueueAnalyzer could not be initialized.")
else:
    st.info("No historical queue data available yet. Please wait for the `queue_collector.py` worker to generate data and upload it to GCS.")

# --- Auto-refresh feature for Streamlit ---
st.markdown("---")
refresh_interval_sec = st.slider("Auto-refresh interval (seconds)", 5, 30, 10)
time.sleep(refresh_interval_sec)
st.rerun()
