import os
import time
import streamlit as st
from datetime import datetime
import pandas as pd
import cv2
import numpy as np
from io import BytesIO

# Import necessary components from queue_utils
from queue_utils import (
    get_gcs_client, # Re-use the consolidated GCS client function
    GCS_BUCKET_NAME,
    GCS_OBJECT_NAME,
    GCS_LIVE_IMAGE_OBJECT_NAME, # For fetching pre-processed image
    TIMEZONE,
    CAMERA_URL, # Used as fallback if no pre-processed image
    QueueAnalyzer # Import QueueAnalyzer for its prediction methods
)

# Initialize timezone from queue_utils
tz = pytz.timezone(TIMEZONE)

# --- Helper Functions for Streamlit App ---

@st.cache_data(ttl=5) # Cache for 5 seconds to avoid constant image re-fetches
def fetch_live_detection_image_from_gcs():
    """Fetches the latest pre-processed image with detections from GCS."""
    gcs_client = get_gcs_client() # Use the centralized GCS client function
    if not gcs_client:
        st.error("GCS_CREDENTIALS_BASE64 environment variable not found. GCS client cannot be initialized for image fetch.")
        return None

    bucket = gcs_client.bucket(GCS_BUCKET_NAME)
    blob = bucket.blob(GCS_LIVE_IMAGE_OBJECT_NAME)

    if blob.exists():
        try:
            image_bytes = blob.download_as_bytes()
            # Use cv2.imdecode to read image bytes into a numpy array
            image_array = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
            return image_array
        except Exception as e:
            st.error(f"Error fetching live detection image from GCS: {e}")
            return None
    else:
        st.info(f"Live detection image '{GCS_LIVE_IMAGE_OBJECT_NAME}' not found in GCS. Waiting for collector.")
        return None

@st.cache_data(ttl=60) # Cache for 60 seconds to avoid constant re-downloads from GCS
def load_queue_history_from_gcs_for_display():
    """Loads queue history from GCS for display in Streamlit."""
    gcs_client = get_gcs_client() # Use the centralized GCS client function
    if not gcs_client:
        return pd.DataFrame(columns=['timestamp', 'person_count']).set_index('timestamp')

    bucket = gcs_client.bucket(GCS_BUCKET_NAME)
    blob = bucket.blob(GCS_OBJECT_NAME)

    if blob.exists():
        try:
            csv_bytes = blob.download_as_bytes()
            df = pd.read_csv(BytesIO(csv_bytes))

            # Ensure timestamp is parsed correctly and converted to the desired timezone
            df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H:%M:%S.%f%z', errors='coerce')
            df.dropna(subset=['timestamp'], inplace=True)
            df['timestamp'] = df['timestamp'].dt.tz_convert(TIMEZONE)
            df.set_index('timestamp', inplace=True)

            # Standardize column name to 'person_count' for display
            if 'count' in df.columns:
                df['count'] = pd.to_numeric(df['count'], errors='coerce').fillna(0)
                df.rename(columns={'count': 'person_count'}, inplace=True)
            elif 'person_count' in df.columns:
                df['person_count'] = pd.to_numeric(df['person_count'], errors='coerce').fillna(0)
            else:
                st.error("CSV must contain 'person_count' or 'count' column. Initializing with zeros.")
                df['person_count'] = 0

            df['person_count'] = df['person_count'].astype(int)

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


# --- Streamlit App Layout ---
st.set_page_config(page_title="Narva Queue Monitor", layout="wide", initial_sidebar_state="collapsed")
st.title("🚶 Narva Queue Monitor")
st.markdown("---")

# Initialize analyzer for prediction methods only (model not loaded here, pass None)
analyzer = QueueAnalyzer(None) # Pass None for model as app.py doesn't do detection

# --- Display Latest Camera Image ---
st.header("Live Camera Feed")
latest_image_placeholder = st.empty()

# Fetch and display the pre-processed image from GCS
image_with_detections = fetch_live_detection_image_from_gcs()

if image_with_detections is not None:
    timestamp_display = datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")

    # Display image with detections (already drawn by collector)
    latest_image_placeholder.image(image_with_detections,
                                   caption=f"Last updated: {timestamp_display} (Live Detections Shown by Collector)",
                                   channels="BGR", use_container_width=True)

    # Display detected people count from the *latest entry in history*
    queue_df_for_live_count = load_queue_history_from_gcs_for_display()
    if not queue_df_for_live_count.empty:
        latest_live_count = queue_df_for_live_count['person_count'].iloc[-1]
        st.metric("Detected People (Live View)", int(latest_live_count))
    else:
        st.info("Waiting for live detection count from collector.")

else:
    # Fallback if no pre-processed image is available from GCS (e.g., collector not running yet)
    st.info("Live detection image is not yet available from the collector. Attempting to display raw feed.")
    try:
        # Use requests directly here as a fallback
        import requests # Import requests for this specific fallback use case
        raw_image_response = requests.get(CAMERA_URL, timeout=10)
        raw_image_response.raise_for_status()
        raw_image_array = cv2.imdecode(np.frombuffer(raw_image_response.content, np.uint8), cv2.IMREAD_COLOR)
        timestamp_display = datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
        latest_image_placeholder.image(raw_image_array, caption=f"Last updated: {timestamp_display} (Raw Feed, No Detections)",
                                       channels="BGR", use_container_width=True)
    except requests.exceptions.RequestException as e:
        latest_image_placeholder.error(f"⚠️ Could not load image from camera feed or GCS: {e}")
    except Exception as e:
        latest_image_placeholder.error(f"⚠️ Error processing raw image: {e}")

st.markdown("---")

# --- Display Historical Data and Predictions ---
st.header("Queue History and Predictions")

# Load queue history for Streamlit display
queue_df = load_queue_history_from_gcs_for_display()

if not queue_df.empty:
    latest_history_count = queue_df['person_count'].iloc[-1]
    st.metric("Latest Recorded People (from History)", int(latest_history_count))

    st.subheader("Queue Trends Over Time")
    st.line_chart(queue_df['person_count'], use_container_width=True)

    # Update analyzer's history_df with the GCS data for predictions
    # Ensure column name matches analyzer's expectation ('count')
    analyzer.history_df = queue_df.rename(columns={'person_count': 'count'})

    st.info(analyzer.predict_trend())
    st.success(f"Best hours to cross: {analyzer.best_hours_to_cross()}")
else:
    st.info("No historical queue data available yet. Please wait for the `queue_collector.py` worker to generate data and upload it to GCS.")

# --- Auto-refresh feature for Streamlit ---
st.markdown("---")
refresh_interval_sec = st.slider("Auto-refresh interval (seconds)", 5, 30, 10)
time.sleep(refresh_interval_sec)
st.rerun()
