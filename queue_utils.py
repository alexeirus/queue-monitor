import os
import pytz
from google.cloud import storage
from google.oauth2 import service_account
import json

# --- GCS Configuration ---
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "narva-queue-monitor")
GCS_OBJECT_NAME = os.getenv("GCS_OBJECT_NAME", "queue_history.csv")
GCS_LIVE_IMAGE_OBJECT_NAME = os.getenv("GCS_LIVE_IMAGE_OBJECT_NAME", "live_detection.jpg")

def get_gcs_client():
    gcs_credentials_base64 = os.getenv("GCS_CREDENTIALS_BASE64")
    if gcs_credentials_base64:
        try:
            credentials_json = json.loads(gcs_credentials_base64)
            credentials = service_account.Credentials.from_service_account_info(credentials_json)
            return storage.Client(credentials=credentials)
        except Exception as e:
            print(f"Error initializing GCS client from credentials: {e}")
            return None
    else:
        print("GCS_CREDENTIALS_BASE64 environment variable not found. GCS client will not be initialized.")
        return None

# --- Camera and Timezone Configuration ---
CAMERA_URL = os.getenv("CAMERA_URL", "http://minsk.webcam.by/narva/current.jpg")
TIMEZONE = os.getenv("TIMEZONE", "Europe/Tallinn") # Narva is in Estonia

# --- Queue Analysis Configuration ---
# Adjusted factor for queues not fully visible (e.g., hidden areas)
# This will now be used with a conditional logic in queue_collector.py
# ADJUSTMENT_FACTOR = 0 # This line can be removed or commented out

# New constants for conditional adjustment
RAW_COUNT_THRESHOLD = 3         # Minimum raw detections to trigger additional pedestrians
ADDITIONAL_PEDESTRIANS = 60     # Number of pedestrians to add if threshold is met

# Region of Interest (ROI) for detection - (y_min, y_max, x_min, x_max)
# These values are based on an assumed common webcam resolution (e.g., 1920x1080)
# and visually inspecting the queue area in your provided images.
# You might need to fine-tune these if your camera feed resolution or perspective changes.
# This crop focuses on the main queue line from the middle to the right.
CROP_REGION = (300, 1080, 600, 1920)

class QueueAnalyzer:
    def __init__(self, history_df):
        self.history_df = history_df # This DF is expected to have 'timestamp' as index and 'count' column

    def predict_trend(self):
        if self.history_df is None or self.history_df.empty or len(self.history_df) < 5:
            return "Not enough data"

        # Use the last few data points to determine a short-term trend
        recent_data = self.history_df['count'].tail(5) # Look at the last 5 data points
        
        # Calculate the average change between consecutive points
        diffs = recent_data.diff().dropna()
        if diffs.empty:
            return "Stable"

        avg_change = diffs.mean()

        if avg_change > 0.5: # Queue is growing
            return "Queue is growing 📈"
        elif avg_change < -0.5: # Queue is shrinking
            return "Queue is shrinking 📉"
        else: # Queue is relatively stable
            return "Queue is stable ↔️"

    # Future prediction methods could go here
    # E.g., predict_future_queue_size, predict_wait_time, etc.
