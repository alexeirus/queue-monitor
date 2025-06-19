import os
import pytz
from google.cloud import storage
from google.oauth2 import service_account
import json
from datetime import datetime, timedelta # Ensure timedelta is imported here for consistency

# --- GCS Configuration ---
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "narva-queue-monitor")
GCS_OBJECT_NAME = os.getenv("GCS_OBJECT_NAME", "queue_history.csv")
GCS_LIVE_IMAGE_OBJECT_NAME = os.getenv("GCS_LIVE_IMAGE_OBJECT_NAME", "live_detection.jpg")

def get_gcs_client():
    gcs_credentials_base64 = os.getenv("GCS_CREDENTIALS_BASE64")
    if gcs_credentials_base64:
        try:
            # Decode the base64 string to bytes, then decode bytes to a UTF-8 string
            # This handles potential differences in how base64 strings are generated/stored
            credentials_json_str = base64.b64decode(gcs_credentials_base64).decode('utf-8')
            credentials_json = json.loads(credentials_json_str)
            credentials = service_account.Credentials.from_service_account_info(credentials_json)
            return storage.Client(credentials=credentials)
        except Exception as e:
            # Added more specific error message for JSON decoding issues
            print(f"Error initializing GCS client from credentials: {e}")
            print("Please check your GCS_CREDENTIALS_BASE64 environment variable. "
                  "It should be a valid base64 encoded JSON string of your service account key.")
            return None
    else:
        print("GCS_CREDENTIALS_BASE64 environment variable not found. GCS client will not be initialized.")
        return None

# --- Camera and Timezone Configuration ---
CAMERA_URL = os.getenv("CAMERA_URL", "https://thumbs.balticlivecam.com/blc/narva.jpg") # <-- CHANGE THIS LINE
TIMEZONE = os.getenv("TIMEZONE", "Europe/Tallinn") # Narva is in Estonia

# --- Queue Analysis Configuration ---
RAW_COUNT_THRESHOLD = 3         # Minimum raw detections to trigger additional pedestrians
ADDITIONAL_PEDESTRIANS = 60     # Number of pedestrians to add if threshold is met

# Region of Interest (ROI) for initial image cropping (zoom-in) - (y_min, y_max, x_min, x_max)
# These values are based on an assumed common webcam resolution (e.g., 1920x1080)
# and visually inspecting the queue area in your provided images.
# You might need to fine-tune these if your camera feed resolution or perspective changes.
# This crop focuses on the main queue line from the middle to the right.
# Assumes original image is at least 1080p (1920x1080)
CROP_REGION = (300, 1080, 600, 1920) # y_start:y_end, x_start:x_end (absolute pixels from original image)

# Region of Interest (ROI) for detections *within the cropped image*
# (x_min_relative, y_min_relative, x_max_relative, y_max_relative)
# These values should be relative to the top-left corner of the `CROP_REGION`
# Based on the previous CROP_REGION (300, 1080, 600, 1920), the cropped image size is:
# Width = 1920 - 600 = 1320 pixels
# Height = 1080 - 300 = 780 pixels
# This DETECTION_ROI_RELATIVE is set to cover the right-hand side, mid-area within this 1320x780 cropped image.
DETECTION_ROI_RELATIVE = [
    800,  # x_min (start 800px from left of cropped image, i.e., 800+600=1400px from left of original)
    400,  # y_min (start 400px from top of cropped image, i.e., 400+300=700px from top of original)
    1300, # x_max (end 1300px from left of cropped image, i.e., 1300+600=1900px from left of original)
    750   # y_max (end 750px from top of cropped image, i.e., 750+300=1050px from top of original)
]


class QueueAnalyzer:
    def __init__(self, history_df):
        self.history_df = history_df # This DF is expected to have 'timestamp' as index and 'count' column

    def predict_trend(self):
        if self.history_df is None or self.history_df.empty or len(self.history_df) < 5:
            return "Not enough data"

        recent_data = self.history_df['count'].tail(5)
        
        diffs = recent_data.diff().dropna()
        if diffs.empty:
            return "Stable"

        avg_change = diffs.mean()

        if avg_change > 0.5:
            return "Queue is growing 📈"
        elif avg_change < -0.5:
            return "Queue is shrinking 📉"
        else:
            return "Queue is stable ↔️"
