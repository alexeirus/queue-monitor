import os
import requests
import time
from datetime import datetime
import pytz
import pandas as pd
import cv2
import numpy as np
import base64
from google.cloud import storage
from io import BytesIO
from PIL import Image

from ultralytics import YOLO

# --- Configuration Constants ---
# (x1, y1, x2, y2) for the queue detection region.
QUEUE_AREA = (390, 320, 1280, 600) # (x1, y1, x2, y2) for the queue detection region (left, top, right, bottom)
MIN_CONFIDENCE = 0.010
MIN_HEIGHT = 5
TIMEZONE = 'Europe/Tallinn'

# GCS Configuration
GCS_BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME", "narva-queue-history-data")
GCS_OBJECT_NAME = "queue_history.csv" # Stores historical count data
GCS_LIVE_IMAGE_OBJECT_NAME = "live_detection_feed.jpg" # Stores the latest image with detections

# Model Configuration
MODEL_URL = "https://ultralytics.com/assets/yolov8s.pt"
MODEL_PATH = "yolov8s.pt" # Local path where the model will be stored
CAMERA_URL = "https://thumbs.balticlivecam.com/blc/narva.jpg" # URL for the live camera feed

# --- Helper function for GCS client ---
def get_gcs_client():
    """Initializes and returns a Google Cloud Storage client.
    Prioritizes Streamlit secrets if running in an app context,
    otherwise falls back to GCS_CREDENTIALS_BASE64 environment variable.
    """
    creds_base64 = None

    # Check if Streamlit is running and if st.secrets is initialized
    try:
        import streamlit as st
        # Check if st.secrets is truly available and initialized by Streamlit
        # st.runtime.exists() ensures we are in an active Streamlit app session
        if hasattr(st, 'secrets') and st.runtime.exists() and "gcs_credentials_base64" in st.secrets: # THIS IS THE CRITICAL LINE
            creds_base64 = st.secrets["gcs_credentials_base64"]
            print("GCS credentials found in Streamlit secrets (app context).")
        else:
            # If Streamlit is present but not running as an app, or secret not found in secrets.toml
            creds_base64 = os.environ.get("GCS_CREDENTIALS_BASE64")
            if creds_base64:
                print("GCS credentials from environment variable (outside app context or secrets.toml issue).")
            else:
                print("GCS credentials not in Streamlit secrets nor environment variable.")

    except ImportError:
        # Streamlit not installed (e.g., in a simple script/worker), directly check environment variable
        creds_base64 = os.environ.get("GCS_CREDENTIALS_BASE64")
        if creds_base64:
            print("GCS credentials from environment variable (Streamlit not imported).")
        else:
            print("GCS credentials not found in environment variable (Streamlit not imported).")


    if creds_base64:
        try:
            creds_json_str = base64.b64decode(creds_base64).decode('utf-8')
            temp_creds_file = '/tmp/gcs_temp_creds.json'
            with open(temp_creds_file, 'w') as f:
                f.write(creds_json_str)
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = temp_creds_file
            client = storage.Client()
            print("GCS client initialized successfully.")
            return client
        except Exception as e:
            print(f"Error initializing GCS client from credentials: {e}")
            return None
    else:
        print("GCS_CREDENTIALS_BASE64 environment variable or Streamlit secret not found. GCS client cannot be initialized.")
        return None

# --- Helper to download YOLO model ---
def download_yolo_model(model_path, model_url):
    """Downloads the YOLOv8s model if it doesn't exist or is incomplete."""
    if not os.path.exists(model_path) or os.path.getsize(model_path) < 10000000: # Check size to ensure it's not a partial download
        print(f"Downloading YOLOv8s model to {model_path}...")
        try:
            r = requests.get(model_url, stream=True)
            r.raise_for_status() # Raise an exception for bad status codes
            with open(model_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("YOLOv8s model downloaded successfully.")
            return True
        except requests.exceptions.RequestException as e:
            print(f"Error downloading model: {e}. Please check the URL or network connection.")
            return False
    print(f"YOLOv8s model already exists at {model_path}.")
    return True

# --- QueueAnalyzer Class ---
class QueueAnalyzer:
    """Manages queue data collection, detection, and prediction."""

    def __init__(self, model):
        self.model = model
        self.tz = pytz.timezone(TIMEZONE)
        self.gcs_client = get_gcs_client()
        self.history_df = self._load_history_from_gcs()

    def _load_history_from_gcs(self):
        """Loads queue history from GCS."""
        if not self.gcs_client:
            print("GCS client not initialized. Cannot load history from GCS.")
            return pd.DataFrame(columns=["timestamp", "count", "day_of_week", "hour"]).set_index('timestamp')

        bucket = self.gcs_client.bucket(GCS_BUCKET_NAME)
        blob = bucket.blob(GCS_OBJECT_NAME)

        if blob.exists():
            try:
                csv_bytes = blob.download_as_bytes()
                df = pd.read_csv(BytesIO(csv_bytes))

                # Ensure timestamp is parsed correctly and timezone-aware
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
                df.dropna(subset=['timestamp'], inplace=True) # Drop rows where timestamp parsing failed

                # If timestamp is naive, assume it's UTC (as it's saved as UTC) then convert to local timezone
                if df['timestamp'].dt.tz is None:
                    df['timestamp'] = df['timestamp'].dt.tz_localize('UTC')
                df['timestamp'] = df['timestamp'].dt.tz_convert(self.tz)
                df.set_index('timestamp', inplace=True)

                if 'count' in df.columns:
                    df['count'] = pd.to_numeric(df['count'], errors='coerce').fillna(0).astype(int)
                else:
                    print("Warning: 'count' column not found in GCS history. Initializing with zeros.")
                    df['count'] = 0
                
                print(f"Loaded {len(df)} entries from GCS: {GCS_OBJECT_NAME}")
                return df
            except pd.errors.EmptyDataError:
                print(f"Warning: GCS object {GCS_OBJECT_NAME} is empty.")
                return pd.DataFrame(columns=["timestamp", "count", "day_of_week", "hour"]).set_index('timestamp')
            except Exception as e:
                print(f"Error loading {GCS_OBJECT_NAME} from GCS: {e}")
                return pd.DataFrame(columns=["timestamp", "count", "day_of_week", "hour"]).set_index('timestamp')
        else:
            print(f"Info: GCS object {GCS_OBJECT_NAME} not found. Starting with a new history.")
            return pd.DataFrame(columns=["timestamp", "count", "day_of_week", "hour"]).set_index('timestamp')

    def save_history(self):
        """Saves current queue history to GCS."""
        if not self.gcs_client:
            print("GCS client not initialized. Cannot save history to GCS.")
            return

        bucket = self.gcs_client.bucket(GCS_BUCKET_NAME)
        blob = bucket.blob(GCS_OBJECT_NAME)

        try:
            df_to_save = self.history_df.copy()
            # Convert to UTC before saving to ensure consistency across timezones/systems
            if df_to_save.index.tz is not None:
                df_to_save.index = df_to_save.index.tz_convert('UTC')
            
            csv_buffer = BytesIO()
            df_to_save.to_csv(csv_buffer, index=True, index_label='timestamp')
            csv_buffer.seek(0) # Rewind to the beginning of the buffer

            blob.upload_from_file(csv_buffer, content_type='text/csv')
            print(f"History saved to GCS: {GCS_OBJECT_NAME}")
        except Exception as e:
            print(f"Error saving history to GCS: {e}")

    def update_history(self, count: int):
        """Adds a new count entry to history and saves it."""
        now = datetime.now(self.tz)
        new_entry_df = pd.DataFrame([{
            "timestamp": now,
            "count": count,
            "day_of_week": now.weekday(), # Monday=0, Sunday=6
            "hour": now.hour
        }]).set_index('timestamp')

        self.history_df = pd.concat([self.history_df, new_entry_df], ignore_index=False)
        self.save_history()

    def fetch_image(self, url: str) -> np.ndarray | None:
        """Fetches and processes an image from a URL, returns the full image."""
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            img = Image.open(BytesIO(response.content))
            img_np = np.array(img)
            # Handle potential RGBA images, convert to BGR for OpenCV
            if img_np.ndim == 3 and img_np.shape[2] == 4:
                img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)
            else:
                img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            return img_cv
        except requests.exceptions.RequestException as e:
            print(f"Network error fetching image from {url}: {e}")
            return None
        except Exception as e:
            print(f"Error processing image from {url}: {e}")
            return None

    def detect_pedestrians(self, image: np.ndarray) -> list[tuple]:
        """Performs pedestrian detection on the given image using the YOLO model."""
        if image is None:
            return []
        
        # Use classes=0 for 'person' in COCO dataset (YOLOv8 default)
        results = self.model(image, classes=0, conf=MIN_CONFIDENCE, imgsz=640, verbose=False)
        detections = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                height = y2 - y1
                if height > MIN_HEIGHT:
                    detections.append((x1, y1, x2, y2))
        return detections

    def predict_trend(self) -> str:
        """Predicts the queue trend based on recent history."""
        if self.history_df.empty or 'count' not in self.history_df.columns:
            return "No data for trend analysis."
            
        if len(self.history_df) < 4: # Need at least 4 data points for a trend
            return "Insufficient data to determine trend."
        recent = self.history_df.tail(4)["count"].values
        delta = recent[-1] - recent[0]
        if delta > 5:
            return "Queue is growing quickly 📈"
        elif delta > 0:
            return "Queue is growing slowly ↗"
        elif delta < -5:
            return "Queue is shrinking quickly 📉"
        elif delta < 0:
            return "Queue is shrinking slowly ↘"
        return "Queue is stable →"

    def best_hours_to_cross(self) -> str:
        """Determines the best hours to cross based on historical averages (from all history, not filtered)."""
        if self.history_df.empty or 'count' not in self.history_df.columns:
            return "No historical data to determine best hours."

        if len(self.history_df) < 24: # Need at least a day's worth of data for good hourly averages
            return "Need more historical data to determine best hours."

        # Ensure index is datetime and timezone-aware before grouping by hour
        if not pd.api.types.is_datetime64_any_dtype(self.history_df.index) or self.history_df.index.tz is None:
            print("Warning: history_df index not datetime or timezone-aware for best_hours_to_cross. Attempting to fix.")
            try:
                temp_index = pd.to_datetime(self.history_df.index, errors='coerce')
                if temp_index.tz is None:
                    temp_index = temp_index.tz_localize('UTC')
                temp_index = temp_index.tz_convert(self.tz)
                self.history_df.index = temp_index
                self.history_df.dropna(subset=[self.history_df.index.name], inplace=True)
            except Exception as e:
                print(f"Error fixing timestamp in best_hours_to_cross: {e}")
                return "Error: Timestamp data format issue preventing best hours calculation."
            
            if self.history_df.empty:
                return "Error: Timestamp data format issue preventing best hours calculation."

        avg_by_hour = self.history_df.groupby(self.history_df.index.hour)["count"].mean()
        if avg_by_hour.empty:
            return "No historical data to determine best hours."
            
        best_hours = avg_by_hour.sort_values().head(3).index.tolist()
        return ", ".join(f"{h:02d}:00-{h+1:02d}:00" for h in best_hours)

    def get_hourly_averages_for_day(self, day_of_week: int) -> pd.DataFrame:
        """
        Returns a DataFrame with average queue counts per hour for a specific day of the week.
        day_of_week: 0 for Monday, 6 for Sunday.
        """
        if self.history_df.empty or 'count' not in self.history_df.columns:
            return pd.DataFrame(columns=['hour', 'average_count'])

        # Ensure 'day_of_week' column is present; it should be from update_history
        if 'day_of_week' not in self.history_df.columns:
            self.history_df['day_of_week'] = self.history_df.index.weekday

        df_filtered = self.history_df[self.history_df['day_of_week'] == day_of_week].copy()

        if df_filtered.empty:
            return pd.DataFrame(columns=['hour', 'average_count'])

        # Group by hour and calculate mean count
        hourly_averages = df_filtered.groupby('hour')['count'].mean().reset_index()
        hourly_averages.columns = ['hour', 'average_count']
        hourly_averages['hour_str'] = hourly_averages['hour'].apply(lambda x: f"{x:02d}:00")
        # Ensure all 24 hours are present, filling missing with 0
        all_hours = pd.DataFrame({'hour': range(24)})
        hourly_averages = pd.merge(all_hours, hourly_averages, on='hour', how='left').fillna(0)
        hourly_averages['average_count'] = hourly_averages['average_count'].astype(int)
        hourly_averages['hour_str'] = hourly_averages['hour'].apply(lambda x: f"{x:02d}:00")

        return hourly_averages.set_index('hour') # Set hour as index for easier plotting

    def get_overall_best_times(self) -> dict:
        """
        Determines the overall best day and best hour(s) to cross based on historical averages.
        Returns a dictionary with 'best_day_name' and 'best_hours'.
        """
        if self.history_df.empty or 'count' not in self.history_df.columns:
            return {"best_day_name": "N/A", "best_hours": "N/A"}

        # Calculate average count for each day-hour combination
        # Ensure 'day_of_week' and 'hour' columns are present and correctly populated
        if 'day_of_week' not in self.history_df.columns or 'hour' not in self.history_df.columns:
            # Re-calculate if somehow missing (should be set in update_history)
            self.history_df['day_of_week'] = self.history_df.index.weekday
            self.history_df['hour'] = self.history_df.index.hour

        hourly_daily_averages = self.history_df.groupby(['day_of_week', 'hour'])['count'].mean().reset_index()
        
        if hourly_daily_averages.empty:
            return {"best_day_name": "N/A", "best_hours": "N/A"}

        # Find the minimum average count
        min_avg_count = hourly_daily_averages['count'].min()

        # Filter for entries that have this minimum count
        best_times_df = hourly_daily_averages[hourly_daily_averages['count'] == min_avg_count]

        # Get the best day(s)
        # Convert weekday integer to name (0=Monday, 6=Sunday)
        day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        
        best_day_indices = sorted(best_times_df['day_of_week'].unique())
        best_day_names = [day_names[i] for i in best_day_indices]
        best_day_string = ", ".join(best_day_names)

        # Get the best hour(s) across these best days
        best_hours_list = sorted(best_times_df['hour'].unique())
        best_hours_string = ", ".join(f"{h:02d}:00-{h+1:02d}:00" for h in best_hours_list)

        return {
            "best_day_name": best_day_string,
            "best_hours": best_hours_string
        }
