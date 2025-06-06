import cv2
import numpy as np
import pandas as pd
from datetime import datetime
import pytz
import os
import requests
from PIL import Image
from io import BytesIO
import time # For the sleep
import base64 # Import base64 for decoding credentials
from google.cloud import storage # Import GCS client

# --- Configuration Constants ---
QUEUE_AREA = (390, 324, 1276, 595)
MIN_CONFIDENCE = 0.025
MIN_HEIGHT = 10
DENSITY_FACTOR = 0.95
HISTORY_FILE = "queue_history.csv" # Still used for local temp storage
TIMEZONE = 'Europe/Tallinn'

# GCS Configuration
GCS_BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME", "queue-monitor-writer-reader") # <--- REPLACE WITH YOUR BUCKET NAME
GCS_OBJECT_NAME = "queue_history.csv" # The name of the file in GCS

# --- Model Configuration (same as in app.py) ---
MODEL_URL = "https://ultralytics.com/assets/yolov8s.pt"
MODEL_PATH = "yolov8s.pt"
CAMERA_URL = "https://thumbs.balticlivecam.com/blc/narva.jpg"

# --- YOLO Model Loading (same as in app.py, moved to top for clarity) ---
from ultralytics import YOLO
import torch.serialization
import ultralytics.nn.tasks
import ultralytics.nn.modules
import ultralytics.nn.modules.conv
import ultralytics.nn.modules.block
import torch.nn.modules.container

# Safe Globals for PyTorch
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

# --- Helper function for GCS client ---
def get_gcs_client():
    creds_base64 = os.environ.get("GCS_CREDENTIALS_BASE64")
    if creds_base64:
        creds_json_str = base64.b64decode(creds_base64).decode('utf-8')
        # Create a temporary file to load credentials from
        # In production, using io.StringIO might be better, but for now, temp file is fine.
        # Alternatively, can use google.oauth2.service_account.Credentials.from_service_account_info
        # import json
        # info = json.loads(creds_json_str)
        # credentials = google.oauth2.service_account.Credentials.from_service_account_info(info)
        # client = storage.Client(credentials=credentials)
        # For simplicity, let's try setting GOOGLE_APPLICATION_CREDENTIALS environment variable
        # which google-cloud-storage library automatically picks up.
        temp_creds_file = 'gcs_temp_creds.json'
        with open(temp_creds_file, 'w') as f:
            f.write(creds_json_str)
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = temp_creds_file
        client = storage.Client()
        # Clean up temp file (optional, but good practice)
        # os.remove(temp_creds_file) # Careful with this if client creation is delayed
        return client
    else:
        print("GCS_CREDENTIALS_BASE64 environment variable not found. GCS client cannot be initialized.")
        return None

# --- Main QueueAnalyzer Class (modified to use GCS for history) ---
class QueueAnalyzer:
    def __init__(self, model):
        self.model = model
        self.tz = pytz.timezone(TIMEZONE)
        self.gcs_client = get_gcs_client() # Initialize GCS client
        self.history_df = self._load_history_from_gcs() # Load history from GCS

    def _load_history_from_gcs(self):
        """
        Loads the queue history from GCS.
        """
        if not self.gcs_client:
            print("GCS client not initialized. Cannot load history from GCS.")
            return pd.DataFrame(columns=["timestamp", "count", "day_of_week", "hour"]).set_index('timestamp')

        bucket = self.gcs_client.bucket(GCS_BUCKET_NAME)
        blob = bucket.blob(GCS_OBJECT_NAME)

        if blob.exists():
            try:
                # Download blob content as bytes and read into pandas
                csv_bytes = blob.download_as_bytes()
                df = pd.read_csv(BytesIO(csv_bytes), dtype={'count': int})

                # Convert timestamp, handling timezone offset
                df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H:%M:%S.%f%z', errors='coerce')
                df.dropna(subset=['timestamp'], inplace=True)
                df['timestamp'] = df['timestamp'].dt.tz_convert(self.tz)
                df.set_index('timestamp', inplace=True)

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
        """Saves the current history DataFrame to GCS."""
        if not self.gcs_client:
            print("GCS client not initialized. Cannot save history to GCS.")
            return

        bucket = self.gcs_client.bucket(GCS_BUCKET_NAME)
        blob = bucket.blob(GCS_OBJECT_NAME)

        try:
            # Convert to UTC before saving to keep a consistent base timezone in the CSV
            df_to_save = self.history_df.copy()
            if df_to_save.index.tz is not None:
                df_to_save.index = df_to_save.index.tz_convert('UTC')
            
            # Save to a BytesIO object first, then upload
            csv_buffer = BytesIO()
            df_to_save.to_csv(csv_buffer, index=True, index_label='timestamp')
            csv_buffer.seek(0) # Rewind to the beginning of the buffer

            blob.upload_from_file(csv_buffer, content_type='text/csv')
            print(f"History saved to GCS: {GCS_OBJECT_NAME}")
        except Exception as e:
            print(f"Error saving history to GCS: {e}")

    def update_history(self, count: int):
        """
        Appends a new count entry to the history and saves it to GCS.
        """
        now = datetime.now(self.tz)
        new_entry_df = pd.DataFrame([{
            "timestamp": now,
            "count": count,
            "day_of_week": now.weekday(),
            "hour": now.hour
        }]).set_index('timestamp')

        self.history_df = pd.concat([self.history_df, new_entry_df], ignore_index=False)
        self.save_history() # Save after each update

    # --- Image Processing (remains largely the same) ---
    def fetch_image(self, url: str):
        # ... (no changes needed here, copy the original content)
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            img = Image.open(BytesIO(response.content))
            img_np = np.array(img)
            if img_np.ndim == 3 and img_np.shape[2] == 4:
                img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)
            else:
                img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

            cropped_img = img_cv[QUEUE_AREA[1]:QUEUE_AREA[3], QUEUE_AREA[0]:QUEUE_AREA[2]]
            return cropped_img
        except requests.exceptions.RequestException as e:
            print(f"Network error fetching image from {url}: {e}")
            return None
        except Exception as e:
            print(f"Error processing image from {url}: {e}")
            return None

    def detect_pedestrians(self, image: np.ndarray):
        # ... (no changes needed here, copy the original content)
        if image is None:
            return []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY_INV, 11, 2)
        results = self.model(image, classes=0, conf=MIN_CONFIDENCE, imgsz=640, verbose=False)
        detections = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                height = y2 - y1
                if height > MIN_HEIGHT:
                    roi = thresh[y1:y2, x1:x2]
                    if np.sum(roi) > 0.05 * 255 * roi.size:
                        detections.append((x1, y1, x2, y2))
        return detections

    def predict_trend(self) -> str:
        # ... (no changes needed here, copy the original content)
        if len(self.history_df) < 4:
            return "Insufficient data"
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
        # ... (no changes needed here, copy the original content)
        if len(self.history_df) < 24:
            return "Need more data"
        if not pd.api.types.is_datetime64_any_dtype(self.history_df.index):
            print("Warning: 'timestamp' column is not datetime. Attempting to re-localize.")
            temp_index = pd.to_datetime(self.history_df.index.astype(str), format='%Y-%m-%d %H:%M:%S.%f%z', errors='coerce')
            temp_index = temp_index.tz_convert(self.tz)
            self.history_df.index = temp_index
            self.history_df.dropna(subset=[self.history_df.index.name], inplace=True)
            if self.history_df.empty:
                return "Error: Timestamp data format issue for best hours."

        avg_by_hour = self.history_df.groupby(self.history_df.index.hour)["count"].mean()
        if avg_by_hour.empty:
            return "No historical data to determine best hours."
        best_hours = avg_by_hour.sort_values().head(3).index.tolist()
        return ", ".join(f"{h}:00-{h+1}:00" for h in best_hours)

# --- Main execution loop for queue_collector.py ---
if __name__ == '__main__':
    # Download model (local for collector)
    try:
        r = requests.get(MODEL_URL, stream=True)
        r.raise_for_status()
        with open(MODEL_PATH, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        print("YOLOv8s model downloaded for collector.")
    except requests.exceptions.RequestException as e:
        print(f"Error downloading model for collector: {e}. Exiting.")
        exit(1)

    model = YOLO(MODEL_PATH)
    analyzer = QueueAnalyzer(model)

    while True:
        print(f"Collecting data at {datetime.now(analyzer.tz)}...")
        image = analyzer.fetch_image(CAMERA_URL)
        if image is not None:
            detections = analyzer.detect_pedestrians(image)
            base_count = len(detections)
            adjusted_count = base_count + 50 if base_count > 0 else 0
            analyzer.update_history(adjusted_count)
            print(f"Detected: {base_count} people, Adjusted: {adjusted_count} people.")
        else:
            print("Failed to fetch image. Skipping update.")

        time.sleep(300) # Wait for 5 minutes (300 seconds)
