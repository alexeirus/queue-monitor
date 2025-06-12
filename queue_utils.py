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

# It's good practice to ensure these imports are consistent
import torch.serialization
import torch.nn.modules.container
import ultralytics.nn.tasks
import ultralytics.nn.modules
import ultralytics.nn.modules.conv
import ultralytics.nn.modules.block
from ultralytics import YOLO

# --- Configuration Constants (Moved here) ---
QUEUE_AREA = (390, 324, 1276, 595)
MIN_CONFIDENCE = 0.025
MIN_HEIGHT = 10
DENSITY_FACTOR = 0.95 # This is used by the collector, not directly for prediction
TIMEZONE = 'Europe/Tallinn'
tz = pytz.timezone(TIMEZONE)

# GCS Configuration
GCS_BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME", "narva-queue-history-data")
GCS_OBJECT_NAME = "queue_history.csv"
GCS_LIVE_IMAGE_OBJECT_NAME = "live_detection_feed.jpg" # New: for pre-processed image

# Model Configuration
MODEL_URL = "https://ultralytics.com/assets/yolov8s.pt"
MODEL_PATH = "yolov8s.pt"
CAMERA_URL = "https://thumbs.balticlivecam.com/blc/narva.jpg"

# --- Safe Globals for PyTorch (Moved here) ---
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

# --- Helper function for GCS client (Consolidated) ---
def get_gcs_client():
    creds_base64 = os.environ.get("GCS_CREDENTIALS_BASE64")
    if creds_base64:
        creds_json_str = base64.b64decode(creds_base64).decode('utf-8')
        temp_creds_file = 'gcs_temp_creds.json'
        with open(temp_creds_file, 'w') as f:
            f.write(creds_json_str)
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = temp_creds_file
        client = storage.Client()
        return client
    else:
        print("GCS_CREDENTIALS_BASE64 environment variable not found. GCS client cannot be initialized.")
        return None

# --- Helper to download YOLO model (moved here, called by collector) ---
def download_yolo_model(model_path, model_url):
    """Downloads the YOLOv8s model if it doesn't exist or is incomplete."""
    if not os.path.exists(model_path) or os.path.getsize(model_path) < 10000000:
        print(f"Downloading YOLOv8s model to {model_path}...")
        try:
            r = requests.get(model_url, stream=True)
            r.raise_for_status()
            with open(model_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("YOLOv8s model downloaded successfully.")
        except requests.exceptions.RequestException as e:
            print(f"Error downloading model: {e}. Please check the URL or network connection.")
            return False
    return True

# --- QueueAnalyzer Class ---
class QueueAnalyzer:
    def __init__(self, model):
        self.model = model
        self.tz = pytz.timezone(TIMEZONE)
        self.gcs_client = get_gcs_client()
        self.history_df = self._load_history_from_gcs()

    def _load_history_from_gcs(self):
        # ... (Your existing _load_history_from_gcs function, no changes needed)
        # Ensure the dtype conversion is robust here:
        if not self.gcs_client:
            print("GCS client not initialized. Cannot load history from GCS.")
            return pd.DataFrame(columns=["timestamp", "count", "day_of_week", "hour"]).set_index('timestamp')

        bucket = self.gcs_client.bucket(GCS_BUCKET_NAME)
        blob = bucket.blob(GCS_OBJECT_NAME)

        if blob.exists():
            try:
                csv_bytes = blob.download_as_bytes()
                df = pd.read_csv(BytesIO(csv_bytes))

                df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H:%M:%S.%f%z', errors='coerce')
                df.dropna(subset=['timestamp'], inplace=True)

                if 'count' in df.columns:
                    df['count'] = pd.to_numeric(df['count'], errors='coerce').fillna(0).astype(int)
                else:
                    print("Warning: 'count' column not found in GCS history. Initializing with zeros.")
                    df['count'] = 0

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
        # ... (Your existing save_history function, no changes needed)
        if not self.gcs_client:
            print("GCS client not initialized. Cannot save history to GCS.")
            return

        bucket = self.gcs_client.bucket(GCS_BUCKET_NAME)
        blob = bucket.blob(GCS_OBJECT_NAME)

        try:
            df_to_save = self.history_df.copy()
            if df_to_save.index.tz is not None:
                df_to_save.index = df_to_save.index.tz_convert('UTC')
            
            csv_buffer = BytesIO()
            df_to_save.to_csv(csv_buffer, index=True, index_label='timestamp')
            csv_buffer.seek(0)

            blob.upload_from_file(csv_buffer, content_type='text/csv')
            print(f"History saved to GCS: {GCS_OBJECT_NAME}")
        except Exception as e:
            print(f"Error saving history to GCS: {e}")

    def update_history(self, count: int):
        # ... (Your existing update_history function, no changes needed)
        now = datetime.now(self.tz)
        new_entry_df = pd.DataFrame([{
            "timestamp": now,
            "count": count,
            "day_of_week": now.weekday(),
            "hour": now.hour
        }]).set_index('timestamp')

        self.history_df = pd.concat([self.history_df, new_entry_df], ignore_index=False)
        self.save_history()

    def fetch_image(self, url: str):
        # ... (Your existing fetch_image function, no changes needed)
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            img = Image.open(BytesIO(response.content))
            img_np = np.array(img)
            if img_np.ndim == 3 and img_np.shape[2] == 4:
                img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)
            else:
                img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

            # NOTE: For collector, we only crop the image for detection.
            # The app.py will fetch the full image if it needs to display it,
            # or fetch a pre-drawn image from GCS if the collector saves one.
            cropped_img = img_cv[QUEUE_AREA[1]:QUEUE_AREA[3], QUEUE_AREA[0]:QUEUE_AREA[2]]
            return cropped_img
        except requests.exceptions.RequestException as e:
            print(f"Network error fetching image from {url}: {e}")
            return None
        except Exception as e:
            print(f"Error processing image from {url}: {e}")
            return None

    def detect_pedestrians(self, image: np.ndarray):
        # ... (Your existing detect_pedestrians function, no changes needed)
        if image is None:
            return []
        # Note: YOLOv8 model inference takes RGB images implicitly, but OpenCV loads BGR by default.
        # Ultralytics handles this conversion internally for you when passing a BGR numpy array.
        results = self.model(image, classes=0, conf=MIN_CONFIDENCE, imgsz=640, verbose=False) # classes=0 for person
        detections = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                height = y2 - y1
                # The ROI check was based on binary thresholding, which might not be strictly needed for YOLOv8
                # If YOLOv8 is already accurate, this extra step might be removed.
                if height > MIN_HEIGHT:
                    # Original ROI check logic - keep for now if it helps filter
                    # This part could be simplified if YOLOv8's confidence is enough
                    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # This means converting again for each detection
                    # thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    #                                cv2.THRESH_BINARY_INV, 11, 2)
                    # roi = thresh[y1:y2, x1:x2]
                    # if np.sum(roi) > 0.05 * 255 * roi.size:
                    detections.append((x1, y1, x2, y2))
        return detections

    def predict_trend(self) -> str:
        # ... (Your existing predict_trend function, no changes needed)
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
        # ... (Your existing best_hours_to_cross function, no changes needed)
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
