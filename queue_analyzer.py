import cv2
import numpy as np
import pandas as pd
from datetime import datetime
import pytz
import os
import requests # Ensure requests is imported at the top for fetch_image
from PIL import Image # Ensure PIL is imported for fetch_image
from io import BytesIO # Ensure BytesIO is imported for fetch_image

# --- Configuration Constants ---
# Define the area of interest in the image (x1, y1, x2, y2)
QUEUE_AREA = (390, 324, 1276, 595)
# Keeping MIN_CONFIDENCE and MIN_HEIGHT as per your request
MIN_CONFIDENCE = 0.015
MIN_HEIGHT = 5
DENSITY_FACTOR = 0.95 # This constant is defined but not used in current logic
HISTORY_FILE = "queue_history.csv"
TIMEZONE = 'Europe/Tallinn' # Estonian time zone

class QueueAnalyzer:
    def __init__(self, model):
        self.model = model
        self.tz = pytz.timezone(TIMEZONE)
        # Load history during initialization.
        # It's good practice to ensure history_df is always a DataFrame.
        self.history_df = self._load_history_from_csv()

    # --- History Management ---
    def _load_history_from_csv(self):
        """
        Loads the queue history from a CSV file.
        This method is private (indicated by leading underscore) as it's an internal utility.
        Handles cases where the file might not exist or is empty.
        """
        if os.path.exists(HISTORY_FILE) and os.path.getsize(HISTORY_FILE) > 0:
            try:
                # Read CSV without parsing dates initially
                df = pd.read_csv(HISTORY_FILE, dtype={'count': int})

                # Convert 'timestamp' column to datetime, handling the timezone offset (+03:00)
                # The '%f' handles microseconds, '%z' handles the timezone offset
                # errors='coerce' will turn any unparseable dates into NaT (Not a Time)
                df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H:%M:%S.%f%z', errors='coerce')

                # Drop rows where the timestamp couldn't be parsed (i.e., are NaT)
                df.dropna(subset=['timestamp'], inplace=True)

                # Now that timestamps are timezone-aware, convert them to the desired TIMEZONE
                # No need for .dt.tz_localize('UTC') first since %z already made it aware
                df['timestamp'] = df['timestamp'].dt.tz_convert(self.tz)

                # Set timestamp as index for easier time-based operations
                df.set_index('timestamp', inplace=True)

                print(f"Loaded {len(df)} entries from {HISTORY_FILE}")
                return df
            except pd.errors.EmptyDataError:
                print(f"Warning: {HISTORY_FILE} is empty.")
                return pd.DataFrame(columns=["timestamp", "count", "day_of_week", "hour"]).set_index('timestamp') # Ensure empty DF has correct index
            except Exception as e:
                print(f"Error loading {HISTORY_FILE}: {e}")
                # Return an empty DataFrame on error to prevent further issues
                return pd.DataFrame(columns=["timestamp", "count", "day_of_week", "hour"]).set_index('timestamp') # Ensure empty DF has correct index
        else:
            print(f"Info: {HISTORY_FILE} not found or is empty. Starting with a new history.")
            return pd.DataFrame(columns=["timestamp", "count", "day_of_week", "hour"]).set_index('timestamp') # Ensure empty DF has correct index

    def save_history(self):
        """Saves the current history DataFrame to the CSV file."""
        try:
            # When saving, ensure the timestamp column (now index) is included and in ISO format for consistency
            # Convert to UTC before saving to keep a consistent base timezone in the CSV
            df_to_save = self.history_df.copy()
            if df_to_save.index.tz is not None:
                df_to_save.index = df_to_save.index.tz_convert('UTC')
            
            df_to_save.to_csv(HISTORY_FILE, index=True, index_label='timestamp') # index=True to save timestamp column
            print(f"History saved to {HISTORY_FILE}")
        except Exception as e:
            print(f"Error saving history to {HISTORY_FILE}: {e}")

    def update_history(self, count: int):
        """
        Appends a new count entry to the history and saves it to CSV.
        Adds type hint for clarity.
        """
        now = datetime.now(self.tz)
        new_entry_df = pd.DataFrame([{
            "timestamp": now,
            "count": count,
            "day_of_week": now.weekday(),
            "hour": now.hour
        }]).set_index('timestamp') # Set index here for consistency

        self.history_df = pd.concat([self.history_df, new_entry_df], ignore_index=False) # ignore_index=False to preserve timestamp index
        self.save_history() # Save after each update

    # --- Image Processing ---
    def fetch_image(self, url: str):
        """
        Fetches an image from the given URL and crops it to the defined queue area.
        Returns the cropped image as an OpenCV BGR array or None on failure.
        Adds type hints and more robust error handling.
        """
        try:
            response = requests.get(url, timeout=10) # Original timeout 10 seconds
            response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
            img = Image.open(BytesIO(response.content))
            img_np = np.array(img) # Convert PIL Image to NumPy array
            # Ensure correct color conversion for OpenCV
            if img_np.ndim == 3 and img_np.shape[2] == 4: # Handle RGBA images
                img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)
            else: # Assume RGB
                img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

            # Crop the image to the defined queue area
            cropped_img = img_cv[QUEUE_AREA[1]:QUEUE_AREA[3], QUEUE_AREA[0]:QUEUE_AREA[2]]
            return cropped_img
        except requests.exceptions.RequestException as e:
            print(f"Network error fetching image from {url}: {e}")
            return None
        except Exception as e:
            print(f"Error processing image from {url}: {e}")
            return None

    def detect_pedestrians(self, image: np.ndarray):
        """
        Detects pedestrians within the given image using the YOLO model.
        Splits image into regions and applies additional filtering based on confidence, height, and pixel density.
        Returns a list of bounding box coordinates (x1, y1, x2, y2).
        Original detection logic preserved as requested.
        """
        if image is None:
            return []

        # Convert image to grayscale for thresholding
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY_INV, 11, 2)
        
        # Perform detection using the YOLO model
        results = self.model(image, classes=0, conf=MIN_CONFIDENCE, imgsz=640, verbose=False) # verbose=False added for cleaner logs

        detections = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                height = y2 - y1
                if height > MIN_HEIGHT:
                    roi = thresh[y1:y2, x1:x2]
                    # Original density check logic preserved
                    if np.sum(roi) > 0.05 * 255 * roi.size:
                        detections.append((x1, y1, x2, y2))
        return detections

    # --- Prediction and Analysis ---
    def predict_trend(self) -> str:
        """
        Predicts the current queue trend based on recent history.
        Returns a descriptive string with emojis.
        Original trend prediction logic preserved.
        """
        if len(self.history_df) < 4: # Original condition
            return "Insufficient data"
        
        recent = self.history_df.tail(4)["count"].values
        delta = recent[-1] - recent[0]
        
        if delta > 5: # Original delta threshold
            return "Queue is growing quickly 📈"
        elif delta > 0:
            return "Queue is growing slowly ↗"
        elif delta < -5: # Original delta threshold
            return "Queue is shrinking quickly 📉"
        elif delta < 0:
            return "Queue is shrinking slowly ↘"
        return "Queue is stable →"

    def best_hours_to_cross(self) -> str:
        if len(self.history_df) < 24:
            return "Need more data"
        # The timestamp conversion safety check should be handled by _load_history_from_gcs now,
        # but leaving it for robustness if other data sources are introduced or issues arise.
        if not pd.api.types.is_datetime64_any_dtype(self.history_df.index) or self.history_df.index.tz is None:
            # This block can usually be removed if _load_history_from_gcs ensures correct dtype and tz
            print("Warning: 'timestamp' index is not datetime with timezone. Attempting to localize.")
            temp_index = pd.to_datetime(self.history_df.index.astype(str), errors='coerce')
            temp_index = temp_index.tz_localize('UTC', errors='coerce').tz_convert(self.tz)  # Assume UTC if no TZ, then convert
            self.history_df.index = temp_index
            self.history_df.dropna(subset=[self.history_df.index.name], inplace=True)
            if self.history_df.empty:
                return "Error: Timestamp index format issue for best hours."

        # Filter data to only include operating hours (7 AM to 11 PM inclusive in Europe/Tallinn)
        # 7 AM is hour 7, 11 PM is hour 23
        operating_hours_df = self.history_df[
            (self.history_df.index.hour >= 7) &
            (self.history_df.index.hour <= 22)  # Changed 23 to 22 to exclude the 23:00-24:00 hour
        ]

        if operating_hours_df.empty:
            # This can happen if all historical data is outside operating hours
            # or if there simply isn't enough data *within* operating hours
            return "No sufficient data within operating hours to determine best times."

        # Group by hour and calculate the mean queue count for only the operating hours
        avg_by_hour = operating_hours_df.groupby(operating_hours_df.index.hour)["count"].mean()

        if avg_by_hour.empty:
            return "No sufficient data within operating hours to determine best times."

        # Get the top 3 hours with the lowest average queue count
        best_hours = avg_by_hour.sort_values().head(3).index.tolist()

        # Format the output string
        return ", ".join(f"{h}:00-{h + 1}:00" for h in best_hours)
