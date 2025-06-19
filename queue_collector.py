import os
import time
from datetime import datetime
import pandas as pd
import cv2
import numpy as np
from io import BytesIO
import requests
import pytz
from ultralytics import YOLO

# Import constants and utilities from queue_utils
from queue_utils import (
    get_gcs_client,
    GCS_BUCKET_NAME,
    GCS_OBJECT_NAME,
    GCS_LIVE_IMAGE_OBJECT_NAME,
    TIMEZONE,
    CAMERA_URL,
    RAW_COUNT_THRESHOLD,       # New import
    ADDITIONAL_PEDESTRIANS,    # New import
    CROP_REGION                # New import
)

# Initialize timezone
tz = pytz.timezone(TIMEZONE)

# --- YOLO Model Configuration ---
# Load a pre-trained YOLOv8n model
# Ensure this model file is accessible in your deployment environment
MODEL_PATH = os.getenv("YOLO_MODEL_PATH", "yolov8n.pt") # Default to yolov8n.pt
try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    print(f"Error loading YOLO model from {MODEL_PATH}: {e}")
    print("Please ensure the model path is correct or yolov8n.pt is available.")
    model = None # Set model to None if loading fails

# Define classes to count (people in COCO dataset)
TARGET_CLASS_IDS = [0] # Class ID 0 corresponds to 'person' in COCO dataset

# Confidence threshold for detections
CONF_THRESHOLD = 0.5

# Region of Interest (ROI) for detections in the original image (x_min, y_min, x_max, y_max)
# This ROI is for filtering detections *after* the initial image crop.
# It should correspond to the area where the queue actually forms *within the CROP_REGION*.
# These values are relative to the *cropped* image.
# You will likely need to adjust these by observing your processed image.
# Example: If your crop is 700x1320 (from CROP_REGION), then these ROI values should be within that.
# For simplicity and to match the prompt's implied broad area,
# let's initially define an ROI that covers most of the cropped queue area.
# This assumes CROP_REGION produces an image roughly (1080-300) x (1920-600) = 780x1320.
# So x goes from 0 to 1320, y goes from 0 to 780 within the cropped image.
# The user's image shows the queue mostly on the right side of the road/parking lot.
# This ROI will encompass that general area within the cropped image.
DETECTION_ROI = [600 - CROP_REGION[2], 300 - CROP_REGION[0], 1920 - CROP_REGION[2], 1080 - CROP_REGION[0]]
# Re-evaluating DETECTION_ROI based on CROP_REGION to be relative to the cropped image:
# x_min = max(0, 600 - CROP_REGION[2])
# y_min = max(0, 300 - CROP_REGION[0])
# x_max = min(1920 - CROP_REGION[2], 1920 - CROP_REGION[2]) # max x of cropped image
# y_max = min(1080 - CROP_REGION[0], 1080 - CROP_REGION[0]) # max y of cropped image

# The DETECTION_ROI should be defined relative to the *cropped* image dimensions.
# Let's assume CROP_REGION is applied first.
# The `results.boxes.xyxy` coordinates are relative to the *input image* fed to the model.
# Since we're cropping the input image, the ROI should be relative to the *cropped* image.
# If CROP_REGION = (y_min_abs, y_max_abs, x_min_abs, x_max_abs), then the cropped image
# dimensions are (y_max_abs - y_min_abs) x (x_max_abs - x_min_abs).
# A simple ROI covering the full cropped image would be [0, 0, (x_max_abs - x_min_abs), (y_max_abs - y_min_abs)].
# However, the user wants detections only in the queue area. Let's make this explicit.
# Based on the image, the queue is primarily on the right side of the main road.
# Let's set the DETECTION_ROI to a reasonable portion of the cropped image.
# These values are *relative to the cropped image* now.
# (x_min, y_min, x_max, y_max) of the specific queue area within the cropped frame.
# This might need fine-tuning. Let's try to focus on the sidewalk area on the right.
# Assuming cropped image is roughly 1320x780 (width x height)
DETECTION_ROI_RELATIVE = [600, 300, 1300, 700] # These values need to be carefully chosen relative to the cropped image.
# Let's simplify DETECTION_ROI to just cover the right side of the image, as that's where the queue is.
# The crop already limits the view. Let's use coordinates within the cropped image.
# If the crop is `frame = frame[300:1080, 600:1920]`, then the cropped image is 780 pixels high and 1320 pixels wide.
# A reasonable DETECTION_ROI within this cropped image focusing on the queue:
# (x_min_relative, y_min_relative, x_max_relative, y_max_relative)
DETECTION_ROI_RELATIVE = [
    800,  # x_min (start further right in the cropped image)
    400,  # y_min (start below the parking lot, higher on the image)
    1300, # x_max (end towards the far right of the cropped image)
    750   # y_max (end towards the bottom of the cropped image)
]


def load_queue_history_from_gcs():
    """Loads existing queue history from GCS."""
    gcs_client = get_gcs_client()
    if not gcs_client:
        return pd.DataFrame(columns=['timestamp', 'count'])

    bucket = gcs_client.bucket(GCS_BUCKET_NAME)
    blob = bucket.blob(GCS_OBJECT_NAME)

    if blob.exists():
        try:
            csv_bytes = blob.download_as_bytes()
            df = pd.read_csv(BytesIO(csv_bytes))
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
        except pd.errors.EmptyDataError:
            return pd.DataFrame(columns=['timestamp', 'count'])
        except Exception as e:
            print(f"Error loading queue history from GCS: {e}")
            return pd.DataFrame(columns=['timestamp', 'count'])
    return pd.DataFrame(columns=['timestamp', 'count'])


def save_queue_history_to_gcs(df):
    """Saves queue history DataFrame to GCS."""
    gcs_client = get_gcs_client()
    if not gcs_client:
        print("GCS client not available. Cannot save queue history.")
        return

    bucket = gcs_client.bucket(GCS_BUCKET_NAME)
    blob = bucket.blob(GCS_OBJECT_NAME)

    try:
        csv_buffer = BytesIO()
        df.to_csv(csv_buffer, index=False)
        blob.upload_from_string(csv_buffer.getvalue(), content_type='text/csv')
        print(f"Queue history updated successfully on GCS at {datetime.now(tz)}")
    except Exception as e:
        print(f"Error saving queue history to GCS: {e}")


def upload_image_to_gcs(image_array, object_name):
    """Uploads an image (numpy array) to GCS."""
    gcs_client = get_gcs_client()
    if not gcs_client:
        print("GCS client not available. Cannot upload image.")
        return

    bucket = gcs_client.bucket(GCS_BUCKET_NAME)
    blob = bucket.blob(object_name)

    try:
        _, encoded_image = cv2.imencode('.jpg', image_array, [cv2.IMWRITE_JPEG_QUALITY, 90]) # 90% quality
        blob.upload_from_string(encoded_image.tobytes(), content_type='image/jpeg')
        print(f"Image '{object_name}' uploaded successfully to GCS.")
    except Exception as e:
        print(f"Error uploading image to GCS: {e}")


def process_frame(frame, model):
    """Processes a single frame for object detection and returns count and annotated frame."""
    if model is None:
        return 0, frame # Return 0 count and original frame if model not loaded

    # Draw the CROP_REGION on the original frame for debugging purposes (optional)
    # This helps visualize where the crop is applied on the full image
    # cv2.rectangle(frame, (CROP_REGION[2], CROP_REGION[0]), (CROP_REGION[3], CROP_REGION[1]), (255, 0, 0), 2) # Blue box for crop area

    # Apply the CROP_REGION to zoom in
    # Ensure the crop region is within frame dimensions
    h, w, _ = frame.shape
    y_min_abs, y_max_abs, x_min_abs, x_max_abs = CROP_REGION
    
    # Adjust crop region if it exceeds image boundaries
    y_min_abs = max(0, y_min_abs)
    y_max_abs = min(h, y_max_abs)
    x_min_abs = max(0, x_min_abs)
    x_max_abs = min(w, x_max_abs)

    cropped_frame = frame[y_min_abs:y_max_abs, x_min_abs:x_max_abs]
    
    if cropped_frame.shape[0] == 0 or cropped_frame.shape[1] == 0:
        print("Warning: Cropped frame has zero dimensions. Check CROP_REGION settings.")
        return 0, frame # Return 0 count and original frame if crop results in empty frame

    # Run YOLO detection on the cropped frame
    results = model(cropped_frame, conf=CONF_THRESHOLD, verbose=False) # verbose=False to reduce console output

    current_raw_count = 0
    annotated_cropped_frame = cropped_frame.copy() # Make a copy to draw on

    if results and len(results) > 0:
        for r in results:
            if r.boxes:
                for box in r.boxes.xyxy:
                    x1, y1, x2, y2 = map(int, box[:4])
                    cls = int(r.boxes.cls[0]) # Class ID

                    if cls in TARGET_CLASS_IDS:
                        # Check if detection is within the DETECTION_ROI_RELATIVE within the cropped image
                        if (x1 >= DETECTION_ROI_RELATIVE[0] and y1 >= DETECTION_ROI_RELATIVE[1] and
                            x2 <= DETECTION_ROI_RELATIVE[2] and y2 <= DETECTION_ROI_RELATIVE[3]):
                            current_raw_count += 1
                            # Draw bounding box (green for detected person)
                            cv2.rectangle(annotated_cropped_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Draw the DETECTION_ROI_RELATIVE on the annotated cropped frame for visual debugging
    cv2.rectangle(annotated_cropped_frame, 
                  (DETECTION_ROI_RELATIVE[0], DETECTION_ROI_RELATIVE[1]),
                  (DETECTION_ROI_RELATIVE[2], DETECTION_ROI_RELATIVE[3]),
                  (0, 0, 255), 2) # Red box for detection ROI

    # Apply conditional adjustment to the count
    current_adjusted_count = current_raw_count
    if current_raw_count >= RAW_COUNT_THRESHOLD:
        current_adjusted_count += ADDITIONAL_PEDESTRIANS

    # Add count text to the annotated cropped frame
    # Use adjusted count for display
    timestamp_str = datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
    text = f"Last updated: {timestamp_str} | Detected: {current_adjusted_count}"
    cv2.putText(annotated_cropped_frame, text, (10, annotated_cropped_frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA) # White text

    return current_adjusted_count, annotated_cropped_frame


def main():
    print("Queue Collector Started.")
    queue_history_df = load_queue_history_from_gcs()

    # Get the period for saving data to GCS from environment variable, default to 5 minutes
    SAVE_PERIOD_MINUTES = int(os.getenv("SAVE_PERIOD_MINUTES", 5))
    last_save_time = datetime.now(tz) - timedelta(minutes=SAVE_PERIOD_MINUTES + 1) # Force initial save

    # Get the image fetch interval from environment variable, default to 10 seconds
    FETCH_INTERVAL_SECONDS = int(os.getenv("FETCH_INTERVAL_SECONDS", 10))

    while True:
        try:
            # Fetch image from URL
            response = requests.get(CAMERA_URL, timeout=10)
            response.raise_for_status() # Raise an exception for HTTP errors
            nparr = np.frombuffer(response.content, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame is None:
                print("Failed to decode image from URL.")
                time.sleep(FETCH_INTERVAL_SECONDS)
                continue

            # Process the frame (detect and annotate)
            adjusted_count, annotated_frame = process_frame(frame, model)

            # Upload the annotated frame to GCS for Streamlit to display
            upload_image_to_gcs(annotated_frame, GCS_LIVE_IMAGE_OBJECT_NAME)

            current_time = datetime.now(tz)

            # Append current data to DataFrame
            new_row = pd.DataFrame([{'timestamp': current_time, 'count': adjusted_count}])
            queue_history_df = pd.concat([queue_history_df, new_row], ignore_index=True)

            # Save to GCS periodically
            if (current_time - last_save_time).total_seconds() >= SAVE_PERIOD_MINUTES * 60:
                save_queue_history_to_gcs(queue_history_df)
                last_save_time = current_time

            print(f"[{current_time.strftime('%Y-%m-%d %H:%M:%S')}] Detected (Adjusted): {adjusted_count} people.")

        except requests.exceptions.RequestException as e:
            print(f"Error fetching image from URL: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

        time.sleep(FETCH_INTERVAL_SECONDS)

if __name__ == "__main__":
    main()
