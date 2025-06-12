import cv2
import numpy as np
import pandas as pd
from datetime import datetime
import time
from PIL import Image
from io import BytesIO
import os

# Import everything from the new queue_utils.py
from queue_utils import (
    YOLO,
    QueueAnalyzer,
    download_yolo_model,
    MODEL_PATH,
    MODEL_URL,
    CAMERA_URL,
    GCS_LIVE_IMAGE_OBJECT_NAME,
    GCS_BUCKET_NAME,
    get_gcs_client,
    QUEUE_AREA # Import QUEUE_AREA for drawing on full image
)

# --- Main execution loop for queue_collector.py ---
if __name__ == '__main__':
    # Download model (local for collector)
    if not download_yolo_model(MODEL_PATH, MODEL_URL):
        print("Exiting collector due to model download failure.")
        exit(1)

    model = YOLO(MODEL_PATH)
    analyzer = QueueAnalyzer(model) # QueueAnalyzer now handles GCS client internally

    gcs_client_collector = get_gcs_client() # Get a GCS client for uploading live image

    while True:
        current_time_str = datetime.now(analyzer.tz).strftime("%Y-%m-%d %H:%M:%S")
        print(f"Collecting data at {current_time_str}...")

        # Fetch the *full* image to draw on for the live feed
        try:
            response = requests.get(CAMERA_URL, timeout=10)
            response.raise_for_status()
            img = Image.open(BytesIO(response.content))
            img_np_full = np.array(img) # Full image
            if img_np_full.ndim == 3 and img_np_full.shape[2] == 4:
                img_cv_full = cv2.cvtColor(img_np_full, cv2.COLOR_RGBA2BGR)
            else:
                img_cv_full = cv2.cvtColor(img_np_full, cv2.COLOR_RGB2BGR)

            # Crop the image for detection only
            cropped_image_for_detection = img_cv_full[QUEUE_AREA[1]:QUEUE_AREA[3], QUEUE_AREA[0]:QUEUE_AREA[2]]

            detections_relative_to_cropped = analyzer.detect_pedestrians(cropped_image_for_detection)

            # Adjust detection coordinates to be relative to the *full* image
            detections_full_image = []
            for (x1, y1, x2, y2) in detections_relative_to_cropped:
                # Add the offset of the cropped area
                detections_full_image.append((x1 + QUEUE_AREA[0], y1 + QUEUE_AREA[1],
                                               x2 + QUEUE_AREA[0], y2 + QUEUE_AREA[1]))

            base_count = len(detections_full_image)
            adjusted_count = base_count + 50 if base_count > 0 else 0
            analyzer.update_history(adjusted_count)
            print(f"Detected: {base_count} people, Adjusted: {adjusted_count} people.")

            # --- Draw detections on the full image for live display ---
            for (x1, y1, x2, y2) in detections_full_image:
                cv2.rectangle(img_cv_full, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Add timestamp to the image for clarity
            cv2.putText(img_cv_full, f"Last updated: {current_time_str} (Live Detections Shown)",
                        (10, img_cv_full.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

            # --- Upload the pre-processed image to GCS ---
            if gcs_client_collector:
                try:
                    # Convert processed image to bytes
                    is_success, buffer = cv2.imencode(".jpg", img_cv_full)
                    if is_success:
                        blob_image = gcs_client_collector.bucket(GCS_BUCKET_NAME).blob(GCS_LIVE_IMAGE_OBJECT_NAME)
                        blob_image.upload_from_string(buffer.tobytes(), content_type='image/jpeg')
                        print(f"Live detection image uploaded to GCS: {GCS_LIVE_IMAGE_OBJECT_NAME}")
                    else:
                        print("Failed to encode image to JPG.")
                except Exception as e:
                    print(f"Error uploading live detection image to GCS: {e}")
            else:
                print("GCS client not available for uploading live image.")

        except requests.exceptions.RequestException as e:
            print(f"Network error fetching image from {CAMERA_URL}: {e}")
            print("Failed to fetch image. Skipping update.")
        except Exception as e:
            print(f"Error during image processing or detection: {e}. Skipping update.")

        time.sleep(300) # Wait for 5 minutes (300 seconds)
