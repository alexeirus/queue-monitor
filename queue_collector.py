import cv2
import numpy as np
from datetime import datetime
import time
from PIL import Image
from io import BytesIO
import os
import requests # <--- THIS IS THE CRUCIAL MISSING IMPORT FOR THE ORIGINAL ERROR

# Import necessary components from queue_utils
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
    QUEUE_AREA, # For drawing the detection area
    ADJUSTMENT_FACTOR # For the adjusted count
)

# --- Main execution loop for queue_collector.py ---
if __name__ == '__main__':
    print("Starting queue_collector.py...")
    # Download model (local for collector)
    if not download_yolo_model(MODEL_PATH, MODEL_URL):
        print("Exiting collector due to model download failure.")
        exit(1)

    # Initialize YOLO model and QueueAnalyzer
    # The QueueAnalyzer handles GCS client internally for history saving
    model = YOLO(MODEL_PATH)
    analyzer = QueueAnalyzer(model) 
    
    # Get a separate GCS client specifically for uploading the live image,
    # as QueueAnalyzer's client is for history.
    gcs_client_for_live_image = get_gcs_client() 

    while True:
        current_time_str = datetime.now(analyzer.tz).strftime("%Y-%m-%d %H:%M:%S")
        print(f"Collecting data at {current_time_str}...")

        try:
            # Fetch the *full* image from the camera URL
            response = requests.get(CAMERA_URL, timeout=10)
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            
            img = Image.open(BytesIO(response.content))
            img_np_full = np.array(img) # Convert PIL image to NumPy array

            # Convert to BGR for OpenCV processing (YOLO will handle internally)
            if img_np_full.ndim == 3 and img_np_full.shape[2] == 4: # RGBA
                img_cv_full = cv2.cvtColor(img_np_full, cv2.COLOR_RGBA2BGR)
            else: # RGB
                img_cv_full = cv2.cvtColor(img_np_full, cv2.COLOR_RGB2BGR)

            # Crop the image for detection only (this is the actual area YOLO will analyze)
            cropped_image_for_detection = img_cv_full[QUEUE_AREA[1]:QUEUE_AREA[3], QUEUE_AREA[0]:QUEUE_AREA[2]]

            # Perform detections using the analyzer
            detections_relative_to_cropped = analyzer.detect_pedestrians(cropped_image_for_detection)

            # Adjust detection coordinates back to be relative to the *full* image for drawing
            detections_full_image = []
            for (x1, y1, x2, y2) in detections_relative_to_cropped:
                # Add the offset of the cropped area's top-left corner
                detections_full_image.append((x1 + QUEUE_AREA[0], y1 + QUEUE_AREA[1],
                                               x2 + QUEUE_AREA[0], y2 + QUEUE_AREA[1]))

            base_count = len(detections_full_image)
            # Apply the adjustment factor (if base_count is 0, adjustment is 0)
            adjusted_count = base_count + ADJUSTMENT_FACTOR if base_count > 0 else 0
            
            # Update the history in GCS via the analyzer
            analyzer.update_history(adjusted_count)
            print(f"Detected: {base_count} people, Adjusted for history: {adjusted_count} people.")

            # --- Draw detections and area on the full image for live display ---
            # Draw bounding boxes
            for (x1, y1, x2, y2) in detections_full_image:
                cv2.rectangle(img_cv_full, (x1, y1), (x2, y2), (0, 255, 0), 2) # Green boxes

            # Draw the QUEUE_AREA rectangle
            cv2.rectangle(img_cv_full, (QUEUE_AREA[0], QUEUE_AREA[1]), (QUEUE_AREA[2], QUEUE_AREA[3]),
                          (0, 0, 255), 2) # Red rectangle for the detection area

            # Add timestamp to the image for clarity
            cv2.putText(img_cv_full, f"Last updated: {current_time_str} | Detected: {base_count} (Raw)",
                        (10, img_cv_full.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

            # --- Upload the pre-processed image to GCS for the app to display ---
            if gcs_client_for_live_image:
                try:
                    # Convert processed image (NumPy array) to JPG bytes
                    is_success, buffer = cv2.imencode(".jpg", img_cv_full)
                    if is_success:
                        blob_image = gcs_client_for_live_image.bucket(GCS_BUCKET_NAME).blob(GCS_LIVE_IMAGE_OBJECT_NAME)
                        blob_image.upload_from_string(buffer.tobytes(), content_type='image/jpeg')
                        print(f"Live detection image uploaded to GCS: {GCS_LIVE_IMAGE_OBJECT_NAME}")
                    else:
                        print("Failed to encode image to JPG.")
                except Exception as e:
                    print(f"Error uploading live detection image to GCS: {e}")
            else:
                print("GCS client not available for uploading live image.")

        except requests.exceptions.RequestException as e:
            print(f"Network error fetching image from {CAMERA_URL}: {e}. Skipping update.")
        except Exception as e:
            print(f"Error during image processing or detection: {e}. Skipping update.")

        time.sleep(300) # Wait for 5 minutes (300 seconds) before next collection
