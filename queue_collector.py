import os
import requests
import time
from datetime import datetime
import pytz
import torch.serialization
import torch.nn.modules.container
import ultralytics.nn.tasks
import ultralytics.nn.modules
import ultralytics.nn.modules.conv
import ultralytics.nn.modules.block
from ultralytics import YOLO
from queue_analyzer import QueueAnalyzer # Assuming queue_analyzer.py is in the same directory

# ✅ Optional SPPF registration
try:
    import ultralytics.nn.modules.common as ul_common
    sppf = ul_common.SPPF
except (ImportError, AttributeError):
    sppf = None

# ✅ Register required globals for PyTorch unpickling
safe_globals = [
    ultralytics.nn.tasks.DetectionModel,
    ultralytics.nn.modules.Conv, # <- Essential for YOLOv8 models
    ultralytics.nn.modules.conv.Conv,
    ultralytics.nn.modules.conv.Concat,
    ultralytics.nn.modules.block.C2f,
    ultralytics.nn.modules.block.Bottleneck,
    torch.nn.modules.container.Sequential
]
if sppf:
    safe_globals.append(sppf)

# Apply the safe globals before any model loading attempts
torch.serialization.add_safe_globals(safe_globals)

# --- Configuration ---
MODEL_URL = "https://ultralytics.com/assets/yolov8s.pt"
MODEL_PATH = "yolov8s.pt"
CAMERA_URL = "https://thumbs.balticlivecam.com/blc/narva.jpg"
TIMEZONE = "Europe/Tallinn"
tz = pytz.timezone(TIMEZONE)

# --- Functions ---

def download_yolo_model(model_path, model_url):
    """Downloads the YOLOv8s model if it doesn't exist or is incomplete."""
    if not os.path.exists(model_path) or os.path.getsize(model_path) < 10000000: # Check size to ensure full download
        print(f"Downloading YOLOv8s model from {model_url}...")
        try:
            r = requests.get(model_url, stream=True)
            r.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
            with open(model_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("YOLOv8s model downloaded successfully.")
        except requests.exceptions.RequestException as e:
            print(f"Error downloading model: {e}. Please check the URL or network connection.")
            # Exit or handle gracefully if model cannot be downloaded
            exit(1) # Or re-raise, or return False to indicate failure

def run_cycle(analyzer):
    """Executes a single cycle of image fetching, detection, and history update."""
    now = datetime.now(tz)
    # Monitor only between 5 AM and 11 PM (inclusive)
    if 5 <= now.hour <= 23:
        print(f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] Running cycle...")
        image = analyzer.fetch_image(CAMERA_URL)
        if image is not None:
            detections = analyzer.detect_pedestrians(image)
            base_count = len(detections)
            # Add 50 people offset if any detection is made, otherwise 0
            adjusted_count = base_count + 50 if base_count > 0 else 0
            analyzer.update_history(adjusted_count)
            print(f"Detected {base_count} people. Adjusted count (including offset): {adjusted_count}.")
        else:
            print("Image fetch failed. Skipping detection for this cycle.")
    else:
        print(f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] Outside monitoring hours (5 AM - 11 PM). Skipping cycle.")

# --- Main Execution ---
if __name__ == "__main__":
    # Ensure the model is downloaded first
    download_yolo_model(MODEL_PATH, MODEL_URL)

    # Load the YOLO model and initialize the analyzer here, once
    try:
        model = YOLO(MODEL_PATH)
        analyzer = QueueAnalyzer(model)
        print("YOLO model loaded and QueueAnalyzer initialized successfully.")
    except Exception as e:
        print(f"Error loading YOLO model or initializing QueueAnalyzer: {e}")
        print("Exiting. Please ensure the model file is valid and dependencies are met.")
        exit(1) # Exit if the core components can't be loaded

    # Start the continuous monitoring loop
    while True:
        run_cycle(analyzer)
        time.sleep(180) # Wait for 3 minutes (180 seconds) before the next cycle
