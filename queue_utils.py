# queue_utils.py - REFINED get_gcs_client to handle non-Streamlit contexts
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

# ... (rest of your constants) ...

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
        if hasattr(st, 'secrets') and st.runtime.exists() and "gcs_credentials_base64" in st.secrets:
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

# ... (rest of QueueAnalyzer class and other functions) ...
