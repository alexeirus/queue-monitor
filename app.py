# streamlit_app.py - MODIFIED for new charting and prediction logic
import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
import cv2
import os
from datetime import datetime
import pytz
from io import BytesIO
from PIL import Image

# Import QueueAnalyzer and other utilities
from queue_utils import QueueAnalyzer, download_yolo_model, MODEL_PATH, MODEL_URL, CAMERA_URL, QUEUE_AREA, GCS_BUCKET_NAME, GCS_LIVE_IMAGE_OBJECT_NAME, get_gcs_client

# --- Streamlit Page Configuration ---
st.set_page_config(layout="wide", page_title="Narva Border Queue Status")

# --- Constants ---
TIMEZONE = 'Europe/Tallinn'
tz = pytz.timezone(TIMEZONE)

# --- Global or Cached Resources ---
@st.cache_resource
def load_yolo_model():
    """Caches the YOLO model loading."""
    if download_yolo_model(MODEL_PATH, MODEL_URL):
        from ultralytics import YOLO
        return YOLO(MODEL_PATH)
    return None

@st.cache_resource
def get_analyzer():
    """Caches the QueueAnalyzer instance."""
    model = load_yolo_model()
    if model:
        return QueueAnalyzer(model)
    return None

analyzer = get_analyzer()
if analyzer is None:
    st.error("Failed to load YOLO model or initialize QueueAnalyzer. Please check logs.")
    st.stop()

# --- Streamlit App Layout ---
st.title("Narva Border Queue Status")

col1, col2 = st.columns([2, 1])

with col1:
    st.header("Live Camera Feed")
    live_image_placeholder = st.empty() # Placeholder for dynamic image updates
    last_collector_update_placeholder = st.empty() # Placeholder for collector update timestamp

    gcs_client = get_gcs_client()
    if gcs_client:
        bucket = gcs_client.bucket(GCS_BUCKET_NAME)
        blob = bucket.blob(GCS_LIVE_IMAGE_OBJECT_NAME)
        if blob.exists():
            try:
                # Use a unique URL or query parameter to bust cache and ensure fresh image
                image_url = blob.public_url + f"?t={datetime.now().timestamp()}"
                live_image_placeholder.image(image_url, caption="Live Detection View", use_column_width=True)
                
                # Fetch metadata to get the last update time from GCS object
                blob.reload() # Reload blob metadata to get the latest update_time
                last_update_utc = blob.update_time
                last_update_local = last_update_utc.astimezone(tz)
                last_collector_update_placeholder.caption(f"Last updated: {last_update_local.strftime('%Y-%m-%d %H:%M:%S')} (Live Detections by Collector)")

            except Exception as e:
                st.error(f"Error loading live image from GCS: {e}")
                live_image_placeholder.write("Could not load live image.")
        else:
            live_image_placeholder.write("Live detection image not yet available on GCS.")
    else:
        st.warning("GCS client not initialized. Cannot fetch live image.")
        live_image_placeholder.write("Live image display unavailable.")

with col2:
    st.header("Current Status")

    # Fetch latest count and trend
    current_count = analyzer.history_df['count'].iloc[-1] if not analyzer.history_df.empty else 0
    current_trend = analyzer.predict_trend()

    st.metric(label="Detected People (Live)", value=current_count)
    st.write(f"Trend: {current_trend}")

    st.subheader("Optimal Crossing Times")
    best_times_info = analyzer.get_overall_best_times()
    st.markdown(f"**Best Day(s) to Cross:** <span style='color:green;font-weight:bold;'>{best_times_info['best_day_name']}</span>", unsafe_allow_html=True)
    st.markdown(f"**Best Hour(s) to Cross:** <span style='color:green;font-weight:bold;'>{best_times_info['best_hours']}</span>", unsafe_allow_html=True)
    st.info("Based on historical average queue counts.")


st.markdown("---") # Separator

st.header("Historical Queue Data")

# --- Hourly Queue Trends by Day of Week ---
st.subheader("Hourly Queue Trends by Day of Week (Average Count)")
st.write("These charts show the average number of people in the queue for each hour of the day, broken down by day of the week. Use this to identify historically quieter times.")

day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# Create tabs for each day
tab_objects = st.tabs(day_names)

for i, day_name in enumerate(day_names):
    with tab_objects[i]:
        st.write(f"**Average Queue Count on {day_name}s**")
        hourly_data = analyzer.get_hourly_averages_for_day(i)

        if not hourly_data.empty:
            chart = alt.Chart(hourly_data.reset_index()).mark_line(point=True).encode(
                x=alt.X('hour', axis=alt.Axis(title='Hour of Day (24-hour)', format='.0f', values=list(range(0, 24, 2)))),
                y=alt.Y('average_count', title='Average People in Queue'),
                tooltip=['hour_str', 'average_count']
            ).properties(
                title=f'Hourly Average Queue on {day_name}s'
            ).interactive()
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info(f"No historical data available for {day_name} yet.")


st.markdown("---")

# Daily Queue Trends (The original one, if you wish to keep it, otherwise remove this block)
st.subheader("Daily Queue Trends (Last 7 Days)")
if not analyzer.history_df.empty:
    # Resample to daily mean for a smoother trend over days
    daily_avg_df = analyzer.history_df['count'].resample('D').mean().reset_index()
    daily_avg_df.columns = ['date', 'average_count']

    # Filter for the last 7 days for a relevant trend
    if len(daily_avg_df) > 7:
        daily_avg_df = daily_avg_df.tail(7)

    if not daily_avg_df.empty:
        daily_trend_chart = alt.Chart(daily_avg_df).mark_line(point=True).encode(
            x=alt.X('date', title='Date'),
            y=alt.Y('average_count', title='Average People in Queue'),
            tooltip=[alt.Tooltip('date', format='%Y-%m-%d'), 'average_count']
        ).properties(
            title='Daily Average Queue Count'
        ).interactive()
        st.altair_chart(daily_trend_chart, use_container_width=True)
    else:
        st.info("Not enough data to show daily trends for the last 7 days.")
else:
    st.info("No historical data to display daily trends.")

st.markdown("---")
st.caption("Data collected by queue_collector.py service. All times are in Europe/Tallinn timezone.")
