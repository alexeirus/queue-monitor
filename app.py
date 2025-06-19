# app.py (streamlit_app.py) - MODIFIED
import os
import time
import streamlit as st
from datetime import datetime, timedelta # Import timedelta for range calculations
import pandas as pd
import cv2
import numpy as np
from io import BytesIO
import pytz
import plotly.express as px # Import plotly for better charts

# Import necessary components from queue_utils
from queue_utils import (
    get_gcs_client,
    GCS_BUCKET_NAME,
    GCS_OBJECT_NAME,
    GCS_LIVE_IMAGE_OBJECT_NAME,
    TIMEZONE,
    CAMERA_URL,
    QueueAnalyzer # We'll still use this for its specific prediction methods
)

# Initialize timezone from queue_utils
tz = pytz.timezone(TIMEZONE)

# --- Configuration for Predictive Analytics (NEW) ---
OPERATIONAL_START_HOUR = 7
OPERATIONAL_END_HOUR = 23 # Exclusive, so up to 22:59:59
# These controls how much data *outside* operational hours is included in hourly trends graph
RAMP_UP_HOURS = 2   # e.g., if START_HOUR is 7, data from 5 AM onwards included
RAMP_DOWN_HOURS = 2 # e.g., if END_HOUR is 23, data up to 1 AM (next day) included

# --- Helper Functions for Streamlit App ---

@st.cache_data(ttl=5) # Cache for 5 seconds to avoid constant image re-fetches
def fetch_live_detection_image_from_gcs():
    """Fetches the latest pre-processed image with detections from GCS."""
    gcs_client = get_gcs_client()
    if not gcs_client:
        st.error("GCS_CREDENTIALS_BASE64 environment variable not found. GCS client cannot be initialized for image fetch.")
        return None

    bucket = gcs_client.bucket(GCS_BUCKET_NAME)
    blob = bucket.blob(GCS_LIVE_IMAGE_OBJECT_NAME)

    if blob.exists():
        try:
            image_bytes = blob.download_as_bytes()
            # Use cv2.imdecode to read image bytes into a numpy array
            image_array = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
            return image_array
        except Exception as e:
            st.error(f"Error fetching live detection image from GCS: {e}")
            return None
    else:
        # Return None silently if not found, to avoid spamming info messages
        return None 

@st.cache_data(ttl=60) # Cache for 60 seconds (1 minute) for history data
def load_queue_history_from_gcs_for_display():
    """Loads queue history from GCS for display and analytics in Streamlit."""
    gcs_client = get_gcs_client()
    if not gcs_client:
        return pd.DataFrame(columns=['timestamp', 'count', 'day_of_week', 'hour'])

    bucket = gcs_client.bucket(GCS_BUCKET_NAME)
    blob = bucket.blob(GCS_OBJECT_NAME)

    if blob.exists():
        try:
            csv_bytes = blob.download_as_bytes()
            df = pd.read_csv(BytesIO(csv_bytes))

            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            df.dropna(subset=['timestamp'], inplace=True)
            
            # Ensure timestamp is timezone-aware and set as index
            if df['timestamp'].dt.tz is None: # If naive, assume UTC as it's saved as UTC
                df['timestamp'] = df['timestamp'].dt.tz_localize('UTC')
            df['timestamp'] = df['timestamp'].dt.tz_convert(TIMEZONE) # Convert to local display timezone
            
            # --- CRITICAL ADDITION FOR DISPLAY FIX ---
            # Make the timestamp timezone-naive AFTER converting to the local timezone
            # This prevents Streamlit/Plotly from re-converting to the browser's local time for display
            df['timestamp'] = df['timestamp'].dt.tz_localize(None)
            # --- END OF CRITICAL ADDITION ---

            df.set_index('timestamp', inplace=True)

            if 'count' in df.columns:
                df['count'] = pd.to_numeric(df['count'], errors='coerce').fillna(0).astype(int)
            else:
                st.error("CSV must contain 'count' column. Analytics may be impacted.")
                df['count'] = 0 # Default to 0 if 'count' is missing
            
            # --- ADDED FOR DIAGNOSTICS - KEEP TEMPORARILY ---
            st.write("--- Timezone Diagnostics ---")
            st.write(f"DataFrame Index Timezone (after tz_localize(None)): {df.index.tz}") # Should be None
            if not df.empty:
                st.write(f"First Timestamp in DataFrame: {df.index.min()}")
                st.write(f"Last Timestamp in DataFrame: {df.index.max()}")
                st.write(f"Current System Time (Narva): {datetime.now(tz).strftime('%Y-%m-%d %H:%M:%S %Z%z')}")
            st.write("--------------------------")
            # --- END OF DIAGNOSTICS ---

            return df
        except pd.errors.EmptyDataError:
            return pd.DataFrame(columns=['timestamp', 'count', 'day_of_week', 'hour'])
        except Exception as e:
            st.error(f"Error loading queue history from GCS for display: {e}")
            return pd.DataFrame(columns=['timestamp', 'count', 'day_of_week', 'hour'])
    else:
        return pd.DataFrame(columns=['timestamp', 'count', 'day_of_week', 'hour'])


# --- NEW Predictive Analytics Functions (using Plotly) ---

def analyze_hourly_trends(df):
    if df.empty:
        return None, "No data to analyze hourly trends."

    # Filter to operational hours with ramp-up/down
    # Adjusted modulo for robust handling of hours like 0, 1, 23, 24
    start_hour_with_ramp = (OPERATIONAL_START_HOUR - RAMP_UP_HOURS + 24) % 24
    end_hour_with_ramp = (OPERATIONAL_END_HOUR + RAMP_DOWN_HOURS + 24) % 24

    if start_hour_with_ramp < end_hour_with_ramp:
        df_filtered = df[(df.index.hour >= start_hour_with_ramp) & (df.index.hour < end_hour_with_ramp)].copy()
    else: # Spans across midnight, e.g., 22 to 3
        df_filtered = df[(df.index.hour >= start_hour_with_ramp) | (df.index.hour < end_hour_with_ramp)].copy()


    df_filtered['count'] = pd.to_numeric(df_filtered['count'], errors='coerce')
    df_filtered.dropna(subset=['count'], inplace=True)

    if df_filtered.empty:
        return None, "No data for selected operational hours and ramp-up/down."

    hourly_avg = df_filtered.groupby(df_filtered.index.hour)['count'].mean().reset_index()
    hourly_avg.columns = ['Hour', 'Average Queue']

    # Sort by hour for proper graph display
    hourly_avg.sort_values('Hour', inplace=True)

    fig = px.line(
        hourly_avg,
        x='Hour',
        y='Average Queue',
        title=f'Average Queue Size by Hour (Operational: {OPERATIONAL_START_HOUR}:00-{OPERATIONAL_END_HOUR}:00)',
        labels={'Hour': 'Hour of Day', 'Average Queue': 'Average Queue Size (People)'},
        line_shape='linear',
        markers=True
    )
    fig.update_xaxes(dtick=1) # Ensure hours are integers
    fig.update_layout(hovermode="x unified") # Nice hover effect

    # Identify best times within strict operational hours
    best_hours_df = hourly_avg[
        (hourly_avg['Hour'] >= OPERATIONAL_START_HOUR) &
        (hourly_avg['Hour'] < OPERATIONAL_END_HOUR)
    ]
    if not best_hours_df.empty:
        best_times = best_hours_df.sort_values('Average Queue').head(3)
        best_times_str = ", ".join([f"{int(h):02d}:00-{int(h)+1:02d}:00" for h in best_times['Hour']])
    else:
        best_times_str = "N/A (No data for strict operational hours)"

    return fig, f"**Historically Best Times to Cross (during operational hours):** {best_times_str}"

def analyze_daily_trends(df):
    if df.empty:
        return None, "No data to analyze daily trends."

    # Filter to operational hours to make daily averages more relevant
    df_filtered = df[
        (df.index.hour >= OPERATIONAL_START_HOUR) &
        (df.index.hour < OPERATIONAL_END_HOUR)
    ].copy()

    df_filtered['count'] = pd.to_numeric(df_filtered['count'], errors='coerce')
    df_filtered.dropna(subset=['count'], inplace=True)

    if df_filtered.empty:
        return None, "No data for selected operational hours on any day."

    # Map day_of_week (0=Monday, 6=Sunday) to names for better display
    day_names = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
    # Ensure all days are represented even if no data, for consistent order
    daily_avg_all_days = pd.DataFrame(list(day_names.items()), columns=['day_of_week', 'Day Name'])
    
    # Calculate average and merge
    actual_daily_avg = df_filtered.groupby(df_filtered.index.dayofweek)['count'].mean().reset_index()
    actual_daily_avg.columns = ['day_of_week', 'Average Queue']

    daily_avg_merged = pd.merge(daily_avg_all_days, actual_daily_avg, on='day_of_week', how='left')
    daily_avg_merged['Average Queue'].fillna(0, inplace=True) # Fill missing days with 0 or NaN
    daily_avg_merged.sort_values('day_of_week', inplace=True) # Sort by numeric day for correct order

    fig = px.bar(
        daily_avg_merged,
        x='Day Name',
        y='Average Queue',
        title='Average Queue Size by Day of Week (Operational Hours)',
        labels={'Day Name': 'Day of Week', 'Average Queue': 'Average Queue Size (People)'}
    )
    fig.update_layout(hovermode="x unified")

    # Identify best days (consider only days with actual data for sorting, or handle 0s gracefully)
    best_days_df = daily_avg_merged[daily_avg_merged['Average Queue'] > 0] # Only consider days with actual queue data
    if not best_days_df.empty:
        best_days = best_days_df.sort_values('Average Queue').head(3)
        best_days_str = ", ".join(best_days['Day Name'])
    else:
        best_days_str = "N/A (No significant historical data for days)"

    return fig, f"**Historically Best Days to Cross (during operational hours):** {best_days_str}"

def analyze_queue_movement_speed(df):
    if df.empty:
        return None, "No data for queue movement speed analysis."
    
    # Filter to operational hours to make speed analysis relevant
    df_filtered = df[
        (df.index.hour >= OPERATIONAL_START_HOUR) &
        (df.index.hour < OPERATIONAL_END_HOUR)
    ].copy()

    if df_filtered.empty:
        return None, "No operational data for queue movement speed analysis."

    # Resample to hourly mean, then calculate difference
    df_resampled = df_filtered['count'].resample('1H').mean().ffill() 
    
    if len(df_resampled) < 2:
        return None, "Insufficient data after resampling for movement speed."

    # Calculate the difference in count hour-over-hour
    df_resampled_diff = df_resampled.diff().dropna()
    df_resampled_diff = df_resampled_diff.reset_index()
    df_resampled_diff.columns = ['Hour', 'Change in Queue (People/Hour)']
    df_resampled_diff['Hour'] = df_resampled_diff['Hour'].dt.hour # Extract just the hour number

    fig = px.bar(
        df_resampled_diff,
        x='Hour',
        y='Change in Queue (People/Hour)',
        title='Hourly Change in Queue Size (People/Hour) During Operational Hours',
        labels={'Hour': 'Hour of Day', 'Change in Queue (People/Hour)': 'Change in Queue Size'},
        color='Change in Queue (People/Hour)', # Color by positive/negative change
        color_continuous_scale=px.colors.sequential.RdBu, # Red for decrease, Blue for increase
        # Ensure symmetric color range for better visualization of positive/negative changes
        range_color=[-max(abs(df_resampled_diff['Change in Queue (People/Hour)'].max()), abs(df_resampled_diff['Change in Queue (People/Hour)'].min())),
                     max(abs(df_resampled_diff['Change in Queue (People/Hour)'].max()), abs(df_resampled_diff['Change in Queue (People/Hour)'].min()))] 
    )
    fig.update_layout(hovermode="x unified")
    fig.update_xaxes(dtick=1) # Show hourly ticks
    
    return fig, "This graph shows the approximate change in queue size per hour. Positive values mean the queue is growing, negative values mean it's shrinking. A larger absolute value indicates faster movement."

# --- Streamlit App Layout ---
st.set_page_config(page_title="Narva Queue Monitor", layout="wide", initial_sidebar_state="collapsed")
st.title("🚶 Narva Queue Monitor - Real-time & Predictive Analytics")
st.markdown("---")

# Placeholder for live camera feed and metric
live_section_container = st.container()
with live_section_container:
    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        st.header("Live Camera Feed")
        latest_image_placeholder = st.empty()
    with col2:
        st.header("Current Status")
        live_count_metric_placeholder = st.empty()
        trend_status_placeholder = st.empty()


# --- Display Latest Camera Image and Live Count ---
image_with_detections = fetch_live_detection_image_from_gcs()

with live_section_container:
    with col1:
        if image_with_detections is not None:
            # Use Narva timezone for the live update timestamp
            timestamp_display = datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
            latest_image_placeholder.image(image_with_detections,
                                            caption=f"Last updated: {timestamp_display} (Live Detections by Collector)",
                                            channels="BGR", use_container_width=True)
        else:
            # Fallback if no pre-processed image is available from GCS (e.g., collector not running yet)
            try:
                import requests
                raw_image_response = requests.get(CAMERA_URL, timeout=10)
                raw_image_response.raise_for_status()
                raw_image_array = cv2.imdecode(np.frombuffer(raw_image_response.content, np.uint8), cv2.IMREAD_COLOR)
                # Use Narva timezone for fallback raw image timestamp
                timestamp_display = datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
                latest_image_placeholder.image(raw_image_array, caption=f"Last updated: {timestamp_display} (Raw Feed, No Detections)",
                                                channels="BGR", use_container_width=True)
            except requests.exceptions.RequestException as e:
                latest_image_placeholder.error(f"⚠️ Could not load image from camera feed or GCS: {e}")
            except Exception as e:
                latest_image_placeholder.error(f"⚠️ Error processing raw image: {e}")

    with col2:
        # Load queue history for the latest count and trend
        queue_df_for_live_info = load_queue_history_from_gcs_for_display()
        if not queue_df_for_live_info.empty and 'count' in queue_df_for_live_info.columns:
            latest_live_count = queue_df_for_live_info['count'].iloc[-1]
            live_count_metric_placeholder.metric("Detected People (Live)", int(latest_live_count))
            
            # Instantiate a temporary analyzer just to get the predict_trend
            # It will use the 'count' column which already includes the ADJUSTMENT_FACTOR
            temp_analyzer = QueueAnalyzer(None) 
            temp_analyzer.history_df = queue_df_for_live_info 
            trend_status_placeholder.write(f"**Trend:** {temp_analyzer.predict_trend()}")
        else:
            live_count_metric_placeholder.info("Waiting for live detection count.")
            trend_status_placeholder.info("Waiting for trend data.")


st.markdown("---")

# --- Display Historical Data and Predictive Analytics ---
st.header("Historical Data and Predictive Analytics")

# Load queue history for Streamlit display and analytics (this is cached)
full_history_df = load_queue_history_from_gcs_for_display()

if not full_history_df.empty:
    # 1. Graph: Queue numbers hour by hour (7-23 with ramp-up/down)
    st.subheader("Hourly Queue Trends")
    hourly_fig, hourly_text = analyze_hourly_trends(full_history_df)
    if hourly_fig:
        st.plotly_chart(hourly_fig, use_container_width=True)
        st.write(hourly_text)
    else:
        st.info(hourly_text)

    # 2. Graph: Best days of the week to cross
    st.subheader("Daily Queue Trends")
    daily_fig, daily_text = analyze_daily_trends(full_history_df)
    if daily_fig:
        st.plotly_chart(daily_fig, use_container_width=True)
        st.write(daily_text)
    else:
        st.info(daily_text)

    # 3. Graph: How fast the queue is moving
    st.subheader("Queue Movement Speed")
    movement_fig, movement_text = analyze_queue_movement_speed(full_history_df)
    if movement_fig:
        st.plotly_chart(movement_fig, use_container_width=True)
        st.write(movement_text)
    else:
        st.info(movement_text)

    # You can still keep the general historical line chart if desired
    st.subheader("Raw Queue Count Over Time")
    st.line_chart(full_history_df['count'], use_container_width=True)


else:
    st.info("No historical queue data available yet. Please wait for the `queue_collector.py` worker to generate data and upload it to GCS.")

# --- Auto-refresh feature control (REMOVED st.rerun()) ---
# The caching (ttl) on the data loading functions handles the updates
st.markdown("---")
st.info("Data for charts and live image updates automatically. The page does not refresh visibly.")
st.caption(f"Live image updates every 5 seconds. Historical data updates every 60 seconds. (controlled by @st.cache_data ttl)")
