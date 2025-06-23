# app.py (streamlit_app.py) - Streamlined for Direct Predictions

import os
import time
import streamlit as st
from datetime import datetime, timedelta
import pandas as pd
import cv2
import numpy as np
from io import BytesIO
import pytz
import plotly.express as px
import plotly.graph_objects as go

# Import necessary components from queue_utils
from queue_utils import (
    get_gcs_client,
    GCS_BUCKET_NAME,
    GCS_OBJECT_NAME,
    GCS_LIVE_IMAGE_OBJECT_NAME,
    TIMEZONE,
    CAMERA_URL,
    QueueAnalyzer # We'll use this extensively for all predictions
)

# Initialize timezone from queue_utils
tz = pytz.timezone(TIMEZONE)

# --- Streamlit Page Configuration ---
st.set_page_config(page_title="Narva Queue Monitor", layout="wide", initial_sidebar_state="collapsed")
st.title("🚶 Narva Queue Monitor - Simplified Analytics & Best Times")
st.markdown("---")

# --- Helper Functions for Streamlit App ---

@st.cache_data(ttl=5) # Cache for 5 seconds to avoid constant image re-fetches
def fetch_live_detection_image_from_gcs():
    """Fetches the latest pre-processed image with detections from GCS."""
    gcs_client = get_gcs_client()
    if not gcs_client:
        # st.error("GCS_CREDENTIALS_BASE64 environment variable not found. GCS client cannot be initialized for image fetch.")
        return None # Return None silently if GCS not configured

    bucket = gcs_client.bucket(GCS_BUCKET_NAME)
    blob = bucket.blob(GCS_LIVE_IMAGE_OBJECT_NAME)

    if blob.exists():
        try:
            image_bytes = blob.download_as_bytes()
            image_array = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
            return image_array
        except Exception as e:
            st.error(f"Error fetching live detection image from GCS: {e}")
            return None
    else:
        return None # No image found yet or not updated

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
            
            if df['timestamp'].dt.tz is None:
                df['timestamp'] = df['timestamp'].dt.tz_localize('UTC')
            df['timestamp'] = df['timestamp'].dt.tz_convert(TIMEZONE)
            
            # Make timestamp naive AFTER converting to local timezone for Plotly compatibility
            df['timestamp'] = df['timestamp'].dt.tz_localize(None)
            
            df.set_index('timestamp', inplace=True)

            if 'count' in df.columns:
                df['count'] = pd.to_numeric(df['count'], errors='coerce').fillna(0).astype(int)
            else:
                st.error("CSV must contain 'count' column. Analytics may be impacted.")
                df['count'] = 0
            
            # Ensure day_of_week and hour columns are present for QueueAnalyzer
            df['day_of_week'] = df.index.weekday
            df['hour'] = df.index.hour

            return df
        except pd.errors.EmptyDataError:
            return pd.DataFrame(columns=['timestamp', 'count', 'day_of_week', 'hour'])
        except Exception as e:
            st.error(f"Error loading queue history from GCS for display: {e}")
            return pd.DataFrame(columns=['timestamp', 'count', 'day_of_week', 'hour'])
    else:
        return pd.DataFrame(columns=['timestamp', 'count', 'day_of_week', 'hour'])

# --- Streamlit App Layout ---
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

# Display Latest Camera Image and Live Count
image_with_detections = fetch_live_detection_image_from_gcs()

with live_section_container:
    with col1:
        if image_with_detections is not None:
            timestamp_display = datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
            latest_image_placeholder.image(image_with_detections,
                                            caption=f"Last updated: {timestamp_display} (Live Detections by Collector)",
                                            channels="BGR", use_container_width=True)
        else:
            # Fallback to raw camera URL if no processed image
            try:
                import requests
                raw_image_response = requests.get(CAMERA_URL, timeout=10)
                raw_image_response.raise_for_status()
                raw_image_array = cv2.imdecode(np.frombuffer(raw_image_response.content, np.uint8), cv2.IMREAD_COLOR)
                timestamp_display = datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
                latest_image_placeholder.image(raw_image_array, caption=f"Last updated: {timestamp_display} (Raw Feed, No Detections)",
                                                channels="BGR", use_container_width=True)
            except requests.exceptions.RequestException as e:
                latest_image_placeholder.error(f"⚠️ Could not load image from camera feed or GCS: {e}")
            except Exception as e:
                latest_image_placeholder.error(f"⚠️ Error processing raw image: {e}")

    with col2:
        # Load queue history for the latest count and trend
        full_history_df = load_queue_history_from_gcs_for_display() # Load here once for entire app

        if not full_history_df.empty and 'count' in full_history_df.columns:
            latest_live_count = full_history_df['count'].iloc[-1]
            live_count_metric_placeholder.metric("Detected People (Live)", int(latest_live_count))
            
            # Instantiate analyzer for trend prediction
            temp_analyzer = QueueAnalyzer(None) 
            temp_analyzer.history_df = full_history_df 
            trend_status_placeholder.write(f"**Trend:** {temp_analyzer.predict_trend()}")
        else:
            live_count_metric_placeholder.info("Waiting for live detection count.")
            trend_status_placeholder.info("Waiting for trend data.")


st.markdown("---")

# --- Predictive Analytics & Historical Data ---
st.header("Best Times to Cross & Historical Trends")

if not full_history_df.empty:
    # Instantiate QueueAnalyzer once with the loaded full history
    analyzer = QueueAnalyzer(None) 
    analyzer.history_df = full_history_df

    # --- Overall Best Times to Cross ---
    st.subheader("Overall Best Times to Cross")
    best_times_info = analyzer.get_overall_best_times()
    
    st.markdown(f"Based on all historical data, the **best day(s)** to cross are typically: <span style='color:green; font-weight:bold; font-size:1.1em;'>{best_times_info['best_day_name']}</span>", unsafe_allow_html=True)
    st.markdown(f"And the **best hour(s)** to cross are often: <span style='color:green; font-weight:bold; font-size:1.1em;'>{best_times_info['best_hours']}</span>", unsafe_allow_html=True)
    st.markdown("---")

    # --- Best Times for Each Day of the Week (NEW REQUEST) ---
    st.subheader("Best Times for Each Day of the Week")
    st.info("Below are the historically quietest hours for each day of the week. Times are for Narva local time (Europe/Tallinn).")
    
    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    
    for i, day_name in enumerate(day_names):
        hourly_averages_for_day = analyzer.get_hourly_averages_for_day(i)
        
        if not hourly_averages_for_day.empty and 'average_count' in hourly_averages_for_day.columns:
            # Find the minimum average count for this day
            min_avg_count = hourly_averages_for_day['average_count'].min()
            
            # Filter for hours that have this minimum count
            best_hours_df = hourly_averages_for_day[hourly_averages_for_day['average_count'] == min_avg_count]
            
            # Get the unique best hours for this day
            best_hours_list = sorted(best_hours_df['hour'].unique())
            
            if best_hours_list:
                best_hours_str = ", ".join(f"{h:02d}:00-{h+1:02d}:00" for h in best_hours_list)
                st.markdown(f"**{day_name}:** Quietest around <span style='color:green; font-weight:bold;'>{best_hours_str}</span> (Avg: {min_avg_count:.0f} people)", unsafe_allow_html=True)
            else:
                st.markdown(f"**{day_name}:** *No specific quiet hours found for this day with current data.*")
        else:
            st.markdown(f"**{day_name}:** *No historical data available for this day.*")
    st.markdown("---")


    # --- Simplified Hourly Trend Chart (Overall) ---
    st.subheader("Average Queue Size by Hour (Overall)")
    # This is similar to analyze_hourly_trends but without the ramp-up/down
    # We can calculate this directly here or enhance a new method in QueueAnalyzer
    if not full_history_df.empty:
        hourly_avg_overall = full_history_df.groupby(full_history_df.index.hour)['count'].mean().reset_index()
        hourly_avg_overall.columns = ['Hour', 'Average Queue']
        hourly_avg_overall.sort_values('Hour', inplace=True)

        fig_hourly_overall = px.line(
            hourly_avg_overall,
            x='Hour',
            y='Average Queue',
            title='Overall Average Queue Size by Hour of Day',
            labels={'Hour': 'Hour of Day (Narva)', 'Average Queue': 'Average Queue Size (People)'},
            line_shape='linear',
            markers=True
        )
        fig_hourly_overall.update_xaxes(dtick=1)
        fig_hourly_overall.update_layout(hovermode="x unified")
        st.plotly_chart(fig_hourly_overall, use_container_width=True)
    else:
        st.info("No data available to display overall hourly trends.")

    # --- Raw Queue Count Over Time (with Zoom) ---
    st.subheader("Raw Queue Count Over Time (Zoomable)")
    df_for_raw_plot = full_history_df.reset_index()

    if not df_for_raw_plot.empty:
        fig_raw = px.line(
            df_for_raw_plot,
            x='timestamp',
            y='count',
            title='Raw Queue Count Over Time',
            labels={'timestamp': 'Date and Time (Narva)', 'count': 'Queue Size (People)'}
        )
        fig_raw.update_xaxes(
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1d", step="day", stepmode="backward"),
                    dict(count=7, label="1w", step="day", stepmode="backward"),
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(step="all")
                ])
            ),
            rangeslider=dict(visible=True),
            type="date"
        )
        fig_raw.update_layout(hovermode="x unified")
        st.plotly_chart(fig_raw, use_container_width=True)
    else:
        st.info("No raw queue data available.")


else:
    st.info("No historical queue data available yet. Please wait for the `queue_collector.py` worker to generate data and upload it to GCS.")

# --- Auto-refresh feature control ---
st.markdown("---")
st.info("Data for charts and live image updates automatically. The page does not refresh visibly.")
st.caption(f"Live image updates every 5 seconds. Historical data updates every 60 seconds. (controlled by @st.cache_data ttl)")
