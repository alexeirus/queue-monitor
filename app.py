# app.py (streamlit_app.py) - Reverted to richer visuals, with enhanced predictions and new per-day hourly charts
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
    QueueAnalyzer, # Ensure QueueAnalyzer is imported
    MODEL_PATH, # Ensure MODEL_PATH is imported for YOLO initialization
    download_yolo_model, # Ensure download_yolo_model is imported
    MODEL_URL # Ensure MODEL_URL is imported for YOLO download
)

# Import YOLO for model loading
from ultralytics import YOLO

# Initialize timezone from queue_utils
tz = pytz.timezone(TIMEZONE)

# --- Configuration for Predictive Analytics ---
OPERATIONAL_START_HOUR = 7
OPERATIONAL_END_HOUR = 23 # Exclusive, so up to 22:59:59
RAMP_UP_HOURS = 2
RAMP_DOWN_HOURS = 2

# --- Helper Functions for Streamlit App ---

@st.cache_data(ttl=5) # Cache for 5 seconds to avoid constant image re-fetches
def fetch_live_detection_image_from_gcs():
    """Fetches the latest pre-processed image with detections from GCS."""
    gcs_client = get_gcs_client()
    if not gcs_client:
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
        return None

@st.cache_data(ttl=60) # Cache for 60 seconds (1 minute) for history data
def load_queue_history_from_gcs_for_display():
    """Loads queue history from GCS for display and analytics in Streamlit."""
    gcs_client = get_gcs_client()
    if not gcs_client:
        st.warning("GCS client not initialized. Cannot load history from GCS.")
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
            
            # Make timezone-naive AFTER local conversion for Streamlit plotting compatibility if needed
            # For Plotly, it often works well with timezone-aware, but if issues arise, this line helps.
            df['timestamp'] = df['timestamp'].dt.tz_localize(None) 
            
            df.set_index('timestamp', inplace=True)

            if 'count' in df.columns:
                df['count'] = pd.to_numeric(df['count'], errors='coerce').fillna(0).astype(int)
            else:
                st.error("CSV must contain 'count' column. Analytics may be impacted.")
                df['count'] = 0
            
            df['day_of_week'] = df.index.weekday # Ensure day_of_week is available
            df['hour'] = df.index.hour # Ensure hour is available

            return df
        except pd.errors.EmptyDataError:
            st.info(f"GCS object {GCS_OBJECT_NAME} is empty. Starting with an empty history for display.")
            return pd.DataFrame(columns=['timestamp', 'count', 'day_of_week', 'hour'])
        except Exception as e:
            st.error(f"Error loading queue history from GCS for display: {e}")
            return pd.DataFrame(columns=['timestamp', 'count', 'day_of_week', 'hour'])
    else:
        st.info(f"GCS object {GCS_OBJECT_NAME} not found. Starting with an empty history for display.")
        return pd.DataFrame(columns=['timestamp', 'count', 'day_of_week', 'hour'])

# --- Predictive Analytics & Charting Functions ---

def analyze_hourly_trends(df):
    if df.empty:
        return None, "No data to analyze hourly trends."

    start_hour_with_ramp = (OPERATIONAL_START_HOUR - RAMP_UP_HOURS + 24) % 24
    end_hour_with_ramp = (OPERATIONAL_END_HOUR + RAMP_DOWN_HOURS + 24) % 24

    if start_hour_with_ramp < end_hour_with_ramp:
        df_filtered = df[(df.index.hour >= start_hour_with_ramp) & (df.index.hour < end_hour_with_ramp)].copy()
    else:
        # Handle overnight operational hours
        df_filtered = df[(df.index.hour >= start_hour_with_ramp) | (df.index.hour < end_hour_with_ramp)].copy()

    df_filtered['count'] = pd.to_numeric(df_filtered['count'], errors='coerce')
    df_filtered.dropna(subset=['count'], inplace=True)

    if df_filtered.empty:
        return None, "No data for selected operational hours and ramp-up/down."

    hourly_avg = df_filtered.groupby(df_filtered.index.hour)['count'].mean().reset_index()
    hourly_avg.columns = ['Hour', 'Average Queue']
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
    fig.update_xaxes(dtick=1)
    fig.update_layout(hovermode="x unified")

    best_hours_df = hourly_avg[
        (hourly_avg['Hour'] >= OPERATIONAL_START_HOUR) &
        (hourly_avg['Hour'] < OPERATIONAL_END_HOUR)
    ]
    if not best_hours_df.empty:
        best_times = best_hours_df.sort_values('Average Queue').head(3)
        best_times_str = ", ".join([f"{int(h):02d}:00-{int(h)+1:02d}:02d" for h in best_times['Hour']])
    else:
        best_times_str = "N/A (No data for strict operational hours)"

    return fig, f"**Historically Best Times to Cross (during operational hours):** {best_times_str}"

def analyze_daily_trends(df):
    if df.empty:
        return None, "No data to analyze daily trends."

    df_filtered = df[
        (df.index.hour >= OPERATIONAL_START_HOUR) &
        (df.index.hour < OPERATIONAL_END_HOUR)
    ].copy()

    df_filtered['count'] = pd.to_numeric(df_filtered['count'], errors='coerce')
    df_filtered.dropna(subset=['count'], inplace=True)

    if df_filtered.empty:
        return None, "No data for selected operational hours on any day."

    day_names = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
    daily_avg_all_days = pd.DataFrame(list(day_names.items()), columns=['day_of_week', 'Day Name'])
    
    actual_daily_avg = df_filtered.groupby(df_filtered.index.dayofweek)['count'].mean().reset_index()
    actual_daily_avg.columns = ['day_of_week', 'Average Queue']

    daily_avg_merged = pd.merge(daily_avg_all_days, actual_daily_avg, on='day_of_week', how='left')
    daily_avg_merged['Average Queue'].fillna(0, inplace=True)
    daily_avg_merged.sort_values('day_of_week', inplace=True)

    fig = px.bar(
        daily_avg_merged,
        x='Day Name',
        y='Average Queue',
        title='Average Queue Size by Day of Week (Operational Hours)',
        labels={'Day Name': 'Day of Week', 'Average Queue': 'Average Queue Size (People)'}
    )
    fig.update_layout(hovermode="x unified")

    best_days_df = daily_avg_merged[daily_avg_merged['Average Queue'] > 0] 
    if not best_days_df.empty:
        best_days = best_days_df.sort_values('Average Queue').head(3)
        best_days_str = ", ".join(best_days['Day Name'])
    else:
        best_days_str = "N/A (No significant historical data for days)"

    return fig, f"**Historically Best Days to Cross (during operational hours):** {best_days_str}"

def analyze_queue_movement_speed(df):
    if df.empty:
        return None, "No data for queue movement speed analysis."
    
    df_filtered = df[
        (df.index.hour >= OPERATIONAL_START_HOUR) &
        (df.index.hour < OPERATIONAL_END_HOUR)
    ].copy()

    if df_filtered.empty:
        return None, "No operational data for queue movement speed analysis."

    df_resampled = df_filtered['count'].resample('1H').mean().ffill()    
    
    if len(df_resampled) < 2:
        return None, "Insufficient data after resampling for movement speed."

    df_resampled_diff = df_resampled.diff().dropna()
    df_resampled_diff = df_resampled_diff.reset_index()
    df_resampled_diff.columns = ['Hour', 'Change in Queue (People/Hour)']
    df_resampled_diff['Hour'] = df_resampled_diff['Hour'].dt.hour

    fig = px.bar(
        df_resampled_diff,
        x='Hour',
        y='Change in Queue (People/Hour)',
        title='Hourly Change in Queue Size (People/Hour) During Operational Hours',
        labels={'Hour': 'Hour of Day', 'Change in Queue (People/Hour)': 'Change in Queue Size'},
        color='Change in Queue (People/Hour)',
        color_continuous_scale=px.colors.sequential.RdBu,
        range_color=[-max(abs(df_resampled_diff['Change in Queue (People/Hour)'].max()), abs(df_resampled_diff['Change in Queue (People/Hour)'].min())),
                     max(abs(df_resampled_diff['Change in Queue (People/Hour)'].max()), abs(df_resampled_diff['Change in Queue (People/Hour)'].min()))]    
    )
    fig.update_layout(hovermode="x unified")
    fig.update_xaxes(dtick=1)
    
    return fig, "This graph shows the approximate change in queue size per hour. Positive values mean the queue is growing, negative values mean it's shrinking. A larger absolute value indicates faster movement."

def analyze_daily_summary_queue(df):
    """
    Generates a Plotly chart showing daily min, max, and average queue counts.
    """
    if df.empty:
        return None, "No data for daily summary analysis."

    daily_summary = df.resample('D')['count'].agg(['min', 'max', 'mean']).reset_index()
    daily_summary.columns = ['Date', 'Min Queue', 'Max Queue', 'Average Queue']

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=daily_summary['Date'],
        y=daily_summary['Max Queue'],
        mode='lines+markers',
        name='Daily Max Queue',
        line=dict(color='red', width=2),
        hovertemplate='<b>Date:</b> %{x|%Y-%m-%d}<br><b>Max Queue:</b> %{y}<extra></extra>'
    ))

    fig.add_trace(go.Scatter(
        x=daily_summary['Date'],
        y=daily_summary['Average Queue'],
        mode='lines+markers',
        name='Daily Average Queue',
        line=dict(color='blue', width=2),
        hovertemplate='<b>Date:</b> %{x|%Y-%m-%d}<br><b>Average Queue:</b> %{y:.2f}<extra></extra>'
    ))

    fig.add_trace(go.Scatter(
        x=daily_summary['Date'],
        y=daily_summary['Min Queue'],
        mode='lines+markers',
        name='Daily Min Queue',
        line=dict(color='green', width=2),
        hovertemplate='<b>Date:</b> %{x|%Y-%m-%d}<br><b>Min Queue:</b> %{y}<extra></extra>'
    ))

    fig.update_layout(
        title='Daily Queue Summary (Min, Max, Average)',
        xaxis_title='Date (Narva)',
        yaxis_title='Queue Size (People)',
        hovermode="x unified"
    )
    
    fig.update_xaxes(
        rangeselector=dict(
            buttons=list([
                dict(count=7, label="1w", step="day", stepmode="backward"),
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=3, label="3m", step="month", stepmode="backward"),
                dict(step="all")
            ])
        ),
        rangeslider=dict(visible=True),
        type="date"
    )

    return fig, "This chart shows the minimum, maximum, and average queue size for each day. Hover for details."

def analyze_hourly_trends_for_each_day(df, analyzer):
    """
    Generates a series of Plotly bar charts, one for each day of the week,
    showing average queue counts per hour, presented in a tabbed format.
    Also displays the best time to cross for each specific day.
    """
    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    
    st.subheader("Hourly Queue Trends for Each Day of the Week")
    st.info("These charts show the average queue size for each hour, broken down by day of the week. Use the tabs below to switch between days. This helps identify the quietest times on specific days.")

    # Create tabs for each day
    tabs = st.tabs(day_names)

    for i, day_name in enumerate(day_names):
        with tabs[i]: # Place content inside the current tab
            hourly_averages = analyzer.get_hourly_averages_for_day(i) # Use the analyzer's method

            if not hourly_averages.empty and 'average_count' in hourly_averages.columns:
                # Calculate and display Best Time to Cross for this specific day
                operational_hourly_averages = hourly_averages[
                    (hourly_averages['hour'] >= OPERATIONAL_START_HOUR) &
                    (hourly_averages['hour'] < OPERATIONAL_END_HOUR)
                ]

                if not operational_hourly_averages.empty:
                    min_queue = operational_hourly_averages['average_count'].min()
                    # Find all hours that have this minimum average count
                    best_hours_for_day = operational_hourly_averages[
                        operational_hourly_averages['average_count'] == min_queue
                    ]['hour'].tolist()

                    best_hours_str = ", ".join([f"{int(h):02d}:00-{int(h)+1:02d}:00" for h in sorted(best_hours_for_day)])
                    st.success(f"**Best Time to Cross on {day_name}:** Approximately **{best_hours_str}** (average queue: {min_queue:.0f} people).")
                else:
                    st.info(f"No sufficient historical data for **{day_name}** during operational hours to determine best times.")

                fig = px.bar(
                    hourly_averages,
                    x='hour',
                    y='average_count',
                    title=f'Average Queue Size for {day_name}',
                    labels={'hour': 'Hour of Day (Narva)', 'average_count': 'Average Queue Size (People)'},
                    text='average_count' # Display value on top of bars
                )
                fig.update_traces(texttemplate='%{text:.0f}', textposition='outside')
                fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide', hovermode="x unified")
                fig.update_xaxes(dtick=1, range=[-0.5, 23.5]) # Ensure all hours are displayed
                fig.update_yaxes(range=[0, hourly_averages['average_count'].max() * 1.1]) # Adjust y-axis for better visibility

                st.plotly_chart(fig, use_container_width=True)
            else:
                st.markdown(f"**{day_name}:** *No sufficient historical data available for this day to generate a detailed hourly trend.*")
    st.markdown("---")


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
            timestamp_display = datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
            latest_image_placeholder.image(image_with_detections,
                                            caption=f"Last updated: {timestamp_display} (Live Detections by Collector)",
                                            channels="BGR", use_container_width=True)
        else:
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
                latest_image_placeholder.error(f"⚠️ An unexpected error occurred while loading image: {e}")

# --- Initialize YOLO Model and QueueAnalyzer ---
# Download the model if it doesn't exist
if not download_yolo_model(MODEL_PATH, MODEL_URL):
    st.error("Failed to download YOLO model. Please check your network connection or MODEL_URL.")
    st.stop() # Stop the app if model cannot be loaded

# Load the YOLO model and initialize QueueAnalyzer
model = YOLO(MODEL_PATH)
analyzer = QueueAnalyzer(model)

# Load historical data for display and analytics
history_df_for_display = load_queue_history_from_gcs_for_display()

# --- Display Current Status and Trend ---
current_queue_count = history_df_for_display['count'].iloc[-1] if not history_df_for_display.empty else 0
with live_section_container:
    with col2:
        live_count_metric_placeholder.metric(label="Current Queue Count", value=f"{current_queue_count} People")
        trend_status_placeholder.markdown(f"**Queue Trend:** {analyzer.predict_trend()}")
        
        # --- Overall Best Day and Hours to Cross (Moved here) ---
        best_overall_times = analyzer.get_overall_best_times()
        st.markdown("##### 🗓️ Overall Best Time to Cross")
        if best_overall_times["best_day_name"] != "N/A":
            st.info(f"The **best day** historically is **{best_overall_times['best_day_name']}**, around **{best_overall_times['best_hours']}** (Narva Time).")
        else:
            st.warning("Not enough historical data to determine overall best day and times to cross.")


st.markdown("---")

# --- Enhanced Predictive Analytics and Charts ---
st.header("Historical Queue Analytics and Predictions")

# Historical Average Queue Size by Hour
hourly_fig, hourly_msg = analyze_hourly_trends(history_df_for_display)
if hourly_fig:
    st.plotly_chart(hourly_fig, use_container_width=True)
    st.caption(hourly_msg)
else:
    st.info(hourly_msg)

st.markdown("---")

# Historical Average Queue Size by Day of Week
daily_fig, daily_msg = analyze_daily_trends(history_df_for_display)
if daily_fig:
    st.plotly_chart(daily_fig, use_container_width=True)
    st.caption(daily_msg)
else:
    st.info(daily_msg)

st.markdown("---")

# Hourly Change in Queue Size
movement_fig, movement_msg = analyze_queue_movement_speed(history_df_for_display)
if movement_fig:
    st.plotly_chart(movement_fig, use_container_width=True)
    st.caption(movement_msg)
else:
    st.info(movement_msg)

st.markdown("---")

# Daily Summary Chart
daily_summary_fig, daily_summary_msg = analyze_daily_summary_queue(history_df_for_display)
if daily_summary_fig:
    st.plotly_chart(daily_summary_fig, use_container_width=True)
    st.caption(daily_summary_msg)
else:
    st.info(daily_summary_msg)

st.markdown("---")

# Per-Day Hourly Charts (Now with tabs and per-day best times!)
analyze_hourly_trends_for_each_day(history_df_for_display, analyzer) # Pass analyzer here

st.markdown("---")
st.caption(f"Data last refreshed at: {datetime.now(tz).strftime('%Y-%m-%d %H:%M:%S %Z%z')}")
st.caption("Queue detections powered by YOLOv8. Historical data stored on Google Cloud Storage.")
