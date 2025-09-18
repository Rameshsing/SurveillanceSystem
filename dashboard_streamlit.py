# dashboard_streamlit.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
import os
from PIL import Image

from db import init_db  # use the app's DB connection to ensure same file

st.set_page_config(layout="wide")
st.title("📊 Smart Surveillance Dashboard")

# Idempotent table creation DDL
DDL = """
CREATE TABLE IF NOT EXISTS TrafficLog (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    time TEXT,
    "in" INTEGER,
    "out" INTEGER,
    camera_id TEXT,
    posture TEXT,
    alert TEXT
);
"""

@st.cache_data
def load_data():
    # Open the same DB used by the app
    conn = init_db()
    # Ensure the table exists (safe if it already does)
    conn.execute(DDL)
    # Read all logs
    df = pd.read_sql_query('SELECT * FROM TrafficLog', conn)
    conn.close()

    # Parse timestamps
    if not df.empty and 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'], errors='coerce')
        df = df.dropna(subset=['time'])

    return df

df = load_data()

# Empty-state handling
if df.empty:
    st.info("No data available yet. Start the pipeline to generate logs, then refresh this page.", icon="ℹ️")
    st.stop()

# Sidebar filters
st.sidebar.header("🔍 Filter Options")

camera_options = sorted(df['camera_id'].dropna().unique().tolist())
selected_camera = st.sidebar.multiselect("Select Camera(s)", camera_options, default=camera_options)

posture_options = sorted(df['posture'].dropna().unique().tolist())
selected_posture = st.sidebar.multiselect("Select Posture(s)", posture_options, default=posture_options)

alert_options = sorted([a for a in df['alert'].dropna().unique().tolist() if str(a).strip() != ""])
selected_alert = st.sidebar.multiselect("Select Alert(s)", alert_options, default=alert_options)

min_time = df['time'].min()
max_time = df['time'].max()
start_time, end_time = st.sidebar.slider(
    "Select Time Range",
    min_value=min_time,
    max_value=max_time,
    value=(min_time, max_time)
)

# Apply filters
mask = (
    df['camera_id'].isin(selected_camera) &
    df['posture'].isin(selected_posture) &
    df['time'].between(start_time, end_time)
)
# Alert filter: include empty alerts when no alert selected means "all"
if selected_alert:
    mask = mask & df['alert'].isin(selected_alert)

filtered_df = df.loc[mask].copy()

st.subheader("📋 Filtered Logs")
st.dataframe(filtered_df.sort_values(by='time', ascending=False), use_container_width=True)

# Charts: IN/OUT trends
st.subheader("📈 People Flow Trends")
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### People IN Count")
    if 'in' in filtered_df.columns:
        in_trend = filtered_df.groupby('time', as_index=False)['in'].sum()
        st.line_chart(in_trend.set_index('time').rename(columns={'in': 'People IN'}))
    else:
        st.warning("Column 'in' not found in data.")

with col2:
    st.markdown("#### People OUT Count")
    if 'out' in filtered_df.columns:
        out_trend = filtered_df.groupby('time', as_index=False)['out'].sum()
        st.line_chart(out_trend.set_index('time').rename(columns={'out': 'People OUT'}))
    else:
        st.warning("Column 'out' not found in data.")

# Alerts trend
st.subheader("🚨 Alert Frequency")
if 'alert' in filtered_df.columns:
    alert_counts = filtered_df['alert'].fillna("").replace("", "No Alert").value_counts()
    st.bar_chart(alert_counts)
else:
    st.warning("Column 'alert' not found in data.")

# Heatmaps and zone counts (optional image/CSV overlays)
LOG_FOLDER = "logs"
st.subheader("🔥 Visual Overlays")
for cam_id in selected_camera:
    heatmap_path = os.path.join(LOG_FOLDER, f"heatmap_{cam_id}.jpg")
    zone_path = os.path.join(LOG_FOLDER, f"zone_heatmap_{cam_id}.jpg")
    zone_csv = os.path.join(LOG_FOLDER, f"zone_counts_{cam_id}.csv")

    st.markdown(f"#### {cam_id} Heatmaps")

    cols = st.columns(2)
    if os.path.exists(heatmap_path):
        with cols:
            st.image(Image.open(heatmap_path), caption=f"Motion Heatmap - {cam_id}", use_column_width=True)
    if os.path.exists(zone_path):
        with cols[22]:
            st.image(Image.open(zone_path), caption=f"Zone Heatmap - {cam_id}", use_column_width=True)

    # Zone counts matrix
    if os.path.exists(zone_csv):
        zone_df = pd.read_csv(zone_csv)
        if 'Zone(Row,Col)' in zone_df.columns and 'Count' in zone_df.columns:
            zone_df[['Row', 'Col']] = zone_df['Zone(Row,Col)'].astype(str).str.strip('()').str.split(',', expand=True).astype(int)
            zone_matrix = zone_df.pivot(index='Row', columns='Col', values='Count')

            st.markdown("##### Zone Occupancy Matrix")
            fig, ax = plt.subplots()
            sns.heatmap(zone_matrix, annot=True, fmt='d', cmap="YlOrRd", ax=ax)
            st.pyplot(fig)
