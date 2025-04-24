import streamlit as st
import pandas as pd
import sqlite3
import os
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

st.set_page_config(layout="wide")
st.title("📊 Smart Surveillance Dashboard")

DB_PATH = "logs.db"
LOG_FOLDER = "logs"

@st.cache_data
def load_data():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM TrafficLog", conn)
    conn.close()
    df['time'] = pd.to_datetime(df['time'])
    return df

df = load_data()

# Sidebar filters
st.sidebar.header("🔍 Filter Options")
camera_options = df['camera_id'].unique().tolist()
selected_camera = st.sidebar.multiselect("Select Camera(s)", camera_options, default=camera_options)

posture_options = df['posture'].unique().tolist()
selected_posture = st.sidebar.multiselect("Select Posture(s)", posture_options, default=posture_options)

alert_options = df['alert'].unique().tolist()
selected_alert = st.sidebar.multiselect("Select Alert(s)", alert_options, default=alert_options)

min_time = df['time'].min()
max_time = df['time'].max()
start_time, end_time = st.sidebar.slider("Select Time Range", min_value=min_time, max_value=max_time,
                                         value=(min_time, max_time))

# Apply filters
filtered_df = df[
    (df['camera_id'].isin(selected_camera)) &
    (df['posture'].isin(selected_posture)) &
    (df['alert'].isin(selected_alert)) &
    (df['time'].between(start_time, end_time))
]

st.subheader("📋 Filtered Logs")
st.dataframe(filtered_df.sort_values(by='time', ascending=False), use_container_width=True)

# Charts: IN/OUT trends
st.subheader("📈 People Flow Trends")
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### People IN Count")
    in_trend = filtered_df.groupby('time')['count_in'].sum().reset_index()
    st.line_chart(in_trend.rename(columns={"count_in": "People IN"}).set_index('time'))

with col2:
    st.markdown("#### People OUT Count")
    out_trend = filtered_df.groupby('time')['count_out'].sum().reset_index()
    st.line_chart(out_trend.rename(columns={"count_out": "People OUT"}).set_index('time'))

# Alerts trend
st.subheader("🚨 Alert Frequency")
alert_counts = filtered_df['alert'].value_counts()
st.bar_chart(alert_counts)

# Heatmaps and zone counts
st.subheader("🔥 Visual Overlays")
for cam_id in selected_camera:
    heatmap_path = os.path.join(LOG_FOLDER, f"heatmap_{cam_id}.jpg")
    zone_path = os.path.join(LOG_FOLDER, f"zone_heatmap_{cam_id}.jpg")
    zone_csv = os.path.join(LOG_FOLDER, f"zone_counts_{cam_id}.csv")

    st.markdown(f"#### {cam_id} Heatmaps")

    cols = st.columns(2)
    if os.path.exists(heatmap_path):
        with cols[0]:
            st.image(Image.open(heatmap_path), caption=f"Motion Heatmap - {cam_id}", use_column_width=True)
    if os.path.exists(zone_path):
        with cols[1]:
            st.image(Image.open(zone_path), caption=f"Zone Heatmap - {cam_id}", use_column_width=True)

    # Zone counts
    if os.path.exists(zone_csv):
        zone_df = pd.read_csv(zone_csv)
        zone_df[['Row', 'Col']] = zone_df['Zone(Row,Col)'].str.strip('()').str.split(',', expand=True).astype(int)
        zone_matrix = zone_df.pivot(index='Row', columns='Col', values='Count')

        st.markdown("##### Zone Occupancy Matrix")
        fig, ax = plt.subplots()
        sns.heatmap(zone_matrix, annot=True, fmt='d', cmap="YlOrRd", ax=ax)
        st.pyplot(fig)

