import streamlit as st
import pandas as pd
import glob
from PIL import Image
import os

st.title("🧠 People Flow Dashboard (Multi-Camera)")

# Heatmap section
st.header("📸 Upload Your Heatmap")
uploaded_img = st.file_uploader("Upload heatmap image", type=["jpg", "png"])
if uploaded_img:
    img = Image.open(uploaded_img)
    st.image(img, caption="Uploaded Heatmap", use_column_width=True)

# Traffic Logs
st.header("📈 Select Camera Traffic Log")
csv_files = glob.glob("logs/traffic_log_*.csv")
if csv_files:
    selected = st.selectbox("Choose Camera Log", csv_files)
    df = pd.read_csv(selected)
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values("time")
    st.line_chart(df.set_index("time")[["in", "out"]])
    st.dataframe(df.tail(10))

if "posture" in df.columns:
    st.bar_chart(df["posture"].value_counts())

if "alert" in df.columns:
    alert_df = df[df["alert"].str.strip() != ""]
    if not alert_df.empty:
        st.subheader("🚨 Detected Alerts")
        st.dataframe(alert_df[["time", "alert", "posture"]].tail(10))
