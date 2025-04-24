import streamlit as st
import pandas as pd
import glob
from PIL import Image
import os

from flask import Flask, render_template, request
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///logs.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

class TrafficLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    time = db.Column(db.String(100))
    count_in = db.Column(db.Integer)
    count_out = db.Column(db.Integer)
    camera_id = db.Column(db.String(50))
    posture = db.Column(db.String(100))
    alert = db.Column(db.String(200))

@app.route('/')
def index():
    camera_id = request.args.get('camera_id', None)
    start_time = request.args.get('start_time', None)
    end_time = request.args.get('end_time', None)

    # Build query filter based on provided filters
    query = TrafficLog.query

    if camera_id:
        query = query.filter(TrafficLog.camera_id == camera_id)

    if start_time and end_time:
        query = query.filter(TrafficLog.time.between(start_time, end_time))

    logs = query.all()
    
    return render_template('index.html', logs=logs)

if __name__ == '__main__':
    app.run(debug=True)

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

if "alert" in df.columns:
    st.subheader("🚨 All Detected Alerts")
    alert_df = df[df["alert"].str.strip() != ""]
    st.dataframe(alert_df[["time", "alert", "posture"]].tail(20))

    # Filter by type (optional)
    st.markdown("### 📌 Filter by Alert Type")
    alert_type = st.selectbox("Alert Type", ["All", "Fall", "Crowd", "Inactivity"])
    if alert_type != "All":
        filtered = alert_df[alert_df["alert"].str.contains(alert_type)]
        st.dataframe(filtered.tail(10))

st.subheader("📥 Download Logs")
csv = df.to_csv(index=False).encode('utf-8')
st.download_button("Download CSV Log", csv, f"log_{selected.split('_')[-1]}", "text/csv")

st.subheader("📊 Alerts Breakdown")
alert_counts = df["alert"].str.split().explode().value_counts()
st.bar_chart(alert_counts)

st.subheader("📷 Saved Heatmaps")
heatmaps = glob.glob("logs/heatmap_*.jpg")
for img_path in heatmaps:
    st.image(Image.open(img_path), caption=os.path.basename(img_path), use_column_width=True)