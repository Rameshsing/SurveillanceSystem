import streamlit as st
import pandas as pd
import time
import os
from PIL import Image

st.title("People Flow Dashboard")

uploaded_csv = st.file_uploader("Upload traffic_log.csv", type="csv")
uploaded_img = st.file_uploader("Upload heatmap image", type=["jpg", "png"])

if uploaded_csv:
    df = pd.read_csv(uploaded_csv)
    st.line_chart(df.set_index("time")[["in", "out"]])

if uploaded_img:
    img = Image.open(uploaded_img)
    st.image(img, caption="Heatmap", use_column_width=True)

if os.path.exists("traffic_log.csv"):
    df = pd.read_csv("traffic_log.csv")
    st.line_chart(df.set_index("time")[["in", "out"]])