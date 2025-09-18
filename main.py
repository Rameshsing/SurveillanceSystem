# main.py
import os
import cv2
import csv
import numpy as np
from datetime import datetime
from db import init_db, insert_log
from dashboard import app as dashboard_app
from core_processing import SurveillanceProcessor
# Remove: from app import send_alert  # This caused the circular import!

camera_feeds = {
    "live_cam": 0,          # default webcam
    # "secondary_cam": 1,   # uncomment if a second camera is connected
}

def main():
    processor = SurveillanceProcessor()
    
    for camera_id, path in camera_feeds.items():
        print(f"Starting surveillance for {camera_id}")
        processor.process_camera_stream(camera_id, path)
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Run either the dashboard OR surveillance, not both simultaneously
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "dashboard":
        dashboard_app.run_server(debug=True)
    else:
        main()
