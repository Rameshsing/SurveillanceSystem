# alert_service.py
from alerts import send_email_alert, send_whatsapp_alert

def send_surveillance_alert(alert_text, camera_id):
    """Send surveillance alerts via multiple channels"""
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    message = f"⚠️ {alert_text} detected on {camera_id} at {timestamp}"
    
    # Send via email and WhatsApp
    send_email_alert("Camera Alert", message)
    send_whatsapp_alert(message)
    print(f"[ALERT] {message}")

def send_alert(alert_type="general", details="", camera_id="unknown"):
    """General alert function for API calls"""
    message = f"Alert ({alert_type}): {details} on {camera_id}"
    send_surveillance_alert(message, camera_id)
