import smtplib
from email.mime.text import MIMEText
from twilio.rest import Client

# Email
def send_email_alert(subject, body, to_email="recipient@example.com"):
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = "alif.rahman.c@gmail.com"
    msg["To"] = to_email
    
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login("alif.rahman.c@gmail.com", "app_password")
        smtp.send_message(msg)

# WhatsApp via Twilio
def send_whatsapp_alert(message):
    account_sid = "your_account_sid"
    auth_token = "your_auth_token"
    client = Client(account_sid, auth_token)

    client.messages.create(
        body=message,
        from_='whatsapp:+14155238886',  # Twilio sandbox number
        to='whatsapp:+880130055542'
    )
    print(f"WhatsApp Alert: {message}")
