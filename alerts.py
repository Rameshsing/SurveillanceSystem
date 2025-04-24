import smtplib
from twilio.rest import Client

def send_advanced_alert(subject, message):
    send_email_alert(subject, message)
    send_whatsapp_alert(message)

def send_email_alert(subject, message):
    sender_email = "your_email@example.com"
    receiver_email = "recipient@example.com"
    password = "your_email_password"
    try:
        server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
        server.login(sender_email, password)
        message = f'Subject: {subject}\n\n{message}'
        server.sendmail(sender_email, receiver_email, message)
        server.quit()
    except Exception as e:
        print(f"Error sending email: {e}")

def send_whatsapp_alert(message):
    account_sid = "your_twilio_account_sid"
    auth_token = "your_twilio_auth_token"
    client = Client(account_sid, auth_token)
    
    message = client.messages.create(
        body=message,
        from_="whatsapp:+8801300155542",
        to="whatsapp:+your_number"
    )
    print(f"WhatsApp message sent: {message.sid}")
