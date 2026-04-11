# ─────────────────────────────────────────────
#  alerts.py  —  Email + SMS notifications
# ─────────────────────────────────────────────

import smtplib
import logging
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from pathlib import Path
from typing import Optional

from config import (
    ENABLE_EMAIL_ALERTS, SMTP_HOST, SMTP_PORT,
    SMTP_USER, SMTP_PASS, ALERT_RECIPIENT,
    ENABLE_SMS_ALERTS,
    TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN,
    TWILIO_FROM, TWILIO_TO,
)

logger = logging.getLogger(__name__)


def send_email_alert(
    timestamp: str,
    litter_label: str,
    snapshot_path: Optional[str] = None,
) -> bool:
    """
    Send email alert with optional snapshot attachment.
    Returns True on success.
    """
    if not ENABLE_EMAIL_ALERTS:
        return False

    try:
        msg = MIMEMultipart()
        msg["From"] = SMTP_USER
        msg["To"] = ALERT_RECIPIENT
        msg["Subject"] = f"[ALERT] Dumping detected — {timestamp}"

        body = f"""
        <html><body>
        <h2 style="color:#c0392b">⚠ Illegal Dumping Detected</h2>
        <table style="font-family:sans-serif;font-size:14px">
          <tr><td><b>Time</b></td><td>{timestamp}</td></tr>
          <tr><td><b>Camera</b></td><td>CAM-01</td></tr>
          <tr><td><b>Item detected</b></td><td>{litter_label}</td></tr>
        </table>
        <p>A snapshot has been saved and attached to this email.</p>
        </body></html>
        """
        msg.attach(MIMEText(body, "html"))

        if snapshot_path and Path(snapshot_path).exists():
            with open(snapshot_path, "rb") as f:
                img = MIMEImage(f.read(), name=Path(snapshot_path).name)
                img.add_header("Content-Disposition", "attachment",
                               filename=Path(snapshot_path).name)
                msg.attach(img)

        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASS)
            server.sendmail(SMTP_USER, ALERT_RECIPIENT, msg.as_string())

        logger.info(f"Email alert sent to {ALERT_RECIPIENT}")
        return True

    except Exception as e:
        logger.error(f"Email alert failed: {e}")
        return False


def send_sms_alert(timestamp: str, litter_label: str) -> bool:
    """
    Send SMS alert via Twilio.
    Returns True on success.
    """
    if not ENABLE_SMS_ALERTS:
        return False

    try:
        from twilio.rest import Client
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        message = client.messages.create(
            body=(
                f"LITTER ALERT | CAM-01\n"
                f"Time: {timestamp}\n"
                f"Item: {litter_label}\n"
                f"Dumping detected — check dashboard."
            ),
            from_=TWILIO_FROM,
            to=TWILIO_TO,
        )
        logger.info(f"SMS sent: {message.sid}")
        return True

    except ImportError:
        logger.warning("twilio not installed — run: pip install twilio")
        return False
    except Exception as e:
        logger.error(f"SMS alert failed: {e}")
        return False


def dispatch_alerts(timestamp: str, litter_label: str,
                    snapshot_path: Optional[str] = None) -> dict:
    """Fire all enabled alert channels. Returns status dict."""
    return {
        "email": send_email_alert(timestamp, litter_label, snapshot_path),
        "sms":   send_sms_alert(timestamp, litter_label),
    }
