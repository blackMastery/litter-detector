# ─────────────────────────────────────────────
#  alerts.py  —  Email + SMS notifications
# ─────────────────────────────────────────────

import base64
import logging
from pathlib import Path
from typing import Optional

try:
    import resend
except ImportError:  # pragma: no cover - handled at runtime
    resend = None

import supabase_client
from config import (
    ENABLE_EMAIL_ALERTS,
    RESEND_API_KEY,
    RESEND_FROM_EMAIL,
    ENABLE_SMS_ALERTS,
    TWILIO_ACCOUNT_SID,
    TWILIO_AUTH_TOKEN,
    TWILIO_FROM,
    TWILIO_TO,
    CAMERA_NAME,
)

logger = logging.getLogger(__name__)


def send_email_alert(
    timestamp: str,
    litter_label: str,
    recipients: list[str],
    snapshot_path: Optional[str] = None,
    enabled_override: Optional[bool] = None,
) -> bool:
    """
    Send email alert with optional snapshot attachment.
    Returns True on success.
    """
    email_enabled = ENABLE_EMAIL_ALERTS if enabled_override is None else enabled_override
    if not email_enabled:
        supabase_client.log_email_attempt(
            litter_label=litter_label,
            recipients=recipients,
            status="skipped",
            error_message="email alerts disabled",
        )
        return False
    if resend is None:
        logger.warning("Email alert skipped: resend package is not installed")
        supabase_client.log_email_attempt(
            litter_label=litter_label,
            recipients=recipients,
            status="skipped",
            error_message="resend package is not installed",
        )
        return False
    if not RESEND_API_KEY or not RESEND_FROM_EMAIL:
        logger.warning("Email alert skipped: RESEND_API_KEY / RESEND_FROM_EMAIL not configured")
        supabase_client.log_email_attempt(
            litter_label=litter_label,
            recipients=recipients,
            status="skipped",
            error_message="RESEND_API_KEY / RESEND_FROM_EMAIL not configured",
        )
        return False
    if not recipients:
        logger.warning("Email alert skipped: no recipient emails configured")
        supabase_client.log_email_attempt(
            litter_label=litter_label,
            recipients=recipients,
            status="skipped",
            error_message="no recipient emails configured",
        )
        return False

    try:
        resend.api_key = RESEND_API_KEY

        body = f"""
        <html><body>
        <h2 style="color:#c0392b">⚠ Illegal Dumping Detected</h2>
        <table style="font-family:sans-serif;font-size:14px">
          <tr><td><b>Time</b></td><td>{timestamp}</td></tr>
          <tr><td><b>Camera</b></td><td>{CAMERA_NAME}</td></tr>
          <tr><td><b>Item detected</b></td><td>{litter_label}</td></tr>
        </table>
        <p>A snapshot has been saved and attached to this email.</p>
        </body></html>
        """
        params: resend.Emails.SendParams = {
            "from": RESEND_FROM_EMAIL,
            "to": recipients,
            "subject": f"[ALERT] Dumping detected - {timestamp}",
            "html": body,
        }

        if snapshot_path and Path(snapshot_path).exists():
            encoded = base64.b64encode(Path(snapshot_path).read_bytes()).decode("utf-8")
            params["attachments"] = [
                {
                    "filename": Path(snapshot_path).name,
                    "content": encoded,
                }
            ]

        response = resend.Emails.send(params)
        provider_message_id = ""
        if isinstance(response, dict):
            provider_message_id = str(response.get("id") or "")
        supabase_client.log_email_attempt(
            litter_label=litter_label,
            recipients=recipients,
            status="sent",
            provider_message_id=provider_message_id,
        )
        logger.info("Email alert sent to %s", ", ".join(recipients))
        return True

    except Exception as e:
        logger.error("Email alert failed: %s", e)
        supabase_client.log_email_attempt(
            litter_label=litter_label,
            recipients=recipients,
            status="failed",
            error_message=str(e),
        )
        return False


def send_sms_alert(
    timestamp: str,
    litter_label: str,
    enabled_override: Optional[bool] = None,
) -> bool:
    """
    Send SMS alert via Twilio.
    Returns True on success.
    """
    sms_enabled = ENABLE_SMS_ALERTS if enabled_override is None else enabled_override
    if not sms_enabled:
        return False

    try:
        from twilio.rest import Client
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        message = client.messages.create(
            body=(
                f"LITTER ALERT | {CAMERA_NAME}\n"
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


def dispatch_alerts(
    timestamp: str,
    litter_label: str,
    recipients: list[str],
    snapshot_path: Optional[str] = None,
    enable_email: Optional[bool] = None,
    enable_sms: Optional[bool] = None,
) -> dict:
    """Fire all enabled alert channels. Returns status dict."""
    return {
        "email": send_email_alert(
            timestamp,
            litter_label,
            recipients,
            snapshot_path,
            enabled_override=enable_email,
        ),
        "sms": send_sms_alert(
            timestamp,
            litter_label,
            enabled_override=enable_sms,
        ),
    }
