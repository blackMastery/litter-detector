# ─────────────────────────────────────────────
#  config.py  —  Phase 0 Litter Detector
# ─────────────────────────────────────────────

# ── Camera ────────────────────────────────────
# Hikvision S04 — update with your camera's IP/credentials
# Format: rtsp://username:password@camera_ip:554/Streaming/Channels/101
RTSP_URL = "rtsp://admin:admin123@192.168.1.100:554/Streaming/Channels/101"

# For local webcam testing (set to 0 for default webcam, or a video file path)
USE_WEBCAM = False
WEBCAM_INDEX = 0
TEST_VIDEO_PATH = ""   # e.g. "test_footage.mp4" — leave empty to use RTSP/webcam

# ── Model ─────────────────────────────────────
MODEL_PATH = "yolov8m"       # yolov8n (fast) | yolov8s | yolov8m (more accurate)
CONFIDENCE_THRESHOLD = 0.45     # default — adjustable in dashboard

# ── Detection targets ─────────────────────────
# Standard COCO classes present in base YOLOv8 — covers most urban litter
# LITTER_CLASSES = [
#     "bottle",
#     "cup",
#     "handbag",      # maps to plastic bag in urban context
#     "backpack",     # sometimes maps to trash bags
#     "suitcase",
#     "sports ball",  # can resemble crumpled objects
# ]

# When fine-tuned on TACO dataset, replace LITTER_CLASSES with:
LITTER_CLASSES = ["bottle","cigarette","cup","plastic_bag","wrapper",
                  "carton","can","paper","polystyrene","trash_pile"]

PERSON_CLASS = "person"

# ── Dumping detection ──────────────────────────
# Distance (pixels) between person centroid and litter centroid to trigger event
PROXIMITY_THRESHOLD = 150

# Seconds before the same zone can trigger another alert
DUMP_COOLDOWN_SECONDS = 10

# ── Alerts ────────────────────────────────────
ENABLE_EMAIL_ALERTS = False
SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 587
SMTP_USER = "your@gmail.com"
SMTP_PASS = "your_app_password"   # use Gmail App Password, not account password
ALERT_RECIPIENT = "supervisor@yourorg.com"

ENABLE_SMS_ALERTS = False
TWILIO_ACCOUNT_SID = "ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
TWILIO_AUTH_TOKEN  = "your_twilio_auth_token"
TWILIO_FROM        = "+1xxxxxxxxxx"
TWILIO_TO          = "+1xxxxxxxxxx"

# ── Storage ───────────────────────────────────
SNAPSHOT_DIR = "snapshots"
LOG_DIR = "logs"
MAX_LOG_ENTRIES = 500   # keep last N entries in session

# ── Display ───────────────────────────────────
STREAM_FPS_LIMIT = 15   # target FPS for Streamlit display (lower = less CPU)
JPEG_QUALITY = 80       # snapshot JPEG quality 1-100
