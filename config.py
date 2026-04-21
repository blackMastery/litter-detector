# ─────────────────────────────────────────────
#  config.py  —  Phase 0 Litter Detector
# ─────────────────────────────────────────────

import os
from dotenv import load_dotenv
load_dotenv()   # reads .env if present; safe to call even if .env doesn't exist

# ── Camera ────────────────────────────────────
# Set RTSP_URL in .env — never hardcode credentials here.
RTSP_URL = os.getenv("RTSP_URL", "rtsp://admin:admin@192.168.1.100:554/Streaming/Channels/101")
CAMERA_NAME = os.getenv("CAMERA_NAME", "CAM-01")

# For local webcam testing (set to 0 for default webcam, or a video file path)
USE_WEBCAM = False
WEBCAM_INDEX = 0
TEST_VIDEO_PATH = ""   # e.g. "test_footage.mp4" — leave empty to use RTSP/webcam

# ── Model ─────────────────────────────────────
GARBAGE_MODEL_PATH = "exp.torchscript"   # custom-trained TorchScript export (single class: garbage)
PERSON_MODEL_PATH  = "yolo11m.pt"        # base YOLO11m, COCO 80 classes — person detection only
CONFIDENCE_THRESHOLD = 0.30     # default — adjustable in dashboard

# ── Preprocessing ──────────────────────────────
PREPROCESS_CLAHE = True    # enhance contrast before inference (helps dark/night scenes)
CLAHE_CLIP_LIMIT  = 2.0    # contrast amplification limit (lower = subtler)
CLAHE_TILE_SIZE   = (8, 8) # grid tile size for localized contrast normalization

# ── Detection targets ─────────────────────────
# Standard COCO classes present in base YOLO11 — covers most urban litter
# LITTER_CLASSES = [
#     "bottle",
#     "cup",
#     "handbag",      # maps to plastic bag in urban context
#     "backpack",     # sometimes maps to trash bags
#     "suitcase",
#     "sports ball",  # can resemble crumpled objects
# ]

# COCO classes present in base yolo11m that approximate urban litter & bulk garbage piles
LITTER_CLASSES = [
    "garbage",      # single class from custom-trained exp.torchscript model
]

PERSON_CLASS = "person"

# ── Ground filtering ───────────────────────────
# Ignore litter detections whose bounding box bottom is above this fraction of
# the frame height — filters out items held by people, on shelves, signs, etc.
# 0.0 = full frame (no filter), 0.35 = only bottom 65% of frame counts as ground
GROUND_FILTER_ENABLED = True
GROUND_ZONE_RATIO     = 0.35    # detections with y2 < (frame_h * ratio) are discarded

# ── Pile detection ─────────────────────────────
# Cluster nearby litter detections and flag as a "garbage pile"
PILE_DETECTION_ENABLED = True
PILE_CLUSTER_DIST      = 200    # pixels — items within this distance belong to same pile
PILE_MIN_ITEMS         = 3      # minimum individual detections to call it a pile

# ── Dumping detection ──────────────────────────
# Distance (pixels) between person centroid and litter centroid to trigger event
PROXIMITY_THRESHOLD = 150

# Seconds before the same zone can trigger another alert
DUMP_COOLDOWN_SECONDS = 10

# ── Alerts ────────────────────────────────────
ENABLE_EMAIL_ALERTS = False
SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 587
SMTP_USER          = os.getenv("SMTP_USER", "")
SMTP_PASS          = os.getenv("SMTP_PASS", "")
ALERT_RECIPIENT    = os.getenv("ALERT_RECIPIENT", "")

ENABLE_SMS_ALERTS = False
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN  = os.getenv("TWILIO_AUTH_TOKEN", "")
TWILIO_FROM        = os.getenv("TWILIO_FROM", "")
TWILIO_TO          = os.getenv("TWILIO_TO", "")

# ── Storage ───────────────────────────────────
SNAPSHOT_DIR  = "snapshots"
LOG_DIR       = "logs"
RECORDING_DIR = "recordings"
MAX_LOG_ENTRIES = 500   # keep last N entries in session

# ── Display ───────────────────────────────────
STREAM_FPS_LIMIT = 15   # target FPS for Streamlit display (lower = less CPU)
JPEG_QUALITY = 80       # snapshot JPEG quality 1-100

# ── Object Tracking ───────────────────────────
TRACKER_MAX_ABSENT    = 15   # frames before a track is pruned (~1.5 s at 10 FPS)
PERSON_HISTORY_FRAMES = 30   # frames of person positions kept for dump detection (~3 s at 10 FPS)
MIN_PROXIMITY_FRAMES  = 8    # person must be within threshold for this many frames to count as dumping (~0.8 s)
