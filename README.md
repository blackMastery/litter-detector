# Litter & Dumping Detector — Phase 0

AI-powered litter and illegal dumping detection using YOLOv8 + Streamlit.
Connects to a live RTSP camera (Hikvision S04) or local webcam.

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure your camera

Edit `config.py`:

```python
# Option A — Hikvision S04 over 4G
RTSP_URL  = "rtsp://admin:yourpassword@192.168.1.100:554/Streaming/Channels/101"
USE_WEBCAM = False

# Option B — Local webcam (for testing)
USE_WEBCAM   = True
WEBCAM_INDEX = 0

# Option C — Test with a video file
TEST_VIDEO_PATH = "test_footage.mp4"
```

### 3. Run

```bash
streamlit run app.py
```

Open your browser at **http://localhost:8501**

---

## Project Structure

```
litter-detector/
├── app.py          # Streamlit dashboard (main entry point)
├── detector.py     # YOLO inference + dumping event logic
├── camera.py       # Thread-safe RTSP/webcam stream handler
├── alerts.py       # Email + SMS notifications
├── logger.py       # CSV logging for detections + incidents
├── config.py       # All settings in one place
├── requirements.txt
├── snapshots/      # Auto-created — dumping event images saved here
└── logs/           # Auto-created — CSV logs saved here
```

---

## Hikvision S04 — Getting the RTSP URL

1. Connect the camera to your 4G SIM and power it on
2. Find the camera IP from your router's DHCP table
   (or use the Hik-Connect app to get the IP)
3. RTSP URL format:
   ```
   rtsp://admin:PASSWORD@CAMERA_IP:554/Streaming/Channels/101
   ```
   - Channel 101 = main stream (highest quality)
   - Channel 102 = sub stream (lower res, faster)
4. Test the URL in VLC before starting the app

---

## How Detection Works

```
Live RTSP frame
      │
      ▼
  YOLOv8 inference
      │
      ├─── Litter detected?  ──►  Amber box drawn on frame
      │                           Logged to detections.csv
      │
      ├─── Person detected?  ──►  Blue box drawn on frame
      │
      └─── Person + Litter within 150px?
                │
                ▼
          DUMPING EVENT
           ├─ Red box + alert label on frame
           ├─ Snapshot saved to /snapshots/
           ├─ Logged to incidents.csv
           ├─ Email alert sent (if enabled)
           └─ SMS alert sent (if enabled)
```

---

## Enabling Alerts

### Email (Resend)
1. Create a Resend account and generate an API key
2. Verify your sending domain/address in Resend
3. Set environment variables (for example in `.env`):
   ```bash
   RESEND_API_KEY=re_xxxxxxxxxxxxxxxxx
   RESEND_FROM_EMAIL=alerts@yourdomain.com
   ```
4. In `config.py`, enable email alerts:
   ```python
   ENABLE_EMAIL_ALERTS = True
   ```
5. In the app sidebar under **Alerts**, enter recipient emails (comma/newline separated)

### SMS (Twilio)
1. Create a free Twilio account at twilio.com
2. Get a phone number and your Account SID + Auth Token
3. Set in `config.py`:
   ```python
   ENABLE_SMS_ALERTS    = True
   TWILIO_ACCOUNT_SID  = "ACxxxxxxxxxx"
   TWILIO_AUTH_TOKEN   = "your_token"
   TWILIO_FROM         = "+1xxxxxxxxxx"
   TWILIO_TO           = "+1xxxxxxxxxx"
   ```

---

## Fine-tuning on TACO Dataset (Phase 1 prep)

For better litter detection accuracy, fine-tune YOLOv8 on the TACO dataset:

```bash
# Download TACO
git clone https://github.com/pedropro/TACO.git
cd TACO && pip install -r requirements.txt
python download.py

# Fine-tune
yolo detect train \
  data=taco.yaml \
  model=yolov8n.pt \
  epochs=50 \
  imgsz=640

# Update config.py
MODEL_PATH = "runs/detect/train/weights/best.pt"
```

---

## Tested on

- Python 3.11
- macOS 14 / Ubuntu 22.04
- NVIDIA GPU (CUDA) for faster inference — CPU also works
- Hikvision DS-2XS6A47G1-LZS/C36S80 (S04) over 4G LTE
