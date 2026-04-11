# ─────────────────────────────────────────────
#  logger.py  —  Detection + incident logging
# ─────────────────────────────────────────────

import csv
import os
import time
import logging
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

from config import LOG_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os.makedirs(LOG_DIR, exist_ok=True)

DETECTION_LOG = os.path.join(LOG_DIR, "detections.csv")
INCIDENT_LOG  = os.path.join(LOG_DIR, "incidents.csv")


def _ensure_csv(path: str, headers: list[str]):
    if not Path(path).exists():
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(headers)


def log_detection(label: str, confidence: float, det_type: str):
    _ensure_csv(DETECTION_LOG, ["date", "time", "type", "label", "confidence"])
    with open(DETECTION_LOG, "a", newline="") as f:
        csv.writer(f).writerow([
            time.strftime("%Y-%m-%d"),
            time.strftime("%H:%M:%S"),
            det_type,
            label,
            f"{confidence:.2f}",
        ])


def log_incident(timestamp: str, litter_label: str,
                 snapshot_path: str, person_conf: float, litter_conf: float):
    _ensure_csv(INCIDENT_LOG, [
        "date", "time", "litter_label",
        "person_conf", "litter_conf", "snapshot_path"
    ])
    with open(INCIDENT_LOG, "a", newline="") as f:
        csv.writer(f).writerow([
            time.strftime("%Y-%m-%d"),
            timestamp,
            litter_label,
            f"{person_conf:.2f}",
            f"{litter_conf:.2f}",
            snapshot_path,
        ])
    logger.info(f"Incident logged: {timestamp} — {litter_label}")


def read_incidents_csv() -> list[dict]:
    if not Path(INCIDENT_LOG).exists():
        return []
    with open(INCIDENT_LOG, newline="") as f:
        return list(csv.DictReader(f))


def read_detections_csv() -> list[dict]:
    if not Path(DETECTION_LOG).exists():
        return []
    with open(DETECTION_LOG, newline="") as f:
        return list(csv.DictReader(f))
