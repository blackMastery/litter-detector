# ─────────────────────────────────────────────
#  detector.py  —  YOLO inference + dumping logic
# ─────────────────────────────────────────────

import cv2
import time
import os
import numpy as np
from ultralytics import YOLO
from dataclasses import dataclass, field
from typing import Optional
import logging

from config import (
    MODEL_PATH, CONFIDENCE_THRESHOLD,
    LITTER_CLASSES, PERSON_CLASS,
    PROXIMITY_THRESHOLD, DUMP_COOLDOWN_SECONDS,
    SNAPSHOT_DIR, JPEG_QUALITY,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os.makedirs(SNAPSHOT_DIR, exist_ok=True)


# ── Data classes ──────────────────────────────

@dataclass
class Detection:
    label: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int
    timestamp: str = field(default_factory=lambda: time.strftime("%H:%M:%S"))

    @property
    def center(self):
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)

    @property
    def type(self):
        if self.label == PERSON_CLASS:
            return "person"
        return "litter"


@dataclass
class DumpEvent:
    timestamp: str
    snapshot_path: str
    litter_label: str
    person_conf: float
    litter_conf: float
    date: str = field(default_factory=lambda: time.strftime("%Y-%m-%d"))


# ── Detector class ────────────────────────────

class LitterDetector:
    def __init__(self):
        logger.info(f"Loading model: {MODEL_PATH}")
        self.model = YOLO(MODEL_PATH)
        self._last_dump_time = 0
        self._frame_count = 0
        logger.info("Model loaded.")

    def _euclidean(self, c1, c2) -> float:
        return float(np.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2))

    def _is_dumping(self, person: Detection, litter: Detection) -> bool:
        return self._euclidean(person.center, litter.center) < PROXIMITY_THRESHOLD

    def _save_snapshot(self, frame: np.ndarray) -> str:
        ts = time.strftime("%Y%m%d_%H%M%S")
        path = os.path.join(SNAPSHOT_DIR, f"dump_{ts}_{self._frame_count}.jpg")
        cv2.imwrite(path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        return path

    def _draw_box(self, frame, det: Detection, color, thickness=2):
        x1, y1, x2, y2 = det.x1, det.y1, det.x2, det.y2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        label_text = f"{det.label} {det.confidence:.0%}"
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
        cv2.putText(frame, label_text, (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    def _draw_dump_alert(self, frame, person: Detection):
        x1, y1, x2, y2 = person.x1, person.y1, person.x2, person.y2
        # Thick red box
        cv2.rectangle(frame, (x1 - 4, y1 - 4), (x2 + 4, y2 + 4), (0, 0, 220), 4)
        # Alert label
        alert = "!! DUMPING DETECTED"
        (tw, th), _ = cv2.getTextSize(alert, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        cv2.rectangle(frame, (x1, y2 + 4), (x1 + tw + 8, y2 + th + 16), (0, 0, 220), -1)
        cv2.putText(frame, alert, (x1 + 4, y2 + th + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

    def process_frame(
        self,
        frame: np.ndarray,
        confidence_override: Optional[float] = None,
    ) -> tuple[np.ndarray, list[Detection], list[Detection], Optional[DumpEvent]]:
        """
        Run detection on a single frame.

        Returns:
            annotated_frame  — frame with all bounding boxes drawn
            litter_dets      — list of litter Detection objects
            person_dets      — list of person Detection objects
            dump_event       — DumpEvent if triggered, else None
        """
        self._frame_count += 1
        conf = confidence_override if confidence_override is not None else CONFIDENCE_THRESHOLD

        results = self.model(frame, conf=conf, verbose=False)[0]

        litter_dets: list[Detection] = []
        person_dets: list[Detection] = []

        for box in results.boxes:
            label = self.model.names[int(box.cls)]
            confidence = float(box.conf)
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            det = Detection(label, confidence, x1, y1, x2, y2)

            if label in LITTER_CLASSES:
                litter_dets.append(det)
            elif label == PERSON_CLASS:
                person_dets.append(det)

        # ── Draw litter boxes (amber)
        for det in litter_dets:
            self._draw_box(frame, det, (30, 165, 240))   # BGR amber

        # ── Check dumping + draw person boxes
        dump_event: Optional[DumpEvent] = None
        now = time.time()
        cooldown_ok = (now - self._last_dump_time) > DUMP_COOLDOWN_SECONDS

        for person in person_dets:
            dumping = any(self._is_dumping(person, l) for l in litter_dets)

            if dumping:
                self._draw_box(frame, person, (0, 80, 220), thickness=2)  # red-tinted
                self._draw_dump_alert(frame, person)

                if cooldown_ok and dump_event is None:
                    self._last_dump_time = now
                    snap_path = self._save_snapshot(frame)
                    closest = min(litter_dets,
                                  key=lambda l: self._euclidean(person.center, l.center))
                    dump_event = DumpEvent(
                        timestamp=time.strftime("%H:%M:%S"),
                        snapshot_path=snap_path,
                        litter_label=closest.label,
                        person_conf=person.confidence,
                        litter_conf=closest.confidence,
                    )
            else:
                self._draw_box(frame, person, (220, 130, 50))  # BGR blue

        # ── Timestamp overlay on frame
        ts = time.strftime("%Y-%m-%d  %H:%M:%S")
        cv2.putText(frame, f"CAM-01  |  {ts}",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (200, 200, 200), 1, cv2.LINE_AA)

        return frame, litter_dets, person_dets, dump_event
