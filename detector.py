# ─────────────────────────────────────────────
#  detector.py  —  YOLO11 inference + dumping logic
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
    GARBAGE_MODEL_PATH, PERSON_MODEL_PATH, CONFIDENCE_THRESHOLD,
    LITTER_CLASSES, PERSON_CLASS,
    PROXIMITY_THRESHOLD, DUMP_COOLDOWN_SECONDS,
    SNAPSHOT_DIR, JPEG_QUALITY,
    PREPROCESS_CLAHE, CLAHE_CLIP_LIMIT, CLAHE_TILE_SIZE,
    GROUND_FILTER_ENABLED, GROUND_ZONE_RATIO,
    PILE_DETECTION_ENABLED, PILE_CLUSTER_DIST, PILE_MIN_ITEMS,
    TRACKER_MAX_ABSENT,
)
from tracker import ObjectTracker

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
    timestamp: str       = field(default_factory=lambda: time.strftime("%H:%M:%S"))
    track_id: Optional[int] = field(default=None)
    is_new: bool            = field(default=True)

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
        logger.info(f"Loading garbage model: {GARBAGE_MODEL_PATH}")
        self.garbage_model = YOLO(GARBAGE_MODEL_PATH)
        logger.info(f"Loading person model: {PERSON_MODEL_PATH}")
        self.person_model = YOLO(PERSON_MODEL_PATH)
        self._last_dump_time = 0
        self._frame_count = 0
        self._clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_SIZE)
        self._tracker = ObjectTracker(max_absent=TRACKER_MAX_ABSENT)
        logger.info("Both models loaded.")

    def _euclidean(self, c1, c2) -> float:
        return float(np.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2))

    def _clahe_enhance(self, frame: np.ndarray) -> np.ndarray:
        """Apply CLAHE on the L channel (LAB) to improve contrast in dark frames."""
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_eq = self._clahe.apply(l)
        return cv2.cvtColor(cv2.merge([l_eq, a, b]), cv2.COLOR_LAB2BGR)

    def _is_on_ground(self, det: Detection, frame_height: int) -> bool:
        """Return True if the detection's bottom edge is in the ground zone."""
        return det.y2 >= frame_height * GROUND_ZONE_RATIO

    def _find_piles(self, dets: list[Detection]) -> list[list[Detection]]:
        """
        Group detections into piles using greedy distance clustering.
        Returns only groups that meet PILE_MIN_ITEMS threshold.
        """
        if not dets:
            return []
        assigned = [False] * len(dets)
        piles = []
        for i, anchor in enumerate(dets):
            if assigned[i]:
                continue
            group = [anchor]
            assigned[i] = True
            for j, candidate in enumerate(dets):
                if assigned[j]:
                    continue
                if self._euclidean(anchor.center, candidate.center) < PILE_CLUSTER_DIST:
                    group.append(candidate)
                    assigned[j] = True
            if len(group) >= PILE_MIN_ITEMS:
                piles.append(group)
        return piles

    def _draw_pile(self, frame: np.ndarray, pile: list[Detection]):
        """Draw a single encompassing box around a garbage pile."""
        x1 = min(d.x1 for d in pile)
        y1 = min(d.y1 for d in pile)
        x2 = max(d.x2 for d in pile)
        y2 = max(d.y2 for d in pile)
        color = (0, 200, 80)   # BGR green
        cv2.rectangle(frame, (x1 - 6, y1 - 6), (x2 + 6, y2 + 6), color, 3)
        label = f"GARBAGE PILE  ({len(pile)} items)"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        cv2.rectangle(frame, (x1 - 6, y1 - th - 18), (x1 + tw + 4, y1 - 6), color, -1)
        cv2.putText(frame, label, (x1 - 2, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

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
        tid_str = f" #{det.track_id}" if det.track_id is not None else ""
        label_text = f"{det.label}{tid_str} {det.confidence:.0%}"
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

        if PREPROCESS_CLAHE:
            frame = self._clahe_enhance(frame)

        frame_h = frame.shape[0]
        litter_dets: list[Detection] = []
        person_dets: list[Detection] = []

        # ── Garbage model: only "garbage" class detections
        garbage_results = self.garbage_model(frame, conf=conf, verbose=False)[0]
        for box in garbage_results.boxes:
            label = self.garbage_model.names[int(box.cls)]
            if label not in LITTER_CLASSES:
                continue
            confidence = float(box.conf)
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            det = Detection(label, confidence, x1, y1, x2, y2)
            if GROUND_FILTER_ENABLED and not self._is_on_ground(det, frame_h):
                continue
            litter_dets.append(det)

        # ── Person model: only "person" class detections
        person_results = self.person_model(frame, conf=conf, verbose=False)[0]
        for box in person_results.boxes:
            label = self.person_model.names[int(box.cls)]
            if label != PERSON_CLASS:
                continue
            confidence = float(box.conf)
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            person_dets.append(Detection(label, confidence, x1, y1, x2, y2))

        # ── Track objects across frames; sets track_id and is_new on each Detection
        self._tracker.update(litter_dets + person_dets, self._frame_count)

        # ── Draw pile boxes (green) before individual boxes so they sit underneath
        if PILE_DETECTION_ENABLED:
            for pile in self._find_piles(litter_dets):
                self._draw_pile(frame, pile)

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

    def reset_tracker(self):
        self._tracker.reset()
