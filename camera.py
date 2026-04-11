# ─────────────────────────────────────────────
#  camera.py  —  RTSP / webcam stream handler
# ─────────────────────────────────────────────

import cv2
import time
import threading
import logging
from typing import Optional
import numpy as np

from config import (
    RTSP_URL, USE_WEBCAM, WEBCAM_INDEX, TEST_VIDEO_PATH,
    STREAM_FPS_LIMIT,
)

logger = logging.getLogger(__name__)


class CameraStream:
    """
    Thread-safe camera wrapper.
    Reads frames in a background thread so the main thread never
    blocks waiting for a frame — important for smooth Streamlit updates.

    Usage:
        cam = CameraStream()
        cam.start()
        frame = cam.read()
        cam.stop()
    """

    def __init__(self):
        self._cap: Optional[cv2.VideoCapture] = None
        self._frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._connected = False
        self._error: Optional[str] = None
        self._frame_count = 0

    # ── Public API ────────────────────────────

    def start(self) -> bool:
        """Open camera and start background read thread. Returns True on success."""
        source = self._get_source()
        logger.info(f"Opening stream: {source}")
        self._cap = cv2.VideoCapture(source)

        if TEST_VIDEO_PATH:
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        else:
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self._cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
            self._cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)

        if not self._cap.isOpened():
            self._error = f"Could not open stream: {source}"
            logger.error(self._error)
            return False

        self._connected = True
        self._running = True
        self._error = None
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()
        logger.info("Camera stream started.")
        return True

    def read(self) -> Optional[np.ndarray]:
        """Return the latest frame (or None if not yet available)."""
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def stop(self):
        """Stop the background thread and release the capture."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        if self._cap:
            self._cap.release()
        self._connected = False
        logger.info("Camera stream stopped.")

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def error(self) -> Optional[str]:
        return self._error

    @property
    def frame_count(self) -> int:
        return self._frame_count

    def get_info(self) -> dict:
        if self._cap and self._cap.isOpened():
            return {
                "width":  int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "fps":    round(self._cap.get(cv2.CAP_PROP_FPS), 1),
                "source": self._get_source(),
            }
        return {}

    # ── Internal ──────────────────────────────

    def _get_source(self):
        if TEST_VIDEO_PATH:
            return TEST_VIDEO_PATH
        if USE_WEBCAM:
            return WEBCAM_INDEX
        return RTSP_URL

    def _read_loop(self):
        min_interval = 1.0 / max(STREAM_FPS_LIMIT, 1)
        while self._running:
            t0 = time.time()
            ret, frame = self._cap.read()
            if not ret:
                logger.warning("Frame read failed — attempting reconnect…")
                self._connected = False
                self._reconnect()
                continue
            self._connected = True
            self._frame_count += 1
            with self._lock:
                self._frame = frame
            elapsed = time.time() - t0
            sleep = min_interval - elapsed
            if sleep > 0:
                time.sleep(sleep)

    def _reconnect(self):
        """Try to reconnect to the stream every 3 seconds."""
        for attempt in range(1, 6):
            logger.info(f"Reconnect attempt {attempt}/5…")
            time.sleep(3)
            if self._cap:
                self._cap.release()
            source = self._get_source()
            self._cap = cv2.VideoCapture(source)
            if self._cap.isOpened():
                self._connected = True
                logger.info("Reconnected.")
                return
        self._error = "Stream unavailable after 5 reconnect attempts."
        logger.error(self._error)
        self._running = False
