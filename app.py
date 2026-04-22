# ─────────────────────────────────────────────
#  app.py  —  Phase 0 Litter & Dumping Detector
#  Run:  streamlit run app.py
# ─────────────────────────────────────────────

import streamlit as st
import cv2
import time
import numpy as np
from PIL import Image
from pathlib import Path

from detector import LitterDetector, DumpEvent
from camera import CameraStream
from alerts import dispatch_alerts
import logger as event_logger
from config import (
    CONFIDENCE_THRESHOLD, SNAPSHOT_DIR, LOG_DIR, RECORDING_DIR,
    ENABLE_EMAIL_ALERTS, ENABLE_SMS_ALERTS,
    CAMERA_NAME,
)

import os
os.makedirs(RECORDING_DIR, exist_ok=True)

# ─────────────────────────────────────────────
#  Page config
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Litter & Dumping Detector",
    page_icon="🗑️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  Custom CSS
# ─────────────────────────────────────────────

st.markdown("""
<style>
/* Hide default Streamlit header */
#MainMenu, footer, header { visibility: hidden; }

/* Status badges */
.badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 6px;
    font-size: 13px;
    font-weight: 600;
    letter-spacing: .03em;
}
.badge-clear   { background:#E1F5EE; color:#085041; }
.badge-litter  { background:#FAEEDA; color:#633806; }
.badge-dump    { background:#FCEBEB; color:#501313; }
.badge-person  { background:#E6F1FB; color:#042C53; }

/* Incident card */
.incident-card {
    border: 1px solid #F09595;
    border-left: 4px solid #E24B4A;
    border-radius: 8px;
    padding: 12px 16px;
    margin-bottom: 10px;
    background: #FCEBEB;
}
.incident-title { font-weight: 600; color: #A32D2D; font-size: 14px; }
.incident-meta  { color: #791F1F; font-size: 12px; margin-top: 4px; }

/* Metric tweaks */
[data-testid="stMetric"] label { font-size: 12px !important; }
[data-testid="stMetricValue"] { font-size: 26px !important; }

</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  Session state initialisation
# ─────────────────────────────────────────────

def _init_state():
    defaults = {
        "running":          False,
        "detector":         None,
        "camera":           None,
        "dump_events":      [],
        "stat_litter":      0,
        "stat_persons":     0,
        "stat_incidents":   0,
        "session_start":    None,
        "confidence":       CONFIDENCE_THRESHOLD,
        "video_source":     None,
        "alert_status":     None,   # None | {"email": bool, "sms": bool, "time": str}
        "recording":        False,
        "recorder":         None,   # cv2.VideoWriter instance
        "recording_path":   None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ─────────────────────────────────────────────
#  Cached file helpers
# ─────────────────────────────────────────────

@st.cache_data(ttl=60)
def _list_snapshots(snap_dir: str) -> list[tuple[str, str]]:
    files = sorted(
        Path(snap_dir).glob("*.jpg"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return [(str(f), f.stem) for f in files]

@st.cache_data(ttl=30)
def _list_recordings(rec_dir: str) -> list[str]:
    files = sorted(
        Path(rec_dir).glob("*.mp4"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return [f.name for f in files]

@st.cache_data(ttl=30)
def _list_test_footage(footage_dir: str) -> list[str]:
    p = Path(footage_dir)
    if not p.exists():
        return []
    return sorted(f.name for f in p.glob("*.mp4"))

@st.cache_data(ttl=30)
def _read_incidents_csv_bytes(csv_path: str) -> bytes | None:
    p = Path(csv_path)
    if not p.exists():
        return None
    return p.read_bytes()


# ─────────────────────────────────────────────
#  Sidebar
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Settings")

    st.session_state.confidence = st.slider(
        "Confidence threshold",
        min_value=0.20, max_value=0.90,
        value=st.session_state.confidence, step=0.05,
        format="%.0f%%",
        help="Lower = more detections, higher = fewer false positives"
    )

    st.markdown("---")
    st.markdown("### 🎬 Video Source")
    _source_options = ["🎥 Live Camera (RTSP/Webcam)"] + _list_test_footage("test_footage")
    _selected = st.selectbox(
        "Source", _source_options,
        disabled=st.session_state.running,
        help="Pick a test clip or use the live camera feed"
    )
    st.session_state.video_source = (
        None if _selected == _source_options[0]
        else str(Path("test_footage") / _selected)
    )

    st.markdown("---")
    st.markdown("### 🔔 Alerts")
    st.markdown(
        f"Email: {'✅ enabled' if ENABLE_EMAIL_ALERTS else '⬜ disabled'}  \n"
        f"SMS: {'✅ enabled' if ENABLE_SMS_ALERTS else '⬜ disabled'}"
    )
    _alert = st.session_state.alert_status
    if _alert:
        _parts = []
        if ENABLE_EMAIL_ALERTS:
            _parts.append("📧 " + ("✅ sent" if _alert["email"] else "❌ failed"))
        if ENABLE_SMS_ALERTS:
            _parts.append("📱 " + ("✅ sent" if _alert["sms"] else "❌ failed"))
        if _parts:
            st.caption(f"Last alert ({_alert['time']}): " + "  ".join(_parts))
    st.caption("Configure credentials in `.env`")

    st.markdown("---")
    st.markdown("### 🎞 Recording")
    if not st.session_state.recording:
        if st.button("⏺ Start recording", use_container_width=True,
                     disabled=not st.session_state.running):
            st.session_state.recording = True
    else:
        if st.button("⏹ Stop recording", use_container_width=True, type="secondary"):
            if st.session_state.recorder:
                st.session_state.recorder.release()
                st.session_state.recorder = None
            st.session_state.recording = False
        st.caption(f"Saving to `{Path(st.session_state.recording_path).name}`" if st.session_state.recording_path else "Initialising…")

    _rec_names = _list_recordings(RECORDING_DIR)
    if _rec_names:
        _chosen = st.selectbox("Download recording", _rec_names, label_visibility="collapsed")
        _rec_path = Path(RECORDING_DIR) / _chosen
        st.download_button("📥 Download", data=lambda p=_rec_path: p.read_bytes(),
                           file_name=_chosen, mime="video/mp4", use_container_width=True)

    st.markdown("---")
    st.markdown("### 📁 Export")

    _incidents_bytes = _read_incidents_csv_bytes(str(Path(LOG_DIR) / "incidents.csv"))
    if _incidents_bytes is not None:
        st.download_button("📥 Incidents", data=_incidents_bytes,
                           file_name="incidents.csv", mime="text/csv", use_container_width=True)
    else:
        st.button("📥 Incidents", disabled=True, use_container_width=True)

    st.markdown("---")
    st.markdown("### ℹ️ System")
    if st.session_state.camera and st.session_state.camera.is_connected:
        info = st.session_state.camera.get_info()
        st.caption(
            f"Source: `{str(info.get('source',''))[:30]}`  \n"
            f"Resolution: {info.get('width','?')}×{info.get('height','?')}  \n"
            f"Stream FPS: {info.get('fps','?')}"
        )
    else:
        st.caption("Camera not connected")


# ─────────────────────────────────────────────
#  Header
# ─────────────────────────────────────────────

st.markdown("## 🗑️ AI Litter & Dumping Detector")
st.caption(f"Phase 0 — Proof of Concept | {CAMERA_NAME}")

# Control buttons
col_btn1, col_btn2, col_btn3, col_space = st.columns([1, 1, 1, 5])

with col_btn1:
    if not st.session_state.running:
        if st.button("▶ Start", type="primary", use_container_width=True):
            # Initialise detector + camera
            if st.session_state.detector is None:
                with st.spinner("Loading model…"):
                    try:
                        st.session_state.detector = LitterDetector()
                    except Exception as e:
                        st.error(f"Model load failed: {e}\n\nCheck that `{__import__('config').GARBAGE_MODEL_PATH}` and `{__import__('config').PERSON_MODEL_PATH}` exist in the project folder.")
                        st.stop()

            cam = CameraStream(source=st.session_state.video_source)
            if cam.start():
                st.session_state.camera = cam
                st.session_state.running = True
                st.session_state.session_start = time.time()
                st.rerun()
            else:
                st.error(f"Camera error: {cam.error}")
    else:
        if st.button("⏹ Stop", type="secondary", use_container_width=True):
            if st.session_state.camera:
                st.session_state.camera.stop()
            if st.session_state.detector:
                st.session_state.detector.reset_tracker()
            if st.session_state.recorder:
                st.session_state.recorder.release()
                st.session_state.recorder = None
            st.session_state.recording = False
            st.session_state.running = False
            st.rerun()

with col_btn2:
    if st.button("🗑 Reset", use_container_width=True):
        st.session_state.dump_events   = []
        st.session_state.stat_litter   = 0
        st.session_state.stat_persons  = 0
        st.session_state.stat_incidents = 0
        if st.session_state.detector:
            st.session_state.detector.reset_tracker()
        st.rerun()

with col_btn3:
    st.link_button("📂 Snapshots", f"file://{Path(SNAPSHOT_DIR).resolve()}",
                   use_container_width=True)


# ─────────────────────────────────────────────
#  Live view — fragment streams WebSocket deltas
#  to the browser every 100 ms without triggering
#  a full page re-render.
# ─────────────────────────────────────────────

@st.fragment(run_every=0.1)
def _live_view():
    import threading

    cam  = st.session_state.camera
    det  = st.session_state.detector
    conf = st.session_state.confidence

    annotated    = None
    litter_dets  = []
    person_dets  = []
    dump_event   = None

    # ── Run detection ─────────────────────────
    if st.session_state.running:
        if cam is None or not cam.is_connected:
            st.session_state.running = False
        else:
            frame = cam.read()
            if frame is not None:
                annotated, litter_dets, person_dets, dump_event = det.process_frame(
                    frame, confidence_override=conf
                )

                # ── Session recording ─────────────────
                if st.session_state.recording:
                    if st.session_state.recorder is None:
                        h, w = annotated.shape[:2]
                        rec_path = str(Path(RECORDING_DIR) /
                                       f"session_{time.strftime('%Y%m%d_%H%M%S')}.mp4")
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        st.session_state.recorder = cv2.VideoWriter(rec_path, fourcc, 10, (w, h))
                        st.session_state.recording_path = rec_path
                    st.session_state.recorder.write(annotated)

                for l in litter_dets:
                    if l.is_new:
                        st.session_state.stat_litter += 1
                for p in person_dets:
                    if p.is_new:
                        st.session_state.stat_persons += 1
                if dump_event:
                    ev_dict = {
                        "timestamp":     dump_event.timestamp,
                        "litter_label":  dump_event.litter_label,
                        "person_conf":   dump_event.person_conf,
                        "litter_conf":   dump_event.litter_conf,
                        "snapshot_path": dump_event.snapshot_path,
                    }
                    st.session_state.dump_events.append(ev_dict)
                    st.session_state.stat_incidents += 1
                    event_logger.log_incident(
                        dump_event.timestamp, dump_event.litter_label,
                        dump_event.snapshot_path,
                        dump_event.person_conf, dump_event.litter_conf,
                    )
                    def _alert_and_record(ts, label, snap):
                        result = dispatch_alerts(ts, label, snap)
                        st.session_state.alert_status = {
                            "email": result.get("email", False),
                            "sms":   result.get("sms", False),
                            "time":  ts,
                        }
                    threading.Thread(
                        target=_alert_and_record,
                        args=(dump_event.timestamp, dump_event.litter_label,
                              dump_event.snapshot_path),
                        daemon=True,
                    ).start()

    # ── Status badge ──────────────────────────
    dump_occurred = dump_event is not None
    if not st.session_state.running:
        st.markdown(
            '<span class="badge badge-clear">⏸ Detection stopped</span>',
            unsafe_allow_html=True)
    elif dump_occurred:
        st.markdown(
            '<span class="badge badge-dump">🚨 DUMPING IN PROGRESS — Snapshot saved</span>',
            unsafe_allow_html=True)
    elif litter_dets:
        st.markdown(
            f'<span class="badge badge-litter">⚠️ {len(litter_dets)}'
            ' litter item(s) detected in frame</span>',
            unsafe_allow_html=True)
    elif person_dets:
        st.markdown(
            f'<span class="badge badge-person">👤 {len(person_dets)}'
            ' person(s) in view — monitoring</span>',
            unsafe_allow_html=True)
    else:
        st.markdown(
            '<span class="badge badge-clear">✅ Area clear</span>',
            unsafe_allow_html=True)

    st.markdown("---")

    left_col, right_col = st.columns([3, 2], gap="medium")

    # ── Left panel ────────────────────────────
    with left_col:
        st.markdown("### 📹 Live Feed")
        if annotated is not None:
            frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            st.image(frame_rgb, use_container_width=True, channels="RGB")
        else:
            st.info("Press **▶ Start** to connect to the camera.")

        st.markdown("### 📊 Session Stats")
        m1, m2, m3 = st.columns(3)
        _uptime = ""
        if st.session_state.session_start:
            _secs = int(time.time() - st.session_state.session_start)
            _uptime = f"{_secs//60:02d}:{_secs%60:02d}"
        m1.metric("Litter items",  st.session_state.stat_litter)
        m2.metric("🚨 Incidents",  st.session_state.stat_incidents)
        m3.metric("Uptime",        _uptime or "—")

    # ── Right panel ───────────────────────────
    with right_col:
        st.markdown("### 🚨 Dumping Incidents")
        events = st.session_state.dump_events[::-1]
        if not events:
            st.info("No incidents recorded this session.")
        else:
            html_parts = []
            for i, ev in enumerate(events):
                n = len(events) - i
                html_parts.append(f"""
                <div class="incident-card">
                  <div class="incident-title">Dumping Event #{n}</div>
                  <div class="incident-meta">
                    🕐 {ev['timestamp']} &nbsp;|&nbsp;
                    📦 {ev['litter_label']} &nbsp;|&nbsp;
                    {CAMERA_NAME}
                  </div>
                  <div class="incident-meta">
                    Person conf: {ev['person_conf']:.0%} &nbsp;|&nbsp;
                    Litter conf: {ev['litter_conf']:.0%}
                  </div>
                </div>
                """)
            st.markdown("".join(html_parts), unsafe_allow_html=True)

        st.markdown("### 📸 Latest Snapshot")
        snap_events = st.session_state.dump_events
        if not snap_events:
            st.caption("Snapshots appear here when a dumping event is detected.")
        else:
            latest = snap_events[-1]
            snap_path = latest.get("snapshot_path")
            if snap_path and Path(snap_path).exists():
                img = Image.open(snap_path)
                st.image(img,
                         caption=f"Event at {latest['timestamp']} — {latest['litter_label']}",
                         use_container_width=True)


_live_view()


# ─────────────────────────────────────────────
#  Snapshot Gallery
# ─────────────────────────────────────────────

st.markdown("---")
st.markdown("### 📸 Snapshot Gallery")

_snap_items = _list_snapshots(SNAPSHOT_DIR)
if not _snap_items:
    st.caption("No snapshots yet. Snapshots are saved automatically when a dumping event is detected.")
else:
    _cols_per_row = 4
    for i in range(0, len(_snap_items), _cols_per_row):
        _row = _snap_items[i:i + _cols_per_row]
        for col, (snap_path_str, snap_stem) in zip(st.columns(_cols_per_row), _row):
            with col:
                st.image(Path(snap_path_str), use_container_width=True, caption=snap_stem)
