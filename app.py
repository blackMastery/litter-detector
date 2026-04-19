# ─────────────────────────────────────────────
#  app.py  —  Phase 0 Litter & Dumping Detector
#  Run:  streamlit run app.py
# ─────────────────────────────────────────────

import streamlit as st
import cv2
import time
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
import io

from detector import LitterDetector, DumpEvent
from camera import CameraStream
from alerts import dispatch_alerts
import logger as event_logger
from config import (
    CONFIDENCE_THRESHOLD, SNAPSHOT_DIR, LOG_DIR,
    ENABLE_EMAIL_ALERTS, ENABLE_SMS_ALERTS,
    MAX_LOG_ENTRIES,
)

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

/* Scrollable log */
.log-table { max-height: 300px; overflow-y: auto; }
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
        "log_entries":      [],
        "dump_events":      [],
        "stat_litter":      0,
        "stat_persons":     0,
        "stat_incidents":   0,
        "session_start":    None,
        "confidence":       CONFIDENCE_THRESHOLD,
        "video_source":     None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


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
    _test_footage = sorted(Path("test_footage").glob("*.mp4"))
    _source_options = ["🎥 Live Camera (RTSP/Webcam)"] + [f.name for f in _test_footage]
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
    st.caption("Configure in `config.py`")

    st.markdown("---")
    st.markdown("### 📁 Export")

    col_a, col_b = st.columns(2)
    with col_a:
        incidents_csv = Path(LOG_DIR) / "incidents.csv"
        if incidents_csv.exists():
            with open(incidents_csv, "rb") as f:
                st.download_button("📥 Incidents", f, "incidents.csv", "text/csv")
        else:
            st.button("📥 Incidents", disabled=True)

    with col_b:
        detections_csv = Path(LOG_DIR) / "detections.csv"
        if detections_csv.exists():
            with open(detections_csv, "rb") as f:
                st.download_button("📥 Detections", f, "detections.csv", "text/csv")
        else:
            st.button("📥 Detections", disabled=True)

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
st.caption("Phase 0 — Proof of Concept | CAM-01 | Hikvision S04")

# Control buttons
col_btn1, col_btn2, col_btn3, col_space = st.columns([1, 1, 1, 5])

with col_btn1:
    if not st.session_state.running:
        if st.button("▶ Start", type="primary", use_container_width=True):
            # Initialise detector + camera
            if st.session_state.detector is None:
                with st.spinner("Loading model…"):
                    st.session_state.detector = LitterDetector()

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
            st.session_state.running = False
            st.rerun()

with col_btn2:
    if st.button("🗑 Clear logs", use_container_width=True):
        st.session_state.log_entries   = []
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
                ts = time.strftime("%H:%M:%S")
                for l in litter_dets:
                    st.session_state.log_entries.append(
                        [ts, "litter", l.label, f"{l.confidence:.0%}"])
                    if l.is_new:
                        st.session_state.stat_litter += 1
                        event_logger.log_detection(l.label, l.confidence, "litter")
                for p in person_dets:
                    st.session_state.log_entries.append(
                        [ts, "person", p.label, f"{p.confidence:.0%}"])
                    if p.is_new:
                        st.session_state.stat_persons += 1
                        event_logger.log_detection(p.label, p.confidence, "person")
                if len(st.session_state.log_entries) > MAX_LOG_ENTRIES:
                    st.session_state.log_entries = (
                        st.session_state.log_entries[-MAX_LOG_ENTRIES:])
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
                    st.session_state.log_entries.append(
                        [dump_event.timestamp, "⚠ DUMP",
                         dump_event.litter_label, "EVENT"])
                    event_logger.log_incident(
                        dump_event.timestamp, dump_event.litter_label,
                        dump_event.snapshot_path,
                        dump_event.person_conf, dump_event.litter_conf,
                    )
                    threading.Thread(
                        target=dispatch_alerts,
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
            st.image(frame_rgb, use_column_width=True, channels="RGB")
        else:
            st.info("Press **▶ Start** to connect to the camera.")

        st.markdown("### 📊 Session Stats")
        m1, m2, m3, m4 = st.columns(4)
        _uptime = ""
        if st.session_state.session_start:
            _secs = int(time.time() - st.session_state.session_start)
            _uptime = f"{_secs//60:02d}:{_secs%60:02d}"
        m1.metric("Litter items",  st.session_state.stat_litter)
        m2.metric("Persons",       st.session_state.stat_persons)
        m3.metric("🚨 Incidents",  st.session_state.stat_incidents)
        m4.metric("Uptime",        _uptime or "—")

        st.markdown("### 📋 Detection Log")
        _entries = st.session_state.log_entries[-50:][::-1]
        if not _entries:
            st.caption("No detections yet.")
        else:
            _df = pd.DataFrame(_entries, columns=["Time", "Type", "Label", "Confidence"])
            st.dataframe(_df, use_container_width=True, hide_index=True,
                         height=min(300, len(_df) * 35 + 38))

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
                    CAM-01
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
                         use_column_width=True)


_live_view()
