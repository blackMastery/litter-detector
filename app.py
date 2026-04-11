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
        format="%.0%%",
        help="Lower = more detections, higher = fewer false positives"
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

            cam = CameraStream()
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
            st.session_state.running = False
            st.rerun()

with col_btn2:
    if st.button("🗑 Clear logs", use_container_width=True):
        st.session_state.log_entries   = []
        st.session_state.dump_events   = []
        st.session_state.stat_litter   = 0
        st.session_state.stat_persons  = 0
        st.session_state.stat_incidents = 0
        st.rerun()

with col_btn3:
    st.link_button("📂 Snapshots", f"file://{Path(SNAPSHOT_DIR).resolve()}",
                   use_container_width=True)


# ─────────────────────────────────────────────
#  Status bar
# ─────────────────────────────────────────────

status_placeholder = st.empty()

def render_status(litter_count=0, person_count=0, dump=False):
    if not st.session_state.running:
        status_placeholder.markdown(
            '<span class="badge badge-clear">⏸ Detection stopped</span>',
            unsafe_allow_html=True)
    elif dump:
        status_placeholder.markdown(
            '<span class="badge badge-dump">🚨 DUMPING IN PROGRESS — Snapshot saved</span>',
            unsafe_allow_html=True)
    elif litter_count > 0:
        status_placeholder.markdown(
            f'<span class="badge badge-litter">⚠️ {litter_count} litter item(s) detected in frame</span>',
            unsafe_allow_html=True)
    elif person_count > 0:
        status_placeholder.markdown(
            f'<span class="badge badge-person">👤 {person_count} person(s) in view — monitoring</span>',
            unsafe_allow_html=True)
    else:
        status_placeholder.markdown(
            '<span class="badge badge-clear">✅ Area clear</span>',
            unsafe_allow_html=True)

render_status()

st.markdown("---")


# ─────────────────────────────────────────────
#  Main layout
# ─────────────────────────────────────────────

left_col, right_col = st.columns([3, 2], gap="medium")

# ── Left: live feed + stats ──

with left_col:
    st.markdown("### 📹 Live Feed")
    feed_placeholder = st.empty()
    feed_placeholder.info("Press **▶ Start** to connect to the camera.")

    st.markdown("### 📊 Session Stats")
    m1, m2, m3, m4 = st.columns(4)
    stat_litter_ph   = m1.empty()
    stat_persons_ph  = m2.empty()
    stat_incidents_ph = m3.empty()
    stat_uptime_ph   = m4.empty()

    def render_stats():
        uptime = ""
        if st.session_state.session_start:
            secs = int(time.time() - st.session_state.session_start)
            uptime = f"{secs//60:02d}:{secs%60:02d}"
        stat_litter_ph.metric("Litter items",  st.session_state.stat_litter)
        stat_persons_ph.metric("Persons",       st.session_state.stat_persons)
        stat_incidents_ph.metric("🚨 Incidents", st.session_state.stat_incidents)
        stat_uptime_ph.metric("Uptime",         uptime or "—")

    render_stats()

    st.markdown("### 📋 Detection Log")
    log_placeholder = st.empty()

    def render_log():
        entries = st.session_state.log_entries[-50:][::-1]
        if not entries:
            log_placeholder.caption("No detections yet.")
            return
        df = pd.DataFrame(entries, columns=["Time", "Type", "Label", "Confidence"])
        log_placeholder.dataframe(
            df, use_container_width=True, hide_index=True,
            height=min(300, len(df) * 35 + 38),
        )

    render_log()


# ── Right: incidents ──

with right_col:
    st.markdown("### 🚨 Dumping Incidents")
    incidents_placeholder = st.empty()

    def render_incidents():
        events = st.session_state.dump_events[::-1]
        if not events:
            incidents_placeholder.info("No incidents recorded this session.")
            return

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
        incidents_placeholder.markdown("".join(html_parts), unsafe_allow_html=True)

    render_incidents()

    # Latest snapshot
    st.markdown("### 📸 Latest Snapshot")
    snap_placeholder = st.empty()

    def render_snapshot():
        events = st.session_state.dump_events
        if not events:
            snap_placeholder.caption("Snapshots appear here when a dumping event is detected.")
            return
        latest = events[-1]
        snap_path = latest.get("snapshot_path")
        if snap_path and Path(snap_path).exists():
            img = Image.open(snap_path)
            snap_placeholder.image(
                img, caption=f"Event at {latest['timestamp']} — {latest['litter_label']}",
                use_column_width=True,
            )

    render_snapshot()


# ─────────────────────────────────────────────
#  Detection loop
# ─────────────────────────────────────────────

if st.session_state.running:
    cam    = st.session_state.camera
    det    = st.session_state.detector
    conf   = st.session_state.confidence

    if cam is None or not cam.is_connected:
        st.error("Camera disconnected.")
        st.session_state.running = False
        st.rerun()

    frame = cam.read()
    if frame is None:
        st.warning("Waiting for first frame…")
        time.sleep(0.2)
        st.rerun()

    # ── Run inference
    annotated, litter_dets, person_dets, dump_event = det.process_frame(
        frame, confidence_override=conf
    )

    # ── Display frame
    frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    feed_placeholder.image(frame_rgb, use_column_width=True, channels="RGB")

    # ── Update status
    dump_occurred = dump_event is not None
    render_status(len(litter_dets), len(person_dets), dump_occurred)

    # ── Log detections
    ts = time.strftime("%H:%M:%S")
    for l in litter_dets:
        st.session_state.log_entries.append([ts, "litter", l.label, f"{l.confidence:.0%}"])
        st.session_state.stat_litter += 1
        event_logger.log_detection(l.label, l.confidence, "litter")

    for p in person_dets:
        st.session_state.log_entries.append([ts, "person", p.label, f"{p.confidence:.0%}"])
        st.session_state.stat_persons += 1
        event_logger.log_detection(p.label, p.confidence, "person")

    # Trim log
    if len(st.session_state.log_entries) > MAX_LOG_ENTRIES:
        st.session_state.log_entries = st.session_state.log_entries[-MAX_LOG_ENTRIES:]

    # ── Handle dump event
    if dump_occurred:
        ev_dict = {
            "timestamp":    dump_event.timestamp,
            "litter_label": dump_event.litter_label,
            "person_conf":  dump_event.person_conf,
            "litter_conf":  dump_event.litter_conf,
            "snapshot_path": dump_event.snapshot_path,
        }
        st.session_state.dump_events.append(ev_dict)
        st.session_state.stat_incidents += 1
        st.session_state.log_entries.append(
            [dump_event.timestamp, "⚠ DUMP", dump_event.litter_label, "EVENT"])

        event_logger.log_incident(
            dump_event.timestamp, dump_event.litter_label,
            dump_event.snapshot_path, dump_event.person_conf, dump_event.litter_conf,
        )

        # Fire alerts (non-blocking — runs in background thread)
        import threading
        threading.Thread(
            target=dispatch_alerts,
            args=(dump_event.timestamp, dump_event.litter_label, dump_event.snapshot_path),
            daemon=True,
        ).start()

    # ── Refresh stats + panels
    render_stats()
    render_log()
    render_incidents()
    render_snapshot()

    # ── Loop
    time.sleep(0.05)
    st.rerun()
