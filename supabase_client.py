# ─────────────────────────────────────────────
#  supabase_client.py  —  Supabase backend integration
# ─────────────────────────────────────────────

import logging
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import config

logger = logging.getLogger(__name__)

_client = None
_camera_id: Optional[str] = None


def get_client():
    global _client
    if _client is not None:
        return _client
    if not (config.SUPABASE_URL and config.SUPABASE_SERVICE_KEY):
        return None
    try:
        from supabase import create_client
        _client = create_client(config.SUPABASE_URL, config.SUPABASE_SERVICE_KEY)
        return _client
    except Exception as e:
        logger.warning(f"Supabase client init failed: {e}")
        return None


def get_camera_id() -> Optional[str]:
    global _camera_id
    if _camera_id is not None:
        return _camera_id
    client = get_client()
    if client is None:
        return None
    try:
        result = client.table("cameras").upsert(
            {"name": config.CAMERA_NAME, "rtsp_url": config.RTSP_URL},
            on_conflict="name",
        ).execute()
        _camera_id = result.data[0]["id"]
        return _camera_id
    except Exception as e:
        logger.warning(f"Camera upsert failed: {e}")
        return None


def upload_incident(timestamp: str, litter_label: str, snapshot_path: str,
                    person_conf: float, litter_conf: float) -> bool:
    client = get_client()
    if client is None:
        return False
    try:
        cam_id = get_camera_id()
        occurred_at = datetime.now(timezone.utc).isoformat()

        snapshot_id = None
        snap_p = Path(snapshot_path)
        if snap_p.exists():
            storage_key = f"{config.CAMERA_NAME}/{snap_p.name}"
            client.storage.from_(config.SUPABASE_SNAPSHOT_BUCKET).upload(
                storage_key,
                snap_p.read_bytes(),
                {"content-type": "image/jpeg", "upsert": "true"},
            )
            snap_row = client.table("snapshots").insert({
                "camera_id": cam_id,
                "storage_path": storage_key,
                "captured_at": occurred_at,
                "file_size_bytes": snap_p.stat().st_size,
            }).execute()
            snapshot_id = snap_row.data[0]["id"]

        client.table("incidents").insert({
            "camera_id": cam_id,
            "snapshot_id": snapshot_id,
            "occurred_at": occurred_at,
            "litter_label": litter_label,
            "litter_confidence": round(litter_conf, 3),
            "person_confidence": round(person_conf, 3),
        }).execute()
        return True
    except Exception as e:
        logger.warning(f"Incident sync failed: {e}")
        return False


_VIDEO_CONTENT_TYPES = {
    ".mp4":  "video/mp4",
    ".mov":  "video/quicktime",
    ".avi":  "video/x-msvideo",
    ".mkv":  "video/x-matroska",
}


def upload_video(filename: str, data: bytes) -> Optional[str]:
    client = get_client()
    if client is None:
        return None
    try:
        ctype = _VIDEO_CONTENT_TYPES.get(Path(filename).suffix.lower(),
                                         "application/octet-stream")
        client.storage.from_(config.SUPABASE_UPLOADS_BUCKET).upload(
            filename, data, {"content-type": ctype, "upsert": "true"},
        )
        return filename
    except Exception as e:
        logger.warning(f"Video upload failed: {e}")
        return None


def list_uploaded_videos() -> list[dict]:
    client = get_client()
    if client is None:
        return []
    try:
        objs = client.storage.from_(config.SUPABASE_UPLOADS_BUCKET).list()
        return [
            {
                "name": o["name"],
                "size": (o.get("metadata") or {}).get("size", 0),
                "created_at": o.get("created_at"),
            }
            for o in objs if not o["name"].startswith(".")
        ]
    except Exception as e:
        logger.warning(f"List uploads failed: {e}")
        return []


def download_video_to_temp(name: str) -> Optional[Path]:
    client = get_client()
    if client is None:
        return None
    try:
        data = client.storage.from_(config.SUPABASE_UPLOADS_BUCKET).download(name)
        suffix = Path(name).suffix or ".mp4"
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.write(data)
        tmp.close()
        return Path(tmp.name)
    except Exception as e:
        logger.warning(f"Video download failed: {e}")
        return None
