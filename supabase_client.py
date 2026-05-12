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


def is_configured() -> bool:
    return bool(config.SUPABASE_URL and config.SUPABASE_SERVICE_KEY)


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


def snapshot_public_url(storage_path: str) -> str:
    """Public URL for a file in the snapshots bucket (bucket must be public or URL will 403)."""
    client = get_client()
    if client is None:
        return ""
    return client.storage.from_(config.SUPABASE_SNAPSHOT_BUCKET).get_public_url(
        storage_path
    )


def list_cloud_snapshots(limit: int = 48) -> list[dict]:
    """
    Rows from `snapshots` (newest first) with image URL and optional labels from related rows.
    Each dict: id, storage_path, captured_at, file_size_bytes, camera_name, litter_label, public_url.
    """
    client = get_client()
    if client is None:
        return []
    try:
        sel = (
            "id, storage_path, captured_at, file_size_bytes, "
            "cameras(name), incidents(litter_label)"
        )
        res = (
            client.table("snapshots")
            .select(sel)
            .order("captured_at", desc=True)
            .limit(limit)
            .execute()
        )
        rows = res.data or []
    except Exception as e:
        logger.warning(f"List cloud snapshots (with joins) failed: {e}")
        try:
            res = (
                client.table("snapshots")
                .select("id, storage_path, captured_at, file_size_bytes")
                .order("captured_at", desc=True)
                .limit(limit)
                .execute()
            )
            rows = res.data or []
        except Exception as e2:
            logger.warning(f"List cloud snapshots failed: {e2}")
            return []

    out: list[dict] = []
    for row in rows:
        cam = row.get("cameras")
        cam_name = None
        if isinstance(cam, dict):
            cam_name = cam.get("name")
        inc = row.get("incidents")
        litter_label = None
        if isinstance(inc, list) and inc:
            litter_label = inc[0].get("litter_label")
        elif isinstance(inc, dict):
            litter_label = inc.get("litter_label")

        path = row["storage_path"]
        out.append(
            {
                "id": row["id"],
                "storage_path": path,
                "captured_at": row["captured_at"],
                "file_size_bytes": row.get("file_size_bytes"),
                "camera_name": cam_name,
                "litter_label": litter_label,
                "public_url": snapshot_public_url(path),
            }
        )
    return out


def load_email_recipients() -> list[str]:
    """Load persisted email recipients for the current camera."""
    client = get_client()
    if client is None:
        return []
    try:
        cam_id = get_camera_id()
        if not cam_id:
            return []
        res = (
            client.table("notification_settings")
            .select("email_recipients")
            .eq("camera_id", cam_id)
            .limit(1)
            .execute()
        )
        if not res.data:
            return []
        recipients = res.data[0].get("email_recipients") or []
        if not isinstance(recipients, list):
            return []
        return [str(r).strip() for r in recipients if str(r).strip()]
    except Exception as e:
        logger.warning(f"Load recipients failed: {e}")
        return []


def save_email_recipients(recipients: list[str]) -> bool:
    """Persist email recipients for the current camera."""
    client = get_client()
    if client is None:
        return False
    try:
        cam_id = get_camera_id()
        if not cam_id:
            return False
        cleaned = [str(r).strip() for r in recipients if str(r).strip()]
        client.table("notification_settings").upsert(
            {
                "camera_id": cam_id,
                "email_recipients": cleaned,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
            on_conflict="camera_id",
        ).execute()
        return True
    except Exception as e:
        logger.warning(f"Save recipients failed: {e}")
        return False


def load_rtsp_url() -> str:
    """Load the persisted RTSP URL for the current camera."""
    client = get_client()
    if client is None:
        return ""
    try:
        res = (
            client.table("cameras")
            .select("rtsp_url")
            .eq("name", config.CAMERA_NAME)
            .limit(1)
            .execute()
        )
        if not res.data:
            return ""
        return str(res.data[0].get("rtsp_url") or "").strip()
    except Exception as e:
        logger.warning(f"Load RTSP URL failed: {e}")
        return ""


def save_rtsp_url(url: str) -> bool:
    """Persist the RTSP URL for the current camera."""
    global _camera_id
    client = get_client()
    if client is None:
        return False
    try:
        client.table("cameras").upsert(
            {"name": config.CAMERA_NAME, "rtsp_url": url.strip()},
            on_conflict="name",
        ).execute()
        _camera_id = None
        return True
    except Exception as e:
        logger.warning(f"Save RTSP URL failed: {e}")
        return False


def log_email_attempt(
    *,
    litter_label: str,
    recipients: list[str],
    status: str,
    provider_message_id: str = "",
    error_message: str = "",
) -> bool:
    """Persist an outbound email alert attempt."""
    client = get_client()
    if client is None:
        return False
    try:
        cam_id = get_camera_id()
        cleaned = [str(r).strip() for r in recipients if str(r).strip()]
        client.table("email_logs").insert(
            {
                "camera_id": cam_id,
                "occurred_at": datetime.now(timezone.utc).isoformat(),
                "litter_label": litter_label,
                "recipients": cleaned,
                "status": status,
                "provider": "resend",
                "provider_message_id": provider_message_id or None,
                "error_message": error_message or None,
            }
        ).execute()
        return True
    except Exception as e:
        logger.warning(f"Log email attempt failed: {e}")
        return False


def list_email_logs(limit: int = 50) -> list[dict]:
    """Fetch recent email alert logs (newest first)."""
    client = get_client()
    if client is None:
        return []
    try:
        res = (
            client.table("email_logs")
            .select(
                "id, occurred_at, litter_label, recipients, status, "
                "provider, provider_message_id, error_message"
            )
            .order("occurred_at", desc=True)
            .limit(limit)
            .execute()
        )
        return res.data or []
    except Exception as e:
        logger.warning(f"List email logs failed: {e}")
        return []
