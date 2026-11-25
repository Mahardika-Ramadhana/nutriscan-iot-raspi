"""Offline-first IoT transports (Supabase, HTTP REST, MQTT) with exponential backoff."""

from __future__ import annotations

import base64
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any, Protocol
import uuid

import numpy as np
import requests

from nutriscan.ai.detector import ScanAnalysis
from nutriscan.core.config import AppConfig, HttpSyncConfig, MqttSyncConfig, SupabaseSyncConfig, get_secret
from nutriscan.core.logger import get_logger
from nutriscan.db.database import Database

logger = get_logger("CloudSync")


class TransportError(RuntimeError):
    """Raised when an external IoT transport fails to deliver a payload."""


class RemoteTransport(Protocol):
    """Protocol for remote IoT payload dispatchers."""

    name: str

    def send(self, payload: dict[str, Any], image_path: Path | None) -> None:
        """Transmit payload (and optional image) or raise TransportError."""


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def save_frame_image(frame: np.ndarray, output_path: Path) -> Path:
    """Persist a BGR numpy image frame to JPEG using OpenCV or Pillow fallback."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import cv2

        ok = cv2.imwrite(str(output_path), frame)
        if ok:
            return output_path
    except ImportError:
        pass

    from PIL import Image

    rgb = frame[:, :, ::-1] if frame.ndim == 3 and frame.shape[2] >= 3 else frame
    Image.fromarray(rgb.astype(np.uint8)).save(output_path, format="JPEG", quality=90)
    return output_path


def _encode_image_base64(image_path: Path | None) -> str | None:
    if image_path is None or not image_path.is_file():
        return None
    return base64.b64encode(image_path.read_bytes()).decode("ascii")


def build_scan_payload(
    device_id: str,
    location: str,
    analysis: ScanAnalysis,
    scan_uuid: str | None = None,
    timestamp_utc: str | None = None,
    image_filename: str | None = None,
) -> dict[str, Any]:
    """Create an immutable versioned telemetry JSON payload for NutriScan MBG."""
    data = analysis.to_dict()
    return {
        "schema_version": "1.0",
        "scan_id": scan_uuid or str(uuid.uuid4()),
        "device_id": device_id,
        "location": location,
        "timestamp_utc": timestamp_utc or utc_now_iso(),
        "image_filename": image_filename,
        **data,
    }


class SupabaseTransport:
    """Supabase upsert transport adapted from DICA CloudSync."""

    name = "supabase"

    def __init__(self, config: SupabaseSyncConfig, client: Any = None) -> None:
        self._config = config
        self._client = client

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        url = get_secret(self._config.url_env, self._config.url)
        key = get_secret(self._config.key_env)
        if not url or not key:
            raise TransportError("Kredensial Supabase belum diset (env URL/KEY kosong)")
        try:
            from supabase import create_client
        except ImportError as exc:
            raise TransportError("Paket 'supabase' tidak terpasang") from exc
        self._client = create_client(url, key)
        return self._client

    def send(self, payload: dict[str, Any], image_path: Path | None) -> None:
        client = self._get_client()
        nutrition = payload.get("total_gizi") or {}
        row = {
            "scan_uuid": payload["scan_id"],
            "device_id": payload["device_id"],
            "location": payload["location"],
            "timestamp_utc": payload["timestamp_utc"],
            "berat_total_g": payload.get("berat_total_g"),
            "jumlah_deteksi": payload.get("jumlah_deteksi", 0),
            "komponen": payload.get("komponen", {}),
            "total_gizi": nutrition,
            "energi_kcal": nutrition.get("energi_kcal"),
            "protein_g": nutrition.get("protein_g"),
            "lemak_g": nutrition.get("lemak_g"),
            "karbohidrat_g": nutrition.get("karbohidrat_g"),
            "has_occlusion": payload.get("has_occlusion", False),
            "image_filename": image_path.name if image_path else payload.get("image_filename"),
        }
        try:
            client.table(self._config.table).upsert(row, on_conflict="scan_uuid").execute()
        except Exception as exc:
            raise TransportError(f"Supabase upsert gagal: {exc}") from exc


class HttpRestTransport:
    """HTTP REST POST transport supporting multipart/form-data and JSON+base64."""

    name = "http"

    def __init__(self, config: HttpSyncConfig) -> None:
        self._config = config

    def send(self, payload: dict[str, Any], image_path: Path | None) -> None:
        headers: dict[str, str] = {}
        token = get_secret(self._config.token_env)
        if token:
            headers["Authorization"] = f"Bearer {token}"

        try:
            if self._config.mode == "multipart":
                form_data = {"payload": json.dumps(payload, ensure_ascii=False)}
                if self._config.include_image and image_path is not None and image_path.is_file():
                    with image_path.open("rb") as img_f:
                        response = requests.post(
                            self._config.endpoint,
                            headers=headers,
                            data=form_data,
                            files={"image": (image_path.name, img_f, "image/jpeg")},
                            timeout=self._config.timeout_s,
                            verify=self._config.verify_tls,
                        )
                else:
                    response = requests.post(
                        self._config.endpoint,
                        headers=headers,
                        data=form_data,
                        timeout=self._config.timeout_s,
                        verify=self._config.verify_tls,
                    )
            else:
                body = dict(payload)
                if self._config.include_image:
                    b64 = _encode_image_base64(image_path)
                    if b64 is not None:
                        body["image_base64"] = b64
                response = requests.post(
                    self._config.endpoint,
                    headers=headers,
                    json=body,
                    timeout=self._config.timeout_s,
                    verify=self._config.verify_tls,
                )
        except requests.RequestException as exc:
            raise TransportError(f"HTTP POST gagal: {exc}") from exc

        if response.status_code < 200 or response.status_code >= 300:
            raise TransportError(
                f"HTTP endpoint membalas status {response.status_code}: {response.text[:200]}"
            )


class MqttTransport:
    """MQTT telemetry publisher using paho-mqtt (v1 & v2 compatible)."""

    name = "mqtt"

    def __init__(self, config: MqttSyncConfig, device_id: str) -> None:
        self._config = config
        self._topic = config.topic.format(device_id=device_id)

    def send(self, payload: dict[str, Any], image_path: Path | None) -> None:
        try:
            import paho.mqtt.client as mqtt
        except ImportError as exc:
            raise TransportError("Paket 'paho-mqtt' tidak terpasang") from exc

        body = dict(payload)
        if self._config.include_image_base64:
            b64 = _encode_image_base64(image_path)
            if b64 is not None:
                body["image_base64"] = b64
        message_str = json.dumps(body, ensure_ascii=False)

        callback_ver = getattr(mqtt, "CallbackAPIVersion", None)
        client = (
            mqtt.Client(callback_api_version=callback_ver.VERSION2)
            if callback_ver is not None
            else mqtt.Client()
        )
        username = get_secret(self._config.username_env)
        password = get_secret(self._config.password_env)
        if username:
            client.username_pw_set(username, password or None)
        if self._config.use_tls:
            client.tls_set()

        try:
            client.connect(
                self._config.broker,
                self._config.port,
                keepalive=max(10, int(self._config.connect_timeout_s)),
            )
            client.loop_start()
            info = client.publish(self._topic, payload=message_str, qos=self._config.qos)
            info.wait_for_publish(timeout=self._config.connect_timeout_s)
        except Exception as exc:
            raise TransportError(f"Publish MQTT ke {self._topic} gagal: {exc}") from exc
        finally:
            try:
                client.loop_stop()
                client.disconnect()
            except Exception:
                pass


class CloudSync:
    """Offline-first scan recorder & multi-transport synchronization engine."""

    def __init__(
        self,
        config: AppConfig,
        db: Database,
        transports: list[RemoteTransport] | None = None,
    ) -> None:
        self._config = config
        self._db = db
        self._transports = transports if transports is not None else self._build_transports()

    def _build_transports(self) -> list[RemoteTransport]:
        active: list[RemoteTransport] = []
        sync_cfg = self._config.sync
        if sync_cfg.supabase.enabled:
            active.append(SupabaseTransport(sync_cfg.supabase))
        if sync_cfg.http.enabled:
            active.append(HttpRestTransport(sync_cfg.http))
        if sync_cfg.mqtt.enabled:
            active.append(MqttTransport(sync_cfg.mqtt, self._config.device.id))
        return active

    def has_remote_transports(self) -> bool:
        return len(self._transports) > 0

    def _persist_images(
        self,
        scan_uuid: str,
        raw_frame: np.ndarray | None,
        annotated_frame: np.ndarray | None,
    ) -> Path | None:
        storage_cfg = self._config.storage
        saved_primary: Path | None = None
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        short_id = scan_uuid[:8]

        if storage_cfg.save_raw_image and raw_frame is not None:
            raw_path = storage_cfg.capture_dir / f"scan_{stamp}_{short_id}_raw.jpg"
            save_frame_image(raw_frame, raw_path)
            saved_primary = raw_path

        if storage_cfg.save_annotated_image and annotated_frame is not None:
            ann_path = storage_cfg.capture_dir / f"scan_{stamp}_{short_id}_hasil.jpg"
            save_frame_image(annotated_frame, ann_path)
            saved_primary = ann_path

        return saved_primary

    def record_and_sync(
        self,
        analysis: ScanAnalysis,
        raw_frame: np.ndarray | None = None,
        annotated_frame: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Store scan to SQLite first, save optional captures, and dispatch online if configured."""
        scan_uuid = str(uuid.uuid4())
        image_path = self._persist_images(scan_uuid, raw_frame, annotated_frame)
        payload = build_scan_payload(
            device_id=self._config.device.id,
            location=self._config.device.location,
            analysis=analysis,
            scan_uuid=scan_uuid,
            image_filename=image_path.name if image_path else None,
        )

        initial_status = "pending" if self.has_remote_transports() else "local_only"
        self._db.save_scan(payload, image_path=image_path, sync_status=initial_status)
        logger.info(
            "Scan %s disimpan ke SQLite (status=%s, berat=%.1fg, items=%d)",
            scan_uuid[:8],
            initial_status,
            payload.get("berat_total_g") or 0.0,
            payload["jumlah_deteksi"],
        )

        if self.has_remote_transports():
            self.sync_pending(limit=10)

        return payload

    def sync_pending(
        self,
        limit: int = 20,
        now: float | None = None,
    ) -> dict[str, Any]:
        """Flush due offline records through all enabled remote transports."""
        if not self.has_remote_transports():
            return {
                "status": "local_only",
                "message": "Tidak ada transport cloud yang diaktifkan (Supabase/HTTP/MQTT off)",
                "sent": 0,
                "failed": 0,
            }

        ts = time.time() if now is None else now
        due_rows = self._db.fetch_pending_scans(now=ts, limit=limit)
        sent = 0
        failed = 0

        for item in due_rows:
            scan_uuid = item["scan_uuid"]
            payload = item["payload"]
            img_path = Path(item["image_path"]) if item["image_path"] else None

            try:
                for transport in self._transports:
                    transport.send(payload, img_path)
                self._db.mark_synced(scan_uuid, utc_now_iso())
                sent += 1
            except Exception as exc:
                failed += 1
                retry_count = int(item["retry_count"])
                sync_cfg = self._config.sync
                delay = min(
                    sync_cfg.retry_base_delay_s * (2**retry_count),
                    sync_cfg.retry_max_delay_s,
                )
                dead_letter = sync_cfg.max_retries > 0 and (retry_count + 1) >= sync_cfg.max_retries
                self._db.mark_retry(
                    scan_uuid,
                    error=str(exc),
                    next_retry_at=ts + delay,
                    dead_letter=dead_letter,
                )
                logger.warning(
                    "Sync scan %s gagal (%s). Retry berikutnya dalam %.1f s%s",
                    scan_uuid[:8],
                    exc,
                    delay,
                    " [DEAD_LETTER]" if dead_letter else "",
                )

        return {
            "status": "success" if failed == 0 else "partial",
            "message": f"Terkirim: {sent}, Gagal/antre ulang: {failed}",
            "sent": sent,
            "failed": failed,
        }
