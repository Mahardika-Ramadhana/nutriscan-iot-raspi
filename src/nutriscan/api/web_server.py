"""Flask REST API and live nutrition dashboard server (adapted from DICA)."""

from __future__ import annotations

import csv
from functools import wraps
import hmac
import io
import logging
from typing import Any

import numpy as np

from nutriscan.core.config import get_secret
from nutriscan.core.logger import get_logger

logger = get_logger("WebServer")


def validate_bearer_token(auth_header: str | None, expected_key: str) -> bool:
    """Validate Authorization: Bearer <token> using constant-time comparison."""
    if not expected_key or not auth_header:
        return False
    prefix = "Bearer "
    if not auth_header.startswith(prefix):
        return False
    provided = auth_header[len(prefix) :].strip()
    return hmac.compare_digest(provided, expected_key)


def encode_jpeg(frame: np.ndarray | None) -> bytes:
    """Encode a BGR numpy frame into JPEG bytes using OpenCV or Pillow fallback."""
    if frame is None or frame.size == 0:
        return b""
    try:
        import cv2

        ok, buf = cv2.imencode(".jpg", frame)
        if ok:
            return buf.tobytes()
    except ImportError:
        pass

    from PIL import Image

    rgb = frame[:, :, ::-1] if frame.ndim == 3 and frame.shape[2] >= 3 else frame
    out = io.BytesIO()
    Image.fromarray(rgb.astype(np.uint8)).save(out, format="JPEG", quality=85)
    return out.getvalue()


def build_status_payload(main_app: Any) -> dict[str, Any]:
    """Assemble read-only JSON status snapshot for /api/status."""
    snap = main_app.sm.snapshot()
    queue_stats = main_app.db.get_queue_stats()
    return {
        "device_id": main_app.config.device.id,
        "location": main_app.config.device.location,
        "state": snap.state.value,
        "current_weight_g": snap.current_weight_g,
        "locked_weight_g": snap.locked_weight_g,
        "stable_elapsed_s": round(snap.stable_elapsed_s, 2),
        "has_occlusion": snap.has_occlusion,
        "last_error": snap.last_error,
        "last_scan": snap.last_scan_payload,
        "queue_stats": queue_stats,
    }


def format_scans_csv(scans: list[dict[str, Any]]) -> str:
    """Serialize scan history rows into CSV text."""
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(
        [
            "ID",
            "ScanUUID",
            "TimestampUTC",
            "DeviceID",
            "BeratTotal_g",
            "Komponen",
            "Energi_kcal",
            "Protein_g",
            "Lemak_g",
            "Karbohidrat_g",
            "SyncStatus",
        ]
    )
    for row in scans:
        comp_names = ", ".join(
            f"{name} ({info.get('estimasi_gram', 0)}g)"
            for name, info in (row.get("komponen") or {}).items()
        )
        gizi = row.get("total_gizi") or {}
        writer.writerow(
            [
                row.get("id", ""),
                row.get("scan_id", ""),
                row.get("timestamp_utc", ""),
                row.get("device_id", ""),
                row.get("berat_total_g", ""),
                comp_names,
                gizi.get("energi_kcal", ""),
                gizi.get("protein_g", ""),
                gizi.get("lemak_g", ""),
                gizi.get("karbohidrat_g", ""),
                row.get("sync_status", ""),
            ]
        )
    return output.getvalue()


def create_app(main_app: Any) -> Any:
    """Create configured Flask application with lazy Flask import."""
    from flask import Flask, Response, jsonify, render_template, request

    logging.getLogger("werkzeug").setLevel(logging.ERROR)
    app = Flask(__name__, template_folder="templates")

    def require_api_key(fn):
        @wraps(fn)
        def decorated(*args, **kwargs):
            expected_key = get_secret(main_app.config.web.api_key_env)
            if not expected_key:
                return (
                    jsonify(
                        {
                            "status": "error",
                            "message": f"API key belum dikonfigurasi pada env {main_app.config.web.api_key_env}",
                        }
                    ),
                    503,
                )
            if not validate_bearer_token(request.headers.get("Authorization"), expected_key):
                return jsonify({"status": "error", "message": "Akses ditolak: API Key tidak valid"}), 401
            return fn(*args, **kwargs)

        return decorated

    @app.route("/")
    def index():
        return render_template(
            "index.html",
            device_id=main_app.config.device.id,
            location=main_app.config.device.location,
        )

    @app.route("/api/status")
    def status():
        return jsonify(build_status_payload(main_app))

    @app.route("/api/image")
    def image():
        data = encode_jpeg(getattr(main_app, "last_drawn_frame_bgr", None))
        return Response(data, mimetype="image/jpeg")

    @app.route("/api/image_raw")
    def image_raw():
        data = encode_jpeg(getattr(main_app, "last_raw_frame_bgr", None))
        return Response(data, mimetype="image/jpeg")

    @app.route("/api/scans")
    def api_scans():
        limit = request.args.get("limit", default=20, type=int)
        safe_limit = max(1, min(limit, 200))
        return jsonify(main_app.db.get_recent_scans(limit=safe_limit))

    @app.route("/api/export_csv")
    def export_csv():
        csv_text = format_scans_csv(main_app.db.get_all_scans())
        return Response(
            csv_text,
            mimetype="text/csv",
            headers={"Content-Disposition": "attachment; filename=riwayat_nutriscan_mbg.csv"},
        )

    @app.route("/api/trigger", methods=["POST"])
    @require_api_key
    def trigger():
        if main_app.trigger_manual_scan():
            return jsonify({"status": "success", "message": "Scan manual dipicu"})
        return jsonify({"status": "error", "message": "Perangkat sedang sibuk memproses"}), 409

    @app.route("/api/tare", methods=["POST"])
    @require_api_key
    def tare():
        offset = main_app.tare_scale()
        return jsonify({"status": "success", "tare_offset": offset})

    @app.route("/api/sync_cloud", methods=["POST"])
    @require_api_key
    def sync_cloud():
        result = main_app.cloud_sync.sync_pending()
        return jsonify(result)

    @app.route("/api/remote_snapshot", methods=["POST"])
    @require_api_key
    def remote_snapshot():
        if "image" not in request.files:
            return jsonify({"status": "error", "message": "Field 'image' wajib disertakan"}), 400
        raw_bytes = request.files["image"].read()
        try:
            import cv2

            arr = np.frombuffer(raw_bytes, np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        except ImportError:
            from PIL import Image

            with Image.open(io.BytesIO(raw_bytes)) as im:
                rgb = np.asarray(im.convert("RGB"), dtype=np.uint8)
                frame = rgb[:, :, ::-1].copy()

        if frame is None or frame.size == 0:
            return jsonify({"status": "error", "message": "Format gambar tidak valid"}), 400

        weight_g = main_app.sm.current_weight_g
        if "weight" in request.form:
            try:
                weight_g = float(request.form["weight"])
            except ValueError:
                return jsonify({"status": "error", "message": "Nilai 'weight' tidak valid"}), 400

        if not main_app.sm.trigger_processing(weight_g):
            return jsonify({"status": "error", "message": "State machine tidak siap"}), 429

        payload = main_app.process_scan(locked_weight_g=weight_g, frame=frame)
        return jsonify({"status": "success", "scan": payload})

    return app


def start_web_server(main_app: Any) -> None:
    """Start Flask web server on configured host and port."""
    app = create_app(main_app)
    app.run(
        host=main_app.config.web.host,
        port=main_app.config.web.port,
        threaded=True,
        use_reloader=False,
    )
