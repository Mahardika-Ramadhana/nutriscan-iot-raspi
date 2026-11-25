from dataclasses import replace
import io
from pathlib import Path
import sys
import types
import numpy as np
from PIL import Image

from nutriscan.api.web_server import (
    build_status_payload,
    create_app,
    encode_jpeg,
    format_scans_csv,
    validate_bearer_token,
)
from nutriscan.core.config import load_config
from nutriscan.core.state_machine import NutriScanStateMachine
from nutriscan.db.database import Database


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_validate_bearer_token():
    assert validate_bearer_token("Bearer secret-xyz", "secret-xyz") is True
    assert validate_bearer_token("Bearer wrong", "secret-xyz") is False
    assert validate_bearer_token("Basic secret-xyz", "secret-xyz") is False
    assert validate_bearer_token("Bearer anything", "") is False
    assert validate_bearer_token(None, "secret-xyz") is False


def test_encode_jpeg_and_status_and_csv(tmp_path: Path):
    assert encode_jpeg(None) == b""
    frame = np.full((32, 48, 3), 120, dtype=np.uint8)
    jpg_bytes = encode_jpeg(frame)
    assert jpg_bytes.startswith(b"\xff\xd8")

    cfg = load_config(_repo_root() / "data" / "config.example.json")
    cfg = replace(cfg, storage=replace(cfg.storage, db_path=tmp_path / "test.db"))
    db = Database(cfg.storage.db_path)
    db.init_db()
    db.save_scan(
        {
            "scan_id": "scan-99",
            "device_id": cfg.device.id,
            "location": cfg.device.location,
            "timestamp_utc": "2026-11-25T10:00:00Z",
            "berat_total_g": 280.0,
            "jumlah_deteksi": 1,
            "komponen": {"Nasi": {"estimasi_gram": 280.0}},
            "total_gizi": {"energi_kcal": 504.0, "protein_g": 8.4, "lemak_g": 0.84, "karbohidrat_g": 111.44},
        },
        sync_status="local_only",
    )

    sm = NutriScanStateMachine()
    sm.update_weight(280.0, 0.5)
    dummy_app = type("DummyApp", (), {"config": cfg, "sm": sm, "db": db})()

    status = build_status_payload(dummy_app)
    assert status["device_id"] == cfg.device.id
    assert status["current_weight_g"] == 280.0
    assert status["queue_stats"]["total"] == 1

    csv_text = format_scans_csv(db.get_all_scans())
    assert "scan-99" in csv_text
    assert "Nasi (280.0g)" in csv_text
    assert "504.0" in csv_text


def test_create_app_routes_and_auth_with_fake_flask(tmp_path: Path, monkeypatch):
    class FakeRequest:
        headers = {}
        args = types.SimpleNamespace(get=lambda _k, default=20, type=int: default)
        files = {}
        form = {}

    fake_req = FakeRequest()

    class FakeFlaskApp:
        def __init__(self, _name, template_folder=None):
            self.routes = {}

        def route(self, rule, methods=("GET",)):
            def deco(fn):
                self.routes[(rule, tuple(methods))] = fn
                return fn

            return deco

    fake_flask = types.SimpleNamespace(
        Flask=FakeFlaskApp,
        Response=lambda data, mimetype=None, headers=None: (data, mimetype, headers),
        jsonify=lambda data: data,
        render_template=lambda tpl, **kw: {"template": tpl, **kw},
        request=fake_req,
    )
    monkeypatch.setitem(sys.modules, "flask", fake_flask)

    cfg = load_config(_repo_root() / "data" / "config.example.json")
    cfg = replace(cfg, storage=replace(cfg.storage, db_path=tmp_path / "web.db"))
    db = Database(cfg.storage.db_path)
    db.init_db()
    sm = NutriScanStateMachine()

    dummy_app = types.SimpleNamespace(
        config=cfg,
        sm=sm,
        db=db,
        last_drawn_frame_bgr=np.zeros((16, 16, 3), dtype=np.uint8),
        last_raw_frame_bgr=np.zeros((16, 16, 3), dtype=np.uint8),
        trigger_manual_scan=lambda: True,
        tare_scale=lambda: 1234.0,
        cloud_sync=types.SimpleNamespace(sync_pending=lambda: {"status": "local_only", "sent": 0, "failed": 0}),
        process_scan=lambda locked_weight_g, frame: {"scan_id": "remote-1", "berat_total_g": locked_weight_g},
    )

    flask_app = create_app(dummy_app)

    # Read endpoints
    assert flask_app.routes[("/", ("GET",))]()[ "template"] == "index.html"
    assert flask_app.routes[("/api/status", ("GET",))]()[ "device_id"] == cfg.device.id
    assert flask_app.routes[("/api/image", ("GET",))]()[0].startswith(b"\xff\xd8")
    assert flask_app.routes[("/api/image_raw", ("GET",))]()[0].startswith(b"\xff\xd8")
    assert flask_app.routes[("/api/scans", ("GET",))]() == []
    assert "ScanUUID" in flask_app.routes[("/api/export_csv", ("GET",))]()[0]

    # Missing API key env -> 503
    monkeypatch.delenv(cfg.web.api_key_env, raising=False)
    body, code = flask_app.routes[("/api/tare", ("POST",))]()
    assert code == 503

    # Wrong token -> 401
    monkeypatch.setenv(cfg.web.api_key_env, "super-secret")
    fake_req.headers = {"Authorization": "Bearer wrong"}
    body, code = flask_app.routes[("/api/tare", ("POST",))]()
    assert code == 401

    # Valid token -> succeeds
    fake_req.headers = {"Authorization": "Bearer super-secret"}
    assert flask_app.routes[("/api/tare", ("POST",))]()[ "tare_offset"] == 1234.0
    assert flask_app.routes[("/api/trigger", ("POST",))]()[ "status"] == "success"
    assert flask_app.routes[("/api/sync_cloud", ("POST",))]()[ "status"] == "local_only"

    # Remote snapshot upload
    img_buf = io.BytesIO()
    Image.new("RGB", (20, 20), color=(0, 255, 0)).save(img_buf, format="JPEG")
    img_buf.seek(0)
    fake_req.files = {"image": img_buf}
    fake_req.form = {"weight": "333.5"}
    snap_res = flask_app.routes[("/api/remote_snapshot", ("POST",))]()
    assert snap_res["scan"]["berat_total_g"] == 333.5

